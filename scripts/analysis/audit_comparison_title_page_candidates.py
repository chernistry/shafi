from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

try:
    from run_experiment_gate import _page_id_parts

    from shafi.core.pipeline import (
        _DIFC_CASE_ID_RE,
        RAGPipelineBuilder,
        _is_case_issue_date_name_compare_query,
        _is_common_judge_compare_query,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_experiment_gate import _page_id_parts
    from src.shafi.core.pipeline import (  # type: ignore[reportMissingImports]
        _DIFC_CASE_ID_RE,
        RAGPipelineBuilder,
        _is_case_issue_date_name_compare_query,
        _is_common_judge_compare_query,
    )

JsonDict = dict[str, Any]


def _is_case_monetary_claim_compare_query(question: str, *, answer_type: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    if answer_type.strip().lower() != "name":
        return False
    if len(_DIFC_CASE_ID_RE.findall(question or "")) != 2:
        return False
    if "claim" not in q:
        return False
    return any(
        phrase in q
        for phrase in (
            "higher monetary claim",
            "lower monetary claim",
            "greater monetary claim",
            "largest monetary claim",
            "smallest monetary claim",
            "higher claim",
            "lower claim",
        )
    )


def _is_case_party_overlap_compare_query(question: str, *, answer_type: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    if answer_type.strip().lower() not in {"boolean", "name"}:
        return False
    if len(_DIFC_CASE_ID_RE.findall(question or "")) < 2:
        return False
    if any(
        phrase in q
        for phrase in (
            "same legal",
            "same parties",
            "same party",
            "same entities",
            "main party common to both",
            "main party to both",
            "appeared in both",
            "appears in both",
            "appears as a main party in both",
            "named as a main party in both",
            "as parties",
        )
    ):
        return True
    has_party_subject = any(
        token in q for token in ("party", "parties", "claimant", "defendant", "entity", "entities", "individual")
    )
    has_overlap_signal = any(token in q for token in ("common", "same", "appeared", "appears", "named", "both"))
    return has_party_subject and has_overlap_signal


@dataclass(frozen=True)
class SourceSignal:
    label: str
    answer_changed: bool
    page1_doc_hits: int
    page1_doc_ids: list[str]
    used_page_ids: list[str]


@dataclass(frozen=True)
class ComparisonCandidate:
    question_id: str
    question: str
    answer_type: str
    compare_kind: str
    refs: list[str]
    support_shape_class: str
    title_page_signal: bool
    minimal_required_page1_count: int
    resolved_doc_count: int
    baseline_used_page1_doc_hits: int
    baseline_context_page1_doc_hits: int
    baseline_retrieved_page1_doc_hits: int
    missing_used_page1_doc_ids: list[str]
    missing_used_page1_titles: list[str]
    baseline_used_page_ids: list[str]
    baseline_context_page_ids: list[str]
    baseline_retrieved_page_ids: list[str]
    source_signals: list[SourceSignal]
    opportunity_score: float
    recommendation: str
    notes: str
    submission_policy: str = "NO_SUBMIT_WITHOUT_USER_APPROVAL"


def _load_json(path: Path) -> JsonDict | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_results(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    out: dict[str, JsonDict] = {}
    for row_obj in payload:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        case = cast("JsonDict", row.get("case")) if isinstance(row.get("case"), dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _page_map(page_ids: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for page_id in page_ids:
        parts = _page_id_parts(page_id)
        if parts is None:
            continue
        doc_id, page = parts
        out.setdefault(doc_id, set()).add(page)
    return out


def _coerce_page_ids(row: JsonDict, key: str) -> list[str]:
    telemetry_obj = row.get("telemetry")
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    value = telemetry.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _comparison_kind(*, question: str, answer_type: str) -> str | None:
    if _is_common_judge_compare_query(question):
        return "judge_overlap"
    if _is_case_issue_date_name_compare_query(question, answer_type=answer_type):
        return "issue_date_compare"
    if _is_case_monetary_claim_compare_query(question, answer_type=answer_type):
        return "monetary_claim_compare"
    if _is_case_party_overlap_compare_query(question, answer_type=answer_type):
        return "party_overlap"
    return None


def _comparison_refs(question: str) -> list[str]:
    refs = RAGPipelineBuilder._paired_support_question_refs(question)
    if len(refs) >= 2:
        return refs[:2]
    case_refs: list[str] = []
    seen: set[str] = set()
    for prefix, number, year in _DIFC_CASE_ID_RE.findall(question or ""):
        ref = f"{prefix.upper()} {int(number):03d}/{year}"
        if ref in seen:
            continue
        seen.add(ref)
        case_refs.append(ref)
    return case_refs[:2]


def _minimal_required_page1_count(record: JsonDict) -> int:
    value = record.get("minimal_required_support_pages")
    if not isinstance(value, list):
        return 0
    count = 0
    for item in value:
        parts = _page_id_parts(str(item))
        if parts is not None and parts[1] == 1:
            count += 1
    return count


def _title_page_signal(record: JsonDict) -> bool:
    note = str(record.get("notes") or "").casefold()
    question = str(record.get("question") or "").casefold()
    return any(
        token in note or token in question
        for token in ("title page", "cover page", "caption", "first page", "page 1", "title/cover")
    )


def _source_signal(
    *,
    label: str,
    baseline_row: JsonDict,
    source_row: JsonDict | None,
    resolved_doc_ids: list[str],
) -> SourceSignal:
    if source_row is None:
        return SourceSignal(
            label=label,
            answer_changed=False,
            page1_doc_hits=0,
            page1_doc_ids=[],
            used_page_ids=[],
        )
    baseline_answer = str(baseline_row.get("answer_text") or "").strip()
    source_answer = str(source_row.get("answer_text") or "").strip()
    source_used = _coerce_page_ids(source_row, "used_page_ids")
    source_map = _page_map(source_used)
    page1_doc_ids = [doc_id for doc_id in resolved_doc_ids if 1 in source_map.get(doc_id, set())]
    return SourceSignal(
        label=label,
        answer_changed=baseline_answer != source_answer,
        page1_doc_hits=len(page1_doc_ids),
        page1_doc_ids=page1_doc_ids,
        used_page_ids=source_used,
    )


def _recommendation(
    *,
    missing_used_doc_count: int,
    baseline_retrieved_page1_doc_hits: int,
    baseline_context_page1_doc_hits: int,
    source_signals: list[SourceSignal],
    compare_kind: str,
    title_page_signal: bool,
) -> str:
    non_drifting_sources = [
        signal for signal in source_signals if not signal.answer_changed and signal.page1_doc_hits >= max(1, missing_used_doc_count)
    ]
    if (
        missing_used_doc_count >= 1
        and compare_kind in {"party_overlap", "judge_overlap", "issue_date_compare", "monetary_claim_compare"}
        and (title_page_signal or baseline_retrieved_page1_doc_hits >= 1 or baseline_context_page1_doc_hits >= 1)
        and non_drifting_sources
    ):
        return "PROMISING"
    if missing_used_doc_count >= 1 and (baseline_retrieved_page1_doc_hits >= 1 or title_page_signal):
        return "WATCH"
    return "REPORT_ONLY"


def _score(
    *,
    compare_kind: str,
    missing_used_doc_count: int,
    baseline_retrieved_page1_doc_hits: int,
    baseline_context_page1_doc_hits: int,
    title_page_signal: bool,
    non_drifting_source_hits: int,
    minimal_required_page1_count: int,
) -> float:
    kind_weight = {
        "party_overlap": 4.0,
        "judge_overlap": 3.5,
        "issue_date_compare": 2.5,
        "monetary_claim_compare": 2.0,
    }.get(compare_kind, 1.0)
    return (
        missing_used_doc_count * 5.0
        + baseline_retrieved_page1_doc_hits * 1.5
        + baseline_context_page1_doc_hits * 1.0
        + minimal_required_page1_count * 1.0
        + (2.0 if title_page_signal else 0.0)
        + non_drifting_source_hits * 3.0
        + kind_weight
    )


def _render_markdown(
    *,
    baseline_label: str,
    source_labels: list[str],
    candidates: list[ComparisonCandidate],
) -> str:
    lines = [
        "# Comparison Title-Page Candidate Audit",
        "",
        f"- baseline_label: `{baseline_label}`",
        f"- source_labels: `{', '.join(source_labels) if source_labels else 'none'}`",
        f"- candidates: `{len(candidates)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | QID | Kind | Recommendation | Missing Used P1 Docs | Baseline Retrieved P1 Docs | Baseline Context P1 Docs | Source Rescue |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(candidates, start=1):
        source_rescue = ", ".join(
            f"{signal.label}:{signal.page1_doc_hits}"
            for signal in row.source_signals
            if not signal.answer_changed and signal.page1_doc_hits > 0
        ) or "n/a"
        lines.append(
            "| "
            f"{index} | `{row.question_id}` | `{row.compare_kind}` | `{row.recommendation}` | "
            f"{len(row.missing_used_page1_doc_ids)} | {row.baseline_retrieved_page1_doc_hits} | "
            f"{row.baseline_context_page1_doc_hits} | {source_rescue} |"
        )

    if candidates:
        lines.append("")
        lines.append("## Top candidates")
        lines.append("")
        for row in candidates[:10]:
            lines.extend(
                [
                    f"### `{row.question_id}`",
                    "",
                    f"- question: {row.question}",
                    f"- compare_kind: `{row.compare_kind}`",
                    f"- refs: {', '.join(f'`{ref}`' for ref in row.refs) or 'n/a'}",
                    f"- support_shape_class: `{row.support_shape_class}`",
                    f"- baseline_used_page_ids: `{row.baseline_used_page_ids}`",
                    f"- missing_used_page1_titles: `{row.missing_used_page1_titles}`",
                    f"- title_page_signal: `{row.title_page_signal}`",
                    f"- minimal_required_page1_count: `{row.minimal_required_page1_count}`",
                    f"- source_signals: `{[asdict(signal) for signal in row.source_signals]}`",
                    f"- notes: {row.notes or 'n/a'}",
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit comparison/title-page candidate opportunities from baseline and source raw-results.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--truth-audit", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--source-raw-results", action="append", default=[], help="label=/absolute/path/to/raw_results.json")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-seed-qids", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_obj = _load_json(args.questions.resolve())
    if not isinstance(questions_obj, list):
        raise ValueError(f"Expected JSON list in {args.questions}")
    truth_obj = _load_json(args.truth_audit.resolve())
    if not isinstance(truth_obj, dict):
        raise ValueError(f"Expected JSON object in {args.truth_audit}")
    records_obj = truth_obj.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Expected records array in {args.truth_audit}")

    baseline_raw = _load_raw_results(args.baseline_raw_results.resolve())
    source_payloads: list[tuple[str, dict[str, JsonDict]]] = []
    for item in args.source_raw_results:
        label, sep, raw_path = item.partition("=")
        if not sep or not label.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --source-raw-results entry: {item}")
        source_payloads.append((label.strip(), _load_raw_results(Path(raw_path).expanduser().resolve())))

    question_map: dict[str, JsonDict] = {}
    for row_obj in questions_obj:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("id") or row.get("question_id") or "").strip()
        if qid:
            question_map[qid] = row

    candidates: list[ComparisonCandidate] = []
    for row_obj in records_obj:
        if not isinstance(row_obj, dict):
            continue
        record = cast("JsonDict", row_obj)
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        question_row = question_map.get(qid, {})
        question = str(question_row.get("question") or record.get("question") or "").strip()
        answer_type = str(question_row.get("answer_type") or record.get("answer_type") or "").strip()
        compare_kind = _comparison_kind(question=question, answer_type=answer_type)
        if compare_kind is None:
            continue

        baseline_row = baseline_raw.get(qid)
        if baseline_row is None:
            continue
        resolved_obj = record.get("resolved_doc_titles")
        resolved_doc_titles = cast("dict[str, str]", resolved_obj) if isinstance(resolved_obj, dict) else {}
        if len(resolved_doc_titles) < 2:
            continue

        baseline_used = _coerce_page_ids(baseline_row, "used_page_ids")
        baseline_context = _coerce_page_ids(baseline_row, "context_page_ids")
        baseline_retrieved = _coerce_page_ids(baseline_row, "retrieved_page_ids")
        used_map = _page_map(baseline_used)
        context_map = _page_map(baseline_context)
        retrieved_map = _page_map(baseline_retrieved)

        resolved_doc_ids = list(resolved_doc_titles.keys())
        baseline_used_page1_doc_hits = sum(1 for doc_id in resolved_doc_ids if 1 in used_map.get(doc_id, set()))
        baseline_context_page1_doc_hits = sum(1 for doc_id in resolved_doc_ids if 1 in context_map.get(doc_id, set()))
        baseline_retrieved_page1_doc_hits = sum(1 for doc_id in resolved_doc_ids if 1 in retrieved_map.get(doc_id, set()))
        missing_doc_ids = [doc_id for doc_id in resolved_doc_ids if 1 not in used_map.get(doc_id, set())]
        missing_titles = [resolved_doc_titles[doc_id] for doc_id in missing_doc_ids]
        source_signals = [
            _source_signal(
                label=label,
                baseline_row=baseline_row,
                source_row=payload.get(qid),
                resolved_doc_ids=resolved_doc_ids,
            )
            for label, payload in source_payloads
        ]
        non_drifting_source_hits = sum(
            1 for signal in source_signals if not signal.answer_changed and signal.page1_doc_hits >= max(1, len(missing_doc_ids))
        )
        title_signal = _title_page_signal(record)
        recommendation = _recommendation(
            missing_used_doc_count=len(missing_doc_ids),
            baseline_retrieved_page1_doc_hits=baseline_retrieved_page1_doc_hits,
            baseline_context_page1_doc_hits=baseline_context_page1_doc_hits,
            source_signals=source_signals,
            compare_kind=compare_kind,
            title_page_signal=title_signal,
        )
        candidates.append(
            ComparisonCandidate(
                question_id=qid,
                question=question,
                answer_type=answer_type,
                compare_kind=compare_kind,
                refs=_comparison_refs(question),
                support_shape_class=str(record.get("support_shape_class") or ""),
                title_page_signal=title_signal,
                minimal_required_page1_count=_minimal_required_page1_count(record),
                resolved_doc_count=len(resolved_doc_titles),
                baseline_used_page1_doc_hits=baseline_used_page1_doc_hits,
                baseline_context_page1_doc_hits=baseline_context_page1_doc_hits,
                baseline_retrieved_page1_doc_hits=baseline_retrieved_page1_doc_hits,
                missing_used_page1_doc_ids=missing_doc_ids,
                missing_used_page1_titles=missing_titles,
                baseline_used_page_ids=baseline_used,
                baseline_context_page_ids=baseline_context,
                baseline_retrieved_page_ids=baseline_retrieved,
                source_signals=source_signals,
                opportunity_score=_score(
                    compare_kind=compare_kind,
                    missing_used_doc_count=len(missing_doc_ids),
                    baseline_retrieved_page1_doc_hits=baseline_retrieved_page1_doc_hits,
                    baseline_context_page1_doc_hits=baseline_context_page1_doc_hits,
                    title_page_signal=title_signal,
                    non_drifting_source_hits=non_drifting_source_hits,
                    minimal_required_page1_count=_minimal_required_page1_count(record),
                ),
                recommendation=recommendation,
                notes=str(record.get("notes") or "").strip(),
            )
        )

    ranked = sorted(
        candidates,
        key=lambda row: (
            {"PROMISING": 2, "WATCH": 1}.get(row.recommendation, 0),
            row.opportunity_score,
            row.question_id,
        ),
        reverse=True,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "baseline_label": args.baseline_label,
                "source_labels": [label for label, _ in source_payloads],
                "records": [asdict(row) for row in ranked],
                "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(
        _render_markdown(
            baseline_label=args.baseline_label,
            source_labels=[label for label, _ in source_payloads],
            candidates=ranked,
        ),
        encoding="utf-8",
    )
    if args.out_seed_qids:
        strong_qids = [row.question_id for row in ranked if row.recommendation == "PROMISING"]
        args.out_seed_qids.parent.mkdir(parents=True, exist_ok=True)
        args.out_seed_qids.write_text(json.dumps(strong_qids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
