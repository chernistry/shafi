from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]
JsonList = list[JsonDict]
_MISS_PACK_FAMILIES = ("explicit_page", "title_page", "same_doc", "mixed_doc", "ocr_risk")

_PAGE2_TERMS = ("page 2", "second page")
_TITLE_TERMS = ("title page", "cover page", "caption", "header", "first page", "page 1")
_PARTY_TERMS = ("party", "parties", "claimant", "claimants", "applicant", "respondent", "judge", "judges")
_ARTICLE_TERMS = ("article", "schedule", "definition", "definitions")
_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ARB|ENF|TCD)\s+\d+/\d+\b", re.IGNORECASE)


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_raw_results(path: Path) -> dict[str, JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON list in {path}")
    out: dict[str, JsonDict] = {}
    for raw in cast("list[object]", obj):
        if not isinstance(raw, dict):
            continue
        item = cast("JsonDict", raw)
        case_obj = item.get("case")
        if not isinstance(case_obj, dict):
            continue
        case = cast("JsonDict", case_obj)
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = item
    return out


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in cast("list[object]", value) if (text := str(item).strip())]


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _page_doc(page_id: str) -> str:
    if "_" not in page_id:
        return page_id
    return page_id.rsplit("_", 1)[0]


def _target_doc_family(gold_pages: list[str], *, ocr_risk: bool) -> str:
    if ocr_risk:
        return "ocr_risk"
    gold_docs = {_page_doc(page_id) for page_id in gold_pages}
    return "multi_doc" if len(gold_docs) >= 2 else "single_doc"


def _infer_doc_family(*, question: str, failure_class: str, support_shape_class: str) -> str:
    q = re.sub(r"\s+", " ", question).strip().lower()
    ref_count = len(_CASE_REF_RE.findall(question))
    if any(term in q for term in _PAGE2_TERMS):
        return "explicit_page_two"
    if any(term in q for term in _TITLE_TERMS):
        if ref_count >= 2 and any(term in q for term in _PARTY_TERMS):
            return "comparison_title_party"
        return "single_doc_title_cover"
    if ref_count >= 2 and any(term in q for term in _PARTY_TERMS):
        return "comparison_party_metadata"
    if support_shape_class == "named_metadata":
        return "named_metadata_single_doc"
    if any(term in q for term in _ARTICLE_TERMS):
        return "statute_article_metadata"
    if failure_class == "support_undercoverage":
        return "generic_support_undercoverage"
    return "other"


def _failure_stage(
    *,
    gold_pages: list[str],
    retrieved_pages: list[str],
    context_pages: list[str],
    used_pages: list[str],
) -> str:
    gold_set = set(gold_pages)
    if gold_set.intersection(used_pages):
        return "retained_to_used"
    if gold_set.intersection(context_pages):
        return "lost_after_context"
    if gold_set.intersection(retrieved_pages):
        return "lost_before_context"

    gold_docs = {_page_doc(page_id) for page_id in gold_pages}
    used_docs = {_page_doc(page_id) for page_id in used_pages}
    context_docs = {_page_doc(page_id) for page_id in context_pages}
    retrieved_docs = {_page_doc(page_id) for page_id in retrieved_pages}

    if gold_docs.intersection(used_docs):
        return "wrong_page_used_same_doc"
    if gold_docs.intersection(context_docs):
        return "wrong_page_context_same_doc"
    if gold_docs.intersection(retrieved_docs):
        return "wrong_page_retrieved_same_doc"
    return "gold_doc_missing_from_retrieval"


def build_page_trace_ledger(
    *,
    raw_results_path: Path,
    benchmark_path: Path,
    scaffold_path: Path,
    qids: set[str] | None = None,
) -> JsonDict:
    raw_results = _load_raw_results(raw_results_path)
    benchmark = _load_json(benchmark_path)
    scaffold = _load_json(scaffold_path)

    benchmark_cases = _coerce_dict_list(benchmark.get("cases"))
    scaffold_records = {
        str(record.get("question_id") or "").strip(): record
        for record in _coerce_dict_list(scaffold.get("records"))
        if str(record.get("question_id") or "").strip()
    }

    records: JsonList = []
    failure_stage_counts: dict[str, int] = {}
    stage_examples: dict[str, list[str]] = {}
    trusted_case_count = 0
    false_positive_case_count = 0
    gold_in_retrieved_count = 0
    gold_in_reranked_count = 0
    gold_in_used_count = 0

    for case in benchmark_cases:
        qid = str(case.get("question_id") or "").strip()
        if not qid or (qids is not None and qid not in qids):
            continue
        raw = raw_results.get(qid)
        if raw is None:
            continue
        gold_pages = _coerce_str_list(case.get("gold_page_ids"))
        if not gold_pages:
            continue

        telemetry = cast("JsonDict", raw.get("telemetry") or {})
        retrieved_pages = _coerce_str_list(telemetry.get("retrieved_page_ids"))
        context_pages = _coerce_str_list(telemetry.get("context_page_ids"))
        used_pages = _coerce_str_list(telemetry.get("used_page_ids"))
        false_positive_pages = [page_id for page_id in used_pages if page_id not in set(gold_pages)]
        record = scaffold_records.get(qid, {})
        question = str(record.get("question") or cast("JsonDict", raw.get("case") or {}).get("question") or "").strip()
        failure_class = str(record.get("failure_class") or "").strip()
        support_shape_class = str(record.get("support_shape_class") or "").strip()
        route_family = str(record.get("route_family") or "").strip()
        ocr_risk = bool(case.get("ocr_risk") or record.get("ocr_risk"))
        gold_in_retrieved = bool(set(gold_pages).intersection(retrieved_pages))
        gold_in_reranked = bool(set(gold_pages).intersection(context_pages))
        gold_in_used = bool(set(gold_pages).intersection(used_pages))
        failure_stage = _failure_stage(
            gold_pages=gold_pages,
            retrieved_pages=retrieved_pages,
            context_pages=context_pages,
            used_pages=used_pages,
        )

        if str(case.get("trust_tier") or "").strip().lower() == "trusted":
            trusted_case_count += 1
        if gold_in_retrieved:
            gold_in_retrieved_count += 1
        if gold_in_reranked:
            gold_in_reranked_count += 1
        if gold_in_used:
            gold_in_used_count += 1
        if false_positive_pages:
            false_positive_case_count += 1
        failure_stage_counts[failure_stage] = failure_stage_counts.get(failure_stage, 0) + 1
        stage_examples.setdefault(failure_stage, [])
        if len(stage_examples[failure_stage]) < 5:
            stage_examples[failure_stage].append(qid)

        records.append(
            {
                "qid": qid,
                "question": question,
                "gold_pages": gold_pages,
                "gold_in_retrieved": gold_in_retrieved,
                "gold_in_reranked": gold_in_reranked,
                "gold_in_used": gold_in_used,
                "retrieved_pages": retrieved_pages,
                "context_pages": context_pages,
                "used_pages": used_pages,
                "false_positive_pages": false_positive_pages,
                "doc_family": _infer_doc_family(
                    question=question,
                    failure_class=failure_class,
                    support_shape_class=support_shape_class,
                ),
                "target_doc_ids": sorted({_page_doc(page_id) for page_id in gold_pages}),
                "target_doc_family": _target_doc_family(gold_pages, ocr_risk=ocr_risk),
                "route": route_family,
                "failure_stage": failure_stage,
                "page_budget_overrun": max(0, len(used_pages) - max(1, len(gold_pages))),
                "wrong_document_risk": bool(case.get("wrong_document_risk")),
                "ocr_risk": ocr_risk,
                "trust_tier": str(case.get("trust_tier") or "").strip() or "unknown",
            }
        )

    cases_scored = len(records)
    explained_count = sum(1 for record in records if str(record.get("failure_stage") or "").strip())
    return {
        "source_raw_results": str(raw_results_path),
        "source_benchmark": str(benchmark_path),
        "source_scaffold": str(scaffold_path),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "summary": {
            "cases_scored": cases_scored,
            "trusted_case_count": trusted_case_count,
            "gold_in_retrieved_count": gold_in_retrieved_count,
            "gold_in_reranked_count": gold_in_reranked_count,
            "gold_in_used_count": gold_in_used_count,
            "false_positive_case_count": false_positive_case_count,
            "failure_stage_counts": failure_stage_counts,
            "stage_examples": stage_examples,
            "explained_ratio": float(explained_count / cases_scored) if cases_scored else 0.0,
        },
        "records": records,
    }


def _miss_pack_family(record: JsonDict) -> str:
    if bool(record.get("ocr_risk")):
        return "ocr_risk"
    doc_family = str(record.get("doc_family") or "").strip()
    failure_stage = str(record.get("failure_stage") or "").strip()
    target_doc_family = str(record.get("target_doc_family") or "").strip()
    if doc_family == "explicit_page_two":
        return "explicit_page"
    if doc_family in {"comparison_title_party", "single_doc_title_cover", "named_metadata_single_doc"}:
        return "title_page"
    if bool(record.get("wrong_document_risk")) or target_doc_family == "multi_doc":
        return "mixed_doc"
    if failure_stage in {"wrong_page_used_same_doc", "wrong_page_context_same_doc", "wrong_page_retrieved_same_doc"}:
        return "same_doc"
    return "same_doc"


def build_bounded_miss_pack(*, ledger: JsonDict, max_per_family: int = 8) -> JsonDict:
    records = _coerce_dict_list(ledger.get("records"))
    miss_records = [
        record
        for record in records
        if (not bool(record.get("gold_in_used"))) or _coerce_str_list(record.get("false_positive_pages"))
    ]
    miss_records.sort(
        key=lambda record: (
            0 if str(record.get("trust_tier") or "") == "trusted" else 1,
            -len(_coerce_str_list(record.get("false_positive_pages"))),
            -_coerce_int(record.get("page_budget_overrun")),
            str(record.get("qid") or ""),
        )
    )

    family_counts = {family: 0 for family in _MISS_PACK_FAMILIES}
    selected_counts = {family: 0 for family in _MISS_PACK_FAMILIES}
    cases: JsonList = []
    for record in miss_records:
        family = _miss_pack_family(record)
        family_counts[family] += 1
        if selected_counts[family] >= max_per_family:
            continue
        selected_counts[family] += 1
        cases.append(
            {
                "qid": str(record.get("qid") or ""),
                "question_family": str(record.get("doc_family") or ""),
                "target_doc_family": str(record.get("target_doc_family") or ""),
                "target_doc_ids": _coerce_str_list(record.get("target_doc_ids")),
                "failure_stage": str(record.get("failure_stage") or ""),
                "miss_family": family,
                "route": str(record.get("route") or ""),
                "trust_tier": str(record.get("trust_tier") or ""),
                "wrong_document_risk": bool(record.get("wrong_document_risk")),
                "ocr_risk": bool(record.get("ocr_risk")),
                "gold_pages": _coerce_str_list(record.get("gold_pages")),
                "used_pages": _coerce_str_list(record.get("used_pages")),
                "false_positive_pages": _coerce_str_list(record.get("false_positive_pages")),
            }
        )

    return {
        "source_page_trace_ledger": ledger.get("source_raw_results"),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "summary": {
            "max_per_family": max_per_family,
            "selected_case_count": len(cases),
            "miss_case_count": len(miss_records),
            "family_counts": family_counts,
            "selected_family_counts": selected_counts,
            "selected_qids": [str(case.get("qid") or "") for case in cases],
        },
        "cases": cases,
    }


def _render_miss_pack_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload.get("summary") or {})
    cases = _coerce_dict_list(payload.get("cases"))
    lines = [
        "# Bounded Miss Pack",
        "",
        f"- source_page_trace_ledger: `{payload.get('source_page_trace_ledger')}`",
        f"- submission_policy: `{payload.get('submission_policy')}`",
        "",
        "## Summary",
        "",
        f"- selected_case_count: `{summary.get('selected_case_count')}`",
        f"- miss_case_count: `{summary.get('miss_case_count')}`",
        f"- max_per_family: `{summary.get('max_per_family')}`",
        f"- selected_qids: `{', '.join(_coerce_str_list(summary.get('selected_qids')))}`",
        "",
        "## Families",
        "",
    ]
    family_counts = cast("dict[str, object]", summary.get("family_counts") or {})
    selected_counts = cast("dict[str, object]", summary.get("selected_family_counts") or {})
    for family in _MISS_PACK_FAMILIES:
        lines.append(
            f"- {family}: total=`{family_counts.get(family, 0)}` selected=`{selected_counts.get(family, 0)}`"
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| qid | miss_family | question_family | target_doc_family | failure_stage | trust |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("qid") or ""),
                    str(case.get("miss_family") or ""),
                    str(case.get("question_family") or ""),
                    str(case.get("target_doc_family") or ""),
                    str(case.get("failure_stage") or ""),
                    str(case.get("trust_tier") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload.get("summary") or {})
    records = _coerce_dict_list(payload.get("records"))
    lines = [
        "# Page Trace Ledger",
        "",
        f"- source_raw_results: `{payload.get('source_raw_results')}`",
        f"- source_benchmark: `{payload.get('source_benchmark')}`",
        f"- source_scaffold: `{payload.get('source_scaffold')}`",
        f"- submission_policy: `{payload.get('submission_policy')}`",
        "",
        "## Summary",
        "",
        f"- cases_scored: `{summary.get('cases_scored')}`",
        f"- trusted_case_count: `{summary.get('trusted_case_count')}`",
        f"- gold_in_retrieved_count: `{summary.get('gold_in_retrieved_count')}`",
        f"- gold_in_reranked_count: `{summary.get('gold_in_reranked_count')}`",
        f"- gold_in_used_count: `{summary.get('gold_in_used_count')}`",
        f"- false_positive_case_count: `{summary.get('false_positive_case_count')}`",
        f"- explained_ratio: `{cast('float', summary.get('explained_ratio') or 0.0):.3f}`",
        "",
        "## Failure Stages",
        "",
    ]
    failure_stage_counts = cast("dict[str, object]", summary.get("failure_stage_counts") or {})
    for stage in sorted(failure_stage_counts):
        lines.append(f"- {stage}: `{failure_stage_counts[stage]}`")

    lines.extend(
        [
            "",
            "## Records",
            "",
            "| qid | trust | route | family | failure_stage | gold_in_retrieved | gold_in_reranked | gold_in_used | overrun |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("qid") or "")[:8],
                    str(record.get("trust_tier") or ""),
                    str(record.get("route") or ""),
                    str(record.get("doc_family") or ""),
                    str(record.get("failure_stage") or ""),
                    str(record.get("gold_in_retrieved") or False),
                    str(record.get("gold_in_reranked") or False),
                    str(record.get("gold_in_used") or False),
                    str(record.get("page_budget_overrun") or 0),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-QID page-trace ledger for benchmark grounding failures.")
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--qids-file", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--miss-pack-json", type=Path, default=None)
    parser.add_argument("--miss-pack-md", type=Path, default=None)
    parser.add_argument("--miss-pack-max-per-family", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qids: set[str] | None = None
    if args.qids_file is not None and args.qids_file.exists():
        qids = {
            line.strip()
            for line in args.qids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    payload = build_page_trace_ledger(
        raw_results_path=args.raw_results,
        benchmark_path=args.benchmark,
        scaffold_path=args.scaffold,
        qids=qids,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    if args.miss_pack_json is not None and args.miss_pack_md is not None:
        miss_pack = build_bounded_miss_pack(ledger=payload, max_per_family=args.miss_pack_max_per_family)
        args.miss_pack_json.parent.mkdir(parents=True, exist_ok=True)
        args.miss_pack_md.parent.mkdir(parents=True, exist_ok=True)
        args.miss_pack_json.write_text(json.dumps(miss_pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.miss_pack_md.write_text(_render_miss_pack_markdown(miss_pack) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
