# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

try:
    from score_page_benchmark import CaseScore, _load_benchmark, _score_case
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest/module import
    from scripts.score_page_benchmark import CaseScore, _load_benchmark, _score_case

JsonDict = dict[str, object]


@dataclass(frozen=True)
class BenchmarkSummary:
    cases: int
    page_f_beta: float


@dataclass(frozen=True)
class SeedCaseDelta:
    question_id: str
    gold_page_ids: list[str]
    baseline_used_page_ids: list[str]
    candidate_used_page_ids: list[str]
    baseline_context_page_ids: list[str]
    candidate_context_page_ids: list[str]
    baseline_used_hit: bool
    candidate_used_hit: bool
    baseline_context_hit: bool
    candidate_context_hit: bool
    candidate_used_equivalent_hit: bool
    candidate_context_equivalent_hit: bool


@dataclass(frozen=True)
class ExperimentRecord:
    timestamp_utc: str
    label: str
    baseline_label: str
    recommendation: str
    answer_changed_count: int
    retrieval_page_projection_changed_count: int
    benchmark_all_baseline: float
    benchmark_all_candidate: float
    benchmark_trusted_baseline: float
    benchmark_trusted_candidate: float
    baseline_page_p95: int | None
    candidate_page_p95: int | None
    improved_seed_cases: list[str]
    regressed_seed_cases: list[str]
    notes: list[str]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_json_list(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [cast("JsonDict", item) for item in obj if isinstance(item, dict)]


def _json_stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _submission_answers_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Submission at {path} is missing 'answers'")
    out: dict[str, JsonDict] = {}
    for raw in answers_obj:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("question_id") or "").strip()
        if qid:
            out[qid] = cast("JsonDict", raw)
    return out


def _raw_results_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json_list(path)
    out: dict[str, JsonDict] = {}
    for raw in payload:
        case_obj = raw.get("case")
        case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _eval_case_from_raw(raw_case: JsonDict) -> JsonDict:
    case_obj = raw_case.get("case")
    case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
    return {
        "question_id": str(case.get("case_id") or case.get("question_id") or "").strip(),
        "answer": raw_case.get("answer_text"),
        "telemetry": raw_case.get("telemetry") if isinstance(raw_case.get("telemetry"), dict) else {},
    }


def _score_benchmark(raw_results_path: Path, benchmark_path: Path) -> tuple[BenchmarkSummary, BenchmarkSummary]:
    benchmark_cases = _load_benchmark(benchmark_path)
    raw_results = _raw_results_by_id(raw_results_path)
    scores: list[CaseScore] = []
    for benchmark_case in benchmark_cases:
        eval_case = raw_results.get(benchmark_case.question_id)
        scores.append(_score_case(benchmark_case, _eval_case_from_raw(eval_case) if eval_case is not None else None, beta=2.5))
    all_summary = _summarize_scores(scores)
    trusted_summary = _summarize_scores([score for score in scores if score.trust_tier == "trusted"])
    return all_summary, trusted_summary


def _summarize_scores(scores: list[CaseScore]) -> BenchmarkSummary:
    if not scores:
        return BenchmarkSummary(cases=0, page_f_beta=0.0)
    return BenchmarkSummary(
        cases=len(scores),
        page_f_beta=sum(score.f_beta for score in scores) / len(scores),
    )


def _answer_changed_count(baseline_submission: dict[str, JsonDict], candidate_submission: dict[str, JsonDict]) -> int:
    count = 0
    for qid, baseline in baseline_submission.items():
        candidate = candidate_submission.get(qid)
        if candidate is None:
            continue
        if _json_stable(baseline.get("answer")) != _json_stable(candidate.get("answer")):
            count += 1
    return count


def _retrieval_projection_changed_count(
    baseline_submission: dict[str, JsonDict],
    candidate_submission: dict[str, JsonDict],
) -> int:
    count = 0
    for qid, baseline in baseline_submission.items():
        candidate = candidate_submission.get(qid)
        if candidate is None:
            continue
        baseline_pages = _retrieved_chunk_pages_projection(baseline)
        candidate_pages = _retrieved_chunk_pages_projection(candidate)
        if _json_stable(baseline_pages) != _json_stable(candidate_pages):
            count += 1
    return count


def _retrieved_chunk_pages_projection(answer_record: JsonDict) -> list[JsonDict]:
    telemetry_obj = answer_record.get("telemetry")
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    retrieval_obj = telemetry.get("retrieval")
    retrieval = cast("JsonDict", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
    pages_obj = retrieval.get("retrieved_chunk_pages")
    if not isinstance(pages_obj, list):
        return []
    return [cast("JsonDict", page) for page in pages_obj if isinstance(page, dict)]


def _page_p95(preflight_path: Path | None) -> int | None:
    if preflight_path is None:
        return None
    payload = _load_json(preflight_path)
    distribution_obj = payload.get("page_count_distribution")
    distribution = cast("JsonDict", distribution_obj) if isinstance(distribution_obj, dict) else {}
    value = distribution.get("p95")
    return int(value) if isinstance(value, int | float) else None


def _scaffold_records_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        return {}
    out: dict[str, JsonDict] = {}
    for raw in records_obj:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("question_id") or raw.get("case_id") or "").strip()
        if qid:
            out[qid] = cast("JsonDict", raw)
    return out


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _page_id_parts(page_id: str) -> tuple[str, int] | None:
    text = str(page_id).strip()
    if "_" not in text:
        return None
    doc_id, _, page_part = text.rpartition("_")
    try:
        return doc_id, int(page_part)
    except ValueError:
        return None


def _normalize_title(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def _page_title_map(record: JsonDict) -> dict[str, str]:
    out: dict[str, str] = {}
    previews_obj = record.get("support_page_previews")
    previews = previews_obj if isinstance(previews_obj, list) else []
    for raw in previews:
        if not isinstance(raw, dict):
            continue
        doc_id = str(raw.get("doc_id") or "").strip()
        page = raw.get("page")
        title = _normalize_title(str(raw.get("doc_title") or "").strip())
        if not doc_id or not isinstance(page, int | float) or not title:
            continue
        out[f"{doc_id}_{int(page)}"] = title

    resolved_obj = record.get("resolved_doc_titles")
    resolved = resolved_obj if isinstance(resolved_obj, dict) else {}
    for page_id in _coerce_str_list(record.get("minimal_required_support_pages")):
        if page_id in out:
            continue
        parts = _page_id_parts(page_id)
        if parts is None:
            continue
        doc_id, _page_num = parts
        title = _normalize_title(resolved.get(doc_id))
        if title:
            out[page_id] = title
    return out


def _record_title_set(record: JsonDict) -> set[str]:
    titles = set(_page_title_map(record).values())
    resolved_obj = record.get("resolved_doc_titles")
    resolved = resolved_obj if isinstance(resolved_obj, dict) else {}
    for raw in resolved.values():
        title = _normalize_title(str(raw))
        if title:
            titles.add(title)
    return titles


def _page_title_equivalent_hit(
    *,
    gold_page_ids: list[str],
    candidate_page_ids: list[str],
    gold_title_map: dict[str, str],
    candidate_title_map: dict[str, str],
    gold_record_titles: set[str],
) -> bool:
    if not gold_page_ids or not candidate_page_ids:
        return False
    gold_specs: set[tuple[int, str]] = set()
    gold_page_numbers: set[int] = set()
    for gold_page_id in gold_page_ids:
        parts = _page_id_parts(gold_page_id)
        if parts is None:
            continue
        _doc_id, page_num = parts
        gold_page_numbers.add(page_num)
        title = gold_title_map.get(gold_page_id)
        if title:
            gold_specs.add((page_num, title))
    if not gold_specs and gold_page_numbers and gold_record_titles:
        gold_specs = {(page_num, title) for page_num in gold_page_numbers for title in gold_record_titles}
    if not gold_specs:
        return False

    for candidate_page_id in candidate_page_ids:
        parts = _page_id_parts(candidate_page_id)
        if parts is None:
            continue
        _doc_id, page_num = parts
        title = candidate_title_map.get(candidate_page_id)
        if title and (page_num, title) in gold_specs:
            return True
    return False


def _seed_case_deltas(
    *,
    baseline_scaffold_path: Path,
    candidate_scaffold_path: Path | None,
    baseline_raw_results_path: Path,
    candidate_raw_results_path: Path,
    seed_qids: list[str],
) -> list[SeedCaseDelta]:
    baseline_scaffold_records = _scaffold_records_by_id(baseline_scaffold_path)
    candidate_scaffold_records = (
        _scaffold_records_by_id(candidate_scaffold_path) if candidate_scaffold_path is not None else {}
    )
    baseline_raw = _raw_results_by_id(baseline_raw_results_path)
    candidate_raw = _raw_results_by_id(candidate_raw_results_path)
    deltas: list[SeedCaseDelta] = []
    for qid in seed_qids:
        baseline_scaffold_record = baseline_scaffold_records.get(qid)
        if baseline_scaffold_record is None:
            continue
        candidate_scaffold_record = candidate_scaffold_records.get(qid, {})
        gold_page_ids = _coerce_str_list(baseline_scaffold_record.get("minimal_required_support_pages"))
        baseline_title_map = _page_title_map(baseline_scaffold_record)
        candidate_title_map = _page_title_map(candidate_scaffold_record)
        gold_record_titles = _record_title_set(baseline_scaffold_record)
        baseline_case = baseline_raw.get(qid, {})
        candidate_case = candidate_raw.get(qid, {})
        baseline_telemetry = cast("JsonDict", baseline_case.get("telemetry")) if isinstance(baseline_case.get("telemetry"), dict) else {}
        candidate_telemetry = cast("JsonDict", candidate_case.get("telemetry")) if isinstance(candidate_case.get("telemetry"), dict) else {}
        baseline_used = _coerce_str_list(baseline_telemetry.get("used_page_ids"))
        candidate_used = _coerce_str_list(candidate_telemetry.get("used_page_ids"))
        baseline_context = _coerce_str_list(baseline_telemetry.get("context_page_ids"))
        candidate_context = _coerce_str_list(candidate_telemetry.get("context_page_ids"))
        gold_set = set(gold_page_ids)
        candidate_used_equivalent_hit = _page_title_equivalent_hit(
            gold_page_ids=gold_page_ids,
            candidate_page_ids=candidate_used,
            gold_title_map=baseline_title_map,
            candidate_title_map=candidate_title_map,
            gold_record_titles=gold_record_titles,
        )
        candidate_context_equivalent_hit = _page_title_equivalent_hit(
            gold_page_ids=gold_page_ids,
            candidate_page_ids=candidate_context,
            gold_title_map=baseline_title_map,
            candidate_title_map=candidate_title_map,
            gold_record_titles=gold_record_titles,
        )
        deltas.append(
            SeedCaseDelta(
                question_id=qid,
                gold_page_ids=gold_page_ids,
                baseline_used_page_ids=baseline_used,
                candidate_used_page_ids=candidate_used,
                baseline_context_page_ids=baseline_context,
                candidate_context_page_ids=candidate_context,
                baseline_used_hit=bool(gold_set.intersection(baseline_used)),
                candidate_used_hit=bool(gold_set.intersection(candidate_used)),
                baseline_context_hit=bool(gold_set.intersection(baseline_context)),
                candidate_context_hit=bool(gold_set.intersection(candidate_context)),
                candidate_used_equivalent_hit=candidate_used_equivalent_hit,
                candidate_context_equivalent_hit=candidate_context_equivalent_hit,
            )
        )
    return deltas


def _recommendation(
    *,
    baseline_trusted: BenchmarkSummary,
    candidate_trusted: BenchmarkSummary,
    answer_changed_count: int,
    baseline_page_p95: int | None,
    candidate_page_p95: int | None,
) -> tuple[str, list[str]]:
    notes: list[str] = []
    if candidate_trusted.page_f_beta + 1e-9 < baseline_trusted.page_f_beta:
        notes.append("trusted hidden-G benchmark regressed")
        return "NO_SUBMIT", notes
    if baseline_page_p95 is not None and candidate_page_p95 is not None and candidate_page_p95 > baseline_page_p95 + 1:
        notes.append("page p95 widened beyond safe bound")
        return "NO_SUBMIT", notes
    if answer_changed_count > 0:
        notes.append("answer drift detected; candidate is exploratory only")
        return "EXPERIMENTAL_NO_SUBMIT", notes
    notes.append("trusted benchmark non-inferior and no answer drift detected")
    return "PROMISING", notes


def _render_report(
    *,
    label: str,
    baseline_label: str,
    answer_changed_count: int,
    retrieval_projection_changed_count: int,
    baseline_all: BenchmarkSummary,
    candidate_all: BenchmarkSummary,
    baseline_trusted: BenchmarkSummary,
    candidate_trusted: BenchmarkSummary,
    baseline_page_p95: int | None,
    candidate_page_p95: int | None,
    seed_deltas: list[SeedCaseDelta],
    recommendation: str,
    notes: list[str],
) -> str:
    improved = [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.candidate_used_hit and not delta.baseline_used_hit)
            or (delta.candidate_context_hit and not delta.baseline_context_hit)
        )
    ]
    equivalent = [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.candidate_used_equivalent_hit and not delta.candidate_used_hit)
            or (delta.candidate_context_equivalent_hit and not delta.candidate_context_hit)
        )
    ]
    regressed = [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.baseline_used_hit and not (delta.candidate_used_hit or delta.candidate_used_equivalent_hit))
            or (delta.baseline_context_hit and not (delta.candidate_context_hit or delta.candidate_context_equivalent_hit))
        )
    ]

    lines = [
        "# Experiment Gate Report",
        "",
        f"- Label: `{label}`",
        f"- Baseline: `{baseline_label}`",
        f"- Recommendation: `{recommendation}`",
        "",
        "## Drift",
        "",
        f"- Answer changes vs baseline: `{answer_changed_count}`",
        f"- Retrieval-page projection changes vs baseline: `{retrieval_projection_changed_count}`",
        "",
        "## Hidden-G Benchmark",
        "",
        f"- Baseline all-cases F_beta(2.5): `{baseline_all.page_f_beta:.4f}`",
        f"- Candidate all-cases F_beta(2.5): `{candidate_all.page_f_beta:.4f}`",
        f"- Baseline trusted F_beta(2.5): `{baseline_trusted.page_f_beta:.4f}`",
        f"- Candidate trusted F_beta(2.5): `{candidate_trusted.page_f_beta:.4f}`",
        "",
        "## Page Shape",
        "",
        f"- Baseline page p95: `{baseline_page_p95 if baseline_page_p95 is not None else 'n/a'}`",
        f"- Candidate page p95: `{candidate_page_p95 if candidate_page_p95 is not None else 'n/a'}`",
        "",
        "## Anchor Seed Slice",
        "",
        f"- Improved seed cases: `{len(improved)}`",
        f"- Equivalent seed cases: `{len(equivalent)}`",
        f"- Regressed seed cases: `{len(regressed)}`",
    ]
    if improved:
        lines.extend([f"- Improved IDs: `{', '.join(improved)}`", ""])
    if equivalent:
        lines.extend([f"- Equivalent IDs: `{', '.join(equivalent)}`", ""])
    if regressed:
        lines.extend([f"- Regressed IDs: `{', '.join(regressed)}`", ""])
    lines.extend(["## Seed Case Details", ""])
    if not seed_deltas:
        lines.append("- None")
    else:
        for delta in seed_deltas:
            lines.extend(
                [
                    f"- `{delta.question_id}`",
                    f"  - gold={delta.gold_page_ids}",
                    f"  - baseline_used={delta.baseline_used_page_ids}",
                    f"  - candidate_used={delta.candidate_used_page_ids}",
                    f"  - baseline_context={delta.baseline_context_page_ids}",
                    f"  - candidate_context={delta.candidate_context_page_ids}",
                    f"  - candidate_used_equivalent={delta.candidate_used_equivalent_hit}",
                    f"  - candidate_context_equivalent={delta.candidate_context_equivalent_hit}",
                ]
            )
    lines.extend(["", "## Notes", ""])
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _report_payload(
    *,
    label: str,
    baseline_label: str,
    answer_changed_count: int,
    retrieval_projection_changed_count: int,
    baseline_all: BenchmarkSummary,
    candidate_all: BenchmarkSummary,
    baseline_trusted: BenchmarkSummary,
    candidate_trusted: BenchmarkSummary,
    baseline_page_p95: int | None,
    candidate_page_p95: int | None,
    seed_deltas: list[SeedCaseDelta],
    recommendation: str,
    notes: list[str],
) -> JsonDict:
    improved = [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.candidate_used_hit and not delta.baseline_used_hit)
            or (delta.candidate_context_hit and not delta.baseline_context_hit)
        )
    ]
    equivalent = [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.candidate_used_equivalent_hit and not delta.candidate_used_hit)
            or (delta.candidate_context_equivalent_hit and not delta.candidate_context_hit)
        )
    ]
    regressed = [
        delta.question_id
        for delta in seed_deltas
        if (
            (delta.baseline_used_hit and not (delta.candidate_used_hit or delta.candidate_used_equivalent_hit))
            or (delta.baseline_context_hit and not (delta.candidate_context_hit or delta.candidate_context_equivalent_hit))
        )
    ]
    return {
        "label": label,
        "baseline_label": baseline_label,
        "recommendation": recommendation,
        "answer_changed_count": answer_changed_count,
        "retrieval_page_projection_changed_count": retrieval_projection_changed_count,
        "benchmark_all_baseline": baseline_all.page_f_beta,
        "benchmark_all_candidate": candidate_all.page_f_beta,
        "benchmark_trusted_baseline": baseline_trusted.page_f_beta,
        "benchmark_trusted_candidate": candidate_trusted.page_f_beta,
        "baseline_page_p95": baseline_page_p95,
        "candidate_page_p95": candidate_page_p95,
        "improved_seed_cases": improved,
        "equivalent_seed_cases": equivalent,
        "regressed_seed_cases": regressed,
        "seed_deltas": [asdict(delta) for delta in seed_deltas],
        "notes": notes,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }


def _append_ledger(path: Path, record: ExperimentRecord) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
            experiments = cast("list[object]", payload["experiments"])
        else:
            experiments = []
    else:
        experiments = []
    experiments.append(asdict(record))
    path.write_text(json.dumps({"experiments": experiments}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _seed_qids(args: argparse.Namespace) -> list[str]:
    qids: list[str] = []
    seen: set[str] = set()
    raw_values: list[str] = []
    raw_values.extend(str(qid).strip() for qid in args.seed_qid)
    seed_qids_file = getattr(args, "seed_qids_file", None)
    if isinstance(seed_qids_file, Path):
        raw_values.extend(seed_qids_file.read_text(encoding="utf-8").splitlines())
    elif not raw_values:
        benchmark_path = getattr(args, "benchmark", None)
        if isinstance(benchmark_path, Path):
            default_seed_qids = benchmark_path.with_name(f"{benchmark_path.stem}_qids.txt")
            if default_seed_qids.exists():
                raw_values.extend(default_seed_qids.read_text(encoding="utf-8").splitlines())
    for raw in raw_values:
        text = str(raw).strip()
        if not text or text.startswith("#") or text in seen:
            continue
        seen.add(text)
        qids.append(text)
    return qids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded experiment gate for branch-vs-baseline RAG artifacts.")
    parser.add_argument("--label", required=True, help="Short label for the candidate experiment.")
    parser.add_argument("--baseline-label", required=True, help="Short label for the baseline artifact.")
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--candidate-raw-results", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--candidate-scaffold", type=Path, default=None)
    parser.add_argument("--baseline-preflight", type=Path, default=None)
    parser.add_argument("--candidate-preflight", type=Path, default=None)
    parser.add_argument("--seed-qid", action="append", default=[])
    parser.add_argument("--seed-qids-file", "--seed-qid-file", dest="seed_qids_file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--ledger-json", type=Path, default=None)
    args = parser.parse_args()

    baseline_submission = _submission_answers_by_id(args.baseline_submission)
    candidate_submission = _submission_answers_by_id(args.candidate_submission)
    answer_changed_count = _answer_changed_count(baseline_submission, candidate_submission)
    retrieval_projection_changed_count = _retrieval_projection_changed_count(baseline_submission, candidate_submission)
    baseline_all, baseline_trusted = _score_benchmark(args.baseline_raw_results, args.benchmark)
    candidate_all, candidate_trusted = _score_benchmark(args.candidate_raw_results, args.benchmark)
    baseline_page_p95 = _page_p95(args.baseline_preflight)
    candidate_page_p95 = _page_p95(args.candidate_preflight)
    seed_deltas = _seed_case_deltas(
        baseline_scaffold_path=args.scaffold,
        candidate_scaffold_path=args.candidate_scaffold,
        baseline_raw_results_path=args.baseline_raw_results,
        candidate_raw_results_path=args.candidate_raw_results,
        seed_qids=_seed_qids(args),
    )
    recommendation, notes = _recommendation(
        baseline_trusted=baseline_trusted,
        candidate_trusted=candidate_trusted,
        answer_changed_count=answer_changed_count,
        baseline_page_p95=baseline_page_p95,
        candidate_page_p95=candidate_page_p95,
    )
    report = _render_report(
        label=args.label,
        baseline_label=args.baseline_label,
        answer_changed_count=answer_changed_count,
        retrieval_projection_changed_count=retrieval_projection_changed_count,
        baseline_all=baseline_all,
        candidate_all=candidate_all,
        baseline_trusted=baseline_trusted,
        candidate_trusted=candidate_trusted,
        baseline_page_p95=baseline_page_p95,
        candidate_page_p95=candidate_page_p95,
        seed_deltas=seed_deltas,
        recommendation=recommendation,
        notes=notes,
    )
    report_payload = _report_payload(
        label=args.label,
        baseline_label=args.baseline_label,
        answer_changed_count=answer_changed_count,
        retrieval_projection_changed_count=retrieval_projection_changed_count,
        baseline_all=baseline_all,
        candidate_all=candidate_all,
        baseline_trusted=baseline_trusted,
        candidate_trusted=candidate_trusted,
        baseline_page_p95=baseline_page_p95,
        candidate_page_p95=candidate_page_p95,
        seed_deltas=seed_deltas,
        recommendation=recommendation,
        notes=notes,
    )
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.ledger_json is not None:
        record = ExperimentRecord(
            timestamp_utc=datetime.now(UTC).isoformat(),
            label=args.label,
            baseline_label=args.baseline_label,
            recommendation=recommendation,
            answer_changed_count=answer_changed_count,
            retrieval_page_projection_changed_count=retrieval_projection_changed_count,
            benchmark_all_baseline=baseline_all.page_f_beta,
            benchmark_all_candidate=candidate_all.page_f_beta,
            benchmark_trusted_baseline=baseline_trusted.page_f_beta,
            benchmark_trusted_candidate=candidate_trusted.page_f_beta,
            baseline_page_p95=baseline_page_p95,
            candidate_page_p95=candidate_page_p95,
            improved_seed_cases=cast("list[str]", report_payload["improved_seed_cases"]),
            regressed_seed_cases=cast("list[str]", report_payload["regressed_seed_cases"]),
            notes=notes,
        )
        _append_ledger(args.ledger_json, record)


if __name__ == "__main__":
    main()
