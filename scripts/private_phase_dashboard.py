#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from shafi.submission.common import classify_unanswerable_answer, select_submission_used_pages

if TYPE_CHECKING:
    from collections.abc import Iterable

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class CaseSnapshot:
    question_id: str
    question: str
    answer_text: str
    answer_type: str
    route_family: str
    model_name: str
    doc_refs: list[str]
    retrieved_page_ids: list[str]
    context_page_ids: list[str]
    context_chunk_count: int
    used_pages: list[str]
    ttft_ms: float | None


def _coerce_dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in cast("list[object]", value) if (text := str(item).strip())]


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _load_json_list(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [cast("JsonDict", row) for row in cast("list[object]", obj) if isinstance(row, dict)]


def _load_questions(path: Path | None) -> dict[str, JsonDict]:
    if path is None or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        return {}
    out: dict[str, JsonDict] = {}
    for row_obj in cast("list[object]", obj):
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("id") or row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_truth_audit(path: Path | None) -> dict[str, JsonDict]:
    if path is None or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    records_obj = cast("JsonDict", obj).get("records")
    if not isinstance(records_obj, list):
        return {}
    out: dict[str, JsonDict] = {}
    for row_obj in cast("list[object]", records_obj):
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("question_id") or row.get("case_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _percentile(values: Iterable[float], p: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, p)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * weight)


def _infer_route_family(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    if normalized == "strict-extractor":
        return "strict"
    if normalized == "structured-extractor":
        return "structured"
    if normalized == "premise-guard":
        return "premise_guard"
    return "model"


def _build_snapshot(
    *,
    row: JsonDict,
    questions: dict[str, JsonDict],
    truth_audit: dict[str, JsonDict],
) -> CaseSnapshot:
    case = _coerce_dict(row.get("case"))
    telemetry = _coerce_dict(row.get("telemetry"))
    qid = str(case.get("case_id") or case.get("question_id") or telemetry.get("question_id") or "").strip()
    question_row = questions.get(qid, {})
    truth_row = truth_audit.get(qid, {})
    question = str(case.get("question") or question_row.get("question") or truth_row.get("question") or "").strip()
    answer_text = str(row.get("answer_text") or row.get("answer") or "").strip()
    answer_type = str(case.get("answer_type") or telemetry.get("answer_type") or question_row.get("answer_type") or truth_row.get("answer_type") or "free_text").strip() or "free_text"
    model_name = str(telemetry.get("model_llm") or telemetry.get("model_name") or "").strip()
    route_family = str(truth_row.get("route_family") or "").strip() or _infer_route_family(model_name)
    ttft_ms = _coerce_float(telemetry.get("ttft_ms"))
    used_pages = select_submission_used_pages(cast("dict[str, object]", telemetry))
    doc_refs = _coerce_str_list(telemetry.get("doc_refs"))
    retrieved_page_ids = _coerce_str_list(telemetry.get("retrieved_page_ids"))
    context_page_ids = _coerce_str_list(telemetry.get("context_page_ids"))
    context_chunk_count = int(telemetry.get("context_chunk_count") or len(_coerce_str_list(telemetry.get("context_chunk_ids"))) or 0)
    return CaseSnapshot(
        question_id=qid,
        question=question,
        answer_text=answer_text,
        answer_type=answer_type,
        route_family=route_family,
        model_name=model_name,
        doc_refs=doc_refs,
        retrieved_page_ids=retrieved_page_ids,
        context_page_ids=context_page_ids,
        context_chunk_count=max(0, context_chunk_count),
        used_pages=used_pages,
        ttft_ms=ttft_ms,
    )


def _classify_root_cause(snapshot: CaseSnapshot) -> str | None:
    strict_null, free_text_null = classify_unanswerable_answer(snapshot.answer_text, snapshot.answer_type)
    is_null = strict_null or free_text_null
    if not is_null and snapshot.used_pages:
        return None
    if (
        strict_null
        and snapshot.answer_type.strip().lower() != "free_text"
        and snapshot.context_chunk_count > 0
        and not snapshot.retrieved_page_ids
        and not snapshot.context_page_ids
        and not snapshot.used_pages
    ):
        return "strict_null_telemetry_reset"
    if not snapshot.retrieved_page_ids and not snapshot.context_page_ids and not snapshot.used_pages:
        return "retrieval_miss"
    if snapshot.retrieved_page_ids and snapshot.context_chunk_count <= 0 and not snapshot.context_page_ids:
        return "context_miss"
    if snapshot.retrieved_page_ids and snapshot.context_chunk_count > 0 and not snapshot.context_page_ids and not snapshot.used_pages:
        return "page_miss"
    if snapshot.context_page_ids and not snapshot.used_pages:
        return "answerer_miss"
    if is_null and snapshot.used_pages:
        return "answerer_miss"
    return None


def _case_summary(snapshot: CaseSnapshot) -> JsonDict:
    strict_null, free_text_null = classify_unanswerable_answer(snapshot.answer_text, snapshot.answer_type)
    root_cause = _classify_root_cause(snapshot)
    return {
        "question_id": snapshot.question_id,
        "question": snapshot.question,
        "answer_type": snapshot.answer_type,
        "route_family": snapshot.route_family,
        "model_name": snapshot.model_name,
        "doc_ref_count": len(snapshot.doc_refs),
        "doc_refs": snapshot.doc_refs,
        "retrieved_page_count": len(snapshot.retrieved_page_ids),
        "context_page_count": len(snapshot.context_page_ids),
        "context_chunk_count": snapshot.context_chunk_count,
        "used_page_count": len(snapshot.used_pages),
        "used_pages": snapshot.used_pages,
        "ttft_ms": snapshot.ttft_ms,
        "null_answer": strict_null or free_text_null,
        "root_cause": root_cause,
    }


def _summarize_run(*, label: str, rows: list[JsonDict], questions: dict[str, JsonDict], truth_audit: dict[str, JsonDict], high_page_threshold: int) -> JsonDict:
    snapshots = [_build_snapshot(row=row, questions=questions, truth_audit=truth_audit) for row in rows]
    route_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    ttft_values: list[float] = []
    null_cases: list[JsonDict] = []
    empty_page_cases: list[JsonDict] = []
    high_page_cases: list[JsonDict] = []
    root_cause_counts: Counter[str] = Counter()

    for snapshot in snapshots:
        route_counts[snapshot.route_family or "unknown"] += 1
        answer_type_counts[snapshot.answer_type or "unknown"] += 1
        model_counts[snapshot.model_name or "unknown"] += 1
        case_info = _case_summary(snapshot)
        root_cause = str(case_info.get("root_cause") or "").strip()
        if root_cause:
            root_cause_counts[root_cause] += 1
        if snapshot.ttft_ms is not None:
            ttft_values.append(snapshot.ttft_ms)
        if case_info["null_answer"]:
            null_cases.append(case_info)
        if len(snapshot.used_pages) == 0:
            empty_page_cases.append(case_info)
        if len(snapshot.used_pages) > high_page_threshold:
            high_page_cases.append(case_info)

    ttft_p95 = _percentile(ttft_values, 0.95)
    slow_ttft_cases = [
        _case_summary(snapshot)
        for snapshot in snapshots
        if snapshot.ttft_ms is not None and ttft_p95 is not None and snapshot.ttft_ms >= ttft_p95
    ]
    slow_ttft_cases = sorted(
        slow_ttft_cases,
        key=lambda row: float(row.get("ttft_ms") or 0.0),
        reverse=True,
    )[:10]

    return {
        "label": label,
        "total_cases": len(snapshots),
        "route_counts": dict(route_counts),
        "answer_type_counts": dict(answer_type_counts),
        "model_counts": dict(model_counts),
        "root_cause_counts": dict(root_cause_counts),
        "null_answer_count": len(null_cases),
        "empty_used_page_count": len(empty_page_cases),
        "high_page_count_case_count": len(high_page_cases),
        "ttft_p50_ms": None if not ttft_values else round(_percentile(ttft_values, 0.50) or 0.0, 1),
        "ttft_p95_ms": None if ttft_p95 is None else round(ttft_p95, 1),
        "anomalies": {
            "null_answers": null_cases[:10],
            "empty_used_pages": empty_page_cases[:10],
            "high_page_count": high_page_cases[:10],
            "slow_ttft": slow_ttft_cases,
        },
        "cases_by_id": {snapshot.question_id: _case_summary(snapshot) for snapshot in snapshots if snapshot.question_id},
    }


def _diff_runs(*, baseline: JsonDict, candidate: JsonDict) -> JsonDict:
    baseline_cases = cast("JsonDict", baseline.get("cases_by_id") or {})
    candidate_cases = cast("JsonDict", candidate.get("cases_by_id") or {})
    shared_qids = sorted(set(baseline_cases).intersection(candidate_cases))
    route_drifts: list[JsonDict] = []
    model_drifts: list[JsonDict] = []
    page_count_changes: list[JsonDict] = []
    null_answer_state_changes: list[JsonDict] = []

    for qid in shared_qids:
        base = _coerce_dict(baseline_cases.get(qid))
        cand = _coerce_dict(candidate_cases.get(qid))
        if str(base.get("route_family") or "") != str(cand.get("route_family") or ""):
            route_drifts.append(
                {
                    "question_id": qid,
                    "question": cand.get("question") or base.get("question") or "",
                    "baseline_route": base.get("route_family"),
                    "candidate_route": cand.get("route_family"),
                }
            )
        if str(base.get("model_name") or "") != str(cand.get("model_name") or ""):
            model_drifts.append(
                {
                    "question_id": qid,
                    "question": cand.get("question") or base.get("question") or "",
                    "baseline_model": base.get("model_name"),
                    "candidate_model": cand.get("model_name"),
                }
            )
        if int(base.get("used_page_count") or 0) != int(cand.get("used_page_count") or 0):
            page_count_changes.append(
                {
                    "question_id": qid,
                    "question": cand.get("question") or base.get("question") or "",
                    "baseline_used_page_count": base.get("used_page_count"),
                    "candidate_used_page_count": cand.get("used_page_count"),
                }
            )
        if bool(base.get("null_answer")) != bool(cand.get("null_answer")):
            null_answer_state_changes.append(
                {
                    "question_id": qid,
                    "question": cand.get("question") or base.get("question") or "",
                    "baseline_null_answer": base.get("null_answer"),
                    "candidate_null_answer": cand.get("null_answer"),
                }
            )

    return {
        "shared_case_count": len(shared_qids),
        "route_drift_count": len(route_drifts),
        "model_drift_count": len(model_drifts),
        "used_page_count_changed_count": len(page_count_changes),
        "null_answer_state_changed_count": len(null_answer_state_changes),
        "ttft_p50_delta_ms": round(float(candidate.get("ttft_p50_ms") or 0.0) - float(baseline.get("ttft_p50_ms") or 0.0), 1),
        "ttft_p95_delta_ms": round(float(candidate.get("ttft_p95_ms") or 0.0) - float(baseline.get("ttft_p95_ms") or 0.0), 1),
        "route_drifts": route_drifts[:20],
        "model_drifts": model_drifts[:20],
        "used_page_count_changes": page_count_changes[:20],
        "null_answer_state_changes": null_answer_state_changes[:20],
    }


def _render_markdown(*, run_a: JsonDict, run_b: JsonDict | None, diff: JsonDict | None) -> str:
    lines = [
        "# Private-Phase Telemetry Dashboard",
        "",
        "## Run A",
        "",
        f"- `label`: `{run_a['label']}`",
        f"- `total_cases`: `{run_a['total_cases']}`",
        f"- `null_answer_count`: `{run_a['null_answer_count']}`",
        f"- `empty_used_page_count`: `{run_a['empty_used_page_count']}`",
        f"- `high_page_count_case_count`: `{run_a['high_page_count_case_count']}`",
        f"- `root_cause_counts`: `{run_a.get('root_cause_counts')}`",
        f"- `ttft_p50_ms`: `{run_a['ttft_p50_ms']}`",
        f"- `ttft_p95_ms`: `{run_a['ttft_p95_ms']}`",
        "",
        "## Run A Anomalies",
        "",
    ]
    anomalies_a = cast("JsonDict", run_a.get("anomalies") or {})
    for key in ("null_answers", "empty_used_pages", "high_page_count", "slow_ttft"):
        rows = cast("list[JsonDict]", anomalies_a.get(key) or [])
        lines.append(f"- `{key}`: `{len(rows)}`")

    if run_b is not None:
        lines.extend(
            [
                "",
                "## Run B",
                "",
                f"- `label`: `{run_b['label']}`",
                f"- `total_cases`: `{run_b['total_cases']}`",
                f"- `null_answer_count`: `{run_b['null_answer_count']}`",
                f"- `empty_used_page_count`: `{run_b['empty_used_page_count']}`",
                f"- `high_page_count_case_count`: `{run_b['high_page_count_case_count']}`",
                f"- `root_cause_counts`: `{run_b.get('root_cause_counts')}`",
                f"- `ttft_p50_ms`: `{run_b['ttft_p50_ms']}`",
                f"- `ttft_p95_ms`: `{run_b['ttft_p95_ms']}`",
            ]
        )

    if diff is not None:
        lines.extend(
            [
                "",
                "## Diff",
                "",
                f"- `shared_case_count`: `{diff['shared_case_count']}`",
                f"- `route_drift_count`: `{diff['route_drift_count']}`",
                f"- `model_drift_count`: `{diff['model_drift_count']}`",
                f"- `used_page_count_changed_count`: `{diff['used_page_count_changed_count']}`",
                f"- `null_answer_state_changed_count`: `{diff['null_answer_state_changed_count']}`",
                f"- `ttft_p50_delta_ms`: `{diff['ttft_p50_delta_ms']}`",
                f"- `ttft_p95_delta_ms`: `{diff['ttft_p95_delta_ms']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one telemetry raw-results run and optionally diff it against another.")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, default=None)
    parser.add_argument("--label-a", default="run_a")
    parser.add_argument("--label-b", default="run_b")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--truth-audit", type=Path, default=None)
    parser.add_argument("--high-page-threshold", type=int, default=4)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = _load_questions(args.questions)
    truth_audit = _load_truth_audit(args.truth_audit)
    run_a = _summarize_run(
        label=str(args.label_a),
        rows=_load_json_list(args.run_a),
        questions=questions,
        truth_audit=truth_audit,
        high_page_threshold=max(1, int(args.high_page_threshold)),
    )
    run_b = None
    diff = None
    if args.run_b is not None:
        run_b = _summarize_run(
            label=str(args.label_b),
            rows=_load_json_list(args.run_b),
            questions=questions,
            truth_audit=truth_audit,
            high_page_threshold=max(1, int(args.high_page_threshold)),
        )
        diff = _diff_runs(baseline=run_a, candidate=run_b)

    payload: JsonDict = {
        "run_a": run_a,
        "run_b": run_b,
        "diff": diff,
        "high_page_threshold": max(1, int(args.high_page_threshold)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(run_a=run_a, run_b=run_b, diff=diff), encoding="utf-8")


if __name__ == "__main__":
    main()
