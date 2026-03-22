#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _coerce_object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return cast("list[object]", value)


def _load_questions(path: Path) -> dict[str, JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _coerce_object_list(payload)
    out: dict[str, JsonDict] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("id") or row.get("question_id") or "").strip()
        if not qid:
            continue
        out[qid] = row
    return out


def _load_truth_audit(path: Path | None) -> dict[str, JsonDict]:
    if path is None:
        return {}
    payload_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        return {}
    payload = cast("JsonDict", payload_obj)
    rows = _coerce_object_list(payload.get("records"))
    out: dict[str, JsonDict] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_canary(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Canary payload must be an object: {path}")
    return cast("JsonDict", payload)


def _runtime_recommendation(
    *,
    baseline_concurrency: int,
    candidate_concurrency: int,
    answer_drift_count: int,
    page_drift_count: int,
    model_drift_count: int,
    missing_case_count: int,
) -> str:
    if answer_drift_count > 0 or page_drift_count > 0 or model_drift_count > 0 or missing_case_count > 0:
        return "query_concurrency=1"
    if candidate_concurrency <= max(1, baseline_concurrency):
        return "query_concurrency=1_stable_only"
    return "query_concurrency>1_allowed"


def _build_report(*, canary: JsonDict, questions: dict[str, JsonDict], truth_audit: dict[str, JsonDict]) -> JsonDict:
    drift_ids = [str(item).strip() for item in _coerce_object_list(canary.get("answer_drift_case_ids")) if str(item).strip()]
    rows: list[JsonDict] = []
    answer_type_counter: Counter[str] = Counter()
    route_family_counter: Counter[str] = Counter()
    support_shape_counter: Counter[str] = Counter()

    for qid in drift_ids:
        question_row = questions.get(qid, {})
        truth_row = truth_audit.get(qid, {})
        answer_type = str(question_row.get("answer_type") or truth_row.get("answer_type") or "").strip() or "unknown"
        route_family = str(truth_row.get("route_family") or "").strip() or "unknown"
        support_shape = str(truth_row.get("support_shape_class") or "").strip() or "unknown"
        row = {
            "question_id": qid,
            "question": str(question_row.get("question") or truth_row.get("question") or "").strip(),
            "answer_type": answer_type,
            "route_family": route_family,
            "support_shape_class": support_shape,
            "question_refs": [
                str(item).strip()
                for item in _coerce_object_list(truth_row.get("question_refs"))
                if str(item).strip()
            ],
            "audit_priority": int(truth_row.get("audit_priority") or 0),
        }
        rows.append(row)
        answer_type_counter[answer_type] += 1
        route_family_counter[route_family] += 1
        support_shape_counter[support_shape] += 1

    total_cases = int(canary.get("total_cases") or 0)
    answer_drift_count = int(canary.get("answer_drift_count") or len(rows))
    page_drift_count = int(canary.get("page_drift_count") or 0)
    model_drift_count = int(canary.get("model_drift_count") or 0)
    missing_case_count = len(_coerce_object_list(canary.get("missing_case_ids")))
    drift_rate = answer_drift_count / total_cases if total_cases > 0 else 0.0
    baseline_concurrency = int(canary.get("baseline_concurrency") or 0)
    candidate_concurrency = int(canary.get("candidate_concurrency") or 0)

    return {
        "baseline_concurrency": baseline_concurrency,
        "candidate_concurrency": candidate_concurrency,
        "total_cases": total_cases,
        "answer_drift_count": answer_drift_count,
        "page_drift_count": page_drift_count,
        "model_drift_count": model_drift_count,
        "missing_case_count": missing_case_count,
        "answer_drift_rate": round(drift_rate, 4),
        "by_answer_type": dict(answer_type_counter),
        "by_route_family": dict(route_family_counter),
        "by_support_shape_class": dict(support_shape_counter),
        "drift_cases": rows,
        "runtime_recommendation": _runtime_recommendation(
            baseline_concurrency=baseline_concurrency,
            candidate_concurrency=candidate_concurrency,
            answer_drift_count=answer_drift_count,
            page_drift_count=page_drift_count,
            model_drift_count=model_drift_count,
            missing_case_count=missing_case_count,
        ),
    }


def _render_markdown(report: JsonDict) -> str:
    lines = [
        "# Concurrency Drift Audit",
        "",
        f"- `baseline_concurrency`: `{report['baseline_concurrency']}`",
        f"- `candidate_concurrency`: `{report['candidate_concurrency']}`",
        f"- `total_cases`: `{report['total_cases']}`",
        f"- `answer_drift_count`: `{report['answer_drift_count']}`",
        f"- `page_drift_count`: `{report['page_drift_count']}`",
        f"- `model_drift_count`: `{report['model_drift_count']}`",
        f"- `missing_case_count`: `{report['missing_case_count']}`",
        f"- `answer_drift_rate`: `{report['answer_drift_rate']}`",
        f"- `runtime_recommendation`: `{report['runtime_recommendation']}`",
        "",
        "## Breakdown",
        "",
        f"- `by_answer_type`: `{report['by_answer_type']}`",
        f"- `by_route_family`: `{report['by_route_family']}`",
        f"- `by_support_shape_class`: `{report['by_support_shape_class']}`",
        "",
        "## Drift Cases",
        "",
    ]

    drift_cases = cast("list[JsonDict]", report.get("drift_cases") or [])
    for row in drift_cases:
        lines.extend(
            [
                f"### `{row['question_id']}`",
                f"- `answer_type`: `{row['answer_type']}`",
                f"- `route_family`: `{row['route_family']}`",
                f"- `support_shape_class`: `{row['support_shape_class']}`",
                f"- `audit_priority`: `{row['audit_priority']}`",
                f"- `question_refs`: `{row['question_refs']}`",
                f"- question: {row['question']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--truth-audit", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    canary = _load_canary(args.canary.resolve())
    questions = _load_questions(args.questions.resolve())
    truth_audit = _load_truth_audit(args.truth_audit.resolve() if args.truth_audit else None)
    report = _build_report(canary=canary, questions=questions, truth_audit=truth_audit)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "concurrency_drift_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "concurrency_drift_report.md").write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
