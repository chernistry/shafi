from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


@dataclass(frozen=True)
class DriftRecord:
    question_id: str
    answer_type: str
    route_family: str
    question: str
    baseline_answer: str
    candidate_answer: str
    baseline_model_name: str
    candidate_model_name: str


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", payload)


def _submission_answers(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Submission at {path} is missing 'answers'")
    out: dict[str, JsonDict] = {}
    for raw in cast("list[object]", answers_obj):
        if not isinstance(raw, dict):
            continue
        answer = cast("JsonDict", raw)
        question_id = str(answer.get("question_id") or "").strip()
        if question_id:
            out[question_id] = answer
    return out


def _scaffold_records(path: Path | None) -> dict[str, JsonDict]:
    if path is None:
        return {}
    payload = _load_json(path)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {path} is missing 'records'")
    out: dict[str, JsonDict] = {}
    for raw in cast("list[object]", records_obj):
        if not isinstance(raw, dict):
            continue
        record = cast("JsonDict", raw)
        question_id = str(record.get("question_id") or "").strip()
        if question_id:
            out[question_id] = record
    return out


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _render_answer(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _route_family(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized == "strict-extractor":
        return "strict"
    if normalized == "structured-extractor":
        return "structured"
    if normalized == "premise-guard":
        return "premise_guard"
    return "model"


def build_drift_records(
    *,
    baseline_submission: dict[str, JsonDict],
    candidate_submission: dict[str, JsonDict],
    scaffold_records: dict[str, JsonDict],
) -> list[DriftRecord]:
    records: list[DriftRecord] = []
    for question_id, baseline_answer in baseline_submission.items():
        candidate_answer = candidate_submission.get(question_id)
        if candidate_answer is None:
            continue
        if _stable_json(baseline_answer.get("answer")) == _stable_json(candidate_answer.get("answer")):
            continue
        scaffold = scaffold_records.get(question_id, {})
        baseline_telemetry = cast("JsonDict", baseline_answer.get("telemetry") or {})
        candidate_telemetry = cast("JsonDict", candidate_answer.get("telemetry") or {})
        baseline_model_name = str(baseline_telemetry.get("model_name") or "").strip()
        candidate_model_name = str(candidate_telemetry.get("model_name") or "").strip()
        records.append(
            DriftRecord(
                question_id=question_id,
                answer_type=str(scaffold.get("answer_type") or "").strip() or "unknown",
                route_family=str(scaffold.get("route_family") or "").strip() or _route_family(candidate_model_name),
                question=str(scaffold.get("question") or "").strip(),
                baseline_answer=_render_answer(baseline_answer.get("answer")),
                candidate_answer=_render_answer(candidate_answer.get("answer")),
                baseline_model_name=baseline_model_name,
                candidate_model_name=candidate_model_name,
            )
        )
    records.sort(key=lambda item: (item.answer_type, item.route_family, item.question_id))
    return records


def render_report(records: list[DriftRecord], *, label: str, baseline_label: str, limit: int) -> str:
    answer_type_counts = Counter(record.answer_type for record in records)
    route_family_counts = Counter(record.route_family for record in records)
    candidate_model_counts = Counter(record.candidate_model_name for record in records)
    lines = [
        "# Answer Drift Report",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Candidate: `{label}`",
        f"- Changed answers: `{len(records)}`",
        "",
        "## Summary",
        "",
        f"- by_answer_type: `{dict(answer_type_counts)}`",
        f"- by_route_family: `{dict(route_family_counts)}`",
        f"- by_candidate_model: `{dict(candidate_model_counts)}`",
        "",
    ]
    for record in records[:limit]:
        lines.extend(
            [
                f"## {record.question_id}",
                f"- answer_type: `{record.answer_type}`",
                f"- route_family: `{record.route_family}`",
                f"- baseline_model_name: `{record.baseline_model_name}`",
                f"- candidate_model_name: `{record.candidate_model_name}`",
                f"- question: {record.question}",
                f"- baseline_answer: `{record.baseline_answer}`",
                f"- candidate_answer: `{record.candidate_answer}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze answer drift between two submission artifacts.")
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    records = build_drift_records(
        baseline_submission=_submission_answers(args.baseline_submission),
        candidate_submission=_submission_answers(args.candidate_submission),
        scaffold_records=_scaffold_records(args.scaffold),
    )
    report = render_report(records, label=args.label, baseline_label=args.baseline_label, limit=args.limit)
    payload = {
        "label": args.label,
        "baseline_label": args.baseline_label,
        "changed_answer_count": len(records),
        "records": [asdict(record) for record in records],
    }

    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
