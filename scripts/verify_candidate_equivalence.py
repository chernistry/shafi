# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

JsonDict = dict[str, object]


@dataclass(frozen=True)
class CandidateComparison:
    baseline_path: str
    answer_changed_ids: list[str]
    unexpected_answer_changed_ids: list[str]
    missing_allowed_answer_ids: list[str]
    retrieved_chunk_pages_changed_ids: list[str]
    used_page_ids_changed_ids: list[str]
    page_drift_ids: list[str]
    baseline_question_count: int
    candidate_question_count: int
    lineage_safe: bool
    verdict: str


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _submission_answers_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Submission at {path} is missing 'answers'")
    out: dict[str, JsonDict] = {}
    for raw in answers_obj:
        if not isinstance(raw, dict):
            continue
        question_id = str(raw.get("question_id") or "").strip()
        if question_id:
            out[question_id] = cast("JsonDict", raw)
    return out


def _json_stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _answer_changed_ids(
    baseline_submission: dict[str, JsonDict],
    candidate_submission: dict[str, JsonDict],
) -> list[str]:
    changed: list[str] = []
    for question_id, baseline in baseline_submission.items():
        candidate = candidate_submission.get(question_id)
        if candidate is None:
            continue
        if _json_stable(baseline.get("answer")) != _json_stable(candidate.get("answer")):
            changed.append(question_id)
    return sorted(changed)


def _retrieved_chunk_pages_projection(answer_record: JsonDict) -> list[JsonDict]:
    telemetry_obj = answer_record.get("telemetry")
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    retrieval_obj = telemetry.get("retrieval")
    retrieval = cast("JsonDict", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
    pages_obj = retrieval.get("retrieved_chunk_pages")
    if not isinstance(pages_obj, list):
        return []
    return [cast("JsonDict", page) for page in pages_obj if isinstance(page, dict)]


def _used_page_ids(answer_record: JsonDict) -> list[str]:
    telemetry_obj = answer_record.get("telemetry")
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    used_page_ids_obj = telemetry.get("used_page_ids")
    if not isinstance(used_page_ids_obj, list):
        return []
    return [str(item).strip() for item in used_page_ids_obj if str(item).strip()]


def _changed_projection_ids(
    baseline_submission: dict[str, JsonDict],
    candidate_submission: dict[str, JsonDict],
    *,
    value_getter: Callable[[JsonDict], object],
) -> list[str]:
    changed: list[str] = []
    for question_id, baseline in baseline_submission.items():
        candidate = candidate_submission.get(question_id)
        if candidate is None:
            continue
        if _json_stable(value_getter(baseline)) != _json_stable(value_getter(candidate)):
            changed.append(question_id)
    return sorted(changed)


def compare_candidate(
    *,
    baseline_path: Path,
    candidate_path: Path,
    allowed_answer_ids: set[str],
) -> CandidateComparison:
    baseline_submission = _submission_answers_by_id(baseline_path)
    candidate_submission = _submission_answers_by_id(candidate_path)
    answer_changed_ids = _answer_changed_ids(baseline_submission, candidate_submission)
    retrieved_chunk_pages_changed_ids = _changed_projection_ids(
        baseline_submission,
        candidate_submission,
        value_getter=_retrieved_chunk_pages_projection,
    )
    used_page_ids_changed_ids = _changed_projection_ids(
        baseline_submission,
        candidate_submission,
        value_getter=_used_page_ids,
    )
    page_drift_ids = sorted(set(retrieved_chunk_pages_changed_ids) | set(used_page_ids_changed_ids))
    unexpected_answer_changed_ids = sorted(
        question_id for question_id in answer_changed_ids if question_id not in allowed_answer_ids
    )
    missing_allowed_answer_ids = sorted(
        question_id for question_id in allowed_answer_ids if question_id not in set(answer_changed_ids)
    )
    lineage_safe = (
        len(baseline_submission) == len(candidate_submission)
        and not unexpected_answer_changed_ids
        and not missing_allowed_answer_ids
        and not page_drift_ids
    )
    verdict = "lineage_safe_exactness_candidate" if lineage_safe else "lineage_unsafe_candidate"
    return CandidateComparison(
        baseline_path=str(baseline_path),
        answer_changed_ids=answer_changed_ids,
        unexpected_answer_changed_ids=unexpected_answer_changed_ids,
        missing_allowed_answer_ids=missing_allowed_answer_ids,
        retrieved_chunk_pages_changed_ids=retrieved_chunk_pages_changed_ids,
        used_page_ids_changed_ids=used_page_ids_changed_ids,
        page_drift_ids=page_drift_ids,
        baseline_question_count=len(baseline_submission),
        candidate_question_count=len(candidate_submission),
        lineage_safe=lineage_safe,
        verdict=verdict,
    )


def render_report(
    *,
    candidate_path: Path,
    practical_public_state: str | None,
    comparisons: list[CandidateComparison],
    allowed_answer_ids: list[str],
) -> str:
    safe_baselines = [comparison.baseline_path for comparison in comparisons if comparison.lineage_safe]
    lines = [
        "# Candidate Lineage And Equivalence Report",
        "",
        f"- Candidate: `{candidate_path}`",
        f"- Allowed answer delta IDs: `{', '.join(allowed_answer_ids) if allowed_answer_ids else '(none)'}`",
    ]
    if practical_public_state:
        lines.append(f"- Practical public champion state: `{practical_public_state}`")
    lines.extend(
        [
            f"- Safe baselines: `{', '.join(safe_baselines) if safe_baselines else '(none)'}`",
            "",
        ]
    )

    for comparison in comparisons:
        lines.extend(
            [
                f"## {Path(comparison.baseline_path).name}",
                f"- verdict: `{comparison.verdict}`",
                f"- lineage_safe: `{comparison.lineage_safe}`",
                f"- baseline_question_count: `{comparison.baseline_question_count}`",
                f"- candidate_question_count: `{comparison.candidate_question_count}`",
                f"- answer_changed_count: `{len(comparison.answer_changed_ids)}`",
                f"- answer_changed_ids: `{', '.join(comparison.answer_changed_ids) if comparison.answer_changed_ids else '(none)'}`",
                f"- unexpected_answer_changed_ids: `{', '.join(comparison.unexpected_answer_changed_ids) if comparison.unexpected_answer_changed_ids else '(none)'}`",
                f"- missing_allowed_answer_ids: `{', '.join(comparison.missing_allowed_answer_ids) if comparison.missing_allowed_answer_ids else '(none)'}`",
                f"- retrieved_chunk_pages_changed_count: `{len(comparison.retrieved_chunk_pages_changed_ids)}`",
                f"- used_page_ids_changed_count: `{len(comparison.used_page_ids_changed_ids)}`",
                f"- page_drift_count: `{len(comparison.page_drift_ids)}`",
                f"- page_drift_ids: `{', '.join(comparison.page_drift_ids) if comparison.page_drift_ids else '(none)'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether a candidate artifact is lineage-safe relative to one or more baselines.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--allowed-answer-id", action="append", default=[])
    parser.add_argument("--practical-public-state", default="")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    allowed_answer_ids = sorted({str(question_id).strip() for question_id in args.allowed_answer_id if str(question_id).strip()})
    comparisons = [
        compare_candidate(
            baseline_path=baseline_path,
            candidate_path=args.candidate,
            allowed_answer_ids=set(allowed_answer_ids),
        )
        for baseline_path in args.baseline
    ]
    report = render_report(
        candidate_path=args.candidate,
        practical_public_state=str(args.practical_public_state or "").strip() or None,
        comparisons=comparisons,
        allowed_answer_ids=allowed_answer_ids,
    )

    payload = {
        "candidate_path": str(args.candidate),
        "allowed_answer_ids": allowed_answer_ids,
        "practical_public_state": str(args.practical_public_state or "").strip(),
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "safe_baselines": [comparison.baseline_path for comparison in comparisons if comparison.lineage_safe],
    }

    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
