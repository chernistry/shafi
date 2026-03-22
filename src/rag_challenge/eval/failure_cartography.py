"""Failure cartography orchestrator over historical run artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, cast

from rag_challenge.eval.failure_cartography_models import (
    FailureLedger,
    FailureRecord,
    FailureTaxonomy,
    JsonDict,
    ReviewedGoldenCase,
    RunFailureObservation,
    RunObservation,
)
from rag_challenge.eval.failure_cartography_rules import (
    answers_match,
    classify_miss,
    compute_drift,
    infer_doc_family,
    page_docs,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_reviewed_golden(path: Path) -> dict[str, ReviewedGoldenCase]:
    """Load reviewed golden labels by question id.

    Args:
        path: Reviewed golden JSON path.

    Returns:
        dict[str, ReviewedGoldenCase]: Question keyed reviewed labels.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected reviewed golden list: {path}")
    out: dict[str, ReviewedGoldenCase] = {}
    for raw in cast("list[object]", payload):
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            continue
        out[qid] = ReviewedGoldenCase(
            question_id=qid,
            question=str(row.get("question") or "").strip(),
            answer_type=str(row.get("answer_type") or "free_text").strip() or "free_text",
            golden_answer=str(row.get("golden_answer") or "").strip(),
            golden_page_ids=_str_list(row.get("golden_page_ids")),
            trust_tier=str(row.get("trust_tier") or "").strip(),
            label_weight=_as_float(row.get("label_weight"), default=1.0),
        )
    return out


def discover_run_artifacts(runs_dir: Path) -> list[Path]:
    """Discover supported historical run artifacts.

    Args:
        runs_dir: Root directory containing JSON run artifacts.

    Returns:
        list[Path]: Sorted artifact paths.
    """

    raw_results = sorted(runs_dir.rglob("raw_results*.json"))
    eval_runs = sorted(path for path in runs_dir.rglob("eval*.json") if "judg" not in path.name.lower())
    return sorted({*raw_results, *eval_runs})


def load_run_observations(path: Path, reviewed: dict[str, ReviewedGoldenCase]) -> list[RunObservation]:
    """Load reviewed observations from one historical artifact.

    Args:
        path: Artifact path.
        reviewed: Reviewed golden lookup.

    Returns:
        list[RunObservation]: Reviewed observations present in the artifact.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    run_label = path.stem
    observations: list[RunObservation] = []
    if isinstance(payload, list):
        for raw in cast("list[object]", payload):
            if not isinstance(raw, dict):
                continue
            row = cast("JsonDict", raw)
            case = _dict(row.get("case"))
            telemetry = _dict(row.get("telemetry"))
            qid = str(case.get("case_id") or telemetry.get("question_id") or row.get("question_id") or "").strip()
            if qid not in reviewed:
                continue
            observations.append(
                RunObservation(
                    run_label=run_label,
                    source_path=str(path),
                    question_id=qid,
                    question=str(case.get("question") or reviewed[qid].question).strip(),
                    answer_type=str(case.get("answer_type") or reviewed[qid].answer_type).strip() or reviewed[qid].answer_type,
                    predicted_answer=str(row.get("answer_text") or "").strip(),
                    used_page_ids=_str_list(telemetry.get("used_page_ids")),
                    retrieved_page_ids=_str_list(telemetry.get("retrieved_page_ids")),
                )
            )
        return observations
    if not isinstance(payload, dict):
        return observations
    payload_dict = cast("JsonDict", payload)
    cases_obj = payload_dict.get("cases")
    if not isinstance(cases_obj, list):
        return observations
    cases = cast("list[object]", cases_obj)
    for raw in cases:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        qid = str(row.get("question_id") or row.get("case_id") or "").strip()
        if qid not in reviewed:
            continue
        observations.append(
            RunObservation(
                run_label=run_label,
                source_path=str(path),
                question_id=qid,
                question=str(row.get("question") or reviewed[qid].question).strip(),
                answer_type=str(row.get("answer_type") or reviewed[qid].answer_type).strip() or reviewed[qid].answer_type,
                predicted_answer=str(row.get("answer") or "").strip(),
                used_page_ids=[],
                retrieved_page_ids=[],
            )
        )
    return observations


def build_failure_ledger(*, reviewed: dict[str, ReviewedGoldenCase], observations: list[RunObservation]) -> FailureLedger:
    """Build an aggregated failure ledger across reviewed questions.

    Args:
        reviewed: Reviewed golden lookup.
        observations: Historical run observations.

    Returns:
        FailureLedger: Aggregated ledger and summaries.
    """

    by_qid: dict[str, list[RunObservation]] = defaultdict(list)
    for observation in observations:
        by_qid[observation.question_id].append(observation)

    records: list[FailureRecord] = []
    failure_counter: Counter[str] = Counter()
    doc_family_counter: Counter[str] = Counter()
    hardest: list[tuple[str, int]] = []
    for qid, runs in sorted(by_qid.items()):
        golden = reviewed[qid]
        run_failures: list[RunFailureObservation] = []
        per_record_counter: Counter[str] = Counter()
        document_ids = {
            *page_docs(golden.golden_page_ids),
            *[doc for obs in runs for doc in page_docs(obs.used_page_ids or obs.retrieved_page_ids)],
        }
        for observation in runs:
            classified = classify_miss(golden, observation)
            for item in classified:
                per_record_counter[item.value] += 1
                failure_counter[item.value] += 1
            run_failures.append(
                RunFailureObservation(
                    run_label=observation.run_label,
                    source_path=observation.source_path,
                    predicted_answer=observation.predicted_answer,
                    used_page_ids=sorted(observation.used_page_ids),
                    retrieved_page_ids=sorted(observation.retrieved_page_ids),
                    failure_types=[item.value for item in classified],
                    answer_correct=answers_match(
                        answer_type=golden.answer_type,
                        predicted=observation.predicted_answer,
                        golden=golden.golden_answer,
                    ),
                )
            )
        doc_family = infer_doc_family(golden.question, sorted(document_ids))
        doc_family_counter[doc_family] += 1
        drift = compute_drift(runs)
        hardest.append((qid, sum(per_record_counter.values())))
        records.append(
            FailureRecord(
                question_id=qid,
                question=golden.question,
                answer_type=golden.answer_type,
                doc_family=doc_family,
                document_ids=sorted(document_ids),
                golden_answer=golden.golden_answer,
                golden_page_ids=golden.golden_page_ids,
                failure_types=sorted(per_record_counter),
                failure_type_counts=dict(sorted(per_record_counter.items())),
                drift=drift,
                run_observations=run_failures,
            )
        )
    hardest_sorted = [qid for qid, _score in sorted(hardest, key=lambda item: (-item[1], item[0]))[:20]]
    summary: JsonDict = {
        "reviewed_questions": len(records),
        "run_observations": len(observations),
        "failure_type_counts": dict(sorted(failure_counter.items())),
        "doc_family_counts": dict(sorted(doc_family_counter.items())),
        "unclassified_rate": 0.0,
        "top_20_hardest_question_ids": hardest_sorted,
        "records_with_answer_drift": len([record for record in records if record.drift.answer_drift_count > 0]),
        "records_with_page_drift": len([record for record in records if record.drift.page_drift_count > 0]),
    }
    return FailureLedger(records=records, summary=summary)


def render_summary_markdown(ledger: FailureLedger) -> str:
    """Render a short markdown summary for operator review.

    Args:
        ledger: Failure ledger payload.

    Returns:
        str: Markdown report.
    """

    summary = ledger.summary
    lines = [
        "# Closed-World Failure Cartography",
        "",
        f"- reviewed_questions: `{summary['reviewed_questions']}`",
        f"- run_observations: `{summary['run_observations']}`",
        f"- records_with_answer_drift: `{summary['records_with_answer_drift']}`",
        f"- records_with_page_drift: `{summary['records_with_page_drift']}`",
        "",
        "## Failure Type Counts",
        "",
    ]
    for key, value in cast("dict[str, int]", summary["failure_type_counts"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Document Family Counts", ""])
    for key, value in cast("dict[str, int]", summary["doc_family_counts"]).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top 20 Hardest Question IDs", ""])
    for qid in cast("list[str]", summary["top_20_hardest_question_ids"]):
        lines.append(f"- `{qid}`")
    return "\n".join(lines).rstrip() + "\n"


def _dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in cast("list[object]", value) if (text := str(item).strip())]


def _as_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


__all__ = [
    "FailureLedger",
    "FailureRecord",
    "FailureTaxonomy",
    "ReviewedGoldenCase",
    "RunFailureObservation",
    "RunObservation",
    "answers_match",
    "build_failure_ledger",
    "classify_miss",
    "compute_drift",
    "discover_run_artifacts",
    "infer_doc_family",
    "load_reviewed_golden",
    "load_run_observations",
    "render_summary_markdown",
]
