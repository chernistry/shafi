# pyright: reportPrivateUsage=false
"""Local question-run capture helpers for reviewed grounding evaluation."""

from __future__ import annotations

import json
from statistics import fmean
from typing import TYPE_CHECKING

from scripts.build_platform_truth_audit import build_truth_audit_scaffold, render_truth_audit_workbook

from rag_challenge.ingestion.pipeline import IngestionPipeline
from rag_challenge.submission.common import load_cases, select_submission_used_pages
from rag_challenge.submission.platform import (
    PlatformCaseResult,
    _build_platform_submission_payload,
    _repair_anomalous_results,
    _run_questions,
    _serialize_platform_case_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def _percentile_from_sorted(values: list[int], percentile: float) -> int:
    """Return a nearest-rank percentile from sorted integers.

    Args:
        values: Sorted integer values.
        percentile: Percentile in the closed interval `[0, 100]`.

    Returns:
        Percentile value, or `0` for an empty list.
    """
    if not values:
        return 0
    if percentile <= 0:
        return values[0]
    if percentile >= 100:
        return values[-1]
    rank = max(0, min(len(values) - 1, round((percentile / 100.0) * (len(values) - 1))))
    return values[rank]


def summarize_results(
    results: list[PlatformCaseResult],
    *,
    questions_path: Path,
    docs_dir: Path | None,
    anomaly_repairs: dict[str, object] | None = None,
    truth_audit_path: Path | None = None,
) -> dict[str, object]:
    """Build a compact summary for one captured run.

    Args:
        results: Completed question results.
        questions_path: Source question set path.
        docs_dir: Source document directory, if available.
        anomaly_repairs: Optional anomaly-repair payload.
        truth_audit_path: Optional truth-audit scaffold path.

    Returns:
        JSON-serializable run summary.
    """
    used_page_counts = sorted(len(select_submission_used_pages(result.telemetry)) for result in results)
    total_ms_values = [int(result.total_ms) for result in results]
    null_answers = sum(1 for result in results if result.answer_text.strip().lower() in {"", "null", "none"})
    return {
        "questions_path": str(questions_path),
        "docs_dir": str(docs_dir) if docs_dir is not None else "",
        "case_count": len(results),
        "answer_null_count": null_answers,
        "page_count_distribution": {
            "min": min(used_page_counts) if used_page_counts else 0,
            "max": max(used_page_counts) if used_page_counts else 0,
            "mean": fmean(used_page_counts) if used_page_counts else 0.0,
            "p50": _percentile_from_sorted(used_page_counts, 50),
            "p95": _percentile_from_sorted(used_page_counts, 95),
            "zero_count": sum(1 for count in used_page_counts if count == 0),
        },
        "total_ms_distribution": {
            "min": min(total_ms_values) if total_ms_values else 0,
            "max": max(total_ms_values) if total_ms_values else 0,
            "mean": fmean(total_ms_values) if total_ms_values else 0.0,
            "p50": _percentile_from_sorted(sorted(total_ms_values), 50),
            "p95": _percentile_from_sorted(sorted(total_ms_values), 95),
        },
        "anomaly_repair_report": anomaly_repairs or {},
        "truth_audit_path": str(truth_audit_path) if truth_audit_path is not None else "",
    }


async def _ingest_documents(doc_dir: Path) -> None:
    """Run a fresh ingest for one document directory.

    Args:
        doc_dir: Document directory to ingest.

    Raises:
        RuntimeError: If the ingest pipeline fails.
    """
    pipeline = IngestionPipeline()
    try:
        await pipeline.run(doc_dir)
    finally:
        await pipeline.close()


async def capture_query_artifacts(
    *,
    questions_path: Path,
    raw_results_path: Path,
    submission_path: Path,
    summary_path: Path,
    docs_dir: Path | None = None,
    truth_audit_path: Path | None = None,
    truth_audit_workbook_path: Path | None = None,
    concurrency: int = 1,
    fail_fast: bool = False,
    ingest_doc_dir: Path | None = None,
) -> dict[str, object]:
    """Run local questions and write telemetry-rich artifacts.

    Args:
        questions_path: Source question JSON path.
        raw_results_path: Target raw-results JSON path.
        submission_path: Target submission JSON path.
        summary_path: Target summary JSON path.
        docs_dir: Optional docs directory for truth-audit scaffold generation.
        truth_audit_path: Optional truth-audit scaffold target path.
        truth_audit_workbook_path: Optional truth-audit workbook target path.
        concurrency: Question execution concurrency.
        fail_fast: Whether to stop on the first execution failure.
        ingest_doc_dir: Optional document directory to ingest before querying.

    Returns:
        JSON-serializable summary payload.
    """
    if ingest_doc_dir is not None:
        await _ingest_documents(ingest_doc_dir)

    cases = load_cases(questions_path)
    results = await _run_questions(cases, concurrency=concurrency, fail_fast=fail_fast)
    repaired_results, anomaly_repairs = await _repair_anomalous_results(results, fail_fast=fail_fast)

    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    raw_results_path.write_text(
        json.dumps([_serialize_platform_case_result(result) for result in repaired_results], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.write_text(
        json.dumps(_build_platform_submission_payload(repaired_results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    resolved_truth_audit_path: Path | None = None
    if docs_dir is not None and truth_audit_path is not None:
        truth_audit_path.parent.mkdir(parents=True, exist_ok=True)
        scaffold = build_truth_audit_scaffold(
            questions_path=questions_path,
            submission_path=submission_path,
            docs_dir=docs_dir,
            existing_scaffold_path=truth_audit_path if truth_audit_path.exists() else None,
        )
        truth_audit_path.write_text(
            json.dumps(scaffold, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        resolved_truth_audit_path = truth_audit_path
        if truth_audit_workbook_path is not None:
            truth_audit_workbook_path.parent.mkdir(parents=True, exist_ok=True)
            truth_audit_workbook_path.write_text(
                render_truth_audit_workbook(scaffold),
                encoding="utf-8",
            )

    summary = summarize_results(
        repaired_results,
        questions_path=questions_path,
        docs_dir=docs_dir,
        anomaly_repairs=anomaly_repairs,
        truth_audit_path=resolved_truth_audit_path,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
