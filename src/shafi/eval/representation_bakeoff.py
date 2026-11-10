"""Offline representation bakeoff helpers for embedder/reranker candidates."""

from __future__ import annotations

import csv
import json
import re
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

_MARKDOWN_LINK_RE = re.compile(r"^\[(?P<label>.+?)\]\((?P<url>https?://.+)\)$")


class ExternalRepresentationRow(BaseModel):
    """External benchmark row used as annotation rather than promotion truth.

    Args:
        model_name: Human-readable model name.
        source_url: Source URL from the benchmark row.
        retrieval_score: Retrieval score from the external benchmark.
        dimensions: Embedding dimensionality when present.
        max_tokens: Maximum supported tokens when present.
    """

    model_name: str
    source_url: str = ""
    retrieval_score: float = 0.0
    dimensions: int = 0
    max_tokens: int = 0


class LocalRepresentationMetric(BaseModel):
    """Local workload metric for one model and slice.

    Args:
        model_name: Candidate model name.
        slice_name: Local slice label.
        doc_recall: Document recall or purity proxy.
        page_recall: Page recall or purity proxy.
        same_doc_localization: Same-document page localization score.
        latency_ms: Mean latency for the candidate.
    """

    model_name: str
    slice_name: str
    doc_recall: float
    page_recall: float
    same_doc_localization: float
    latency_ms: float = 0.0


class RepresentationSummary(BaseModel):
    """Aggregated local-first summary for one representation candidate."""

    model_name: str
    slice_count: int
    mean_doc_recall: float
    mean_page_recall: float
    mean_same_doc_localization: float
    mean_latency_ms: float
    external_retrieval_score: float = 0.0
    source_url: str = ""


def load_external_benchmark_csv(path: Path) -> list[ExternalRepresentationRow]:
    """Load external benchmark rows from a CSV export.

    Args:
        path: CSV file exported from an external benchmark.

    Returns:
        list[ExternalRepresentationRow]: Parsed rows for annotation.
    """

    rows: list[ExternalRepresentationRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = str(row.get("Model") or "").strip()
            name, url = _parse_markdown_model_cell(raw_name)
            rows.append(
                ExternalRepresentationRow(
                    model_name=name,
                    source_url=url,
                    retrieval_score=_to_float(row.get("Retrieval")),
                    dimensions=int(_to_float(row.get("Embedding Dimensions"))),
                    max_tokens=int(_to_float(row.get("Max Tokens"))),
                )
            )
    return rows


def load_local_representation_metrics(path: Path) -> list[LocalRepresentationMetric]:
    """Load local bakeoff metrics from JSON or JSONL.

    Args:
        path: JSON or JSONL metrics file.

    Returns:
        list[LocalRepresentationMetric]: Parsed local metrics.
    """

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [LocalRepresentationMetric.model_validate_json(line) for line in raw_text.splitlines() if line.strip()]

    data = json.loads(raw_text)
    if isinstance(data, list):
        return [LocalRepresentationMetric.model_validate(item) for item in cast("list[object]", data)]
    return [LocalRepresentationMetric.model_validate(data)]


def summarize_representation_candidates(
    local_metrics: list[LocalRepresentationMetric],
    *,
    external_rows: list[ExternalRepresentationRow] | None = None,
) -> list[RepresentationSummary]:
    """Aggregate local metrics into local-first candidate summaries.

    Args:
        local_metrics: Local workload metrics grouped by model and slice.
        external_rows: Optional external benchmark annotation rows.

    Returns:
        list[RepresentationSummary]: Summaries ranked by local workload quality.
    """

    external_map = {row.model_name.casefold(): row for row in (external_rows or [])}
    by_model: dict[str, list[LocalRepresentationMetric]] = {}
    for metric in local_metrics:
        by_model.setdefault(metric.model_name, []).append(metric)

    summaries: list[RepresentationSummary] = []
    for model_name, metrics in by_model.items():
        external = external_map.get(model_name.casefold())
        slice_count = len(metrics)
        summaries.append(
            RepresentationSummary(
                model_name=model_name,
                slice_count=slice_count,
                mean_doc_recall=sum(metric.doc_recall for metric in metrics) / slice_count,
                mean_page_recall=sum(metric.page_recall for metric in metrics) / slice_count,
                mean_same_doc_localization=sum(metric.same_doc_localization for metric in metrics) / slice_count,
                mean_latency_ms=sum(metric.latency_ms for metric in metrics) / slice_count,
                external_retrieval_score=external.retrieval_score if external is not None else 0.0,
                source_url=external.source_url if external is not None else "",
            )
        )

    return sorted(
        summaries,
        key=lambda summary: (
            -(
                0.4 * summary.mean_doc_recall
                + 0.4 * summary.mean_page_recall
                + 0.2 * summary.mean_same_doc_localization
            ),
            summary.mean_latency_ms,
            summary.model_name.casefold(),
        ),
    )


def build_bakeoff_markdown(
    summaries: list[RepresentationSummary],
    *,
    external_rows_used: list[Path],
    local_metric_files: list[Path],
) -> str:
    """Render a markdown report for the representation bakeoff.

    Args:
        summaries: Ranked local-first summaries.
        external_rows_used: External benchmark CSVs used for annotation.
        local_metric_files: Local metric files used for ranking.

    Returns:
        str: Markdown report string.
    """

    lines = [
        "# Representation Bakeoff",
        "",
        "## Inputs",
        "",
        f"- external benchmark CSVs: {[str(path) for path in external_rows_used]}",
        f"- local metric files: {[str(path) for path in local_metric_files]}",
        "",
        "## Candidate ranking",
        "",
    ]
    if not summaries:
        lines.extend(
            [
                "- No local workload metrics provided.",
                "- External benchmark rows may be useful for candidate discovery but are not sufficient for promotion.",
            ]
        )
        return "\n".join(lines)

    for summary in summaries:
        lines.extend(
            [
                f"### {summary.model_name}",
                f"- local slice count: {summary.slice_count}",
                f"- mean doc recall: {summary.mean_doc_recall:.4f}",
                f"- mean page recall: {summary.mean_page_recall:.4f}",
                f"- mean same-doc localization: {summary.mean_same_doc_localization:.4f}",
                f"- mean latency ms: {summary.mean_latency_ms:.1f}",
                f"- external retrieval score: {summary.external_retrieval_score:.2f}",
                f"- source: {summary.source_url or 'n/a'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Promotion rule",
            "",
            "- Do not promote based on external benchmark marketing alone.",
            "- Promote only if local workload slices improve on document recall, page recall, and same-doc localization together.",
        ]
    )
    return "\n".join(lines)


def _parse_markdown_model_cell(value: str) -> tuple[str, str]:
    match = _MARKDOWN_LINK_RE.match(value)
    if match:
        return match.group("label"), match.group("url")
    return value, ""


def _to_float(value: object) -> float:
    try:
        return float(str(value or 0).strip())
    except ValueError:
        return 0.0
