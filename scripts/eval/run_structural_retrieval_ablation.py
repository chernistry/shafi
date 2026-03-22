"""Run a reviewed-golden retrieval ablation for additive segment and bridge lanes.

This script measures retrieval-only effects for compiler-driven additive lanes
without touching the frozen submission path. It compares four modes against the
same reviewed cases:

1. baseline: no additive lanes
2. segment_only: segment retrieval enabled
3. bridge_only: bridge-fact retrieval enabled
4. segment_bridge: both additive lanes enabled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from shafi.config import get_settings
from shafi.core.classifier import QueryClassifier
from shafi.core.embedding import EmbeddingClient
from shafi.core.qdrant import QdrantStore
from shafi.core.retriever import HybridRetriever
from shafi.eval.external_segment_shadow import ExternalSegmentFamily, route_external_segment_family
from shafi.eval.failure_cartography import load_reviewed_golden

if TYPE_CHECKING:
    from shafi.models import RetrievedChunk

JsonDict = dict[str, Any]

_TARGET_FAMILIES = {
    ExternalSegmentFamily.TITLE_CAPTION_CLAIMANT,
    ExternalSegmentFamily.EXACT_PROVISION,
    ExternalSegmentFamily.AUTHORITY_DATE_LAW_NUMBER,
}


@dataclass(frozen=True)
class ModeConfig:
    """Feature-flag combination under test.

    Args:
        name: Stable mode name for JSON output.
        enable_segment: Whether segment retrieval is enabled.
        enable_bridge: Whether bridge-fact retrieval is enabled.
    """

    name: str
    enable_segment: bool
    enable_bridge: bool


@dataclass(frozen=True)
class ReviewedCase:
    """Minimal reviewed-golden case record used by the audit.

    Args:
        question_id: Stable question identifier.
        question: Question text.
        gold_page_ids: Reviewed gold page identifiers.
        family: Routed family bucket.
        doc_refs: Extracted document references from the question.
    """

    question_id: str
    question: str
    gold_page_ids: tuple[str, ...]
    family: str
    doc_refs: tuple[str, ...]


@dataclass(frozen=True)
class CaseAudit:
    """Per-question retrieval audit record.

    Args:
        question_id: Stable question identifier.
        family: Routed family bucket.
        gold_page_ids: Reviewed gold pages.
        projected_page_ids: Top unique projected pages from retrieved chunks.
        hit: Whether any projected page overlaps gold.
        precision: Precision over projected pages.
        page_count: Number of projected pages retained.
        segment_used: Whether the segment lane activated with at least one hit.
        segment_hit_count: Count of retrieved segment units.
        bridge_used: Whether the bridge lane activated with at least one hit.
        bridge_hit_count: Count of retrieved bridge facts.
        source_unique_segment: Count of segment-only additions that survived merge.
        source_unique_bridge: Count of bridge-only additions that survived merge.
        source_survivor_segment: Count of final retrieved chunks with a segment source.
        source_survivor_bridge: Count of final retrieved chunks with a bridge source.
    """

    question_id: str
    family: str
    gold_page_ids: list[str]
    projected_page_ids: list[str]
    hit: bool
    precision: float
    page_count: int
    segment_used: bool
    segment_hit_count: int
    bridge_used: bool
    bridge_hit_count: int
    source_unique_segment: int
    source_unique_bridge: int
    source_survivor_segment: int
    source_survivor_bridge: int


_MODES = (
    ModeConfig(name="baseline", enable_segment=False, enable_bridge=False),
    ModeConfig(name="segment_only", enable_segment=True, enable_bridge=False),
    ModeConfig(name="bridge_only", enable_segment=False, enable_bridge=True),
    ModeConfig(name="segment_bridge", enable_segment=True, enable_bridge=True),
)


def _as_int(value: object) -> int:
    """Convert a debug payload scalar to int.

    Args:
        value: Arbitrary debug payload value.

    Returns:
        Integer representation, or `0` when conversion fails.
    """

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


@contextmanager
def _mode_env(mode: ModeConfig) -> Any:
    """Temporarily set additive retrieval flags for one audit mode.

    Args:
        mode: Mode under test.

    Yields:
        None. Environment and cached settings are restored afterwards.
    """

    previous_segment = os.environ.get("PIPELINE_ENABLE_SEGMENT_RETRIEVAL")
    previous_bridge = os.environ.get("PIPELINE_ENABLE_BRIDGE_FACT_RETRIEVAL")
    os.environ["PIPELINE_ENABLE_SEGMENT_RETRIEVAL"] = "true" if mode.enable_segment else "false"
    os.environ["PIPELINE_ENABLE_BRIDGE_FACT_RETRIEVAL"] = "true" if mode.enable_bridge else "false"
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous_segment is None:
            os.environ.pop("PIPELINE_ENABLE_SEGMENT_RETRIEVAL", None)
        else:
            os.environ["PIPELINE_ENABLE_SEGMENT_RETRIEVAL"] = previous_segment
        if previous_bridge is None:
            os.environ.pop("PIPELINE_ENABLE_BRIDGE_FACT_RETRIEVAL", None)
        else:
            os.environ["PIPELINE_ENABLE_BRIDGE_FACT_RETRIEVAL"] = previous_bridge
        get_settings.cache_clear()


def _page_id_from_chunk(chunk: RetrievedChunk) -> str:
    """Project a retrieved chunk to a platform-valid page identifier.

    Args:
        chunk: Retrieved chunk from the hybrid retriever.

    Returns:
        Page identifier, or an empty string if unavailable.
    """

    if not chunk.section_path.startswith("page:"):
        return ""
    try:
        page_num = int(chunk.section_path.split(":", 1)[1])
    except (TypeError, ValueError, IndexError):
        return ""
    return f"{chunk.doc_id}_{page_num}" if page_num > 0 else ""


def _project_page_ids(chunks: list[RetrievedChunk], *, limit: int) -> list[str]:
    """Collapse retrieved chunks into a bounded unique page list.

    Args:
        chunks: Retrieved chunks ordered by retriever score.
        limit: Maximum number of unique pages to retain.

    Returns:
        Unique page identifiers in retrieval order.
    """

    ordered: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        page_id = _page_id_from_chunk(chunk)
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        ordered.append(page_id)
        if len(ordered) >= limit:
            break
    return ordered


def _load_cases(golden_path: Path) -> list[ReviewedCase]:
    """Load target reviewed cases and route them into audit families.

    Args:
        golden_path: Reviewed-golden JSON path.

    Returns:
        Routed reviewed cases for the targeted families only.
    """

    reviewed = load_reviewed_golden(golden_path)
    cases: list[ReviewedCase] = []
    for qid, row in reviewed.items():
        doc_refs = tuple(QueryClassifier.extract_doc_refs(row.question))
        family = route_external_segment_family(row.question, doc_refs=list(doc_refs))
        if family not in _TARGET_FAMILIES:
            continue
        cases.append(
            ReviewedCase(
                question_id=qid,
                question=row.question,
                gold_page_ids=tuple(row.golden_page_ids),
                family=family.value,
                doc_refs=doc_refs,
            )
        )
    return sorted(cases, key=lambda case: (case.family, case.question_id))


async def _collection_info(store: QdrantStore) -> JsonDict:
    """Collect collection existence and point counts for the current store.

    Args:
        store: Qdrant store using the already-selected collection names.

    Returns:
        JSON-serializable collection metadata.
    """

    collections = {
        "chunks": store.collection_name,
        "shadow_chunks": store.shadow_collection_name,
        "pages": store.page_collection_name,
        "segments": store.segment_collection_name,
        "bridge_facts": store.bridge_fact_collection_name,
    }
    out: JsonDict = {}
    for label, collection_name in collections.items():
        exists = await store.client.collection_exists(collection_name)
        count = 0
        if exists:
            result = await store.client.count(collection_name=collection_name, exact=True)
            count = int(getattr(result, "count", 0))
        out[label] = {
            "collection_name": collection_name,
            "exists": exists,
            "point_count": count,
        }
    return out


async def _run_mode(
    *,
    mode: ModeConfig,
    cases: list[ReviewedCase],
    top_k: int,
) -> JsonDict:
    """Execute one retrieval mode across all target reviewed cases.

    Args:
        mode: Feature flags for the current run.
        cases: Target reviewed cases.
        top_k: Maximum unique projected pages kept per case.
    Returns:
        JSON summary for the mode.
    """

    case_rows: list[CaseAudit] = []
    family_buckets: dict[str, list[CaseAudit]] = defaultdict(list)
    errors: list[JsonDict] = []
    with _mode_env(mode):
        store = QdrantStore()
        embedder = EmbeddingClient()
        retriever = HybridRetriever(store=store, embedder=embedder)
        try:
            for case in cases:
                try:
                    retrieved = await retriever.retrieve(
                        case.question,
                        doc_refs=list(case.doc_refs),
                        top_k=max(8, top_k * 2),
                    )
                    debug = retriever.get_last_retrieval_debug()
                    projected_pages = _project_page_ids(retrieved, limit=top_k)
                    gold_pages = set(case.gold_page_ids)
                    hit = any(page_id in gold_pages for page_id in projected_pages)
                    precision = (
                        sum(1 for page_id in projected_pages if page_id in gold_pages) / len(projected_pages)
                        if projected_pages
                        else 0.0
                    )
                    source_unique = cast("dict[str, object]", debug.get("source_unique_additions") or {})
                    source_survivors = cast("dict[str, object]", debug.get("source_survivors") or {})
                    row = CaseAudit(
                        question_id=case.question_id,
                        family=case.family,
                        gold_page_ids=list(case.gold_page_ids),
                        projected_page_ids=projected_pages,
                        hit=hit,
                        precision=precision,
                        page_count=len(projected_pages),
                        segment_used=bool(debug.get("segment_retrieval_used")),
                        segment_hit_count=_as_int(debug.get("segment_hit_count")),
                        bridge_used=bool(debug.get("bridge_fact_retrieval_used")),
                        bridge_hit_count=_as_int(debug.get("bridge_fact_hit_count")),
                        source_unique_segment=_as_int(source_unique.get("segment")),
                        source_unique_bridge=_as_int(source_unique.get("bridge")),
                        source_survivor_segment=_as_int(source_survivors.get("segment")),
                        source_survivor_bridge=_as_int(source_survivors.get("bridge")),
                    )
                    case_rows.append(row)
                    family_buckets[case.family].append(row)
                except Exception as exc:  # pragma: no cover - surfaced in artifact output
                    errors.append(
                        {
                            "question_id": case.question_id,
                            "family": case.family,
                            "question": case.question,
                            "error": repr(exc),
                        }
                    )
        finally:
            await embedder.close()
            await store.close()

    def summarize(rows: list[CaseAudit]) -> JsonDict:
        """Aggregate one case bucket into rate metrics.

        Args:
            rows: Case-level audit rows for one bucket.

        Returns:
            Aggregated metrics.
        """

        if not rows:
            return {
                "case_count": 0,
                "hit_rate": 0.0,
                "precision": 0.0,
                "avg_page_count": 0.0,
                "segment_used_rate": 0.0,
                "segment_hit_count_avg": 0.0,
                "bridge_used_rate": 0.0,
                "bridge_hit_count_avg": 0.0,
                "segment_unique_additions_avg": 0.0,
                "bridge_unique_additions_avg": 0.0,
                "segment_survivors_avg": 0.0,
                "bridge_survivors_avg": 0.0,
            }
        count = len(rows)
        return {
            "case_count": count,
            "hit_rate": round(sum(1.0 for row in rows if row.hit) / count, 6),
            "precision": round(sum(row.precision for row in rows) / count, 6),
            "avg_page_count": round(sum(row.page_count for row in rows) / count, 6),
            "segment_used_rate": round(sum(1.0 for row in rows if row.segment_used) / count, 6),
            "segment_hit_count_avg": round(sum(row.segment_hit_count for row in rows) / count, 6),
            "bridge_used_rate": round(sum(1.0 for row in rows if row.bridge_used) / count, 6),
            "bridge_hit_count_avg": round(sum(row.bridge_hit_count for row in rows) / count, 6),
            "segment_unique_additions_avg": round(sum(row.source_unique_segment for row in rows) / count, 6),
            "bridge_unique_additions_avg": round(sum(row.source_unique_bridge for row in rows) / count, 6),
            "segment_survivors_avg": round(sum(row.source_survivor_segment for row in rows) / count, 6),
            "bridge_survivors_avg": round(sum(row.source_survivor_bridge for row in rows) / count, 6),
        }

    family_metrics = {family: summarize(rows) for family, rows in sorted(family_buckets.items())}
    return {
        "mode": mode.name,
        "enable_segment_retrieval": mode.enable_segment,
        "enable_bridge_fact_retrieval": mode.enable_bridge,
        "overall": summarize(case_rows),
        "families": family_metrics,
        "cases": [asdict(row) for row in case_rows],
        "errors": errors,
    }


def _render_markdown(payload: JsonDict) -> str:
    """Render a short Markdown summary for operator review.

    Args:
        payload: JSON output generated by this script.

    Returns:
        Markdown summary.
    """

    lines = [
        "# Structural Retrieval Ablation",
        "",
        "## Collections",
        "",
    ]
    collections = cast("dict[str, JsonDict]", payload["collections"])
    for label, info in collections.items():
        lines.append(
            f"- `{label}`: exists=`{info['exists']}` points=`{info['point_count']}` collection=`{info['collection_name']}`"
        )
    lines.extend(["", "## Modes", ""])
    modes = cast("list[JsonDict]", payload["modes"])
    for mode in modes:
        overall = cast("JsonDict", mode["overall"])
        lines.extend(
            [
                f"### {mode['mode']}",
                f"- hit_rate: `{overall['hit_rate']}`",
                f"- precision: `{overall['precision']}`",
                f"- avg_page_count: `{overall['avg_page_count']}`",
                f"- segment_used_rate: `{overall['segment_used_rate']}`",
                f"- bridge_used_rate: `{overall['bridge_used_rate']}`",
                "",
            ]
        )
        families = cast("dict[str, JsonDict]", mode["families"])
        for family, metrics in families.items():
            lines.append(
                f"- `{family}`: hit=`{metrics['hit_rate']}` precision=`{metrics['precision']}` pages=`{metrics['avg_page_count']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def _run(args: argparse.Namespace) -> JsonDict:
    """Run the full ablation across all configured modes.

    Args:
        args: Parsed CLI args.

    Returns:
        Full JSON payload with collection info and mode summaries.
    """

    cases = _load_cases(args.golden)
    store = QdrantStore()
    try:
        payload = {
            "golden_path": str(args.golden),
            "target_case_count": len(cases),
            "target_families": sorted(family.value for family in _TARGET_FAMILIES),
            "collections": await _collection_info(store),
            "modes": [],
        }
        for mode in _MODES:
            mode_payload = await _run_mode(
                mode=mode,
                cases=cases,
                top_k=int(args.projected_top_k),
            )
            cast("list[JsonDict]", payload["modes"]).append(mode_payload)
        return payload
    finally:
        await store.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description="Run a reviewed-golden structural retrieval ablation.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path(".sdd/golden/reviewed/reviewed_all_100.json"),
        help="Reviewed golden JSON path.",
    )
    parser.add_argument(
        "--projected-top-k",
        type=int,
        default=4,
        help="Maximum unique projected pages retained per case.",
    )
    parser.add_argument("--out-json", type=Path, required=True, help="Path for the JSON output artifact.")
    parser.add_argument("--out-md", type=Path, required=True, help="Path for the Markdown summary artifact.")
    return parser.parse_args()


def main() -> int:
    """Run the CLI entrypoint.

    Returns:
        Zero on success.
    """

    args = parse_args()
    payload = asyncio.run(_run(args))
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
