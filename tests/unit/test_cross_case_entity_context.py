"""Tests for ensure_cross_case_entity_context — cross-case entity retrieval helper.

Validates that when a query references two case IDs (e.g. "Is there any main party
that appeared in both cases CFI 035/2025 and CFI 081/2023?"), the function promotes
at least one chunk from each referenced case doc to the front of the context window.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from shafi.models import DocType, RankedChunk, RetrievedChunk


def _retrieved(chunk_id: str, doc_id: str, doc_title: str, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        doc_type=DocType.CASE_LAW,
        text=text or f"Text from {doc_title}.",
        score=0.9,
    )


def _ranked(chunk_id: str, doc_id: str, doc_title: str, text: str = "") -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        doc_type=DocType.CASE_LAW,
        text=text or f"Text from {doc_title}.",
        retrieval_score=0.9,
        rerank_score=0.8,
    )


def _make_pipeline(chunks_raw: list[RetrievedChunk]) -> MagicMock:
    """Build a mock pipeline builder with real case_ref_identity_score behaviour."""
    from shafi.core.pipeline.retrieval_seed_selection import case_ref_identity_score
    from shafi.core.pipeline.support_query_primitives import normalize_support_text

    pipeline = MagicMock()
    pipeline.normalize_support_text.side_effect = normalize_support_text

    def _case_score(*, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        return case_ref_identity_score(pipeline, ref=ref, chunk=chunk)

    pipeline.case_ref_identity_score.side_effect = _case_score

    def _raw_to_ranked(raw: RetrievedChunk) -> RankedChunk:
        return RankedChunk(
            chunk_id=raw.chunk_id,
            doc_id=raw.doc_id,
            doc_title=raw.doc_title,
            doc_type=raw.doc_type,
            text=raw.text,
            retrieval_score=raw.score,
            rerank_score=0.0,
        )

    pipeline.raw_to_ranked.side_effect = _raw_to_ranked
    return pipeline


class TestEnsureCrossCaseEntityContext:
    """Unit tests for ensure_cross_case_entity_context."""

    def test_promotes_both_case_chunks_when_two_refs_present(self) -> None:
        """Both referenced case docs should appear in the top context slots."""
        from shafi.core.pipeline.retrieval_boolean_handlers import (
            ensure_cross_case_entity_context,
        )

        # Two case docs; dense retrieval only found chunks from CFI 081/2023
        cfi_035_chunk = _retrieved("c1", "doc-035", "CFI 035/2025 Alpha v Beta")
        cfi_081_a = _retrieved("c2", "doc-081", "CFI 081/2023 Gamma v Delta", text="Case CFI 081/2023 — Gamma v Delta")
        cfi_081_b = _retrieved("c3", "doc-081", "CFI 081/2023 Gamma v Delta", text="Judgment CFI 081/2023 parties list")

        # Reranked order from dense retrieval: only CFI 081 chunks in top-N
        reranked = [
            _ranked("c2", "doc-081", "CFI 081/2023 Gamma v Delta"),
            _ranked("c3", "doc-081", "CFI 081/2023 Gamma v Delta"),
        ]
        retrieved_all = [cfi_081_a, cfi_081_b, cfi_035_chunk]

        pipeline = _make_pipeline(retrieved_all)

        query = "Is there any main party that appeared in both cases CFI 035/2025 and CFI 081/2023 at any point?"
        result = ensure_cross_case_entity_context(
            pipeline, query=query, reranked=reranked, retrieved=retrieved_all, top_n=4
        )

        doc_ids = [chunk.doc_id for chunk in result]
        assert "doc-035" in doc_ids, "CFI 035/2025 chunk missing from context"
        assert "doc-081" in doc_ids, "CFI 081/2023 chunk missing from context"

    def test_no_op_when_single_case_ref(self) -> None:
        """When query has only 1 case ref, the function returns reranked unchanged."""
        from shafi.core.pipeline.retrieval_boolean_handlers import (
            ensure_cross_case_entity_context,
        )

        chunk = _retrieved("c1", "doc-035", "CFI 035/2025 Alpha v Beta")
        reranked = [_ranked("c1", "doc-035", "CFI 035/2025 Alpha v Beta")]
        pipeline = _make_pipeline([chunk])

        query = "What is the outcome of CFI 035/2025?"
        result = ensure_cross_case_entity_context(pipeline, query=query, reranked=reranked, retrieved=[chunk], top_n=3)
        assert result == reranked[:3]

    def test_no_op_when_no_case_refs(self) -> None:
        """When query has no case refs, return reranked unchanged."""
        from shafi.core.pipeline.retrieval_boolean_handlers import (
            ensure_cross_case_entity_context,
        )

        reranked = [_ranked("c1", "d1", "Employment Law")]
        pipeline = _make_pipeline([])

        result = ensure_cross_case_entity_context(
            pipeline, query="What is the law number?", reranked=reranked, retrieved=[], top_n=3
        )
        assert result == reranked[:3]

    def test_respects_top_n_limit(self) -> None:
        """Result should not exceed top_n chunks."""
        from shafi.core.pipeline.retrieval_boolean_handlers import (
            ensure_cross_case_entity_context,
        )

        raw_chunks = [_retrieved(f"c{i}", f"doc-{i}", f"CA 00{i}/2025 X v Y") for i in range(5)]
        reranked = [_ranked(f"c{i}", f"doc-{i}", f"CA 00{i}/2025 X v Y") for i in range(5)]

        pipeline = _make_pipeline(raw_chunks)
        query = "Are there common parties in CA 001/2025 and CA 002/2025?"
        result = ensure_cross_case_entity_context(
            pipeline, query=query, reranked=reranked, retrieved=raw_chunks, top_n=3
        )
        assert len(result) <= 3

    def test_empty_retrieved_returns_empty(self) -> None:
        """Empty retrieved list should return empty or reranked[:top_n] gracefully."""
        from shafi.core.pipeline.retrieval_boolean_handlers import (
            ensure_cross_case_entity_context,
        )

        pipeline = _make_pipeline([])
        result = ensure_cross_case_entity_context(
            pipeline,
            query="CFI 035/2025 and CFI 081/2023",
            reranked=[],
            retrieved=[],
            top_n=4,
        )
        assert result == []
