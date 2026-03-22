"""Unit tests for the RetrieverEscalationPolicy."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag_challenge.core.retrieval_escalation import (
    EscalationResult,
    RetrieverEscalationPolicy,
    _merge_chunks,
    extract_title_terms,
)
from rag_challenge.models import DocType, RetrievedChunk


def _make_chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    """Create a minimal RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc_1",
        doc_title="Test Document",
        doc_type=DocType.STATUTE,
        text=f"text for {chunk_id}",
        score=score,
        page_number=1,
        section_path="",
    )


class TestExtractTitleTerms:
    def test_extracts_law_number(self) -> None:
        terms = extract_title_terms("What does Law No. 5 of 2004 say?")
        assert len(terms) >= 1
        assert any("5" in t and "2004" in t for t in terms)

    def test_extracts_regulation(self) -> None:
        terms = extract_title_terms("Under Regulation 3/2020, what is required?")
        assert len(terms) >= 1

    def test_no_titles(self) -> None:
        terms = extract_title_terms("What is the claim value?")
        assert terms == []

    def test_multiple_titles(self) -> None:
        terms = extract_title_terms("Compare Law No. 1 of 2020 and Decree 5/2019")
        assert len(terms) >= 2


class TestMergeChunks:
    def test_empty_base(self) -> None:
        new = [_make_chunk("a", 0.8)]
        merged, added = _merge_chunks([], new)
        assert len(merged) == 1
        assert added == 1

    def test_no_overlap(self) -> None:
        base = [_make_chunk("a", 0.8)]
        new = [_make_chunk("b", 0.6)]
        merged, added = _merge_chunks(base, new)
        assert len(merged) == 2
        assert added == 1

    def test_overlap_keeps_higher_score(self) -> None:
        base = [_make_chunk("a", 0.5)]
        new = [_make_chunk("a", 0.9)]
        merged, added = _merge_chunks(base, new)
        assert len(merged) == 1
        assert merged[0].score == 0.9
        assert added == 0  # same chunk_id, not "new"

    def test_overlap_keeps_existing_if_higher(self) -> None:
        base = [_make_chunk("a", 0.9)]
        new = [_make_chunk("a", 0.3)]
        merged, added = _merge_chunks(base, new)
        assert len(merged) == 1
        assert merged[0].score == 0.9
        assert added == 0

    def test_sorted_by_score_desc(self) -> None:
        base = [_make_chunk("a", 0.3)]
        new = [_make_chunk("b", 0.9), _make_chunk("c", 0.6)]
        merged, _ = _merge_chunks(base, new)
        scores = [c.score for c in merged]
        assert scores == sorted(scores, reverse=True)

    def test_purely_additive(self) -> None:
        base = [_make_chunk("a", 0.8), _make_chunk("b", 0.6)]
        new = [_make_chunk("c", 0.1)]
        merged, added = _merge_chunks(base, new)
        base_ids = {c.chunk_id for c in base}
        merged_ids = {c.chunk_id for c in merged}
        assert base_ids.issubset(merged_ids), "Merge must never remove base chunks"
        assert added == 1


class TestRetrieverEscalationPolicy:
    def _mock_retriever(self, side_effect: list[list[RetrievedChunk]] | None = None) -> AsyncMock:
        retriever = AsyncMock()
        if side_effect is not None:
            retriever.retrieve = AsyncMock(side_effect=side_effect)
        else:
            retriever.retrieve = AsyncMock(return_value=[])
        return retriever

    @pytest.mark.asyncio
    async def test_no_escalation_above_threshold(self) -> None:
        retriever = self._mock_retriever()
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)
        initial = [_make_chunk("a", 0.8)]

        result = await policy.retrieve_with_escalation(
            query="test query",
            initial_chunks=initial,
            top_rerank_score=0.7,  # above threshold
        )

        assert result.chunks == initial
        assert result.levels_applied == []
        retriever.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_escalation_below_threshold(self) -> None:
        level1_new = [_make_chunk("b", 0.6)]
        level2_new = [_make_chunk("c", 0.4)]
        retriever = self._mock_retriever(side_effect=[level1_new, level2_new])
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)
        initial = [_make_chunk("a", 0.8)]

        result = await policy.retrieve_with_escalation(
            query="What is the claim value?",  # no title terms → no L3
            initial_chunks=initial,
            top_rerank_score=0.3,  # below threshold
        )

        assert len(result.chunks) == 3  # a + b + c
        assert 1 in result.levels_applied
        assert 2 in result.levels_applied
        assert 3 not in result.levels_applied  # no title terms
        assert result.chunks_added_per_level[1] == 1
        assert result.chunks_added_per_level[2] == 1

    @pytest.mark.asyncio
    async def test_escalation_with_title_terms(self) -> None:
        level1_new = [_make_chunk("b", 0.6)]
        level2_new = [_make_chunk("c", 0.4)]
        level3_new = [_make_chunk("d", 0.3)]
        retriever = self._mock_retriever(side_effect=[level1_new, level2_new, level3_new])
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)
        initial = [_make_chunk("a", 0.8)]

        result = await policy.retrieve_with_escalation(
            query="What does Law No. 5 of 2004 require?",
            initial_chunks=initial,
            top_rerank_score=0.2,
        )

        assert len(result.chunks) == 4  # a + b + c + d
        assert 3 in result.levels_applied

    @pytest.mark.asyncio
    async def test_escalation_preserves_initial_chunks(self) -> None:
        """Escalation must never remove initial chunks (purely additive)."""
        retriever = self._mock_retriever(side_effect=[[], []])
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)
        initial = [_make_chunk("a", 0.8), _make_chunk("b", 0.6)]

        result = await policy.retrieve_with_escalation(
            query="test query",
            initial_chunks=initial,
            top_rerank_score=0.1,
        )

        result_ids = {c.chunk_id for c in result.chunks}
        initial_ids = {c.chunk_id for c in initial}
        assert initial_ids.issubset(result_ids)

    @pytest.mark.asyncio
    async def test_escalation_deduplicates(self) -> None:
        """Same chunk_id returned by multiple levels → keep highest score."""
        level1 = [_make_chunk("a", 0.9)]  # same id as initial but higher score
        level2 = [_make_chunk("a", 0.3)]  # same id, lower score
        retriever = self._mock_retriever(side_effect=[level1, level2])
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)
        initial = [_make_chunk("a", 0.5)]

        result = await policy.retrieve_with_escalation(
            query="test query",
            initial_chunks=initial,
            top_rerank_score=0.2,
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].score == 0.9  # highest score wins

    @pytest.mark.asyncio
    async def test_threshold_property(self) -> None:
        retriever = self._mock_retriever()
        policy = RetrieverEscalationPolicy(retriever, threshold=0.42)
        assert policy.threshold == 0.42

    @pytest.mark.asyncio
    async def test_custom_prefetch_limits(self) -> None:
        retriever = self._mock_retriever(side_effect=[[], []])
        policy = RetrieverEscalationPolicy(
            retriever,
            threshold=0.5,
            prefetch_dense=200,
            prefetch_sparse=200,
        )

        await policy.retrieve_with_escalation(
            query="test query",
            initial_chunks=[],
            top_rerank_score=0.1,
        )

        # Level 1 call should use custom prefetch limits
        call_kwargs = retriever.retrieve.call_args_list[0].kwargs
        assert call_kwargs.get("prefetch_dense") == 200
        assert call_kwargs.get("prefetch_sparse") == 200

    @pytest.mark.asyncio
    async def test_doc_refs_passed_to_retriever(self) -> None:
        retriever = self._mock_retriever(side_effect=[[], []])
        policy = RetrieverEscalationPolicy(retriever, threshold=0.5)

        await policy.retrieve_with_escalation(
            query="test query",
            initial_chunks=[],
            top_rerank_score=0.1,
            doc_refs=["DIFC/2020/001"],
        )

        # L1 and L2 should both pass doc_refs
        for call in retriever.retrieve.call_args_list:
            assert call.kwargs.get("doc_refs") == ["DIFC/2020/001"]
