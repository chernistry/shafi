# pyright: reportPrivateUsage=false
"""Unit tests for grounding evidence selector core logic.

Tests cover the pure/deterministic functions in evidence_selector.py:
- _normalize_answer_value
- answer_requires_empty_grounding
- _chunk_id_to_page_id
- _page_ids_from_context_chunks / _page_ids_from_chunk_ids
- _should_activate_sidecar
- _score_candidates
- _select_minimal_pages
- _map_support_fact_results
- _select_doc_scope
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from shafi.core.grounding.evidence_selector import (
    GroundingEvidenceSelector,
    _normalize_answer_value,
    answer_requires_empty_grounding,
)
from shafi.models.schemas import (
    DocType,
    QueryScopePrediction,
    RankedChunk,
    RetrievedPage,
    RetrievedSupportFact,
    ScopeMode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_selector() -> GroundingEvidenceSelector:
    """Create a GroundingEvidenceSelector with minimal mocked dependencies."""
    settings = MagicMock()
    settings.enable_trained_page_scorer = False
    settings.trained_page_scorer_model_path = ""
    return GroundingEvidenceSelector(
        retriever=MagicMock(),
        store=MagicMock(),
        embedder=MagicMock(),
        sparse_encoder=None,
        pipeline_settings=settings,
    )


def _make_scope(
    mode: ScopeMode = ScopeMode.COMPARE_PAIR,
    page_budget: int = 2,
    target_page_roles: list[str] | None = None,
    hard_anchor_strings: list[str] | None = None,
    should_force_empty: bool = False,
) -> QueryScopePrediction:
    return QueryScopePrediction(
        scope_mode=mode,
        page_budget=page_budget,
        target_page_roles=target_page_roles or [],
        hard_anchor_strings=hard_anchor_strings or [],
        should_force_empty_grounding_on_null=should_force_empty,
    )


def _make_page(page_id: str, score: float = 0.5) -> RetrievedPage:
    doc_id, _, page_num_str = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id,
        page_num=int(page_num_str),
        score=score,
    )


def _make_fact(
    page_id: str,
    score: float = 0.5,
    fact_type: str = "general",
    normalized_value: str = "",
    page_role: str = "",
) -> RetrievedSupportFact:
    doc_id, _, page_num_str = page_id.rpartition("_")
    return RetrievedSupportFact(
        fact_id=f"fact_{page_id}",
        doc_id=doc_id,
        page_id=page_id,
        page_num=int(page_num_str),
        fact_type=fact_type,
        normalized_value=normalized_value,
        page_role=page_role,
        score=score,
    )


def _make_chunk(doc_id: str, page_idx: int, chunk_idx: int = 0) -> RankedChunk:
    """Create a RankedChunk with properly formatted chunk_id.

    chunk_id format: doc_id:page_idx:chunk_idx:hash (page_idx is 0-indexed).
    """
    return RankedChunk(
        chunk_id=f"{doc_id}:{page_idx}:{chunk_idx}:abc123",
        doc_id=doc_id,
        doc_title="Test Doc",
        doc_type=DocType.CASE_LAW,
        text="test text",
        retrieval_score=0.5,
        rerank_score=0.5,
    )


# ===========================================================================
# _normalize_answer_value
# ===========================================================================


class TestNormalizeAnswerValue:
    """Tests for the _normalize_answer_value standalone function."""

    def test_empty_string_returns_empty(self) -> None:
        assert _normalize_answer_value("", "text") == ""

    def test_none_like_returns_empty(self) -> None:
        assert _normalize_answer_value("", "boolean") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert _normalize_answer_value("   ", "text") == ""

    def test_boolean_yes_normalized(self) -> None:
        assert _normalize_answer_value("Yes", "boolean") == "true"
        assert _normalize_answer_value("yes", "boolean") == "true"
        assert _normalize_answer_value("TRUE", "boolean") == "true"
        assert _normalize_answer_value("true", "boolean") == "true"

    def test_boolean_no_normalized(self) -> None:
        assert _normalize_answer_value("No", "boolean") == "false"
        assert _normalize_answer_value("no", "boolean") == "false"
        assert _normalize_answer_value("FALSE", "boolean") == "false"
        assert _normalize_answer_value("false", "boolean") == "false"

    def test_boolean_null_returns_empty(self) -> None:
        assert _normalize_answer_value("null", "boolean") == ""
        assert _normalize_answer_value("None", "boolean") == ""
        assert _normalize_answer_value("no information", "boolean") == ""

    def test_boolean_non_canonical_passes_through(self) -> None:
        result = _normalize_answer_value("maybe", "boolean")
        assert result == "maybe"

    def test_non_boolean_type_no_normalization(self) -> None:
        assert _normalize_answer_value("Yes", "text") == "Yes"
        assert _normalize_answer_value("No", "number") == "No"

    def test_strips_trailing_punctuation(self) -> None:
        assert _normalize_answer_value("answer.", "text") == "answer"
        assert _normalize_answer_value("answer;", "text") == "answer"
        assert _normalize_answer_value("answer:", "text") == "answer"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_answer_value("hello   world", "text") == "hello world"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert _normalize_answer_value("  hello  ", "text") == "hello"


# ===========================================================================
# answer_requires_empty_grounding
# ===========================================================================


class TestAnswerRequiresEmptyGrounding:
    """Tests for the answer_requires_empty_grounding function."""

    def test_null_answers(self) -> None:
        assert answer_requires_empty_grounding("null") is True
        assert answer_requires_empty_grounding("None") is True
        assert answer_requires_empty_grounding("N/A") is True
        assert answer_requires_empty_grounding("no information") is True

    def test_empty_string(self) -> None:
        assert answer_requires_empty_grounding("") is True

    def test_long_null_phrase(self) -> None:
        assert answer_requires_empty_grounding("There is no information on this question") is True

    def test_real_answers_not_empty(self) -> None:
        assert answer_requires_empty_grounding("Yes") is False
        assert answer_requires_empty_grounding("42") is False
        assert answer_requires_empty_grounding("John Smith") is False

    def test_case_insensitive(self) -> None:
        assert answer_requires_empty_grounding("NULL") is True
        assert answer_requires_empty_grounding("No Information") is True

    def test_whitespace_stripped(self) -> None:
        assert answer_requires_empty_grounding("  null  ") is True

    def test_punctuation_stripped(self) -> None:
        assert answer_requires_empty_grounding("null.") is True
        assert answer_requires_empty_grounding("N/A;") is True


# ===========================================================================
# _chunk_id_to_page_id
# ===========================================================================


class TestChunkIdToPageId:
    """Tests for the _chunk_id_to_page_id static method."""

    def test_standard_conversion(self) -> None:
        # 0-indexed page_idx → 1-indexed page_num
        assert GroundingEvidenceSelector._chunk_id_to_page_id("docA:0:0:abc") == "docA_1"
        assert GroundingEvidenceSelector._chunk_id_to_page_id("docA:4:2:xyz") == "docA_5"

    def test_no_colon_returns_empty(self) -> None:
        assert GroundingEvidenceSelector._chunk_id_to_page_id("malformed") == ""

    def test_non_numeric_page_idx_returns_empty(self) -> None:
        assert GroundingEvidenceSelector._chunk_id_to_page_id("docA:abc:0:hash") == ""

    def test_empty_doc_id_returns_empty(self) -> None:
        assert GroundingEvidenceSelector._chunk_id_to_page_id(":0:0:hash") == ""

    def test_single_colon(self) -> None:
        assert GroundingEvidenceSelector._chunk_id_to_page_id("docA:3") == "docA_4"

    def test_whitespace_stripped(self) -> None:
        assert GroundingEvidenceSelector._chunk_id_to_page_id(" docA : 2 :0:h") == "docA_3"


# ===========================================================================
# _page_ids_from_context_chunks / _page_ids_from_chunk_ids
# ===========================================================================


class TestPageIdsFromContextChunks:
    """Tests for _page_ids_from_context_chunks and _page_ids_from_chunk_ids."""

    def test_basic_extraction(self) -> None:
        chunks = [_make_chunk("docA", 0), _make_chunk("docA", 4)]
        result = GroundingEvidenceSelector._page_ids_from_context_chunks(
            context_chunks=chunks,
            allowed_doc_ids={"docA"},
        )
        assert result == ["docA_1", "docA_5"]

    def test_deduplication(self) -> None:
        chunks = [_make_chunk("docA", 0), _make_chunk("docA", 0)]
        result = GroundingEvidenceSelector._page_ids_from_context_chunks(
            context_chunks=chunks,
            allowed_doc_ids={"docA"},
        )
        assert result == ["docA_1"]

    def test_doc_scope_filtering(self) -> None:
        chunks = [_make_chunk("docA", 0), _make_chunk("docB", 1)]
        result = GroundingEvidenceSelector._page_ids_from_context_chunks(
            context_chunks=chunks,
            allowed_doc_ids={"docA"},  # docB excluded
        )
        assert result == ["docA_1"]

    def test_page_ids_from_chunk_ids_basic(self) -> None:
        chunk_ids = ["docA:0:0:h1", "docA:2:1:h2"]
        result = GroundingEvidenceSelector._page_ids_from_chunk_ids(
            chunk_ids=chunk_ids,
            allowed_doc_ids={"docA"},
        )
        assert result == ["docA_1", "docA_3"]

    def test_page_ids_from_chunk_ids_filters_docs(self) -> None:
        chunk_ids = ["docA:0:0:h1", "docB:1:0:h2"]
        result = GroundingEvidenceSelector._page_ids_from_chunk_ids(
            chunk_ids=chunk_ids,
            allowed_doc_ids={"docA"},
        )
        assert result == ["docA_1"]

    def test_malformed_chunk_ids_skipped(self) -> None:
        chunk_ids = ["malformed", "docA:0:0:h"]
        result = GroundingEvidenceSelector._page_ids_from_chunk_ids(
            chunk_ids=chunk_ids,
            allowed_doc_ids={"docA"},
        )
        assert result == ["docA_1"]


# ===========================================================================
# _should_activate_sidecar
# ===========================================================================


class TestShouldActivateSidecar:
    """Tests for the _should_activate_sidecar static method."""

    def test_compare_pair_always_active(self) -> None:
        scope = _make_scope(mode=ScopeMode.COMPARE_PAIR)
        chunks = [_make_chunk("docA", 0)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="Compare the judge in CFI 001/2024 and CFI 002/2024",
                scope=scope,
                answer_type="boolean",
                context_chunks=chunks,
            )
            is True
        )

    def test_full_case_files_always_active(self) -> None:
        scope = _make_scope(mode=ScopeMode.FULL_CASE_FILES)
        chunks = [_make_chunk("docA", 0)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="Look through all documents for the case outcome",
                scope=scope,
                answer_type="text",
                context_chunks=chunks,
            )
            is True
        )

    def test_negative_unanswerable_always_active(self) -> None:
        scope = _make_scope(mode=ScopeMode.NEGATIVE_UNANSWERABLE)
        chunks = []
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="What was the jury finding?",
                scope=scope,
                answer_type="text",
                context_chunks=chunks,
            )
            is True
        )

    def test_explicit_page_active_single_doc(self) -> None:
        scope = _make_scope(mode=ScopeMode.EXPLICIT_PAGE)
        chunks = [_make_chunk("docA", 0)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="What is stated on page 2?",
                scope=scope,
                answer_type="text",
                context_chunks=chunks,
            )
            is True
        )

    def test_explicit_page_inactive_multi_doc(self) -> None:
        scope = _make_scope(mode=ScopeMode.EXPLICIT_PAGE)
        chunks = [_make_chunk("docA", 0), _make_chunk("docB", 1)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="What is stated on page 2?",
                scope=scope,
                answer_type="text",
                context_chunks=chunks,
            )
            is False
        )

    def test_single_field_authority_active(self) -> None:
        scope = _make_scope(mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC)
        chunks = [_make_chunk("docA", 0)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="What is the date of issue of the law?",
                scope=scope,
                answer_type="number",
                context_chunks=chunks,
            )
            is True
        )

    def test_single_field_generic_strict_query_activates_sidecar(self) -> None:
        """Single-doc generic query now ALWAYS activates sidecar for page_budget=2 recall floor.

        17/20 correct-G=0 cases had only 1 page (wrong page from legacy path).
        Sidecar with page_budget=2 gives top-2 candidates, fixing these cases.
        """
        scope = _make_scope(mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC)
        chunks = [_make_chunk("docA", 0)]
        # Generic query without authority patterns → sidecar still activates (changed behavior)
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="What is the amount owed?",
                scope=scope,
                answer_type="number",
                context_chunks=chunks,
            )
            is True
        )

    def test_broad_free_text_active_with_few_docs(self) -> None:
        """Sidecar activates for BROAD_FREE_TEXT with ≤3 docs."""
        scope = _make_scope(mode=ScopeMode.BROAD_FREE_TEXT)
        chunks = [_make_chunk("docA", 0)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="Summarize the law in plain language.",
                scope=scope,
                answer_type="free_text",
                context_chunks=chunks,
            )
            is True
        )

    def test_broad_free_text_inactive_with_many_docs(self) -> None:
        """Sidecar deactivates for BROAD_FREE_TEXT with > 5 docs."""
        scope = _make_scope(mode=ScopeMode.BROAD_FREE_TEXT)
        chunks = [_make_chunk(f"doc{i}", i) for i in range(6)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="Summarize all laws in plain language.",
                scope=scope,
                answer_type="free_text",
                context_chunks=chunks,
            )
            is False
        )

    def test_single_field_active_with_two_docs(self) -> None:
        """Sidecar activates for 2-doc comparison queries (dual-case fix)."""
        scope = _make_scope(mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC)
        chunks = [_make_chunk("lawA", 0), _make_chunk("lawB", 1)]
        assert (
            GroundingEvidenceSelector._should_activate_sidecar(
                query="Was the Employment Law enacted in the same year as the IP Law?",
                scope=scope,
                answer_type="boolean",
                context_chunks=chunks,
            )
            is True
        )


# ===========================================================================
# _score_candidates
# ===========================================================================


class TestScoreCandidates:
    """Tests for the _score_candidates method."""

    def test_page_retrieval_scores_accumulated(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        pages = [_make_page("docA_1", score=0.8), _make_page("docA_2", score=0.6)]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=[],
            page_candidates=pages,
        )

        assert result["docA_1"] == pytest.approx(0.8)
        assert result["docA_2"] == pytest.approx(0.6)

    def test_context_page_ids_get_bonus(self) -> None:
        sel = _make_selector()
        scope = _make_scope()

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=[],
            page_candidates=[],
            context_page_ids={"docA_1", "docA_2"},
        )

        assert result["docA_1"] == pytest.approx(3.0)
        assert result["docA_2"] == pytest.approx(3.0)

    def test_support_fact_weight_is_0_8(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        facts = [_make_fact("docA_3", score=1.0)]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=facts,
            page_candidates=[],
        )

        assert result["docA_3"] == pytest.approx(0.8)

    def test_answer_match_bonus_exact(self) -> None:
        """Answer value equality match gives +1.0 bonus."""
        sel = _make_selector()
        scope = _make_scope()
        facts = [_make_fact("docA_1", score=0.0, normalized_value="12345")]

        result = sel._score_candidates(
            query="test",
            answer_type="number",
            scope=scope,
            answer_value="12345",
            support_candidates=facts,
            page_candidates=[],
        )

        assert result["docA_1"] == pytest.approx(1.0)

    def test_answer_match_no_substring_false_positive(self) -> None:
        """Equality match prevents '5' from matching '15000'."""
        sel = _make_selector()
        scope = _make_scope()
        facts = [_make_fact("docA_1", score=0.0, normalized_value="15000")]

        result = sel._score_candidates(
            query="test",
            answer_type="number",
            scope=scope,
            answer_value="5",
            support_candidates=facts,
            page_candidates=[],
        )

        # "5" != "15000", no answer match bonus
        assert result["docA_1"] == pytest.approx(0.0)

    def test_page_role_match_bonus(self) -> None:
        sel = _make_selector()
        scope = _make_scope(target_page_roles=["ARTICLE_CLAUSE"])
        facts = [_make_fact("docA_1", score=0.0, page_role="ARTICLE_CLAUSE")]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=facts,
            page_candidates=[],
        )

        assert result["docA_1"] == pytest.approx(0.5)

    def test_date_of_issue_bonus(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        facts = [_make_fact("docA_1", score=0.0, fact_type="date_of_issue")]

        result = sel._score_candidates(
            query="What is the date of issue?",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=facts,
            page_candidates=[],
        )

        assert result["docA_1"] == pytest.approx(1.0)

    def test_number_financial_fact_bonus(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        for fact_type in ("claim_amount", "costs_awarded", "penalty"):
            facts = [_make_fact("docA_1", score=0.0, fact_type=fact_type)]
            result = sel._score_candidates(
                query="test",
                answer_type="number",
                scope=scope,
                answer_value="",
                support_candidates=facts,
                page_candidates=[],
            )
            assert result["docA_1"] == pytest.approx(0.5), f"Failed for {fact_type}"

    def test_combined_signals_additive(self) -> None:
        """All signals should be additive for a single page."""
        sel = _make_selector()
        scope = _make_scope(target_page_roles=["ARTICLE_CLAUSE"])
        pages = [_make_page("docA_1", score=0.5)]
        facts = [
            _make_fact(
                "docA_1",
                score=1.0,
                normalized_value="hello",
                page_role="ARTICLE_CLAUSE",
            )
        ]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="hello",
            support_candidates=facts,
            page_candidates=pages,
            context_page_ids={"docA_1"},
        )

        # page: 0.5 + context: 3.0 + support: 0.8*1.0 + answer: 1.0 + role: 0.5 = 5.8
        assert result["docA_1"] == pytest.approx(5.8)

    def test_scores_sorted_descending(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        pages = [
            _make_page("docA_1", score=0.2),
            _make_page("docA_2", score=0.9),
            _make_page("docA_3", score=0.5),
        ]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=[],
            page_candidates=pages,
        )

        keys = list(result.keys())
        assert keys == ["docA_2", "docA_3", "docA_1"]

    def test_empty_candidates_returns_empty(self) -> None:
        sel = _make_selector()
        scope = _make_scope()

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=[],
            page_candidates=[],
        )

        assert result == {}

    def test_context_page_ids_none_handled(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        pages = [_make_page("docA_1", score=0.5)]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=[],
            page_candidates=pages,
            context_page_ids=None,
        )

        assert result["docA_1"] == pytest.approx(0.5)

    def test_explicit_page_context_bonus_filtered(self) -> None:
        """In EXPLICIT_PAGE mode, context bonus skips pages not in candidates."""
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.EXPLICIT_PAGE)
        pages = [_make_page("docA_3", score=0.5)]

        result = sel._score_candidates(
            query="test",
            answer_type="text",
            scope=scope,
            answer_value="",
            support_candidates=[],
            page_candidates=pages,
            context_page_ids={"docA_1", "docA_3"},  # docA_1 not in candidates
        )

        # docA_3 gets both page score and context bonus
        assert result["docA_3"] == pytest.approx(3.5)
        # docA_1 should NOT get context bonus (not in candidate set)
        assert "docA_1" not in result


# ===========================================================================
# _select_minimal_pages
# ===========================================================================


class TestSelectMinimalPages:
    """Tests for the _select_minimal_pages method."""

    def test_empty_scored_returns_empty(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        assert sel._select_minimal_pages(query="", scope=scope, scored={}, page_candidates=[]) == []

    def test_explicit_page_returns_top_1(self) -> None:
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.EXPLICIT_PAGE, page_budget=5)
        scored = {"docA_1": 3.0, "docA_2": 2.0, "docA_3": 1.0}

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert result == ["docA_1"]

    def test_full_case_files_one_per_doc(self) -> None:
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.FULL_CASE_FILES)
        scored = {
            "docA_3": 5.0,
            "docB_2": 4.0,
            "docA_1": 3.0,
            "docB_5": 2.0,
        }

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert set(result) == {"docA_3", "docB_2"}

    def test_compare_pair_one_per_doc_with_budget(self) -> None:
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.COMPARE_PAIR, page_budget=2)
        scored = {
            "docA_1": 5.0,
            "docB_1": 4.0,
            "docC_1": 3.0,
        }

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert result == ["docA_1", "docB_1"]

    def test_compare_pair_budget_1_still_returns_2(self) -> None:
        """max(2, page_budget) means budget=1 → effective budget=2."""
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.COMPARE_PAIR, page_budget=1)
        scored = {
            "docA_1": 5.0,
            "docB_1": 4.0,
            "docC_1": 3.0,
        }

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert len(result) == 2
        assert result == ["docA_1", "docB_1"]

    def test_compare_pair_single_doc_returns_one(self) -> None:
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.COMPARE_PAIR, page_budget=2)
        scored = {"docA_1": 5.0, "docA_2": 3.0}

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert result == ["docA_1"]

    def test_single_field_mode_returns_budget_pages(self) -> None:
        """With page_budget=2, single_field returns 2 pages (gold at rank #2 not ignored)."""
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC, page_budget=2)
        scored = {"docA_1": 5.0, "docA_2": 4.0, "docA_3": 3.0}

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert result == ["docA_1", "docA_2"]

    def test_default_mode_budget_0_returns_empty(self) -> None:
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.SINGLE_FIELD_SINGLE_DOC, page_budget=0)
        scored = {"docA_1": 5.0}

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert result == []

    def test_page_id_with_underscore_in_doc_id(self) -> None:
        """Doc IDs can contain underscores. rpartition('_') handles this."""
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.FULL_CASE_FILES)
        scored = {"doc_with_underscores_3": 5.0, "another_doc_1": 4.0}

        result = sel._select_minimal_pages(query="", scope=scope, scored=scored, page_candidates=[])
        assert set(result) == {"doc_with_underscores_3", "another_doc_1"}


# ===========================================================================
# _map_support_fact_results
# ===========================================================================


class TestMapSupportFactResults:
    """Tests for the _map_support_fact_results static method."""

    def test_empty_result(self) -> None:
        result = SimpleNamespace(points=[])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert mapped == []

    def test_result_with_no_points_attr(self) -> None:
        result = SimpleNamespace()
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert mapped == []

    def test_result_uses_result_attr_fallback(self) -> None:
        point = SimpleNamespace(
            score=0.9,
            payload={
                "fact_id": "f1",
                "doc_id": "docA",
                "page_id": "docA_1",
                "page_num": 1,
                "fact_type": "case_number",
                "normalized_value": "AB 123/2020",
            },
        )
        result = SimpleNamespace(points=None, result=[point])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert len(mapped) == 1
        assert mapped[0].page_id == "docA_1"
        assert mapped[0].score == pytest.approx(0.9)

    def test_maps_all_fields(self) -> None:
        point = SimpleNamespace(
            score=0.75,
            payload={
                "fact_id": "f1",
                "doc_id": "docA",
                "page_id": "docA_2",
                "page_num": 2,
                "doc_title": "Test Case",
                "fact_type": "date_of_issue",
                "normalized_value": "2023-01-01",
                "quote_text": "Issued on January 1",
                "page_role": "TITLE_COVER",
                "page_family": "cover",
            },
        )
        result = SimpleNamespace(points=[point])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)

        assert len(mapped) == 1
        f = mapped[0]
        assert f.fact_id == "f1"
        assert f.doc_id == "docA"
        assert f.page_id == "docA_2"
        assert f.page_num == 2
        assert f.doc_title == "Test Case"
        assert f.fact_type == "date_of_issue"
        assert f.normalized_value == "2023-01-01"
        assert f.quote_text == "Issued on January 1"
        assert f.page_role == "TITLE_COVER"
        assert f.page_family == "cover"
        assert f.score == pytest.approx(0.75)

    def test_skips_points_without_dict_payload(self) -> None:
        good = SimpleNamespace(score=0.5, payload={"fact_id": "f1", "doc_id": "d", "page_id": "d_1"})
        bad_none = SimpleNamespace(score=0.5, payload=None)
        bad_str = SimpleNamespace(score=0.5, payload="not a dict")
        result = SimpleNamespace(points=[bad_none, good, bad_str])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert len(mapped) == 1

    def test_missing_payload_fields_default(self) -> None:
        point = SimpleNamespace(score=0.5, payload={})
        result = SimpleNamespace(points=[point])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert len(mapped) == 1
        f = mapped[0]
        assert f.fact_id == ""
        assert f.doc_id == ""
        assert f.page_num == 0

    def test_invalid_page_num_defaults_to_zero(self) -> None:
        point = SimpleNamespace(score=0.5, payload={"page_num": "not_a_number"})
        result = SimpleNamespace(points=[point])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert mapped[0].page_num == 0

    def test_invalid_score_defaults_to_zero(self) -> None:
        point = SimpleNamespace(score="bad", payload={"fact_id": "f1"})
        result = SimpleNamespace(points=[point])
        mapped = GroundingEvidenceSelector._map_support_fact_results(result)
        assert mapped[0].score == pytest.approx(0.0)


# ===========================================================================
# _select_doc_scope
# ===========================================================================


class TestSelectDocScope:
    """Tests for the _select_doc_scope method."""

    def test_returns_doc_ids_from_context(self) -> None:
        sel = _make_selector()
        scope = _make_scope(mode=ScopeMode.COMPARE_PAIR)
        chunks = [
            _make_chunk("docA", 0),
            _make_chunk("docB", 1),
            _make_chunk("docA", 2),
        ]

        result = sel._select_doc_scope(query="test", scope=scope, context_chunks=chunks)
        assert "docA" in result
        assert "docB" in result

    def test_empty_chunks_returns_empty(self) -> None:
        sel = _make_selector()
        scope = _make_scope()
        result = sel._select_doc_scope(query="test", scope=scope, context_chunks=[])
        assert result == []
