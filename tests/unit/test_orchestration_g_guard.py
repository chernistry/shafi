"""Unit tests for the orchestration G-guard: early-exit lookups with no page IDs
must fall back to RAG rather than emitting an ungrounded non-null answer (G=0).

NOGA noga-125a — 2026-03-21
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from shafi.telemetry import TelemetryCollector


def _minimal_state(*, request_id: str = "req-1", question_id: str = "q-1") -> dict:
    return {
        "query": "When did Law X come into force?",
        "request_id": request_id,
        "question_id": question_id,
        "answer_type": "date",
        "collector": TelemetryCollector(request_id=request_id),
    }


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(simple_model="gpt-4o-mini", strict_max_tokens=150),
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            strict_types_force_simple_model=True,
            boolean_max_tokens=96,
            enable_multi_hop=False,
            db_answerer_enabled=False,
        ),
        verifier=SimpleNamespace(enabled=False),
    )


@pytest.mark.asyncio
async def test_database_lookup_g_guard_empty_source_page_ids_falls_back() -> None:
    """_database_lookup must return {} (RAG fallback) when source_page_ids is empty."""
    from shafi.core.pipeline import RAGPipelineBuilder

    with patch("shafi.core.pipeline.get_settings", return_value=_settings()):
        builder = RAGPipelineBuilder(
            retriever=MagicMock(),
            reranker=MagicMock(),
            generator=MagicMock(),
            classifier=MagicMock(),
        )

    # Mock db_answerer: loaded, returns answer with EMPTY source_page_ids
    db_answerer = MagicMock()
    db_answerer.is_loaded.return_value = True
    field_answer = SimpleNamespace(
        source_page_ids=(),  # empty — the bug condition
        canonical_entity_id="law:x",
        field_type=SimpleNamespace(value="commencement_date"),
        source_doc_id="law_x",
    )
    db_answerer.answer.return_value = field_answer
    db_answerer.format_answer.return_value = "2020-01-01"
    builder._db_answerer = db_answerer  # type: ignore[attr-defined]

    # Mock query_contract
    state = _minimal_state()
    state["query_contract"] = MagicMock()

    result = await builder._database_lookup(state)

    # Must return {} so pipeline falls back to RAG
    assert result == {}, f"Expected fallback (empty dict), got {result!r}"


@pytest.mark.asyncio
async def test_database_lookup_g_guard_with_pages_proceeds_normally() -> None:
    """_database_lookup must proceed normally when source_page_ids is non-empty."""
    from shafi.core.pipeline import RAGPipelineBuilder

    with patch("shafi.core.pipeline.get_settings", return_value=_settings()):
        builder = RAGPipelineBuilder(
            retriever=MagicMock(),
            reranker=MagicMock(),
            generator=MagicMock(),
            classifier=MagicMock(),
        )

    db_answerer = MagicMock()
    db_answerer.is_loaded.return_value = True
    field_answer = SimpleNamespace(
        source_page_ids=("law_x_1",),  # non-empty — normal case
        canonical_entity_id="law:x",
        field_type=SimpleNamespace(value="commencement_date"),
        source_doc_id="law_x",
    )
    db_answerer.answer.return_value = field_answer
    db_answerer.format_answer.return_value = "2020-01-01"
    builder._db_answerer = db_answerer  # type: ignore[attr-defined]

    state = _minimal_state()
    state["query_contract"] = MagicMock()

    result = await builder._database_lookup(state)

    # Must return a non-empty dict (proceed normally, not fall back)
    assert result != {}, "Expected normal DB lookup result, not fallback"
    assert result.get("answer") == "2020-01-01"
    assert result.get("db_answer") is field_answer


@pytest.mark.asyncio
async def test_compare_lookup_g_guard_empty_source_page_ids_falls_back() -> None:
    """_compare_lookup must return {} (RAG fallback) when source_page_ids is empty."""
    from shafi.core.pipeline import RAGPipelineBuilder

    with patch("shafi.core.pipeline.get_settings", return_value=_settings()):
        builder = RAGPipelineBuilder(
            retriever=MagicMock(),
            reranker=MagicMock(),
            generator=MagicMock(),
            classifier=MagicMock(),
        )

    compare_engine = MagicMock()
    compare_result = SimpleNamespace(
        source_page_ids=(),  # empty — the bug condition
        formatted_answer="Yes",
        result_type=SimpleNamespace(value="boolean_compare"),
        source_doc_ids=["doc_a", "doc_b"],
    )
    compare_engine.execute.return_value = compare_result
    builder._compare_engine = compare_engine  # type: ignore[attr-defined]

    state = _minimal_state()
    state["query_contract"] = MagicMock()
    # Attach execution_plan to contract
    state["query_contract"].execution_plan = [SimpleNamespace(value="compare_join")]

    result = await builder._compare_lookup(state)

    assert result == {}, f"Expected fallback (empty dict), got {result!r}"


@pytest.mark.asyncio
async def test_temporal_lookup_g_guard_empty_provenance_page_ids_falls_back() -> None:
    """_temporal_lookup must return {} (RAG fallback) when provenance_page_ids is empty."""
    from shafi.core.pipeline import RAGPipelineBuilder

    with patch("shafi.core.pipeline.get_settings", return_value=_settings()):
        builder = RAGPipelineBuilder(
            retriever=MagicMock(),
            reranker=MagicMock(),
            generator=MagicMock(),
            classifier=MagicMock(),
        )

    temporal_engine = MagicMock()
    temporal_result = SimpleNamespace(
        provenance_page_ids=(),  # empty — the bug condition
        answer_formatted="2020-01-01",
        query_type=SimpleNamespace(value="commencement_date"),
    )
    temporal_engine.answer.return_value = temporal_result
    builder._temporal_engine = temporal_engine  # type: ignore[attr-defined]

    state = _minimal_state()
    state["query_contract"] = MagicMock()

    result = await builder._temporal_lookup(state)

    assert result == {}, f"Expected fallback (empty dict), got {result!r}"
