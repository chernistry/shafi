import time

import pytest

from rag_challenge.telemetry import TelemetryCollector


def test_basic_timing_accumulates_per_stage():
    collector = TelemetryCollector(request_id="req-1")

    with collector.timed("embed"):
        time.sleep(0.01)
    with collector.timed("qdrant"):
        time.sleep(0.01)

    payload = collector.finalize()
    assert payload.request_id == "req-1"
    assert payload.embed_ms >= 8
    assert payload.qdrant_ms >= 8
    assert payload.total_ms >= payload.embed_ms + payload.qdrant_ms


def test_ttft_tracking():
    collector = TelemetryCollector(request_id="req-2")
    time.sleep(0.01)
    collector.mark_first_token()
    time.sleep(0.01)

    payload = collector.finalize()
    assert payload.ttft_ms > 0
    assert payload.total_ms > payload.ttft_ms


def test_ttft_defaults_to_total_if_not_marked():
    collector = TelemetryCollector(request_id="req-3")
    time.sleep(0.002)
    payload = collector.finalize()
    assert payload.ttft_ms == payload.total_ms


def test_chunk_id_tracking_and_subset_chain():
    collector = TelemetryCollector(request_id="req-4")
    collector.set_retrieved_ids(["c1", "c2", "c3"])
    collector.set_context_ids(["c1", "c2"])
    collector.set_cited_ids(["c2"])

    payload = collector.finalize()
    assert payload.retrieved_chunk_ids == ["c1", "c2", "c3"]
    assert payload.context_chunk_ids == ["c1", "c2"]
    assert payload.cited_chunk_ids == ["c2"]
    assert payload.used_page_ids == []
    assert set(payload.cited_chunk_ids).issubset(set(payload.context_chunk_ids))
    assert set(payload.context_chunk_ids).issubset(set(payload.retrieved_chunk_ids))


def test_used_page_ids_derive_from_final_cited_ids() -> None:
    collector = TelemetryCollector(request_id="req-4b")
    collector.set_retrieved_ids(["doc:0:0:a", "doc:1:0:b"])
    collector.set_context_ids(["doc:0:0:a", "doc:1:0:b"])
    collector.set_cited_ids(["doc:1:0:b"])

    payload = collector.finalize()

    assert payload.cited_page_ids == ["doc_2"]
    assert payload.used_page_ids == ["doc_2"]


def test_used_page_ids_can_be_wider_than_cited_page_ids() -> None:
    collector = TelemetryCollector(request_id="req-4c")
    collector.set_retrieved_ids(["doc:0:0:a", "doc:1:0:b"])
    collector.set_context_ids(["doc:0:0:a", "doc:1:0:b"])
    collector.set_cited_ids(["doc:0:0:a"])
    collector.set_used_ids(["doc:0:0:a", "doc:1:0:b"])

    payload = collector.finalize()

    assert payload.cited_page_ids == ["doc_1"]
    assert payload.used_page_ids == ["doc_1", "doc_2"]


def test_token_usage_and_model_tracking():
    collector = TelemetryCollector(request_id="req-5")
    collector.set_token_usage(prompt_tokens=1000, completion_tokens=120, total_tokens=1120)
    collector.set_models(embed="kanon-2", rerank="zerank-2", llm="gpt-4o-mini")
    collector.set_llm_diagnostics(provider="openrouter", finish_reason="stop", malformed_tail_detected=True)
    collector.set_generation_mode("stream")
    collector.set_context_stats(chunk_count=6, budget_tokens=1600)
    collector.set_retried(True)

    payload = collector.finalize()
    assert payload.prompt_tokens == 1000
    assert payload.completion_tokens == 120
    assert payload.total_tokens == 1120
    assert payload.model_embed == "kanon-2"
    assert payload.model_rerank == "zerank-2"
    assert payload.model_llm == "gpt-4o-mini"
    assert payload.llm_provider == "openrouter"
    assert payload.llm_finish_reason == "stop"
    assert payload.generation_mode == "stream"
    assert payload.context_chunk_count == 6
    assert payload.context_budget_tokens == 1600
    assert payload.malformed_tail_detected is True
    assert payload.retried is True


def test_accumulated_timing_on_retry_stage():
    collector = TelemetryCollector(request_id="req-6")
    with collector.timed("qdrant"):
        time.sleep(0.01)
    with collector.timed("qdrant"):
        time.sleep(0.01)

    payload = collector.finalize()
    assert payload.qdrant_ms >= 16


def test_finalize_logs_invalid_subset_chain_but_does_not_fail(caplog: pytest.LogCaptureFixture):
    collector = TelemetryCollector(request_id="req-7")
    collector.set_retrieved_ids(["c1"])
    collector.set_context_ids(["c2"])
    collector.set_cited_ids(["c3"])

    with caplog.at_level("WARNING"):
        payload = collector.finalize()

    assert payload.request_id == "req-7"
    assert "subset" in caplog.text
