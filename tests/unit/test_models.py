import json
import uuid

from pydantic import ValidationError

from rag_challenge.models import (
    ChunkMetadata,
    Citation,
    DocType,
    QueryRequest,
    RAGResponse,
    RankedChunk,
    RetrievedChunk,
    TelemetryPayload,
)


def test_query_request_default_id():
    req = QueryRequest(question="What is the statute of limitations?")
    assert len(req.request_id) == 36  # UUID4
    assert req.question == "What is the statute of limitations?"
    assert req.question_id == ""
    assert req.answer_type == "free_text"


def test_query_request_rejects_empty():
    import pytest

    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_telemetry_serialization_roundtrip():
    t = TelemetryPayload(
        request_id=str(uuid.uuid4()),
        ttft_ms=450,
        total_ms=1200,
        embed_ms=50,
        qdrant_ms=80,
        rerank_ms=250,
        llm_ms=500,
        prompt_tokens=1500,
        completion_tokens=200,
        total_tokens=1700,
        retrieved_chunk_ids=["c1", "c2", "c3"],
        context_chunk_ids=["c1", "c2"],
        model_embed="kanon-2-embedder",
        model_rerank="zerank-2",
        model_llm="gpt-4o-mini",
    )
    data = t.model_dump_json()
    t2 = TelemetryPayload.model_validate_json(data)
    assert t2.ttft_ms == 450
    assert t2.retrieved_chunk_ids == ["c1", "c2", "c3"]


def test_chunk_metadata_frozen():
    cm = ChunkMetadata(
        chunk_id="doc1:s1:0:500:abc",
        doc_id="doc1",
        doc_title="Contract A",
        doc_type=DocType.CONTRACT,
    )
    import pytest

    with pytest.raises(ValidationError):
        cm.chunk_id = "changed"  # type: ignore[misc]


def test_rag_response_full():
    resp = RAGResponse(
        answer="The limitation period is 6 years.",
        citations=[
            Citation(chunk_id="c1", doc_title="Limitation Act", section_path="Section 5"),
        ],
        telemetry=TelemetryPayload(
            request_id="test",
            ttft_ms=400,
            total_ms=1000,
            embed_ms=40,
            qdrant_ms=60,
            rerank_ms=200,
            llm_ms=400,
        ),
    )
    data = json.loads(resp.model_dump_json())
    assert data["citations"][0]["chunk_id"] == "c1"
    assert data["telemetry"]["ttft_ms"] == 400


def test_retrieved_chunk_score():
    rc = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Test",
        doc_type=DocType.STATUTE,
        text="Some legal text",
        score=0.85,
    )
    assert rc.score == 0.85


def test_ranked_chunk_has_both_scores():
    rk = RankedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Test",
        doc_type=DocType.CASE_LAW,
        text="Some legal text",
        retrieval_score=0.85,
        rerank_score=0.92,
    )
    assert rk.rerank_score > rk.retrieval_score
