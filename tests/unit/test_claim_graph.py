from __future__ import annotations

from rag_challenge.core.claim_graph import ClaimGraphBuilder, SupportType
from rag_challenge.models import DocType, RankedChunk


def _chunk(*, chunk_id: str, text: str, section_path: str = "page:1") -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_title="Test Law",
        doc_type=DocType.STATUTE,
        section_path=section_path,
        text=text,
        retrieval_score=0.9,
        rerank_score=0.9,
        doc_summary="",
    )


def test_build_claim_graph_with_custom_extractor() -> None:
    builder = ClaimGraphBuilder(claim_extractor=lambda _text: ["The law applies.", "Payment is due within 30 days."])
    chunks = [
        _chunk(chunk_id="doc1:0:0", text="The law applies."),
        _chunk(chunk_id="doc1:0:1", text="Payment is due within 30 days."),
    ]

    graph = builder.build("The law applies. Payment is due within 30 days.", chunks, [], None)

    assert len(graph.claims) == 2
    assert graph.support_coverage == 1.0
    assert graph.unsupported_claims == []
    assert graph.dependency_edges == [("claim_1", "claim_2")]
    assert graph.claims[0].support_type == SupportType.DIRECTLY_STATED


def test_classify_support_marks_computed_claims() -> None:
    builder = ClaimGraphBuilder()
    evidence = [
        graph_span(chunk_id="doc1:0:0", page_id="doc1_1", span_text="The total is 10.", match_method="exact"),
        graph_span(chunk_id="doc1:0:1", page_id="doc1_2", span_text="The total is 20.", match_method="fuzzy"),
    ]

    assert builder.classify_support("The total is 30.", evidence) == SupportType.COMPUTED


def test_coverage_handles_unsupported_claims() -> None:
    builder = ClaimGraphBuilder(claim_extractor=lambda _text: ["Supported claim.", "Unsupported claim."])
    chunks = [_chunk(chunk_id="doc1:0:0", text="Supported claim.")]

    graph = builder.build("Supported claim. Unsupported claim.", chunks, [], None)

    assert graph.unsupported_claims == ["claim_2"]
    assert graph.support_coverage == 0.5


def graph_span(*, chunk_id: str, page_id: str, span_text: str, match_method: str) -> object:
    """Build a minimal evidence-span-like object for support tests."""

    from rag_challenge.core.span_grounder import EvidenceSpan

    return EvidenceSpan(
        chunk_id=chunk_id,
        page_id=page_id,
        span_text=span_text,
        match_method=match_method,
    )
