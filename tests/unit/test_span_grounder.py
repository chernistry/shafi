from __future__ import annotations

from types import SimpleNamespace

from shafi.core.span_grounder import SpanGrounder
from shafi.models import DocType, RankedChunk


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


def test_exact_match_finds_page_id_and_span() -> None:
    grounder = SpanGrounder()
    chunk = _chunk(chunk_id="doc1:0:0", text="Article 5 provides payment is due within 30 days.")

    spans = grounder.exact_match("payment is due within 30 days", chunk.text, chunk_id=chunk.chunk_id, page_id="doc1_1")

    assert spans
    assert spans[0].match_method == "exact"
    assert spans[0].page_id == "doc1_1"
    assert spans[0].chunk_id == "doc1:0:0"


def test_fuzzy_match_handles_paraphrase() -> None:
    grounder = SpanGrounder()

    spans = grounder.fuzzy_match(
        "payment must be made within thirty days",
        "The invoice must be paid within 30 days of issue.",
        chunk_id="doc1:0:0",
        page_id="doc1_1",
        threshold=0.35,
    )

    assert spans
    assert spans[0].match_method == "fuzzy"


def test_semantic_match_handles_token_overlap() -> None:
    grounder = SpanGrounder()

    spans = grounder.semantic_match(
        "claimant and respondent share a judge",
        "The claimant and respondent share a judge in this matter.",
        chunk_id="doc1:0:0",
        page_id="doc1_1",
        threshold=0.45,
    )

    assert spans
    assert spans[0].match_method == "semantic"


def test_ground_claim_to_spans_upgrades_segment_ids() -> None:
    grounder = SpanGrounder()
    chunk = _chunk(chunk_id="doc1:0:0", text="Article 5 provides payment is due within 30 days.")
    segment = SimpleNamespace(segment_id="seg-1", page_ids=["doc1_1"])

    spans = grounder.ground_claim_to_spans("payment is due within 30 days", [chunk], [segment])

    assert spans
    assert spans[0].segment_id == "seg-1"
    assert spans[0].page_id == "doc1_1"


def test_deduped_empty_grounding_returns_no_spans() -> None:
    grounder = SpanGrounder()
    spans = grounder.ground_claim_to_spans("unsupported claim", [], [])

    assert spans == []
