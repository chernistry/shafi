from __future__ import annotations

from shafi.ingestion.external_segment_payload import ExternalSegmentMetadata, ExternalSegmentRecord
from shafi.ingestion.rich_segment_text import SegmentTextMode, analyze_segment_noise, compose_segment_text


def _segment() -> ExternalSegmentRecord:
    return ExternalSegmentRecord(
        segment_id="doc-a:1:1",
        doc_id="doc-a",
        page_number=1,
        text="Article 16 requires the annual return.",
        title="Operating Law 2018",
        structure_type="paragraph",
        hierarchy=["Operating Law 2018", "Article 16"],
        context_text="Article 16 requires the annual return. The Registrar must receive the annual return every year.",
        embedding_text="article 16 annual return",
        metadata=ExternalSegmentMetadata(
            law_refs=["Operating Law 2018"],
            token_count=6,
            document_descriptor="law/regulation | Operating Law 2018 | 2018",
        ),
    )


def test_compose_segment_text_plain_vs_rich() -> None:
    plain = compose_segment_text(_segment(), mode=SegmentTextMode.PLAIN)
    rich = compose_segment_text(_segment(), mode=SegmentTextMode.RICH, context_char_limit=80)

    assert plain == "Article 16 requires the annual return."
    assert "Operating Law 2018" in rich
    assert "law/regulation | Operating Law 2018 | 2018" in rich
    assert "structure_type: paragraph" in rich
    assert "heading: Article 16" in rich
    assert "The Registrar must receive" in rich


def test_analyze_segment_noise_detects_no_duplicate_headers() -> None:
    analysis = analyze_segment_noise(_segment(), mode=SegmentTextMode.RICH)

    assert analysis.title_repeated is False
    assert analysis.hierarchy_repeated is False
    assert analysis.duplicate_line_count == 0


def test_compose_segment_text_adds_domain_markers() -> None:
    segment = ExternalSegmentRecord(
        segment_id="doc-b:1:1",
        doc_id="doc-b",
        page_number=1,
        text="This Law comes into force on the date specified in the Enactment Notice.",
        title="DIFC Employment Law",
        structure_type="caption",
        hierarchy=["DIFC Employment Law", "Schedule 1"],
        context_text="Issued by: DIFC Authority. Date of Issue: 01 January 2019.",
        embedding_text="enactment notice issued by difc authority law no 2 of 2019",
        metadata=ExternalSegmentMetadata(
            law_refs=["DIFC Employment Law"],
            token_count=12,
            document_descriptor="law/regulation | DIFC Employment Law | 2019",
        ),
    )

    rich = compose_segment_text(segment, mode=SegmentTextMode.RICH, context_char_limit=120)

    assert "structure_type: caption" in rich
    assert "heading: Schedule 1" in rich
    assert "notice: enactment / commencement" in rich
    assert "panel: issued by" in rich
    assert "panel: date" in rich
    assert "panel: law number" in rich
    assert "law_refs: DIFC Employment Law" in rich
