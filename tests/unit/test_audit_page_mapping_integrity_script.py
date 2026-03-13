from __future__ import annotations

from dataclasses import dataclass

from scripts.audit_page_mapping_integrity import (
    ChunkPageMismatch,
    _chunk_id_page_number,
    _find_chunk_page_mismatches,
    _is_contiguous_page_numbers,
)

from rag_challenge.models import DocumentSection


@dataclass(frozen=True)
class _FakeChunk:
    chunk_id: str
    page_number: int | None
    page_type: str | None = None


def test_chunk_id_page_number_uses_second_segment_plus_one() -> None:
    assert _chunk_id_page_number("doc123:0:0:deadbeef") == 1
    assert _chunk_id_page_number("doc123:4:title-anchor:beadfeed") == 5
    assert _chunk_id_page_number("doc123_only") is None


def test_is_contiguous_page_numbers_accepts_dense_sequence() -> None:
    sections = [
        DocumentSection(heading="Page 1", section_path="page:1", text="a", level=0, page_number=1),
        DocumentSection(heading="Page 2", section_path="page:2", text="b", level=0, page_number=2),
    ]
    assert _is_contiguous_page_numbers(sections) is True


def test_is_contiguous_page_numbers_rejects_gap() -> None:
    sections = [
        DocumentSection(heading="Page 1", section_path="page:1", text="a", level=0, page_number=1),
        DocumentSection(heading="Page 3", section_path="page:3", text="b", level=0, page_number=3),
    ]
    assert _is_contiguous_page_numbers(sections) is False


def test_find_chunk_page_mismatches_detects_section_idx_drift() -> None:
    mismatches = _find_chunk_page_mismatches(
        [
            _FakeChunk(chunk_id="doc123:0:0:deadbeef", page_number=1),
            _FakeChunk(chunk_id="doc123:1:0:feedface", page_number=3, page_type="title_anchor"),
        ]
    )

    assert mismatches == [
        ChunkPageMismatch(
            chunk_id="doc123:1:0:feedface",
            page_from_chunk_id=2,
            page_number=3,
            page_type="title_anchor",
        )
    ]
