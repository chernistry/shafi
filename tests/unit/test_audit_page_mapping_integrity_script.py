from __future__ import annotations

from dataclasses import dataclass

from scripts.audit_page_mapping_integrity import (
    ChunkPageMismatch,
    _chunk_id_section_idx,
    _find_chunk_page_mismatches,
    _is_contiguous_page_numbers,
)

from rag_challenge.models import DocumentSection


@dataclass(frozen=True)
class _FakeChunk:
    chunk_id: str
    section_path: str = ""


def test_chunk_id_section_idx_uses_second_segment() -> None:
    assert _chunk_id_section_idx("doc123:0:0:deadbeef") == 0
    assert _chunk_id_section_idx("doc123:4:title-anchor:beadfeed") == 4
    assert _chunk_id_section_idx("doc123_only") is None


def test_is_contiguous_page_numbers_accepts_dense_sequence() -> None:
    sections = [
        DocumentSection(heading="Page 1", section_path="page:1", text="a", level=0),
        DocumentSection(heading="Page 2", section_path="page:2", text="b", level=0),
    ]
    assert _is_contiguous_page_numbers(sections) is True


def test_is_contiguous_page_numbers_rejects_gap() -> None:
    sections = [
        DocumentSection(heading="Page 1", section_path="page:1", text="a", level=0),
        DocumentSection(heading="Page 3", section_path="page:3", text="b", level=0),
    ]
    assert _is_contiguous_page_numbers(sections) is False


def test_find_chunk_page_mismatches_detects_section_idx_drift() -> None:
    mismatches = _find_chunk_page_mismatches(
        [
            _FakeChunk(chunk_id="doc123:0:0:deadbeef", section_path="page:1"),
            _FakeChunk(chunk_id="doc123:1:0:feedface", section_path="page:9"),
        ],
        sections=[
            DocumentSection(heading="Page 1", section_path="page:1", text="a", level=0),
            DocumentSection(heading="Page 2", section_path="page:2", text="b", level=0),
        ],
    )

    assert mismatches == [
        ChunkPageMismatch(
            chunk_id="doc123:1:0:feedface",
            section_idx=1,
            chunk_section_path="page:9",
            parsed_section_path="page:2",
        )
    ]
