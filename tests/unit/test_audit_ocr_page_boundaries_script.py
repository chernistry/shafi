from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.audit_ocr_page_boundaries import (
    OcrBoundaryAudit,
    _audit_document,
    _current_parser_mode,
    _scope_verdict,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_current_parser_mode_marks_multi_page_fallback_as_collapsed() -> None:
    assert (
        _current_parser_mode(actual_pdf_pages=4, text_sufficient=False, page_identity_collapsed=True)
        == "docling_merged_single_page"
    )
    assert (
        _current_parser_mode(actual_pdf_pages=4, text_sufficient=False, page_identity_collapsed=False)
        == "docling_pages_preserved"
    )
    assert (
        _current_parser_mode(actual_pdf_pages=4, text_sufficient=True, page_identity_collapsed=False)
        == "pymupdf_pages_preserved"
    )
    assert (
        _current_parser_mode(actual_pdf_pages=1, text_sufficient=False, page_identity_collapsed=False)
        == "not_applicable_single_page"
    )


def test_scope_verdict_returns_private_watchpoint_when_no_collapsed_docs() -> None:
    audits = [
        OcrBoundaryAudit(
            document="doc.pdf",
            actual_pdf_pages=2,
            extracted_nonblank_pages=2,
            text_sufficient=True,
            fallback_triggered=False,
            page_identity_collapsed=False,
            current_parser_mode="pymupdf_pages_preserved",
        )
    ]
    assert _scope_verdict(audits) == "PRIVATE_SAFE_ONLY_WATCHPOINT"


def test_audit_document_marks_multi_page_fallback_safe_when_pages_are_preserved(tmp_path: Path, monkeypatch) -> None:
    from rag_challenge.ingestion.parser import DocumentParser

    parser = DocumentParser(pdf_text_min_chars=50, pdf_text_min_words=10)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("scripts.audit_ocr_page_boundaries._pdf_page_stats", lambda _path: (4, 1))
    monkeypatch.setattr(parser, "_parse_pdf_pymupdf_pages", lambda _path: ["short"])
    monkeypatch.setattr(parser, "_is_pdf_text_sufficient", lambda _text: False)
    monkeypatch.setattr(parser, "_parse_pdf_docling_pages", lambda _path: ["Recovered page 1", "Recovered page 2"])

    audit = _audit_document(file_path, parser=parser)

    assert audit.actual_pdf_pages == 4
    assert audit.fallback_triggered is True
    assert audit.page_identity_collapsed is False
    assert audit.current_parser_mode == "docling_pages_preserved"
