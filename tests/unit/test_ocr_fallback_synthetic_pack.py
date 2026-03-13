from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from scripts.audit_ocr_page_boundaries import _audit_document, _scope_verdict

from rag_challenge.ingestion.parser import DocumentParser


def test_ocr_fallback_synthetic_pack_reproduces_page_identity_collapse(monkeypatch) -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "ocr_fallback_synthetic_pack.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = cast("list[dict[str, Any]]", payload["cases"])
    parser = DocumentParser(pdf_text_min_chars=50, pdf_text_min_words=10)
    audits = []

    for case in cases:
        document = str(case["document"])
        actual_pdf_pages = int(case["actual_pdf_pages"])
        pymupdf_pages = [str(page) for page in cast("list[object]", case["pymupdf_pages"])]
        text_sufficient = bool(case["text_sufficient"])

        monkeypatch.setattr(
            "scripts.audit_ocr_page_boundaries._pdf_page_stats",
            lambda _path, actual_pdf_pages=actual_pdf_pages, pymupdf_pages=pymupdf_pages: (
                actual_pdf_pages,
                len([page for page in pymupdf_pages if page.strip()]),
            ),
        )
        monkeypatch.setattr(parser, "_parse_pdf_pymupdf_pages", lambda _path, pymupdf_pages=pymupdf_pages: pymupdf_pages)
        monkeypatch.setattr(parser, "_is_pdf_text_sufficient", lambda _text, text_sufficient=text_sufficient: text_sufficient)

        audit = _audit_document(Path(document), parser=parser)
        audits.append(audit)

        assert audit.document == document
        assert audit.fallback_triggered is bool(case["expected_fallback_triggered"])
        assert audit.page_identity_collapsed is bool(case["expected_page_identity_collapsed"])
        assert audit.current_parser_mode == str(case["expected_mode"])

    assert _scope_verdict(audits) == "WARMUP_AND_PRIVATE_RISK"
