import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_challenge.ingestion.parser import DocumentParser
from rag_challenge.models import DocType


def test_parse_txt_statute_extracts_title_type_and_sections(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "limitation_act.txt"
    file_path.write_text(
        "# Limitation Act 2024\n\n"
        "## Section 1 - Short Title\n\n"
        "This Act may be cited as the Limitation Act.\n\n"
        "## Section 2 - Interpretation\n\n"
        "In this Act, action means proceedings in a court.\n",
        encoding="utf-8",
    )

    doc = parser.parse_file(file_path)

    assert doc.title == "Limitation Act 2024"
    assert doc.doc_type == DocType.STATUTE
    assert doc.doc_id
    assert len(doc.sections) >= 2
    assert "Section 1" in doc.sections[0].section_path


def test_parse_txt_contract_extracts_jurisdiction(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "service_agreement.txt"
    file_path.write_text(
        "SERVICE AGREEMENT\n\n"
        "This Agreement is entered into by and between Party A and Party B.\n"
        "WHEREAS the parties hereby agree to the following terms.\n"
        "The contract shall be governed by the laws of the United States.\n",
        encoding="utf-8",
    )

    doc = parser.parse_file(file_path)

    assert doc.doc_type == DocType.CONTRACT
    assert doc.jurisdiction == "US"


def test_parse_directory_skips_failed_files_and_parses_supported(tmp_path: Path):
    parser = DocumentParser()
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text(f"# Document {i}\n\nSection 1 text.\n", encoding="utf-8")
    (tmp_path / "ignore.bin").write_bytes(b"\x00\x01")

    docs = parser.parse_directory(tmp_path)

    assert len(docs) == 3
    assert sorted(doc.title for doc in docs) == ["Document 0", "Document 1", "Document 2"]


def test_deterministic_doc_id(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "sample.txt"
    file_path.write_text("test content", encoding="utf-8")

    doc1 = parser.parse_file(file_path)
    doc2 = parser.parse_file(file_path)

    assert doc1.doc_id == doc2.doc_id


def test_empty_file_handled(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    doc = parser.parse_file(file_path)

    assert doc.doc_type == DocType.OTHER
    assert doc.full_text == ""
    assert doc.sections == []


def test_parse_json_prechunked_preserves_stable_ids(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "starter_doc.json"
    payload = {
        "doc_id": "DOC-123",
        "title": "Sample Contract",
        "doc_type": "contract",
        "jurisdiction": "US",
        "chunks": [
            {"chunk_id": "CH-1", "text": "Alpha", "section_path": "Section 1"},
            {"chunk_id": "CH-2", "text": "Beta", "section_path": "Section 2", "citations": ["X"]},
            {"chunk_id": "", "text": "invalid"},
        ],
    }
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    doc = parser.parse_file(file_path)

    assert doc.doc_id == "DOC-123"
    assert doc.doc_type == DocType.CONTRACT
    assert len(doc.provided_chunks) == 2
    assert [chunk.chunk_id for chunk in doc.provided_chunks] == ["CH-1", "CH-2"]
    assert doc.provided_chunks[1].citations == ["X"]
    assert doc.metadata["source_format"] == "json_prechunked"


def test_parse_json_without_doc_id_uses_deterministic_fallback(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "chunks.json"
    file_path.write_text(json.dumps({"chunks": [{"chunk_id": "c1", "text": "hello"}]}), encoding="utf-8")

    doc = parser.parse_file(file_path)

    assert doc.doc_id
    assert doc.doc_id == "chunks"
    assert doc.provided_chunks[0].chunk_id == "c1"


def test_unsupported_extension_raises(tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "file.csv"
    file_path.write_text("x,y", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        parser.parse_file(file_path)


def test_parse_pdf_uses_pymupdf_fast_path_when_text_is_sufficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parser = DocumentParser(pdf_text_min_chars=20, pdf_text_min_words=3)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(parser, "_parse_pdf_pymupdf_pages", lambda _path: ["This is extracted text from PDF."])
    docling_called = False

    def _docling(_path: Path) -> str:
        nonlocal docling_called
        docling_called = True
        return "docling"

    monkeypatch.setattr(parser, "_parse_pdf_docling", _docling)

    text = parser._parse_pdf(file_path)

    assert "extracted text" in text
    assert docling_called is False


def test_parse_pdf_falls_back_to_docling_when_pymupdf_text_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parser = DocumentParser(pdf_text_min_chars=50, pdf_text_min_words=10)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(parser, "_parse_pdf_pymupdf_pages", lambda _path: ["short"])
    monkeypatch.setattr(parser, "_parse_pdf_docling", lambda _path: "# Parsed by Docling\n\nFull text")

    text = parser._parse_pdf(file_path)

    assert text.startswith("# Parsed by Docling")


def test_parse_pdf_falls_back_to_docling_when_pymupdf_text_is_low_density(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parser = DocumentParser(pdf_text_min_chars=10, pdf_text_min_words=1)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(parser, "_parse_pdf_pymupdf_pages", lambda _path: ["_____ _____ _____ !!!!!"])
    monkeypatch.setattr(parser, "_parse_pdf_docling", lambda _path: "# OCR fallback\n\nRecovered text")

    text = parser._parse_pdf(file_path)

    assert text.startswith("# OCR fallback")


def test_parse_pdf_docling_enables_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    parser = DocumentParser()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    captured: dict[str, bool] = {}

    class _FakePdfPipelineOptions:
        def __init__(self, *, do_ocr: bool) -> None:
            captured["do_ocr"] = do_ocr

    class _FakeDocument:
        @staticmethod
        def export_to_markdown() -> str:
            return "# OCR markdown"

    class _FakeConversionResult:
        document = _FakeDocument()

    class _FakeDocumentConverter:
        def __init__(self, *, format_options: dict[object, object]) -> None:
            captured["format_options"] = bool(format_options)

        def convert(self, _path: str) -> _FakeConversionResult:
            return _FakeConversionResult()

    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf")))
    monkeypatch.setitem(
        sys.modules,
        "docling.datamodel.pipeline_options",
        SimpleNamespace(PdfPipelineOptions=_FakePdfPipelineOptions),
    )
    monkeypatch.setitem(
        sys.modules,
        "docling.document_converter",
        SimpleNamespace(DocumentConverter=_FakeDocumentConverter, PdfFormatOption=lambda pipeline_options: pipeline_options),
    )

    text = parser._parse_pdf_docling(file_path)

    assert text == "# OCR markdown"
    assert captured["do_ocr"] is True
    assert captured["format_options"] is True


def test_parse_pdf_sections_include_page_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parser = DocumentParser(pdf_text_min_chars=10, pdf_text_min_words=1)
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        parser,
        "_read_pdf",
        lambda _path: (
            "First page text\n\nSecond page text",
            ["First page text", "Second page text"],
        ),
    )

    doc = parser.parse_file(file_path)

    assert [section.page_number for section in doc.sections] == [1, 2]
    assert [section.page_type for section in doc.sections] == ["page", "page"]
    assert [section.section_path for section in doc.sections] == ["page:1", "page:2"]
