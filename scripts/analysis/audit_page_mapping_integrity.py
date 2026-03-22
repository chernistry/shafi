# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_challenge.ingestion.chunker import LegalChunker
from rag_challenge.ingestion.parser import DocumentParser

if TYPE_CHECKING:
    from rag_challenge.models import Chunk, DocumentSection


@dataclass(frozen=True)
class ChunkPageMismatch:
    chunk_id: str
    section_idx: int | None
    chunk_section_path: str
    parsed_section_path: str


@dataclass(frozen=True)
class DocumentAudit:
    document: str
    actual_pdf_pages: int
    nonblank_pdf_pages: int
    blank_pdf_pages: int
    fallback_risk: bool
    section_page_numbers: list[int]
    contiguous_section_pages: bool
    chunk_count: int
    page_numbered_chunk_count: int
    chunk_page_mismatches: list[ChunkPageMismatch]


def _chunk_id_section_idx(chunk_id: str) -> int | None:
    parts = str(chunk_id).split(":")
    if len(parts) < 2:
        return None
    if not parts[1].isdigit():
        return None
    return int(parts[1])


def _section_page_number(section: DocumentSection) -> int | None:
    return _section_page_number_from_path(section.section_path)


def _section_page_number_from_path(section_path: str) -> int | None:
    raw = str(section_path or "").strip()
    if not raw.startswith("page:"):
        return None
    page = raw.removeprefix("page:").strip()
    return int(page) if page.isdigit() else None


def _is_contiguous_page_numbers(sections: list[DocumentSection]) -> bool:
    numbers = [_section_page_number(section) for section in sections]
    numbers = [number for number in numbers if number is not None]
    if not numbers:
        return True
    return numbers == list(range(1, len(numbers) + 1))


def _find_chunk_page_mismatches(chunks: list[Chunk], *, sections: list[DocumentSection]) -> list[ChunkPageMismatch]:
    mismatches: list[ChunkPageMismatch] = []
    for chunk in chunks:
        section_idx = _chunk_id_section_idx(chunk.chunk_id)
        if section_idx is None:
            mismatches.append(
                ChunkPageMismatch(
                    chunk_id=chunk.chunk_id,
                    section_idx=None,
                    chunk_section_path=chunk.section_path,
                    parsed_section_path="",
                )
            )
            continue
        if section_idx < 0 or section_idx >= len(sections):
            mismatches.append(
                ChunkPageMismatch(
                    chunk_id=chunk.chunk_id,
                    section_idx=section_idx,
                    chunk_section_path=chunk.section_path,
                    parsed_section_path="",
                )
            )
            continue
        parsed_section_path = sections[section_idx].section_path
        if parsed_section_path != chunk.section_path:
            mismatches.append(
                ChunkPageMismatch(
                    chunk_id=chunk.chunk_id,
                    section_idx=section_idx,
                    chunk_section_path=chunk.section_path,
                    parsed_section_path=parsed_section_path,
                )
            )
    return mismatches


def _pdf_page_text_stats(path: Path) -> tuple[int, int]:
    try:
        import fitz  # pyright: ignore[reportMissingImports,reportMissingTypeStubs]
    except Exception:
        return 0, 0

    fitz_mod: Any = fitz
    actual_pages = 0
    nonblank_pages = 0
    pdf_obj: Any = fitz_mod.open(str(path))
    try:
        for page_obj in list(pdf_obj):
            actual_pages += 1
            page_any: Any = page_obj
            page_text_obj = page_any.get_text("text")
            if isinstance(page_text_obj, str) and page_text_obj.strip():
                nonblank_pages += 1
    finally:
        pdf_obj.close()
    return actual_pages, nonblank_pages


def _audit_document(path: Path, *, parser: DocumentParser, chunker: LegalChunker) -> DocumentAudit:
    actual_pdf_pages, nonblank_pdf_pages = _pdf_page_text_stats(path)
    pymupdf_pages = parser._parse_pdf_pymupdf_pages(path)
    fast_text = "\n\n".join(text for text in pymupdf_pages if text).strip()
    fallback_risk = actual_pdf_pages > 1 and not parser._is_pdf_text_sufficient(fast_text)

    doc = parser.parse_file(path)
    chunks = chunker.chunk_document(doc)
    mismatches = _find_chunk_page_mismatches(chunks, sections=doc.sections)
    section_page_numbers = [
        page_number
        for page_number in (_section_page_number(section) for section in doc.sections)
        if page_number is not None
    ]

    return DocumentAudit(
        document=path.name,
        actual_pdf_pages=actual_pdf_pages,
        nonblank_pdf_pages=nonblank_pdf_pages,
        blank_pdf_pages=max(0, actual_pdf_pages - nonblank_pdf_pages),
        fallback_risk=fallback_risk,
        section_page_numbers=section_page_numbers,
        contiguous_section_pages=_is_contiguous_page_numbers(doc.sections),
        chunk_count=len(chunks),
        page_numbered_chunk_count=sum(1 for chunk in chunks if _section_page_number_from_path(chunk.section_path) is not None),
        chunk_page_mismatches=mismatches,
    )


def _render_markdown(*, audits: list[DocumentAudit], root: Path) -> str:
    fallback_docs = [audit for audit in audits if audit.fallback_risk]
    blank_docs = [audit for audit in audits if audit.blank_pdf_pages > 0]
    noncontiguous_docs = [audit for audit in audits if not audit.contiguous_section_pages]
    mismatch_docs = [audit for audit in audits if audit.chunk_page_mismatches]

    if fallback_docs or blank_docs or noncontiguous_docs or mismatch_docs:
        verdict = "RISK_PRESENT"
    else:
        verdict = "NO_PAGE_BOUNDARY_EVIDENCE"

    lines = [
        "# Page Mapping Integrity Audit",
        "",
        f"- documents_root: `{root}`",
        f"- documents_audited: `{len(audits)}`",
        f"- fallback_risk_docs: `{len(fallback_docs)}`",
        f"- blank_page_docs: `{len(blank_docs)}`",
        f"- noncontiguous_section_docs: `{len(noncontiguous_docs)}`",
        f"- chunk_page_mismatch_docs: `{len(mismatch_docs)}`",
        f"- verdict: `{verdict}`",
        "",
    ]

    if verdict == "NO_PAGE_BOUNDARY_EVIDENCE":
        lines.extend(
            [
                "## Conclusion",
                "",
                "- No evidence that the audited PDF chunks span page boundaries.",
                "- No evidence that `chunk_id -> parsed section_path` drifted on the audited PDFs.",
                "- No OCR fallback collapse was triggered on the audited PDFs.",
                "",
            ]
        )
        return "\n".join(lines) + "\n"

    def add_doc_list(title: str, docs: list[DocumentAudit]) -> None:
        if not docs:
            return
        lines.append(f"## {title}")
        lines.append("")
        for audit in docs:
            lines.append(f"- `{audit.document}`")
        lines.append("")

    add_doc_list("Fallback Risk Docs", fallback_docs)
    add_doc_list("Blank Page Docs", blank_docs)
    add_doc_list("Noncontiguous Section Docs", noncontiguous_docs)
    if mismatch_docs:
        lines.append("## Chunk/Page Mismatches")
        lines.append("")
        for audit in mismatch_docs:
            lines.append(f"- `{audit.document}`")
            for mismatch in audit.chunk_page_mismatches[:10]:
                lines.append(
                    f"  - `{mismatch.chunk_id}` section_idx=`{mismatch.section_idx}` "
                    f"chunk_section_path=`{mismatch.chunk_section_path}` "
                    f"parsed_section_path=`{mismatch.parsed_section_path}`"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PDF page mapping integrity for parser/chunker page IDs.")
    parser.add_argument("--documents", type=Path, required=True, help="Directory of PDF documents to audit.")
    parser.add_argument("--markdown-out", type=Path, required=True, help="Markdown report output path.")
    parser.add_argument("--json-out", type=Path, required=True, help="JSON report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents_dir = args.documents.resolve()
    parser = DocumentParser()
    chunker = LegalChunker()

    audits = [
        _audit_document(path, parser=parser, chunker=chunker)
        for path in sorted(documents_dir.glob("*.pdf"))
    ]

    payload = {
        "documents_root": str(documents_dir),
        "documents_audited": len(audits),
        "audits": [asdict(audit) for audit in audits],
        "fallback_risk_docs": [audit.document for audit in audits if audit.fallback_risk],
        "blank_page_docs": [audit.document for audit in audits if audit.blank_pdf_pages > 0],
        "noncontiguous_section_docs": [audit.document for audit in audits if not audit.contiguous_section_pages],
        "chunk_page_mismatch_docs": [audit.document for audit in audits if audit.chunk_page_mismatches],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }

    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_render_markdown(audits=audits, root=documents_dir), encoding="utf-8")


if __name__ == "__main__":
    main()
