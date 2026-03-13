# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_challenge.ingestion.parser import DocumentParser


@dataclass(frozen=True)
class OcrBoundaryAudit:
    document: str
    actual_pdf_pages: int
    extracted_nonblank_pages: int
    text_sufficient: bool
    fallback_triggered: bool
    page_identity_collapsed: bool
    current_parser_mode: str


def _pdf_page_stats(path: Path) -> tuple[int, int]:
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


def _current_parser_mode(*, actual_pdf_pages: int, text_sufficient: bool, page_identity_collapsed: bool) -> str:
    if actual_pdf_pages <= 1:
        return "not_applicable_single_page"
    if text_sufficient:
        return "pymupdf_pages_preserved"
    if page_identity_collapsed:
        return "docling_merged_single_page"
    return "docling_pages_preserved"


def _audit_document(path: Path, *, parser: DocumentParser) -> OcrBoundaryAudit:
    actual_pdf_pages, _nonblank_pdf_pages = _pdf_page_stats(path)
    pymupdf_pages = parser._parse_pdf_pymupdf_pages(path)
    fast_text = "\n\n".join(text for text in pymupdf_pages if text).strip()
    text_sufficient = parser._is_pdf_text_sufficient(fast_text)
    fallback_triggered = not text_sufficient
    docling_pages = parser._parse_pdf_docling_pages(path) if fallback_triggered else []
    extracted_docling_pages = len([text for text in docling_pages if text.strip()])
    page_identity_collapsed = actual_pdf_pages > 1 and fallback_triggered and extracted_docling_pages <= 1
    return OcrBoundaryAudit(
        document=path.name,
        actual_pdf_pages=actual_pdf_pages,
        extracted_nonblank_pages=len([text for text in pymupdf_pages if text.strip()]),
        text_sufficient=text_sufficient,
        fallback_triggered=fallback_triggered,
        page_identity_collapsed=page_identity_collapsed,
        current_parser_mode=_current_parser_mode(
            actual_pdf_pages=actual_pdf_pages,
            text_sufficient=text_sufficient,
            page_identity_collapsed=page_identity_collapsed,
        ),
    )


def _scope_verdict(audits: list[OcrBoundaryAudit]) -> str:
    if any(audit.page_identity_collapsed for audit in audits):
        return "WARMUP_AND_PRIVATE_RISK"
    return "PRIVATE_SAFE_ONLY_WATCHPOINT"


def _render_markdown(*, audits: list[OcrBoundaryAudit], documents_root: Path) -> str:
    collapsed = [audit for audit in audits if audit.page_identity_collapsed]
    fallback_docs = [audit for audit in audits if audit.fallback_triggered]
    verdict = _scope_verdict(audits)
    lines = [
        "# OCR Page Boundary Audit",
        "",
        f"- documents_root: `{documents_root}`",
        f"- documents_audited: `{len(audits)}`",
        f"- fallback_triggered_docs: `{len(fallback_docs)}`",
        f"- page_identity_collapsed_docs: `{len(collapsed)}`",
        f"- verdict: `{verdict}`",
        "- current_parser_behavior: `when PDF fallback triggers, parser._read_pdf collapses output to a single merged pseudo-page`",
        "",
    ]
    if collapsed:
        lines.extend(["## Collapsed Docs", ""])
        for audit in collapsed:
            lines.append(
                f"- `{audit.document}` | actual_pages=`{audit.actual_pdf_pages}` | "
                f"pymupdf_nonblank_pages=`{audit.extracted_nonblank_pages}` | mode=`{audit.current_parser_mode}`"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Conclusion",
                "",
                "- No warm-up corpus PDFs currently trigger the merged single-page fallback path.",
                "- The code path still exists and remains a private-phase OCR watchpoint until ticket 32 either repairs provenance or the private corpus proves clean.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether OCR fallback still collapses PDF page identity.")
    parser.add_argument("--documents", type=Path, required=True, help="Directory containing PDF documents.")
    parser.add_argument("--markdown-out", type=Path, required=True, help="Markdown report output path.")
    parser.add_argument("--json-out", type=Path, required=True, help="JSON report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents_dir = args.documents.resolve()
    parser = DocumentParser()
    audits = [_audit_document(path, parser=parser) for path in sorted(documents_dir.glob("*.pdf"))]
    payload = {
        "documents_root": str(documents_dir),
        "documents_audited": len(audits),
        "fallback_triggered_docs": [audit.document for audit in audits if audit.fallback_triggered],
        "page_identity_collapsed_docs": [audit.document for audit in audits if audit.page_identity_collapsed],
        "verdict": _scope_verdict(audits),
        "audits": [asdict(audit) for audit in audits],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(
        _render_markdown(audits=audits, documents_root=documents_dir),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
