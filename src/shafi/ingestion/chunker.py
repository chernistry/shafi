from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

import tiktoken

from shafi.config import get_settings
from shafi.models import Chunk, DocumentSection, ParsedDocument

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Legacy citation patterns (kept for compatibility / non-DIFC sources).
_LEGACY_CITATION_RE = re.compile(
    r"(?:§\s*\d+[\w.()/-]*"
    r"|\d+\s+U\.?S\.?C\.?\s*§?\s*\d+[\w.()/-]*"
    r"|\d+\s+[A-Z][A-Za-z.]+\s+\d+"
    r"|[A-Z][A-Za-z]+\s+v\.?\s+[A-Z][A-Za-z]+)"
)

# DIFC / UAE-friendly identifiers (High ROI for grounding):
# - "Law No. 12 of 2004"
# - "CFI 010/2024", "CA 5/2025", etc (normalize leading zeros)
_LAW_NO_RE = re.compile(r"\blaw\s+no\.?\s*(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)
_DIFC_CASE_RE = re.compile(
    r"\b(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*(\d{1,4})\s*[/-]\s*(\d{4})\b",
    re.IGNORECASE,
)
# DIFC law titles used in the dataset/questions, e.g. "Trust Law 2018", "Operating Law 2018".
_LAW_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,10}\s+Law)\s+(\d{4})\b",
    re.IGNORECASE,
)
# DIFC regulations titles, e.g. "Employment Regulations", "Financial Collateral Regulations".
_REG_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,10}\s+Regulations?)\b(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
# Optional intra-document anchors (kept moderate; can be noisy, but helps exact-article questions).
_ARTICLE_RE = re.compile(r"\barticle\s+\d+(?:\s*\([^)]*\))*", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bschedule\s+(\d+)\b", re.IGNORECASE)

_DEFINED_TERM_RE = re.compile(r'"([A-Z][^"]{2,80})"')

_ACRONYMS = {
    "DIFC",
    "DFSA",
    "UAE",
    "AI",
    "AML",
    "CFT",
    "CRS",
    "ICC",
    "IC",
    "LLP",
    "PJSC",
}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_article(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    text = re.sub(r"\barticle\b", "Article", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    # Collapse whitespace around parentheses for canonical matching.
    text = re.sub(r"\s*\(\s*", "(", text)
    text = re.sub(r"\s*\)\s*", ")", text)
    return text.strip()


def _normalize_law_title(raw_title: str, year: str) -> str:
    title = re.sub(r"\s+", " ", raw_title.strip())
    if not title or not year.strip():
        return ""
    stopwords = {
        "the",
        "of",
        "in",
        "under",
        "for",
        "to",
        "and",
        "or",
        "a",
        "an",
        "by",
        "on",
        "at",
        "from",
        "see",
        "compare",
    }

    tokens: list[str] = []
    for word in title.split(" "):
        clean = re.sub(r"[^A-Za-z0-9]", "", word)
        if clean:
            tokens.append(clean)
    if not tokens:
        return ""

    # Heuristic: keep the suffix up to the first boundary stopword when scanning backwards.
    kept_rev: list[str] = []
    for token in reversed(tokens):
        if kept_rev and token.lower() in stopwords:
            break
        kept_rev.append(token)
    kept = list(reversed(kept_rev))
    if not kept:
        return ""

    words: list[str] = []
    for token in kept:
        if any(ch.isdigit() for ch in token):
            words.append(token)
            continue
        upper = token.upper()
        if upper in _ACRONYMS:
            words.append(upper)
        else:
            words.append(token[0].upper() + token[1:].lower())

    if not words:
        return ""
    if words[-1].lower() != "law":
        words.append("Law")
    else:
        words[-1] = "Law"
    return f"{' '.join(words)} {year.strip()}"


def _normalize_reg_title(raw_title: str, year: str | None) -> str:
    title = re.sub(r"\s+", " ", raw_title.strip())
    if not title:
        return ""
    stopwords = {
        "the",
        "of",
        "in",
        "under",
        "for",
        "to",
        "and",
        "or",
        "a",
        "an",
        "by",
        "on",
        "at",
        "from",
        "see",
        "compare",
    }

    tokens: list[str] = []
    for word in title.split(" "):
        clean = re.sub(r"[^A-Za-z0-9]", "", word)
        if clean:
            tokens.append(clean)
    if not tokens:
        return ""

    # Heuristic: keep the suffix up to the first boundary stopword when scanning backwards.
    kept_rev: list[str] = []
    for token in reversed(tokens):
        if kept_rev and token.lower() in stopwords:
            break
        kept_rev.append(token)
    kept = list(reversed(kept_rev))
    if not kept:
        return ""

    words: list[str] = []
    for token in kept:
        if any(ch.isdigit() for ch in token):
            words.append(token)
            continue
        upper = token.upper()
        if upper in _ACRONYMS:
            words.append(upper)
        else:
            words.append(token[0].upper() + token[1:].lower())

    if not words:
        return ""

    last = words[-1].lower()
    if last == "regulation":
        words[-1] = "Regulations"
    elif last != "regulations":
        words.append("Regulations")
    else:
        words[-1] = "Regulations"

    if year is not None and year.strip():
        return f"{' '.join(words)} {year.strip()}"
    return " ".join(words)


def extract_citations(text: str) -> list[str]:
    if not text.strip():
        return []

    found: list[str] = []

    for match in _LAW_NO_RE.finditer(text):
        found.append(f"Law No. {int(match.group(1))} of {match.group(2)}")

    for match in _DIFC_CASE_RE.finditer(text):
        found.append(f"{match.group(1).upper()} {int(match.group(2)):03d}/{match.group(3)}")

    for match in _LAW_TITLE_RE.finditer(text):
        normalized = _normalize_law_title(match.group(1), match.group(2))
        if normalized:
            found.append(normalized)

    for match in _REG_TITLE_RE.finditer(text):
        year = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        normalized = _normalize_reg_title(match.group(1), year)
        if normalized:
            found.append(normalized)

    for match in _ARTICLE_RE.finditer(text):
        normalized = _normalize_article(match.group(0))
        if normalized:
            found.append(normalized)

    for match in _SCHEDULE_RE.finditer(text):
        found.append(f"Schedule {int(match.group(1))}")

    for match in _LEGACY_CITATION_RE.finditer(text):
        raw = match.group(0).strip()
        if raw:
            found.append(raw)

    return _dedupe_preserve_order(found)


class LegalChunker:
    """Structure-aware chunker for legal documents."""

    def __init__(self) -> None:
        self._settings = get_settings().ingestion
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        doc_citations = self._extract_doc_citations(doc)
        if doc.provided_chunks:
            return self._chunks_from_provided(doc, doc_citations=doc_citations)
        if doc.sections:
            return self._chunk_from_sections(doc, doc_citations=doc_citations)
        return self._chunk_flat(doc, doc_citations=doc_citations)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _chunks_from_provided(self, doc: ParsedDocument, *, doc_citations: list[str]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for provided in doc.provided_chunks:
            citations = _dedupe_preserve_order([*doc_citations, *list(provided.citations)])
            chunks.append(
                Chunk(
                    chunk_id=provided.chunk_id,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    doc_type=doc.doc_type,
                    jurisdiction=doc.jurisdiction,
                    section_path=provided.section_path,
                    chunk_text=provided.text,
                    chunk_text_for_embedding=provided.text,
                    doc_summary="",
                    citations=citations,
                    anchors=list(provided.anchors),
                    token_count=self.count_tokens(provided.text),
                )
            )
        return chunks

    def _chunk_from_sections(self, doc: ParsedDocument, *, doc_citations: list[str]) -> list[Chunk]:
        output: list[Chunk] = []
        for section_idx, section in enumerate(doc.sections):
            if not section.text.strip():
                continue
            output.extend(
                self._split_text(
                    text=section.text,
                    doc=doc,
                    section=section,
                    section_idx=section_idx,
                    doc_citations=doc_citations,
                )
            )
        return output if output else self._chunk_flat(doc, doc_citations=doc_citations)

    def _chunk_flat(self, doc: ParsedDocument, *, doc_citations: list[str]) -> list[Chunk]:
        if not doc.full_text.strip():
            return []
        fallback_section = DocumentSection(
            heading=doc.title,
            section_path=doc.title,
            text=doc.full_text,
            level=0,
        )
        return self._split_text(
            text=doc.full_text,
            doc=doc,
            section=fallback_section,
            section_idx=0,
            doc_citations=doc_citations,
        )

    def _split_text(
        self,
        *,
        text: str,
        doc: ParsedDocument,
        section: DocumentSection,
        section_idx: int,
        doc_citations: list[str],
    ) -> list[Chunk]:
        target = max(1, int(self._settings.chunk_size_tokens))
        overlap = max(0, int(self._settings.chunk_overlap_tokens))

        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_count = 0
        chunk_idx = 0

        for para in paragraphs:
            para_count = self.count_tokens(para)
            if para_count == 0:
                continue

            if para_count > target:
                if current_parts:
                    chunks.append(
                        self._make_chunk(
                            text="\n\n".join(current_parts),
                            doc=doc,
                            section=section,
                            section_idx=section_idx,
                            chunk_idx=chunk_idx,
                            doc_citations=doc_citations,
                        )
                    )
                    chunk_idx += 1
                    current_parts = []
                    current_count = 0

                for piece in self._hard_split(para, target=target, overlap=overlap):
                    chunks.append(
                        self._make_chunk(
                            text=piece,
                            doc=doc,
                            section=section,
                            section_idx=section_idx,
                            chunk_idx=chunk_idx,
                            doc_citations=doc_citations,
                        )
                    )
                    chunk_idx += 1
                continue

            if current_parts and (current_count + para_count > target):
                chunks.append(
                    self._make_chunk(
                        text="\n\n".join(current_parts),
                        doc=doc,
                        section=section,
                        section_idx=section_idx,
                        chunk_idx=chunk_idx,
                        doc_citations=doc_citations,
                    )
                )
                chunk_idx += 1
                current_parts, current_count = self._apply_paragraph_overlap(current_parts, overlap)

            current_parts.append(para)
            current_count += para_count

        if current_parts:
            chunks.append(
                self._make_chunk(
                    text="\n\n".join(current_parts),
                    doc=doc,
                    section=section,
                    section_idx=section_idx,
                    chunk_idx=chunk_idx,
                    doc_citations=doc_citations,
                )
            )

        return chunks

    def _apply_paragraph_overlap(self, parts: Sequence[str], overlap_tokens: int) -> tuple[list[str], int]:
        if overlap_tokens <= 0:
            return [], 0

        kept: list[str] = []
        kept_count = 0
        for part in reversed(parts):
            part_count = self.count_tokens(part)
            if part_count <= 0:
                continue
            if kept and (kept_count + part_count > overlap_tokens):
                break
            if not kept and part_count > overlap_tokens:
                # Keep at least one paragraph for continuity if overlap budget is tiny.
                kept.insert(0, part)
                kept_count = part_count
                break
            kept.insert(0, part)
            kept_count += part_count
        return kept, kept_count

    def _hard_split(self, text: str, *, target: int, overlap: int) -> list[str]:
        token_ids = self._encoding.encode(text)
        if not token_ids:
            return []

        pieces: list[str] = []
        start = 0
        step = max(1, target - max(0, overlap))

        while start < len(token_ids):
            end = min(start + target, len(token_ids))
            piece = self._encoding.decode(token_ids[start:end]).strip()
            if piece:
                pieces.append(piece)
            if end >= len(token_ids):
                break
            start += step

        return pieces

    def _make_chunk(
        self,
        *,
        text: str,
        doc: ParsedDocument,
        section: DocumentSection,
        section_idx: int,
        chunk_idx: int,
        doc_citations: list[str],
    ) -> Chunk:
        normalized_text = text.strip()
        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"{doc.doc_id}:{section_idx}:{chunk_idx}:{text_hash}"
        citations = _dedupe_preserve_order([*doc_citations, *extract_citations(normalized_text)])
        anchors = [match.group(1) for match in _DEFINED_TERM_RE.finditer(normalized_text)]

        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            doc_type=doc.doc_type,
            jurisdiction=doc.jurisdiction,
            section_path=section.section_path,
            chunk_text=normalized_text,
            chunk_text_for_embedding=normalized_text,
            doc_summary="",
            citations=citations,
            anchors=anchors,
            token_count=self.count_tokens(normalized_text),
        )

    @staticmethod
    def _extract_doc_citations(doc: ParsedDocument) -> list[str]:
        # Some PDFs have unhelpful titles ("_____"), but doc refs (CFI/CA/Law No...)
        # often appear in the first page(s). Capture doc-level citations once and
        # propagate to all chunks to enable identifier-aware retrieval filters.
        parts: list[str] = []
        if doc.title:
            parts.append(doc.title)

        seed = (doc.full_text or "").strip()
        if seed:
            parts.append(seed[:5000])
        elif doc.sections:
            snippet = "\n\n".join(section.text for section in doc.sections[:3] if section.text)
            if snippet.strip():
                parts.append(snippet[:5000])
        elif doc.provided_chunks:
            snippet = "\n\n".join(chunk.text for chunk in doc.provided_chunks[:3] if chunk.text)
            if snippet.strip():
                parts.append(snippet[:5000])

        return extract_citations("\n".join(parts))

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        parts = re.split(r"\n\s*\n", text)
        paragraphs = [part.strip() for part in parts if part.strip()]
        return paragraphs if paragraphs else [text.strip()] if text.strip() else []
