"""Compiler-driven legal segment extraction for additive retrieval units."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from rag_challenge.models import Chunk, DocType, LegalSegment, SegmentType

if TYPE_CHECKING:
    from rag_challenge.models.legal_objects import BaseLegalObject, CorpusRegistry
    from rag_challenge.models.schemas import ParsedDocument

_HEADING_RE = re.compile(
    r"^(?P<label>Article|Section|Schedule|Part|Chapter)\s+(?P<number>[A-Za-z0-9().-]+)(?:\s*[-:]\s*(?P<title>.+))?$",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(r"\b(?:v\.?|vs\.?|versus)\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\bIT IS HEREBY ORDERED\b", re.IGNORECASE)
_ISSUED_BY_RE = re.compile(r"^\s*Issued by\s*:", re.IGNORECASE)
_DEFINITION_RE = re.compile(r'".{2,80}"\s+means\b|\bDefinitions?\b', re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$")


@dataclass(frozen=True, slots=True)
class ArticleBoundary:
    """Segment boundary detected in the flattened page-line stream."""

    segment_type: SegmentType
    label: str
    title: str
    page_num: int
    line_index: int


def _slugify(text: str) -> str:
    """Build a stable slug for segment identifiers.

    Args:
        text: Raw label text.

    Returns:
        str: Normalized slug.
    """

    return re.sub(r"-+", "-", _SLUG_RE.sub("-", text.casefold())).strip("-") or "segment"


def _normalize_line(text: str) -> str:
    """Normalize one source line for lightweight matching.

    Args:
        text: Raw line text.

    Returns:
        str: Cleaned single-space line.
    """

    return _SPACE_RE.sub(" ", text.strip())


def _page_map(parsed_doc: ParsedDocument) -> dict[int, str]:
    """Extract page-number keyed text from a parsed document.

    Args:
        parsed_doc: Parsed source document.

    Returns:
        dict[int, str]: Page map ordered by page number.
    """

    pages: dict[int, str] = {}
    for section in parsed_doc.sections:
        if not section.section_path.startswith("page:"):
            continue
        try:
            page_num = int(section.section_path.split(":", 1)[1])
        except (ValueError, IndexError):
            continue
        if _is_contents_like_page(section.text):
            continue
        pages[page_num] = section.text
    return dict(sorted(pages.items()))


def _flatten_lines(pages: dict[int, str]) -> list[tuple[int, str]]:
    """Flatten page text into a page-aware line stream.

    Args:
        pages: Page-number keyed text mapping.

    Returns:
        list[tuple[int, str]]: Flattened `(page_num, line)` records.
    """

    flattened: list[tuple[int, str]] = []
    for page_num, page_text in sorted(pages.items()):
        for raw_line in page_text.splitlines():
            line = _normalize_line(raw_line)
            if line:
                flattened.append((page_num, line))
    return flattened


def _is_contents_like_page(text: str) -> bool:
    """Detect table-of-contents style pages that should not become segments.

    Args:
        text: Raw page text.

    Returns:
        bool: True when the page looks like an index or contents listing.
    """

    normalized = text.lower()
    if "table of contents" in normalized or normalized.strip().startswith("contents"):
        return True
    dot_leader_lines = sum(1 for raw_line in text.splitlines() if _DOT_LEADER_RE.search(raw_line.strip()))
    heading_like_lines = sum(
        1
        for raw_line in text.splitlines()
        if raw_line.strip() and _HEADING_RE.match(_normalize_line(raw_line))
    )
    return dot_leader_lines >= 4 and heading_like_lines >= 3


class SegmentCompiler:
    """Compile typed legal segments from parsed documents and corpus objects."""

    def compile_segments(
        self,
        parsed_doc: ParsedDocument,
        corpus_registry: CorpusRegistry,
    ) -> list[LegalSegment]:
        """Compile legal segments for one parsed document.

        Args:
            parsed_doc: Parsed source document.
            corpus_registry: Compiled corpus registry from 1042.

        Returns:
            list[LegalSegment]: Typed additive retrieval segments.
        """

        pages = _page_map(parsed_doc)
        if not pages:
            return []
        doc_object = self._resolve_doc_object(parsed_doc.doc_id, corpus_registry)
        canonical_doc_id = doc_object.object_id if doc_object is not None else parsed_doc.doc_id
        boundaries = self.detect_article_boundaries(pages, doc_type=parsed_doc.doc_type)
        segments = self.merge_cross_page_segments(
            boundaries=boundaries,
            pages=pages,
            doc_id=parsed_doc.doc_id,
            doc_title=parsed_doc.title,
            doc_type=parsed_doc.doc_type,
            canonical_doc_id=canonical_doc_id,
        )

        parent_stack: dict[SegmentType, str] = {}
        finalized: list[LegalSegment] = []
        child_ids: dict[str, list[str]] = {}
        for segment in segments:
            parent_segment_id = ""
            if segment.segment_type is SegmentType.CHAPTER:
                parent_segment_id = parent_stack.get(SegmentType.PART, "")
            elif segment.segment_type not in {SegmentType.PART, SegmentType.CAPTION, SegmentType.ISSUED_BY}:
                parent_segment_id = (
                    parent_stack.get(SegmentType.CHAPTER, "")
                    or parent_stack.get(SegmentType.PART, "")
                )
            if segment.segment_type in {SegmentType.PART, SegmentType.CHAPTER}:
                parent_stack[segment.segment_type] = segment.segment_id
                if segment.segment_type is SegmentType.PART:
                    parent_stack.pop(SegmentType.CHAPTER, None)
            if parent_segment_id:
                child_ids.setdefault(parent_segment_id, []).append(segment.segment_id)
            finalized.append(
                segment.model_copy(
                    update={
                        "parent_segment_id": parent_segment_id,
                        "legal_path": self.build_legal_path(segment, doc_object),
                    }
                )
            )

        return [
            segment.model_copy(update={"child_segment_ids": child_ids.get(segment.segment_id, [])})
            for segment in finalized
        ]

    def detect_article_boundaries(
        self,
        pages: dict[int, str],
        *,
        doc_type: DocType | None = None,
    ) -> list[ArticleBoundary]:
        """Detect structural segment boundaries from page text.

        Args:
            pages: Page-number keyed text mapping.
            doc_type: Optional document type hint.

        Returns:
            list[ArticleBoundary]: Sorted structural boundaries.
        """

        boundaries: list[ArticleBoundary] = []
        flattened = _flatten_lines(pages)
        seen_caption = False
        for line_index, (page_num, line) in enumerate(flattened):
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                heading_name = str(heading_match.group("label") or "").casefold()
                segment_type = {
                    "article": SegmentType.ARTICLE,
                    "section": SegmentType.SECTION,
                    "schedule": SegmentType.SCHEDULE,
                    "part": SegmentType.PART,
                    "chapter": SegmentType.CHAPTER,
                }[heading_name]
                label = f"{heading_match.group('label')} {heading_match.group('number')}".strip()
                title = _normalize_line(str(heading_match.group("title") or ""))
                boundaries.append(
                    ArticleBoundary(
                        segment_type=segment_type,
                        label=label,
                        title=title,
                        page_num=page_num,
                        line_index=line_index,
                    )
                )
                continue
            if doc_type is DocType.CASE_LAW and not seen_caption and page_num == 1 and _CAPTION_RE.search(line):
                seen_caption = True
                boundaries.append(
                    ArticleBoundary(
                        segment_type=SegmentType.CAPTION,
                        label="Caption",
                        title="",
                        page_num=page_num,
                        line_index=line_index,
                    )
                )
                continue
            if _ISSUED_BY_RE.search(line):
                boundaries.append(
                    ArticleBoundary(
                        segment_type=SegmentType.ISSUED_BY,
                        label="Issued by",
                        title="",
                        page_num=page_num,
                        line_index=line_index,
                    )
                )
                continue
            if _ORDER_RE.search(line):
                boundaries.append(
                    ArticleBoundary(
                        segment_type=SegmentType.OPERATIVE_ORDER,
                        label="Operative order",
                        title="",
                        page_num=page_num,
                        line_index=line_index,
                    )
                )
                continue
            if _DEFINITION_RE.search(line):
                boundaries.append(
                    ArticleBoundary(
                        segment_type=SegmentType.DEFINITION,
                        label="Definitions",
                        title="",
                        page_num=page_num,
                        line_index=line_index,
                    )
                )
        deduped: list[ArticleBoundary] = []
        seen_keys: set[tuple[SegmentType, int, int]] = set()
        for boundary in boundaries:
            key = (boundary.segment_type, boundary.page_num, boundary.line_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(boundary)
        return deduped

    def merge_cross_page_segments(
        self,
        *,
        boundaries: list[ArticleBoundary],
        pages: dict[int, str],
        doc_id: str,
        doc_title: str,
        doc_type: DocType,
        canonical_doc_id: str,
    ) -> list[LegalSegment]:
        """Merge boundary-delimited text spans into page-aware legal segments.

        Args:
            boundaries: Sorted structural boundaries.
            pages: Page-number keyed text mapping.
            doc_id: Parent document ID.
            doc_title: Parent document title.
            doc_type: Parent document type.
            canonical_doc_id: Canonical compiled object ID.

        Returns:
            list[LegalSegment]: Compiled segments merged across page boundaries.
        """

        if not boundaries:
            return []
        flattened = _flatten_lines(pages)
        segments: list[LegalSegment] = []
        ordered_boundaries = sorted(boundaries, key=lambda boundary: boundary.line_index)
        for index, boundary in enumerate(ordered_boundaries):
            start = boundary.line_index
            end = ordered_boundaries[index + 1].line_index if index + 1 < len(ordered_boundaries) else len(flattened)
            segment_lines = flattened[start:end]
            if not segment_lines:
                continue
            page_numbers = [page_num for page_num, _line in segment_lines]
            page_ids = [f"{doc_id}_{page_num}" for page_num in sorted(set(page_numbers))]
            text = "\n".join(line for _page_num, line in segment_lines).strip()
            if not text:
                continue
            body_text = "\n".join(line for _page_num, line in segment_lines[1:]).strip()
            if boundary.segment_type in {SegmentType.PART, SegmentType.CHAPTER} and len(body_text) < 24:
                continue
            label_slug = _slugify(boundary.label or boundary.segment_type.value)
            segment_id = f"{doc_id}:segment:{boundary.segment_type.value}:{label_slug}:{page_ids[0]}"
            segments.append(
                LegalSegment(
                    segment_id=segment_id,
                    segment_type=boundary.segment_type,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    canonical_doc_id=canonical_doc_id,
                    legal_path="",
                    text=text,
                    page_ids=page_ids,
                    start_page=min(page_numbers),
                    end_page=max(page_numbers),
                    canonical_entity_ids=[canonical_doc_id] if canonical_doc_id else [],
                    search_text="\n".join(part for part in [doc_title, boundary.label, boundary.title, text] if part),
                )
            )
        return segments

    def build_legal_path(self, segment: LegalSegment, doc_object: BaseLegalObject | None) -> str:
        """Construct a human-readable legal path for one segment.

        Args:
            segment: Segment to describe.
            doc_object: Optional compiled document object.

        Returns:
            str: Hierarchical legal path.
        """

        root = doc_object.title if doc_object is not None else segment.doc_title
        label = segment.segment_type.value.replace("_", " ").title()
        first_line = _normalize_line(segment.text.splitlines()[0]) if segment.text else ""
        if _HEADING_RE.match(first_line):
            label = first_line
        elif first_line and segment.segment_type not in {SegmentType.CAPTION, SegmentType.OPERATIVE_ORDER}:
            label = f"{label}: {first_line[:80]}".strip()
        return " > ".join(part for part in [root, label] if part)

    @staticmethod
    def project_to_pages(segment: LegalSegment) -> list[str]:
        """Project one segment back to platform-valid page IDs.

        Args:
            segment: Segment to project.

        Returns:
            list[str]: Stable page IDs.
        """

        return list(segment.page_ids)

    def annotate_chunks(self, chunks: list[Chunk], segments: list[LegalSegment]) -> list[Chunk]:
        """Assign a best-effort segment ID back-reference to overlapping chunks.

        Args:
            chunks: Existing chunk records.
            segments: Compiled legal segments.

        Returns:
            list[Chunk]: Updated chunks with additive `segment_id` references.
        """

        if not segments:
            return chunks
        page_to_segments: dict[int, list[LegalSegment]] = {}
        for segment in segments:
            for page_id in segment.page_ids:
                try:
                    page_num = int(page_id.rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                page_to_segments.setdefault(page_num, []).append(segment)

        annotated: list[Chunk] = []
        for chunk in chunks:
            page_num = 0
            if chunk.section_path.startswith("page:"):
                try:
                    page_num = int(chunk.section_path.split(":", 1)[1])
                except (ValueError, IndexError):
                    page_num = 0
            match_id = ""
            for segment in page_to_segments.get(page_num, []):
                if not chunk.chunk_text:
                    continue
                if chunk.chunk_text[:80] in segment.text or segment.segment_type in {
                    SegmentType.ARTICLE,
                    SegmentType.SECTION,
                    SegmentType.SCHEDULE,
                }:
                    match_id = segment.segment_id
                    break
            annotated.append(chunk if not match_id else chunk.model_copy(update={"segment_id": match_id}))
        return annotated

    @staticmethod
    def _resolve_doc_object(doc_id: str, corpus_registry: CorpusRegistry) -> BaseLegalObject | None:
        """Resolve one compiled document object from the corpus registry.

        Args:
            doc_id: Parsed document ID.
            corpus_registry: Compiled corpus registry.

        Returns:
            BaseLegalObject | None: Matching compiled object when found.
        """

        stores = (
            cast("dict[str, BaseLegalObject]", corpus_registry.laws),
            cast("dict[str, BaseLegalObject]", corpus_registry.cases),
            cast("dict[str, BaseLegalObject]", corpus_registry.orders),
            cast("dict[str, BaseLegalObject]", corpus_registry.practice_directions),
            cast("dict[str, BaseLegalObject]", corpus_registry.amendments),
            cast("dict[str, BaseLegalObject]", corpus_registry.other_documents),
        )
        for store in stores:
            value = store.get(doc_id)
            if value is not None:
                return value
        return None
