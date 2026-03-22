"""Hard-page visual region detection and registry enrichment."""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from rag_challenge.models.legal_objects import (
    BoundingBox,
    CorpusRegistry,
    RegionType,
    SignatureData,
    TableData,
    VisualRegion,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_TABLE_SPLIT_RE = re.compile(r"\s{2,}|\|")
_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(r"\b(?:v\.?|vs\.?|versus)\b", re.IGNORECASE)
_CASE_ROLE_RE = re.compile(
    r"\b(?:claimant|respondent|appellant|applicants?|defendant|plaintiff|petitioner|appellee|"
    r"accused|prosecution)\b",
    re.IGNORECASE,
)
_NOTICE_RE = re.compile(
    r"\b(?:enactment notice|notice of enactment|issued under|issued by|date of issue|published by)\b",
    re.IGNORECASE,
)
_SIGNATURE_RE = re.compile(
    r"\b(?:signed by|judge|justice|registrar|chief justice|by order of)\b",
    re.IGNORECASE,
)


def _page_text_blocks_factory() -> list[PageTextBlock]:
    """Build a typed empty page-text-block list for Pydantic defaults."""

    return []


class PageTextBlock(BaseModel):
    """One rendered page text block with a bounding box."""

    text: str
    bbox: BoundingBox


class RenderedPage(BaseModel):
    """Lightweight rendered page representation for offline region extraction."""

    page_id: str
    doc_id: str
    width: int
    height: int
    text_blocks: list[PageTextBlock] = Field(default_factory=_page_text_blocks_factory)


class HardPageDetector:
    """Detect pages that benefit from visual-region extraction."""

    def is_hard_page(self, page_text: str, page_metadata: Mapping[str, object] | None = None) -> bool:
        """Return whether the page should enter the regionizer.

        Args:
            page_text: Extracted page text.
            page_metadata: Optional page metadata hints.

        Returns:
            Whether the page should be regionized.
        """
        metadata = dict(page_metadata or {})
        words = [token for token in re.findall(r"[A-Za-z0-9]+", page_text)]
        low_density = len(words) < 25
        table_like = "|" in page_text or "schedule" in page_text.casefold() or "table" in page_text.casefold()
        signature_like = bool(_SIGNATURE_RE.search(page_text))
        caption_like = bool(_CAPTION_RE.search(page_text)) or bool(_CASE_ROLE_RE.search(page_text))
        notice_like = bool(_NOTICE_RE.search(page_text))
        ocr_noise = "�" in page_text or bool(metadata.get("ocr_artifact"))
        return low_density or table_like or signature_like or caption_like or notice_like or ocr_noise


class PageRegionizer:
    """Extract visual regions from rendered hard pages."""

    def extract_regions(self, page_image: RenderedPage, page_id: str) -> list[VisualRegion]:
        """Extract typed visual regions from one rendered page.

        Args:
            page_image: Rendered page with text blocks.
            page_id: Stable page identifier.

        Returns:
            Extracted visual regions.
        """
        regions: list[VisualRegion] = []
        for block in page_image.text_blocks:
            region_type = self._classify_block(block=block, page=page_image)
            if region_type is None:
                continue
            region = VisualRegion(
                region_type=region_type,
                bbox=block.bbox,
                extracted_text=block.text,
                confidence=self._confidence_for_region(region_type),
                page_id=page_id,
                doc_id=page_image.doc_id,
                table_data=self.extract_table_structure(block.text) if region_type is RegionType.TABLE else None,
                signature_data=self.extract_signature_block(block.text) if region_type is RegionType.SIGNATURE else None,
            )
            regions.append(region)
        return regions

    def extract_table_structure(self, region_text: str) -> TableData:
        """Extract structured table rows from a table-like region."""
        lines = [line.strip() for line in region_text.splitlines() if line.strip()]
        if not lines:
            return TableData()
        split_rows = [
            [cell.strip() for cell in _TABLE_SPLIT_RE.split(line) if cell.strip()]
            for line in lines
        ]
        headers = split_rows[0] if split_rows else []
        return TableData(headers=headers, rows=split_rows[1:])

    def extract_signature_block(self, region_text: str) -> SignatureData:
        """Extract signer/authority/date from a signature-like block."""
        lower = region_text.casefold()
        signer = ""
        authority = ""
        if "signed by" in lower:
            signer = region_text.split("Signed by", maxsplit=1)[-1].split("\n", maxsplit=1)[0].strip(" :")
        if "issued by" in lower:
            authority = region_text.split("Issued by", maxsplit=1)[-1].split("\n", maxsplit=1)[0].strip(" :")
        if not signer:
            judge_match = re.search(r"\b(?:judge|justice|registrar|chief justice)\b[:\s-]*(.+)", region_text, re.IGNORECASE)
            if judge_match is not None:
                signer = judge_match.group(1).split("\n", maxsplit=1)[0].strip(" :")
        date_match = _DATE_RE.search(region_text)
        return SignatureData(
            signer=signer,
            authority=authority,
            date=date_match.group(0) if date_match is not None else "",
        )

    def _classify_block(self, *, block: PageTextBlock, page: RenderedPage) -> RegionType | None:
        text = block.text.strip()
        lower = text.casefold()
        top = block.bbox.y / max(page.height, 1)
        bottom = (block.bbox.y + block.bbox.h) / max(page.height, 1)
        if _CAPTION_RE.search(text) or _CASE_ROLE_RE.search(text):
            return RegionType.CAPTION
        if "it is hereby ordered" in lower or "ordered that" in lower:
            return RegionType.OPERATIVE
        if "|" in text or "\t" in text or "schedule" in lower:
            return RegionType.SCHEDULE if "schedule" in lower else RegionType.TABLE
        if _SIGNATURE_RE.search(text) or (_NOTICE_RE.search(text) and "date of issue" in lower):
            return RegionType.SIGNATURE
        if top <= 0.2 and (_NOTICE_RE.search(text) or "law no" in lower or "enactment notice" in lower):
            return RegionType.TITLE
        if top <= 0.15 and text.isupper():
            return RegionType.HEADER
        if top <= 0.2:
            return RegionType.TITLE
        if bottom >= 0.9:
            return RegionType.FOOTER
        return None

    @staticmethod
    def _confidence_for_region(region_type: RegionType) -> float:
        if region_type in {RegionType.CAPTION, RegionType.SIGNATURE, RegionType.OPERATIVE}:
            return 0.9
        if region_type in {RegionType.TABLE, RegionType.SCHEDULE}:
            return 0.85
        return 0.75


class RegionEnricher:
    """Attach extracted visual regions to compiled registry objects."""

    def enrich_corpus_registry(
        self,
        registry: CorpusRegistry,
        regions_by_page: Mapping[str, Sequence[VisualRegion]],
    ) -> CorpusRegistry:
        """Return a registry copy enriched with visual regions.

        Args:
            registry: Source registry.
            regions_by_page: Regions keyed by page ID.

        Returns:
            Registry copy with per-object `visual_regions` populated.
        """
        enriched = registry.model_copy(deep=True)
        for bucket_name in ("laws", "cases", "orders", "practice_directions", "amendments", "other_documents"):
            bucket = getattr(enriched, bucket_name)
            for doc_id, obj in list(bucket.items()):
                visual_regions = {
                    page_id: list(regions_by_page.get(page_id, []))
                    for page_id in obj.page_ids
                    if regions_by_page.get(page_id)
                }
                if not visual_regions:
                    continue
                bucket[doc_id] = obj.model_copy(update={"visual_regions": visual_regions})
        return enriched


def render_region_gallery_html(
    pages: Sequence[RenderedPage],
    regions_by_page: Mapping[str, Sequence[VisualRegion]],
) -> str:
    """Render a compact HTML gallery with overlay boxes for manual review."""
    cards: list[str] = []
    for page in pages:
        overlays: list[str] = []
        for region in regions_by_page.get(page.page_id, []):
            overlays.append(
                f"<div class='region' style='left:{region.bbox.x}px;top:{region.bbox.y}px;"
                f"width:{region.bbox.w}px;height:{region.bbox.h}px;'>"
                f"<span>{html.escape(region.region_type.value)}</span></div>"
            )
        cards.append(
            "<section class='page-card'><h2>{page_id}</h2><div class='canvas' style='width:{width}px;height:{height}px;'>"
            "{overlays}</div><pre>{text}</pre></section>".format(
                page_id=html.escape(page.page_id),
                width=page.width,
                height=page.height,
                overlays="".join(overlays),
                text=html.escape("\n".join(block.text for block in page.text_blocks)),
            )
        )
    return "\n".join(
        [
            "<html><head><style>",
            ".page-card{margin:24px 0;font-family:monospace;}",
            ".canvas{position:relative;border:1px solid #999;background:#fafafa;}",
            ".region{position:absolute;border:2px solid #d24;background:rgba(210,36,68,0.08);overflow:hidden;}",
            ".region span{background:#d24;color:#fff;font-size:11px;padding:2px 4px;display:inline-block;}",
            "</style></head><body>",
            *cards,
            "</body></html>",
        ]
    )
