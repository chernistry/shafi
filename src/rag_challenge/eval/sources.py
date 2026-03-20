from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import fitz  # PyMuPDF
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [text for item in items if (text := str(item).strip())]


def _chunk_id_to_page_id(chunk_id: str) -> str:
    """Convert `doc_id:page_idx:chunk_idx:hash` into `doc_id_{page}` (1-based)."""
    if ":" not in chunk_id and "_" in chunk_id:
        return chunk_id
    parts = chunk_id.split(":")
    if len(parts) < 2:
        return ""
    doc_id = parts[0].strip()
    page_raw = parts[1].strip()
    if not doc_id or not page_raw.isdigit():
        return ""
    return f"{doc_id}_{int(page_raw) + 1}"


def _chunk_ids_to_page_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        page_id = _chunk_id_to_page_id(str(raw).strip())
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        out.append(page_id)
    return out


def select_used_pages(payload: dict[str, object], *, max_pages: int) -> list[str]:
    """Select competition-aligned "used pages" from telemetry payload.

    Mirrors the submission-side logic in ``select_submission_used_pages``
    (submission/common.py) so that local eval scores match platform scores.

    Fallback chain: used_page_ids → used_chunk_ids → cited_page_ids →
    cited_chunk_ids → [] (empty).  Context pages are intentionally excluded
    because the platform submission code never includes them.
    """
    caps = max(0, int(max_pages))

    used_pages = _coerce_str_list(payload.get("used_page_ids"))
    if not used_pages:
        used_chunk_ids = _coerce_str_list(payload.get("used_chunk_ids"))
        used_pages = _chunk_ids_to_page_ids(used_chunk_ids) if used_chunk_ids else []
    if used_pages:
        return used_pages[:caps] if caps > 0 else used_pages

    cited_pages = _coerce_str_list(payload.get("cited_page_ids"))
    if not cited_pages:
        cited_chunk_ids = _coerce_str_list(payload.get("cited_chunk_ids"))
        cited_pages = _chunk_ids_to_page_ids(cited_chunk_ids) if cited_chunk_ids else []
    if cited_pages:
        return cited_pages[:caps] if caps > 0 else cited_pages

    return []


def _parse_page_id(page_id: str) -> tuple[str, int] | None:
    """Parse `doc_id_page` into `(doc_id, page_number)` (1-based)."""
    raw = page_id.strip()
    if not raw:
        return None
    if ":" in raw:
        raw = _chunk_id_to_page_id(raw)
    if "_" not in raw:
        return None

    doc_id, _, page_raw = raw.rpartition("_")
    if not doc_id or not page_raw.isdigit():
        return None
    page = int(page_raw)
    if page <= 0:
        return None
    return doc_id, page


class PdfPageTextProvider:
    def __init__(self, docs_dir: str | Path, *, max_chars_per_page: int) -> None:
        self._docs_dir = Path(docs_dir)
        self._max_chars_per_page = max(0, int(max_chars_per_page))
        self._docs: dict[str, fitz.Document] = {}
        self._page_cache: dict[tuple[str, int], str] = {}

    def close(self) -> None:
        for doc in self._docs.values():
            try:
                doc.close()
            except Exception:
                continue
        self._docs.clear()
        self._page_cache.clear()

    def get_page_text(self, *, doc_id: str, page: int) -> str | None:
        cache_key = (doc_id, page)
        if cache_key in self._page_cache:
            return self._page_cache[cache_key]

        path = self._docs_dir / f"{doc_id}.pdf"
        if not path.exists():
            return None

        doc = self._docs.get(doc_id)
        if doc is None:
            try:
                doc = fitz.open(str(path))
            except Exception:
                logger.debug("Failed to open PDF for sources extraction: %s", path, exc_info=True)
                return None
            self._docs[doc_id] = doc

        if page <= 0 or page > int(doc.page_count):
            return None

        try:
            text = doc.load_page(page - 1).get_text("text")
        except Exception:
            logger.debug("Failed to extract PDF page text: %s page=%d", doc_id, page, exc_info=True)
            return None

        cleaned = str(text or "").strip()
        if self._max_chars_per_page and len(cleaned) > self._max_chars_per_page:
            cleaned = cleaned[: self._max_chars_per_page]
        self._page_cache[cache_key] = cleaned
        return cleaned


def _parse_chunk_idx(chunk_id: str) -> int:
    parts = chunk_id.split(":")
    if len(parts) < 3:
        return 0
    raw = parts[2].strip()
    if not raw.isdigit():
        return 0
    return int(raw)


@dataclass(frozen=True)
class QdrantPageTextFallback:
    qdrant_client: AsyncQdrantClient
    collection: str
    max_chars_per_page: int

    async def get_page_text(self, *, doc_id: str, page: int) -> str | None:
        if page <= 0:
            return None

        selector = Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="section_path", match=MatchValue(value=f"page:{page}")),
            ]
        )

        offset: object = None
        rows: list[tuple[int, str]] = []
        while True:
            records, next_offset = await self.qdrant_client.scroll(
                collection_name=self.collection,
                scroll_filter=selector,
                limit=256,
                offset=offset,
                with_payload=["chunk_id", "chunk_text"],
                with_vectors=False,
            )
            if not records:
                break

            for record in records:
                payload_obj: object = getattr(record, "payload", None)
                if not isinstance(payload_obj, dict):
                    continue
                payload = cast("dict[str, object]", payload_obj)
                chunk_id = str(payload.get("chunk_id") or "").strip()
                chunk_text = str(payload.get("chunk_text") or "")
                chunk_idx = _parse_chunk_idx(chunk_id)
                if chunk_text.strip():
                    rows.append((chunk_idx, chunk_text.strip()))

            offset = next_offset
            if offset is None:
                break

        if not rows:
            return None
        rows.sort(key=lambda it: it[0])
        joined = "\n\n".join(text for _, text in rows).strip()
        cap = max(0, int(self.max_chars_per_page))
        if cap and len(joined) > cap:
            joined = joined[:cap]
        return joined


async def build_sources_text(
    used_pages: list[str],
    *,
    pdf_provider: PdfPageTextProvider,
    qdrant_fallback: QdrantPageTextFallback | None,
    max_chars_total: int,
) -> str:
    """Build a bounded sources_text string for the judge prompt."""
    cap_total = max(0, int(max_chars_total))
    out_parts: list[str] = []
    total = 0
    doc_title_cache: dict[str, str] = {}

    for page_id in used_pages:
        parsed = _parse_page_id(page_id)
        if parsed is None:
            continue
        doc_id, page = parsed

        doc_title = ""
        if qdrant_fallback is not None:
            cached = doc_title_cache.get(doc_id)
            if cached is None:
                try:
                    selector = Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    )
                    records, _next = await qdrant_fallback.qdrant_client.scroll(
                        collection_name=qdrant_fallback.collection,
                        scroll_filter=selector,
                        limit=1,
                        with_payload=["doc_title"],
                        with_vectors=False,
                    )
                except Exception:
                    records = []
                title = ""
                if records:
                    payload_obj: object = getattr(records[0], "payload", None)
                    if isinstance(payload_obj, dict):
                        title = str(cast("dict[str, object]", payload_obj).get("doc_title") or "").strip()
                doc_title_cache[doc_id] = title
                doc_title = title
            else:
                doc_title = cached

        text = pdf_provider.get_page_text(doc_id=doc_id, page=page)
        if text is None and qdrant_fallback is not None:
            try:
                text = await qdrant_fallback.get_page_text(doc_id=doc_id, page=page)
            except Exception:
                logger.debug("Qdrant fallback page text failed: %s_%d", doc_id, page, exc_info=True)
                text = None
        if text is None:
            text = ""

        header = f"### SOURCE {doc_id}_{page} | {doc_title}\n" if doc_title else f"### SOURCE {doc_id}_{page}\n"
        block = header + text.strip() + "\n\n"

        if cap_total and total + len(block) > cap_total:
            remaining = cap_total - total
            if remaining <= len(header):
                break
            allowed_text = max(0, remaining - len(header) - 2)
            truncated = header + text.strip()[:allowed_text] + "\n\n"
            out_parts.append(truncated)
            total += len(truncated)
            break

        out_parts.append(block)
        total += len(block)

        if cap_total and total >= cap_total:
            break

    return "".join(out_parts).strip()
