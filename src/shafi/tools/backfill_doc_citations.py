from __future__ import annotations

import argparse
import asyncio
import logging
import re
from typing import cast

from qdrant_client import AsyncQdrantClient, models

from shafi.config import get_settings
from shafi.config.logging import setup_logging

logger = logging.getLogger(__name__)

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

_LAW_NO_RE = re.compile(r"\blaw\s+no\.?\s*(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)
_DIFC_CASE_RE = re.compile(
    r"\b(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*(\d{1,4})\s*[/-]\s*(\d{4})\b",
    re.IGNORECASE,
)
_LAW_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,10}\s+Law)\s+(\d{4})\b",
    re.IGNORECASE,
)
_REG_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,10}\s+Regulations?)\b(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"\barticle\s+\d+(?:\s*\([^)]*\))*", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bschedule\s+(\d+)\b", re.IGNORECASE)


def _normalize_article(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    text = re.sub(r"\barticle\b", "Article", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
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


def _extract_doc_refs(text: str) -> set[str]:
    refs: set[str] = set()
    for match in _LAW_NO_RE.finditer(text):
        refs.add(f"Law No. {int(match.group(1))} of {match.group(2)}")
    for match in _DIFC_CASE_RE.finditer(text):
        refs.add(f"{match.group(1).upper()} {int(match.group(2)):03d}/{match.group(3)}")
    for match in _LAW_TITLE_RE.finditer(text):
        normalized = _normalize_law_title(match.group(1), match.group(2))
        if normalized:
            refs.add(normalized)
    for match in _REG_TITLE_RE.finditer(text):
        year = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        normalized = _normalize_reg_title(match.group(1), year)
        if normalized:
            refs.add(normalized)
    for match in _ARTICLE_RE.finditer(text):
        normalized = _normalize_article(match.group(0))
        if normalized:
            refs.add(normalized)
    for match in _SCHEDULE_RE.finditer(text):
        refs.add(f"Schedule {int(match.group(1))}")
    return refs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill doc-level DIFC identifiers into Qdrant payload citations[]")
    parser.add_argument("--collection", type=str, default="", help="Override Qdrant collection name")
    parser.add_argument("--limit", type=int, default=10_000_000, help="Max points to scan (safety valve)")
    parser.add_argument("--batch", type=int, default=512, help="Scroll batch size")
    parser.add_argument("--dry-run", action="store_true", help="Compute and report, but do not write")
    return parser


async def _ensure_keyword_index(client: AsyncQdrantClient, *, collection: str, field: str) -> None:
    try:
        await client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        logger.debug("Payload index on '%s' may already exist", field, exc_info=True)


async def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    root_settings = get_settings()
    setup_logging(root_settings.app.log_level, root_settings.app.log_format)

    qdrant_settings = root_settings.qdrant
    collection = args.collection.strip() or qdrant_settings.collection
    dry_run = bool(args.dry_run)

    client = AsyncQdrantClient(
        url=qdrant_settings.url,
        api_key=qdrant_settings.api_key or None,
        timeout=int(qdrant_settings.timeout_s),
        check_compatibility=bool(getattr(qdrant_settings, "check_compatibility", True)),
    )

    try:
        exists = await client.collection_exists(collection)
        if not exists:
            logger.error("Collection does not exist: %s", collection)
            return 2

        # Filtering on citations requires a keyword index in Qdrant Cloud.
        await _ensure_keyword_index(client, collection=collection, field="citations")

        doc_refs_by_id: dict[str, set[str]] = {}
        offset: object = None
        scanned = 0

        logger.info(
            "backfill_scan_start",
            extra={"collection": collection, "dry_run": dry_run, "batch": int(args.batch)},
        )
        while scanned < int(args.limit):
            records, next_offset = await client.scroll(
                collection_name=collection,
                limit=int(args.batch),
                offset=offset,
                with_payload=["doc_id", "doc_title", "chunk_text"],
                with_vectors=False,
            )
            if not records:
                break

            for record in records:
                payload_obj: object = getattr(record, "payload", None)
                if not isinstance(payload_obj, dict):
                    continue
                payload = cast("dict[str, object]", payload_obj)
                doc_id_obj = payload.get("doc_id")
                doc_title_obj = payload.get("doc_title")
                chunk_text_obj = payload.get("chunk_text", "")
                if not isinstance(doc_id_obj, str) or not doc_id_obj.strip():
                    continue
                if not isinstance(doc_title_obj, str) or not doc_title_obj.strip():
                    continue
                doc_id = doc_id_obj.strip()
                title = doc_title_obj.strip()
                chunk_text = chunk_text_obj if isinstance(chunk_text_obj, str) else str(chunk_text_obj)

                # Aggregate strong identifiers across doc title and chunk text.
                refs: set[str] = set()
                refs.update(_extract_doc_refs(title))
                refs.update(_extract_doc_refs(chunk_text))
                if refs:
                    existing = doc_refs_by_id.get(doc_id)
                    if existing is None:
                        existing = cast("set[str]", set())
                        doc_refs_by_id[doc_id] = existing
                    existing.update(refs)
                else:
                    if doc_id not in doc_refs_by_id:
                        doc_refs_by_id[doc_id] = cast("set[str]", set())

            scanned += len(records)
            offset = next_offset
            if offset is None:
                break

        logger.info(
            "backfill_scan_done",
            extra={"collection": collection, "points_scanned": scanned, "docs_found": len(doc_refs_by_id)},
        )

        docs_with_citations = 0
        docs_updated = 0
        for idx, (doc_id, refs) in enumerate(sorted(doc_refs_by_id.items()), start=1):
            citations = sorted(refs) if refs else []
            if not citations:
                continue
            docs_with_citations += 1
            if dry_run:
                continue

            selector = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
            await client.set_payload(
                collection_name=collection,
                payload={"citations": citations},
                points=selector,
                wait=True,
            )
            docs_updated += 1
            if idx % 10 == 0:
                logger.info(
                    "backfill_progress",
                    extra={
                        "docs_seen": idx,
                        "docs_with_citations": docs_with_citations,
                        "docs_updated": docs_updated,
                    },
                )

        logger.info(
            "backfill_done",
            extra={
                "collection": collection,
                "dry_run": dry_run,
                "docs_total": len(doc_refs_by_id),
                "docs_with_citations": docs_with_citations,
                "docs_updated": docs_updated,
            },
        )
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
