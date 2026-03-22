#!/usr/bin/env uv run python3
"""NOAM: Fix case_numbers and party_names for SCT documents in Qdrant.

This script updates all SCT case-law chunks with corrected metadata extraction
using the fixed pipeline functions.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointIdsList, Filter, FieldCondition, MatchValue

from rag_challenge.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("noam_fix_sct")

# Import the extraction functions from pipeline
# We'll replicate them here to avoid import issues
_CASE_ID_RE = re.compile(r"\b(?:CFI|CA|ARB|SCT|TCD|ENF|DEC)\s+\d{3}/\d{4}\b", re.IGNORECASE)
# Bracket format: "[2024] DIFC SCT 415" or "[2024] DIFC CA 012" -> "SCT 415/2024" or "CA 012/2024"
_BRACKET_CASE_RE = re.compile(r"\[(\d{4})\]\s+DIFC\s+(CFI|CA|ARB|SCT|TCD|ENF|DEC)\s+(\d+)", re.IGNORECASE)

def _unique_normalized(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for raw in values:
        normalized = re.sub(r"\s+", " ", raw or "").strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out

def _extract_case_numbers(*texts: str) -> List[str]:
    values = []
    for text in texts:
        # Standard format: "SCT 415/2024"
        values.extend(match.group(0) for match in _CASE_ID_RE.finditer(text or ""))
        # Bracket format: "[2024] DIFC SCT 415" or "[2024] DIFC CA 012" -> "SCT 415/2024" or "CA 012/2024"
        for match in _BRACKET_CASE_RE.finditer(text or ""):
            year, court, number = match.groups()
            values.append(f"{court} {number}/{year}")
    return _unique_normalized(values)

_PARTY_ROLE_RE = re.compile(
    r"\b(?:claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\b[:\s-]*([A-Z][A-Za-z0-9&.,'()/-]{2,80})",
    re.IGNORECASE,
)

def _extract_party_names(text: str) -> List[str]:
    return _unique_normalized([match.group(1) for match in _PARTY_ROLE_RE.finditer(text or "")])

def _extract_case_caption_parties(*texts: str) -> List[str]:
    """Extract party names from case-caption style 'A v B' strings."""
    values = []
    for text in texts:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            continue
        match = re.search(r"(?P<lhs>.+?)\s+(?:v\.?|vs\.?|versus)\s+(?P<rhs>.+)", normalized, re.IGNORECASE)
        if match is None:
            continue

        lhs = re.sub(r"^\S+\s+\d{3}/\d{4}\s+", "", match.group("lhs")).strip(" ,.;:-")
        lhs = _CASE_ID_RE.sub("", lhs).strip(" ,.;:-")
        if lhs:
            values.append(lhs)

        rhs = re.split(r"\b(?:Claim\s+No\.?|COURT\s+OF|SMALL\s+CLAIMS\s+TRIBUNAL|DIGITAL\s+ECONOMY\s+COURT|TECHNOLOGY\s+AND\s+CONSTRUCTION\s+DIVISION|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|\[\d{4}\])\b", match.group("rhs"), maxsplit=1)[0].strip(" ,.;:-")
        numbered_parts = [part.strip(" ,.;:-") for part in re.split(r"\(\d+\)\s*", rhs) if part.strip(" ,.;:-")]
        if numbered_parts:
            values.extend(numbered_parts)
        elif rhs:
            values.append(rhs)

    return _unique_normalized(values)

async def find_bracket_format_documents(client: AsyncQdrantClient, collection: str) -> List[str]:
    """Find all unique doc_ids where doc_title has bracket format like '[2024] DIFC SCT 415' or '[2024] DIFC CA 012'."""
    doc_ids = set()
    next_offset = None
    logger.info(f"Searching for bracket-format case documents in collection '{collection}'")

    # We'll scan all documents and check title pattern
    while True:
        records, next_offset = await client.scroll(
            collection_name=collection,
            limit=500,
            offset=next_offset,
            with_payload=["doc_id", "doc_title"],
            with_vectors=False,
        )
        for rec in records:
            title = rec.payload.get("doc_title", "")
            # Check for bracket pattern: [2024] DIFC <COURT> <NUM>
            if _BRACKET_CASE_RE.search(title):
                doc_ids.add(rec.payload.get("doc_id"))
        if next_offset is None:
            break
        if len(doc_ids) > 100:  # safety limit
            break

    result = list(doc_ids)
    logger.info(f"Found {len(result)} documents with bracket-format case titles")
    return result

async def get_document_chunks(client: AsyncQdrantClient, collection: str, doc_id: str) -> List:
    """Get all chunks for a specific document."""
    chunks = []
    next_offset = None
    while True:
        records, next_offset = await client.scroll(
            collection_name=collection,
            limit=500,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        chunks.extend(records)
        if next_offset is None:
            break
    return chunks

async def main() -> int:
    settings = get_settings()
    client = AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key or None,
    )

    collection = "legal_chunks_private_1792"

    # Verify collection exists
    try:
        await client.get_collection(collection)
    except Exception as e:
        logger.error(f"Collection '{collection}' not accessible: {e}")
        return 1

    # Find documents with bracket-format case titles (SCT, CA, etc.)
    doc_ids = await find_bracket_format_documents(client, collection)
    if not doc_ids:
        logger.warning("No documents with bracket-format case titles found")
        return 0

    # Process each document
    for doc_idx, doc_id in enumerate(doc_ids, 1):
        logger.info(f"[{doc_idx}/{len(doc_ids)}] Processing doc_id={doc_id[:16]}...")

        # Get all chunks for this document
        chunks = await get_document_chunks(client, collection, doc_id)
        if not chunks:
            logger.warning(f"  No chunks found for doc_id={doc_id[:16]}")
            continue

        # Use the first chunk's doc_title to extract metadata
        first_chunk = chunks[0]
        doc_title = first_chunk.payload.get("doc_title", "")
        chunk_text = first_chunk.payload.get("chunk_text", "")

        # Extract corrected metadata
        new_case_numbers = _extract_case_numbers(doc_title, chunk_text)
        new_party_names = _extract_case_caption_parties(doc_title) or _extract_party_names(doc_title)

        if not new_case_numbers and not new_party_names:
            logger.debug(f"  No metadata to update")
            continue

        # Prepare payload updates
        payload_updates = {}
        if new_case_numbers:
            payload_updates["case_numbers"] = new_case_numbers
        if new_party_names:
            payload_updates["party_names"] = new_party_names

        # Update all chunks for this document
        point_ids = [ch.id for ch in chunks]
        await client.set_payload(
            collection_name=collection,
            payload=payload_updates,
            points=PointIdsList(points=point_ids),
            wait=True,
        )

        logger.info(f"  Updated {len(point_ids)} chunks: case_numbers={new_case_numbers}, party_names={new_party_names}")

    logger.info(f"✅ Completed: updated {len(doc_ids)} documents with corrected case_numbers and party_names")
    await client.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
