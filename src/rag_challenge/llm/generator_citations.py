"""Citation helpers for generator post-processing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_challenge.llm.generator_constants import CITE_RE
from rag_challenge.models import Citation, RankedChunk

if TYPE_CHECKING:
    from collections.abc import Sequence


def extract_citations(answer: str, chunks: Sequence[RankedChunk]) -> list[Citation]:
    """Build citation objects from inline cite markers.

    Args:
        answer: Model answer that may contain ``(cite: ...)`` markers.
        chunks: Retrieved chunks available in the current context.

    Returns:
        Citation objects ordered by first appearance.
    """
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    cited_ids = extract_cited_chunk_ids(answer)
    citations: list[Citation] = []
    for chunk_id in cited_ids:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            citations.append(Citation(chunk_id=chunk_id, doc_title="unknown"))
            continue
        citations.append(
            Citation(
                chunk_id=chunk_id,
                doc_title=chunk.doc_title,
                section_path=chunk.section_path or None,
            )
        )
    return citations


def extract_cited_chunk_ids(answer: str) -> list[str]:
    """Extract cited chunk ids from inline cite markers.

    Args:
        answer: Model answer text.

    Returns:
        Ordered unique cited chunk ids.
    """
    ids: list[str] = []
    for match in CITE_RE.finditer(answer):
        for raw_id in re.split(r"[,;]|\s+and\s+", match.group(1)):
            chunk_id = raw_id.strip()
            if chunk_id and chunk_id not in ids:
                ids.append(chunk_id)
    return ids


def sanitize_citations(answer: str, context_chunk_ids: Sequence[str]) -> str:
    """Drop citation markers that point outside the current context.

    Args:
        answer: Model answer text.
        context_chunk_ids: Chunk ids present in the current context.

    Returns:
        Sanitized answer text with invalid cite ids removed.
    """
    if not context_chunk_ids:
        return answer

    valid_ids = set(context_chunk_ids)

    def _replace(match: re.Match[str]) -> str:
        raw_inner = match.group(1)
        good = [cid.strip() for cid in re.split(r"[,;]|\s+and\s+", raw_inner) if cid.strip() in valid_ids]
        if not good:
            return ""
        return f"(cite: {', '.join(good)})"

    sanitized = CITE_RE.sub(_replace, answer).strip()
    return re.sub(r"  +", " ", sanitized)
