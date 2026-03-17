from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk


@dataclass(frozen=True)
class GuardDecision:
    triggered: bool
    term: str = ""
    reason: str = ""


def check_query_premise(
    query: str,
    context_chunks: Sequence[RankedChunk],
    disallowed_terms: Sequence[str],
) -> GuardDecision:
    query_text = query.strip().lower()
    if not query_text:
        return GuardDecision(triggered=False)

    terms = [term.strip().lower() for term in disallowed_terms if term.strip()]
    if not terms:
        return GuardDecision(triggered=False)

    context_text = "\n".join(chunk.text for chunk in context_chunks).lower()
    for term in terms:
        if not _contains_term(query_text, term):
            continue
        if not _context_supports_term(context_text, term):
            return GuardDecision(
                triggered=True,
                term=term,
                reason=f"query mentions '{term}' but retrieved context does not support it",
            )
    return GuardDecision(triggered=False)


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if " " in term:
        return term in text
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return bool(pattern.search(text))


def _context_supports_term(context_text: str, term: str) -> bool:
    if not _contains_term(context_text, term):
        return False

    segments = [segment.strip() for segment in re.split(r"[\n.!?]+", context_text) if segment.strip()]
    for segment in segments:
        if not _contains_term(segment, term):
            continue
        if not _segment_negates_term(segment, term):
            return True
    return False


def _segment_negates_term(segment: str, term: str) -> bool:
    saw = False
    for match in re.finditer(re.escape(term), segment, flags=re.IGNORECASE):
        saw = True
        start = match.start()
        end = match.end()
        before = segment[max(0, start - 48) : start]
        after = segment[end : min(len(segment), end + 24)]
        if not (_has_negation_marker(before) or _has_postfix_negation(after)):
            return False
    return saw


def _has_negation_marker(text: str) -> bool:
    return bool(
        re.search(
            r"\b(no|not|without|none|never|cannot|can't|doesn't|does not|do not|did not|lack|lacks|lacking)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_postfix_negation(text: str) -> bool:
    return bool(re.search(r"\b(is not|are not|was not|were not|does not|did not)\b", text, flags=re.IGNORECASE))
