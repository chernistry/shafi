"""Legal citation pattern extraction for BM25 boosting."""

from __future__ import annotations

import re
from typing import Final

# Legal citation regex patterns (DIFC-specific)
CITATION_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("article", re.compile(r"\bArticle\s+\d+(?:\([^)]+\))*(?:\s+of\s+[A-Z][^.;,]+)?", re.IGNORECASE)),
    ("law_number", re.compile(r"(?:DIFC\s+)?Law\s+No\.\s*\d+\s+of\s+\d{4}", re.IGNORECASE)),
    ("case_number", re.compile(r"\b(?:CFI|SCT|CA|ENF|DEC|TCD|ARB)\s+\d{3}/\d{4}\b", re.IGNORECASE)),
    ("rule", re.compile(r"\bRule\s+\d+(?:\.\d+)*\b", re.IGNORECASE)),
    ("regulation", re.compile(r"\bRegulation\s+\d+(?:\([^)]+\))*\b", re.IGNORECASE)),
    ("schedule", re.compile(r"\bSchedule\s+[A-Z0-9]+\b", re.IGNORECASE)),
    ("section", re.compile(r"\bSection\s+\d+(?:\([^)]+\))*\b", re.IGNORECASE)),
]


def extract_citations(text: str) -> list[str]:
    """Extract all legal citation strings from text.

    Args:
        text: Query or document text

    Returns:
        List of citation strings found (deduplicated, preserving order)
    """
    citations: list[str] = []
    seen: set[str] = set()

    for _pattern_name, pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            citation = match.group(0).strip()
            if citation and citation not in seen:
                citations.append(citation)
                seen.add(citation)

    return citations


def has_legal_citations(text: str) -> bool:
    """Check if text contains any legal citation patterns."""
    return any(pattern.search(text) for _, pattern in CITATION_PATTERNS)
