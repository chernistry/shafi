# pyright: reportPrivateUsage=false
"""Free-text answer quality improvements for better Asst scores.

Strips verbose preamble phrases that waste the 280-character budget,
condenses multi-sentence answers, and normalizes citation artifacts.

Integration:
    Called from ``normalize_free_text_answer()`` in submission/common.py,
    or directly as a post-generation cleanup step.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Preamble patterns that waste the 280-char budget
# ---------------------------------------------------------------------------

# These patterns appear at the START of LLM answers and add no information.
# They're explicitly prohibited by the prompts but LLMs still produce them.
_PREAMBLE_PATTERNS: list[re.Pattern[str]] = [
    # "According to [source/Article X],"
    re.compile(
        r"^(?:According\s+to|As\s+(?:stated|specified|set\s+out|provided|outlined|defined)\s+(?:in|under|by))"
        r"(?:\s+(?:Article|Section|Clause|Schedule|Part|Rule|Regulation)\s+\d+(?:\([^)]*\))?(?:\s+of)?)?[^,]{0,80}?,\s*",
        re.IGNORECASE,
    ),
    # "Based on the [provided sources/available information],"
    re.compile(
        r"^Based\s+on\s+(?:the\s+)?(?:provided\s+)?(?:sources?|information|documents?|context)[^,]{0,40}?,\s*",
        re.IGNORECASE,
    ),
    # NOTE: "Under [Article X]," is NOT stripped — it's an evidence-first
    # opening that cites the governing provision. Stripping it hurts Asst
    # because the LLM judge values grounded answers. (SHAI, 2026-03-22)
    # "In accordance with [the provisions of]..."
    re.compile(
        r"^In\s+accordance\s+with\s+(?:the\s+)?(?:provisions\s+of\s+)?[^,]{0,80}?,\s*",
        re.IGNORECASE,
    ),
    # NOTE: "Pursuant to [Article X]," NOT stripped — evidence-first legal citation.
    # (SHAI, 2026-03-22)
    # "The [answer/short preamble] is [that/as follows]:"
    re.compile(
        r"^The\s+(?:answer|response)\s+(?:to\s+(?:this|the)\s+question\s+)?is\s+(?:that\s+)?",
        re.IGNORECASE,
    ),
    # "Yes/No," followed by actual answer content — keep the Yes/No but strip the comma continuation if verbose
    # (handled separately — do NOT strip boolean lead)
    # "As per [Article X / the Law],"
    re.compile(
        r"^As\s+per\s+(?:(?:Article|Section|Clause)\s+\d+(?:\([^)]*\))?\s+of\s+)?[^,]{0,80}?,\s*",
        re.IGNORECASE,
    ),
    # "It should be noted that" / "It is worth noting that"
    re.compile(
        r"^It\s+(?:should\s+be|is\s+worth)\s+not(?:ed|ing)\s+that\s+",
        re.IGNORECASE,
    ),
    # "In summary," / "To summarize,"
    re.compile(
        r"^(?:In\s+summary|To\s+summar(?:ize|ise)),?\s*",
        re.IGNORECASE,
    ),
]

# Redundant trailing phrases
_TRAILING_PATTERNS: list[re.Pattern[str]] = [
    # "...as per the sources provided."
    re.compile(
        r"\s*(?:,\s*)?as\s+per\s+(?:the\s+)?(?:provided\s+)?(?:sources?|documents?|information)\.?\s*$",
        re.IGNORECASE,
    ),
    # "...according to the available sources."
    re.compile(
        r"\s*(?:,\s*)?according\s+to\s+(?:the\s+)?(?:available\s+|provided\s+)?(?:sources?|documents?|information)\.?\s*$",
        re.IGNORECASE,
    ),
    # "...based on the provided context."
    re.compile(
        r"\s*(?:,\s*)?based\s+on\s+(?:the\s+)?(?:provided\s+)?(?:sources?|context|information|documents?)\.?\s*$",
        re.IGNORECASE,
    ),
]


def strip_verbose_preamble(text: str) -> str:
    """Remove verbose preamble phrases from the start of a free-text answer.

    Returns the cleaned text with the first letter capitalized.
    """
    result = text.strip()
    if not result:
        return result

    for pattern in _PREAMBLE_PATTERNS:
        m = pattern.match(result)
        if m:
            remainder = result[m.end() :].strip()
            if remainder and len(remainder) >= 10:
                # Capitalize the first letter of the remaining text
                result = remainder[0].upper() + remainder[1:] if remainder else remainder
                break

    return result


def strip_trailing_filler(text: str) -> str:
    """Remove redundant trailing phrases like 'as per the sources'."""
    result = text.strip()
    for pattern in _TRAILING_PATTERNS:
        result = pattern.sub("", result).strip()
    return result


def condense_free_text(text: str) -> str:
    """Apply all free-text quality improvements.

    1. Strip preamble
    2. Strip trailing filler
    3. Normalize whitespace

    Does NOT truncate — that's handled by normalize_free_text_answer.
    """
    result = text.strip()
    if not result:
        return result

    result = strip_verbose_preamble(result)
    result = strip_trailing_filler(result)

    # Normalize double spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result
