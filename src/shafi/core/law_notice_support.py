"""Helpers for extracting law-like fields from enactment notices and similar orders."""

from __future__ import annotations

import re

_LAW_NUMBER_YEAR_RE = re.compile(
    r"\b(?:DIFC\s+)?(?:Law|Act|Regulation|Order|Decree|Decision|Resolution|Rule|Rules)\s+No\.?\s*(?P<number>\d+)\s+of\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_EXPLICIT_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_ENACTMENT_DATE_RE = re.compile(
    r"on this\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+day of\s+(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_DATE_CUE_RE = re.compile(
    r"(?:Date of Issue|Date of Re-issue|effective date|effective from|effective on|in force on|"
    r"comes into force on|comes into effect on|comes into operation on|commenc(?:e|ed|ement on)|"
    r"shall come into force on|shall come into effect on|take effect on|takes effect on|promulgated on|issued on)"
    r"[:\s]+(?P<date>[^\n.]+)",
    re.IGNORECASE,
)
_RULER_AUTHORITY_RE = re.compile(
    r"(?:We|I),\s*(?P<authority>[^,\n]+(?:,\s*[^,\n]+){0,3}),\s*Ruler of Dubai",
    re.IGNORECASE,
)
_NOTICE_AUTHORITY_RE = re.compile(
    r"(?:Issued|Administered|Signed|Approved|Made|Promulgated)\s+by\s+(?P<authority>[^.;\n]+)"
    r"|By\s+order\s+of\s+(?P<order_authority>[^.;\n]+)"
    r"|Under\s+the\s+authority\s+of\s+(?P<authority_chain>[^.;\n]+)"
    r"|(?:The\s+)?(?P<authority_lead>DIFC Authority|Dubai Financial Services Authority|"
    r"Dubai International Financial Centre Authority|DFSA|DIFCA|Registrar|President)"
    r"(?:\s+shall\s+(?:administer|issue|enact|make|promulgate|sign)\b.*)?",
    re.IGNORECASE,
)
_AUTHORITY_MARKER_RE = re.compile(
    r"\b(?:DIFC Authority|Dubai Financial Services Authority|Dubai International Financial Centre Authority|"
    r"DFSA|DIFCA|Registrar|President|Ruler of Dubai)\b",
    re.IGNORECASE,
)
_UNDERSCORE_TITLE_RE = re.compile(r"^[_\s.]+$")
_GENERIC_TITLE_RE = re.compile(r"^(?:ENACTMENT NOTICE|CONSOLIDATED VERSION)$", re.IGNORECASE)
_LAWISH_TITLE_RE = re.compile(r"\b(?:law|regulations?|act|code|decree|decision|resolution|rule|rules)\b", re.IGNORECASE)


def normalize_law_like_title(*, title: str, source_text: str) -> str:
    """Return the best law-like title available for a compiled instrument.

    Args:
        title: Current compiled title.
        source_text: Full source text.

    Returns:
        str: Normalized law-like title, or an empty string when unavailable.
    """

    cleaned_title = _clean_text(title)
    if _is_specific_title(cleaned_title):
        return cleaned_title
    for line in source_text.splitlines():
        cleaned_line = _clean_text(line)
        if not _is_specific_title(cleaned_line):
            continue
        if _looks_law_like(cleaned_line):
            return cleaned_line
    return cleaned_title if _is_specific_title(cleaned_title) else ""


def extract_law_number_year(*, title: str, source_text: str) -> tuple[str, str]:
    """Extract a law number and year from title or source text.

    Args:
        title: Current compiled title.
        source_text: Full source text.

    Returns:
        tuple[str, str]: Extracted `(number, year)` strings.
    """

    for candidate in (title, source_text):
        match = _LAW_NUMBER_YEAR_RE.search(candidate or "")
        if match is not None:
            return match.group("number"), match.group("year")
    return "", ""


def extract_enactment_authority(*, source_text: str, fallback: str = "") -> str:
    """Extract a cleaner enacting authority for enactment notices.

    Args:
        source_text: Full source text.
        fallback: Existing extracted authority.

    Returns:
        str: Normalized authority, or the cleaned fallback when no better value exists.
    """

    source_text = source_text or ""
    match = _RULER_AUTHORITY_RE.search(source_text)
    if match is not None:
        return _clean_text(f"{match.group('authority')}, Ruler of Dubai")
    for line in source_text.splitlines():
        cleaned_line = _clean_text(line)
        if not cleaned_line:
            continue
        match = _NOTICE_AUTHORITY_RE.search(cleaned_line)
        if match is None:
            continue
        for group_name in ("authority", "order_authority", "authority_chain", "authority_lead"):
            candidate = match.group(group_name)
            if candidate is None:
                continue
            normalized_candidate = _normalize_authority_candidate(candidate)
            if normalized_candidate and _AUTHORITY_MARKER_RE.search(normalized_candidate):
                return normalized_candidate
    cleaned_fallback = _clean_text(fallback)
    if cleaned_fallback.casefold() in {"", "the"}:
        return ""
    return cleaned_fallback


def extract_enactment_date(*, source_text: str, fallback: str = "") -> str:
    """Extract the operative enactment date from an enactment notice.

    Args:
        source_text: Full source text.
        fallback: Existing extracted date.

    Returns:
        str: Normalized enactment date, or the cleaned fallback.
    """

    match = _DATE_CUE_RE.search(source_text or "")
    if match is not None:
        candidate = _clean_text(match.group("date"))
        explicit = _EXPLICIT_DATE_RE.search(candidate)
        if explicit is not None:
            return _format_date(explicit.group("day"), explicit.group("month"), explicit.group("year"))
    match = _ENACTMENT_DATE_RE.search(source_text or "")
    if match is not None:
        return _format_date(match.group("day"), match.group("month"), match.group("year"))
    explicit = _EXPLICIT_DATE_RE.search(source_text or "")
    if explicit is not None:
        return _format_date(explicit.group("day"), explicit.group("month"), explicit.group("year"))
    cleaned_fallback = _clean_text(fallback)
    if cleaned_fallback:
        fallback_enactment = _ENACTMENT_DATE_RE.search(cleaned_fallback)
        if fallback_enactment is not None:
            return _format_date(
                fallback_enactment.group("day"),
                fallback_enactment.group("month"),
                fallback_enactment.group("year"),
            )
        fallback_explicit = _EXPLICIT_DATE_RE.search(cleaned_fallback)
        if fallback_explicit is not None:
            return _format_date(
                fallback_explicit.group("day"),
                fallback_explicit.group("month"),
                fallback_explicit.group("year"),
            )
    return ""


def is_law_like_order_title(*, title: str, source_text: str) -> bool:
    """Report whether an order likely represents a law-like instrument.

    Args:
        title: Current compiled title.
        source_text: Full source text.

    Returns:
        bool: True when the document should contribute law-title aliases or fields.
    """

    normalized = normalize_law_like_title(title=title, source_text=source_text)
    if normalized and _looks_law_like(normalized):
        return True
    number, year = extract_law_number_year(title=title, source_text=source_text)
    return bool(number and year)


def _is_specific_title(text: str) -> bool:
    """Return whether a title candidate is non-generic and usable.

    Args:
        text: Title candidate.

    Returns:
        bool: True when the title is not placeholder noise.
    """

    if not text:
        return False
    if _UNDERSCORE_TITLE_RE.fullmatch(text):
        return False
    return not _GENERIC_TITLE_RE.fullmatch(text)


def _looks_law_like(text: str) -> bool:
    """Return whether text resembles a law or regulation title.

    Args:
        text: Candidate title.

    Returns:
        bool: True when the title looks like a legal instrument title.
    """

    return bool(_LAWISH_TITLE_RE.search(text))


def _clean_text(text: str) -> str:
    """Normalize whitespace in extracted support text.

    Args:
        text: Raw text value.

    Returns:
        str: Single-space normalized text.
    """

    return " ".join((text or "").split()).strip()


def _normalize_authority_candidate(text: str) -> str:
    """Normalize a candidate authority name extracted from notice text.

    Args:
        text: Raw authority candidate.

    Returns:
        str: Cleaned authority text with a leading article removed.
    """

    cleaned = _clean_text(text)
    cleaned = re.sub(r"(?i)^the\s+", "", cleaned)
    return cleaned.rstrip(" .;,")


def _format_date(day: str, month: str, year: str) -> str:
    """Format a parsed date into the repo's simple day-month-year form.

    Args:
        day: Day component.
        month: Month component.
        year: Year component.

    Returns:
        str: Normalized date text.
    """

    return f"{int(day)} {month} {year}"
