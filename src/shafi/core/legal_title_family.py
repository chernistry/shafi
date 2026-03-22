"""Canonical law-title family normalization for sidecar scope and page metadata."""

from __future__ import annotations

import re
import unicodedata

_LAWISH_TITLE_RE = re.compile(
    r"(?P<title>[A-Z][A-Za-z0-9&,'()/-]+(?:\s+[A-Z][A-Za-z0-9&,'()/-]+){0,10}\s+"
    r"(?:Law|Regulations?|Rules?)(?:\s+Amendment\s+Law|\s+Enactment\s+Notice)?"
    r"(?:\s+No\.?\s*\d+\s+of\s+\d{4}|\s+\d{4})?)",
    re.IGNORECASE,
)
_GENERIC_LAW_SHELL_RE = re.compile(
    r"^(?:(?:the|difc|dubai|ruler of dubai)\s+)*(?:law|regulations?|rules?)\b",
    re.IGNORECASE,
)
_LAW_NUMBER_SUFFIX_RE = re.compile(r"\bNo\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_YEAR_SUFFIX_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_LEADING_ARTICLE_RE = re.compile(r"^(?:(?:the|difc)\s+)+", re.IGNORECASE)
_JURISDICTION_PREFIX_RE = re.compile(
    r"^(?:(?:the|difc|dubai|uae|united arab emirates|ruler of dubai)\s+)+",
    re.IGNORECASE,
)
_ENACTMENT_NOTICE_RE = re.compile(r"\bEnactment Notice\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_QUERY_LEADIN_RE = re.compile(
    r"^(?:according to|under|pursuant to|per|in accordance with)\s+",
    re.IGNORECASE,
)
_QUERY_TAIL_RE = re.compile(r"\b(?:what|which|who|when|where|how)\b.*$", re.IGNORECASE)
_TITLE_ALIAS_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "ip law": ("Intellectual Property Law",),
    "intellectual property law": ("IP Law",),
}


def _clean_title(text: str) -> str:
    """Normalize law-title text while preserving family semantics.

    Args:
        text: Raw title-like string.

    Returns:
        str: Cleaned title string.
    """

    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    cleaned = re.sub(r"^The\s+The\b", "The", cleaned, flags=re.IGNORECASE)
    return cleaned


def _has_numbered_suffix(text: str) -> bool:
    """Return whether a title carries a law-number or year suffix.

    Args:
        text: Cleaned title-like string.

    Returns:
        bool: True when the title includes a numbered law/regulation/rule suffix.
    """

    return bool(_LAW_NUMBER_SUFFIX_RE.search(text) or _YEAR_SUFFIX_RE.search(text))


def _is_generic_law_shell(text: str) -> bool:
    """Return whether a title is a generic law shell without a substantive name.

    Args:
        text: Cleaned title-like string.

    Returns:
        bool: True when the title starts with a jurisdiction prefix and a generic
        law/regulation/rule token instead of a substantive subject name.
    """

    return bool(_GENERIC_LAW_SHELL_RE.match(text or ""))


def _strip_numbered_suffix(text: str) -> str:
    """Remove numbered law suffixes from a canonical title.

    Args:
        text: Canonical title-like string.

    Returns:
        str: Title with law-number/year noise removed.
    """

    cleaned = _LAW_NUMBER_SUFFIX_RE.sub("", text or "")
    cleaned = _YEAR_SUFFIX_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")


def _strip_jurisdiction_prefix(text: str) -> str:
    """Remove safe jurisdiction prefixes from substantive law titles.

    Args:
        text: Canonical title-like string.

    Returns:
        Title with a safe jurisdiction prefix removed when doing so preserves a
        substantive law family; otherwise returns the cleaned input unchanged.
    """

    cleaned = _clean_title(text)
    if not cleaned or _is_generic_law_shell(cleaned):
        return cleaned
    stripped = _JURISDICTION_PREFIX_RE.sub("", cleaned).strip(" ,.;:-")
    return stripped or cleaned


def _safe_title_family_variants(text: str) -> list[str]:
    """Generate narrow alias variants for a law-like title.

    Args:
        text: Raw or normalized title string.

    Returns:
        Ordered unique title variants suitable for alias/key generation.
    """

    cleaned = _clean_title(text)
    if not cleaned:
        return []

    variants: list[str] = [cleaned]
    family = canonical_law_family(cleaned)
    if family:
        variants.append(family)

    for candidate in (cleaned, family):
        stripped = _strip_jurisdiction_prefix(candidate)
        if stripped:
            variants.append(stripped)
        expanded = _TITLE_ALIAS_EXPANSIONS.get(candidate.casefold(), ())
        variants.extend(expanded)
        if stripped:
            variants.extend(_TITLE_ALIAS_EXPANSIONS.get(stripped.casefold(), ()))

    if _is_generic_law_shell(family) and _has_numbered_suffix(family):
        variants.append(_strip_numbered_suffix(family))

    if "amendment law" in family.casefold():
        base = re.sub(r"(?i)\s+amendment law\b", "", family).strip(" ,.;:-")
        if base:
            variants.append(base)
            variants.append(_strip_jurisdiction_prefix(base))

    if "enactment notice" in cleaned.casefold():
        stripped_notice = _ENACTMENT_NOTICE_RE.sub("", family).strip(" ,.;:-")
        if stripped_notice:
            variants.append(stripped_notice)
            variants.append(_strip_jurisdiction_prefix(stripped_notice))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = _clean_title(variant)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def canonical_law_family(title: str) -> str:
    """Build a canonical family label from a raw law-like title.

    Args:
        title: Raw title-like string.

    Returns:
        str: Canonical family label with numbering/year noise removed.
    """

    cleaned = _clean_title(title)
    if _is_generic_law_shell(cleaned) and _has_numbered_suffix(cleaned):
        return cleaned
    cleaned = _LAW_NUMBER_SUFFIX_RE.sub("", cleaned)
    cleaned = _YEAR_SUFFIX_RE.sub("", cleaned)
    cleaned = _ENACTMENT_NOTICE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned


def title_key(title: str) -> str:
    """Build a stopword-stable key for title family matching.

    Args:
        title: Raw or cleaned title.

    Returns:
        str: Canonical comparison key.
    """

    cleaned = canonical_law_family(title)
    cleaned = _LEADING_ARTICLE_RE.sub("", cleaned)
    cleaned = _PUNCT_RE.sub(" ", cleaned.casefold())
    return re.sub(r"\s+", " ", cleaned).strip()


def derive_law_title_aliases(*titles: str) -> list[str]:
    """Generate stable alias variants for one or more law-like titles.

    Args:
        *titles: Raw title strings.

    Returns:
        list[str]: Unique alias variants.
    """

    seen: set[str] = set()
    aliases: list[str] = []

    def add(value: str) -> None:
        cleaned = _clean_title(value)
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        aliases.append(cleaned)

    for raw in titles:
        for variant in _safe_title_family_variants(raw):
            add(variant)
            add(_LEADING_ARTICLE_RE.sub("", variant))

    return aliases


def derive_related_law_families(*titles: str) -> list[str]:
    """Derive related/base family labels for law bundles.

    Args:
        *titles: Raw title strings.

    Returns:
        list[str]: Related family labels.
    """

    seen: set[str] = set()
    families: list[str] = []
    for raw in titles:
        for candidate in _safe_title_family_variants(raw):
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            families.append(candidate)
    return families


def extract_query_law_families(query: str) -> list[str]:
    """Extract canonical law-family keys from a natural-language query.

    Args:
        query: Raw user question.

    Returns:
        list[str]: Canonical family keys found in the query.
    """

    seen: set[str] = set()
    families: list[str] = []
    for match in _LAWISH_TITLE_RE.finditer(query or ""):
        raw_title = _QUERY_LEADIN_RE.sub("", match.group("title")).strip()
        raw_title = _QUERY_TAIL_RE.sub("", raw_title).strip(" ,.;:-")
        for candidate in _safe_title_family_variants(raw_title):
            key = title_key(candidate)
            if key and key not in seen:
                seen.add(key)
                families.append(key)
    for trigger in ("IP Law", "Intellectual Property Law"):
        if not re.search(rf"\b{re.escape(trigger)}\b", query or "", flags=re.IGNORECASE):
            continue
        for candidate in _safe_title_family_variants(trigger):
            key = title_key(candidate)
            if key and key not in seen:
                seen.add(key)
                families.append(key)
    return families
