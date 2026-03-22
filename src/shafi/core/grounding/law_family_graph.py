"""Deterministic law-family bundle helpers for sidecar doc scope."""

from __future__ import annotations

import re
from dataclasses import dataclass

from shafi.core.legal_title_family import (
    derive_law_title_aliases,
    derive_related_law_families,
    extract_query_law_families,
    title_key,
)

_AMENDMENT_KEY_RE = re.compile(r"\s+amendment law\b", re.IGNORECASE)
_ENACTMENT_KEY_RE = re.compile(r"\s+enactment notice\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LawFamilyBundle:
    """Normalized law-family bundle for query or candidate matching.

    Args:
        exact_keys: Exact canonical family keys.
        related_keys: Related/base family keys used for narrow bundle matching.
    """

    exact_keys: tuple[str, ...]
    related_keys: tuple[str, ...]


def build_query_law_family_bundle(query: str) -> LawFamilyBundle:
    """Build a bundle of exact and related law-family keys from a query.

    Args:
        query: Raw user query.

    Returns:
        Query law-family bundle.
    """

    exact_keys = tuple(sorted({key for key in extract_query_law_families(query) if key}))
    related_keys = tuple(sorted({related for key in exact_keys for related in _derive_related_keys(key)}))
    return LawFamilyBundle(exact_keys=exact_keys, related_keys=related_keys)


def build_candidate_law_family_bundle(doc_title: str, law_titles: tuple[str, ...]) -> LawFamilyBundle:
    """Build a bundle of exact and related law-family keys for a document.

    Args:
        doc_title: Candidate document title.
        law_titles: Candidate law-title metadata.

    Returns:
        Candidate law-family bundle.
    """

    exact_keys = {title_key(alias) for alias in derive_law_title_aliases(doc_title, *law_titles) if title_key(alias)}
    related_keys = {
        title_key(alias) for alias in derive_related_law_families(doc_title, *law_titles) if title_key(alias)
    }
    return LawFamilyBundle(
        exact_keys=tuple(sorted(exact_keys)),
        related_keys=tuple(sorted(related_keys | exact_keys)),
    )


def law_family_match_score(query_bundle: LawFamilyBundle, candidate_bundle: LawFamilyBundle) -> float:
    """Score bundle match strength between query and candidate law families.

    Args:
        query_bundle: Query bundle.
        candidate_bundle: Candidate bundle.

    Returns:
        Deterministic narrow match score in ``[0.0, 3.0]``.
    """

    if not query_bundle.exact_keys:
        return 0.0
    query_exact = set(query_bundle.exact_keys)
    query_related = set(query_bundle.related_keys)
    candidate_exact = set(candidate_bundle.exact_keys)
    candidate_related = set(candidate_bundle.related_keys)

    if query_exact & candidate_exact:
        return 3.0
    if query_exact & candidate_related:
        return 2.5
    if query_related & candidate_exact:
        return 2.0
    if query_related & candidate_related:
        return 1.0
    return 0.0


def _derive_related_keys(key: str) -> set[str]:
    """Derive related/base family keys from an already normalized family key.

    Args:
        key: Canonical family key.

    Returns:
        Set of related keys used for narrow bundle matching.
    """

    cleaned = str(key or "").strip()
    if not cleaned:
        return set()
    related = {cleaned}
    related.add(_AMENDMENT_KEY_RE.sub("", cleaned).strip())
    related.add(_ENACTMENT_KEY_RE.sub("", cleaned).strip())
    return {value for value in related if value}
