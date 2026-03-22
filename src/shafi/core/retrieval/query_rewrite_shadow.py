"""Deterministic shadow query rewrite helpers for bounded grounding recovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shafi.core.grounding.scope_policy import extract_case_refs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.models.schemas import RetrievedPage, ScopeMode

_SPACING_RE = re.compile(r"\s+")
_COMPARE_HINT_RE = re.compile(r"\b(compare|common|both|same|between)\b", re.IGNORECASE)
_JURISDICTION_FULL_NAMES = {
    "DIFC": "Dubai International Financial Centre",
    "DFSA": "Dubai Financial Services Authority",
    "UAE": "United Arab Emirates",
}


@dataclass(frozen=True, slots=True)
class ShadowRewriteQuery:
    """One deterministic shadow rewrite candidate.

    Args:
        family: Bounded family label.
        rewritten_query: Deterministically rewritten query text.
        reasons: Short reasons describing the rewrite.
    """

    family: str
    rewritten_query: str
    reasons: tuple[str, ...]


def build_shadow_rewrite_query(
    *,
    query: str,
    family: str,
    hard_anchor_strings: Sequence[str],
    page_candidates: Sequence[RetrievedPage],
    scope_mode: ScopeMode,
) -> ShadowRewriteQuery | None:
    """Build a single deterministic shadow rewrite for bounded families.

    Args:
        query: Raw user query.
        family: Allowed family from the escalation gate.
        hard_anchor_strings: Explicit anchors extracted from the query.
        page_candidates: Current scoped page candidates.
        scope_mode: Grounding scope mode.

    Returns:
        ShadowRewriteQuery | None: One bounded rewrite or ``None``.
    """

    del scope_mode
    base_query = _clean(query)
    additions: list[str] = []
    reasons: list[str] = []
    anchors = [anchor.strip() for anchor in hard_anchor_strings if str(anchor).strip()]
    case_refs = extract_case_refs(query)
    top_page = page_candidates[0] if page_candidates else None

    if family == "compare_authoritative_pair":
        additions.extend(case_refs[:2])
        additions.extend(["official caption", "title page", "claimant", "party"])
        reasons.extend(["compare_case_refs" if case_refs else "compare_family", "caption_title_bias"])
    elif family == "exact_provision":
        additions.extend(anchors[:2] or _top_article_refs(page_candidates))
        additions.extend(["exact article", "official text", "section", "schedule"])
        reasons.extend(["hard_anchor" if anchors else "article_reinforcement", "official_text_bias"])
    elif family == "authority_metadata":
        additions.extend(_top_law_aliases(page_candidates))
        additions.extend(_jurisdiction_terms(query, page_candidates))
        additions.extend(["issued by", "date of issue", "law number", "enactment notice", "official title page"])
        reasons.extend(["authority_field_bias", "official_title_bias"])
    elif family == "title_caption_party":
        additions.extend(case_refs[:2])
        additions.extend(_top_law_aliases(page_candidates))
        additions.extend(["official title", "caption", "claimant", "party", "heading"])
        reasons.extend(["title_caption_bias"])
    elif family == "strict_field_lookup":
        additions.extend(_top_law_aliases(page_candidates))
        additions.extend(_jurisdiction_terms(query, page_candidates))
        additions.extend(["official field", "exact value", "heading block"])
        reasons.extend(["strict_field_lookup"])
    else:
        return None

    if top_page is not None and str(top_page.page_template_family).strip():
        additions.append(top_page.page_template_family.replace("_", " "))
        reasons.append("template_hint")

    rewritten = _append_terms(base_query, additions)
    if not rewritten or _clean(rewritten).casefold() == base_query.casefold():
        return None
    if family == "compare_authoritative_pair" and not (_COMPARE_HINT_RE.search(query) or case_refs):
        return None

    return ShadowRewriteQuery(
        family=family,
        rewritten_query=rewritten,
        reasons=tuple(dict.fromkeys(reason for reason in reasons if reason)),
    )


def _append_terms(base_query: str, additions: Sequence[str]) -> str:
    """Append unique informative terms to a base query.

    Args:
        base_query: Cleaned original query.
        additions: Candidate additions.

    Returns:
        str: Rewritten query with deduplicated additions.
    """

    lower_base = base_query.casefold()
    unique_terms: list[str] = []
    seen: set[str] = set()
    for item in additions:
        term = _clean(item)
        key = term.casefold()
        if not term or key in seen or key in lower_base:
            continue
        seen.add(key)
        unique_terms.append(term)
    if not unique_terms:
        return base_query
    return _clean(f"{base_query} {' '.join(unique_terms[:6])}")


def _top_article_refs(page_candidates: Sequence[RetrievedPage]) -> list[str]:
    """Return the strongest article anchors from top page candidates.

    Args:
        page_candidates: Current scoped page candidates.

    Returns:
        list[str]: Ordered article references.
    """

    refs: list[str] = []
    seen: set[str] = set()
    for page in page_candidates[:3]:
        for ref in page.article_refs:
            cleaned = _clean(ref)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            refs.append(cleaned)
    return refs


def _top_law_aliases(page_candidates: Sequence[RetrievedPage]) -> list[str]:
    """Return top law-title aliases from candidate pages.

    Args:
        page_candidates: Current scoped page candidates.

    Returns:
        list[str]: Ordered law-title aliases.
    """

    aliases: list[str] = []
    seen: set[str] = set()
    for page in page_candidates[:3]:
        for alias in [*page.law_title_aliases, *page.law_titles]:
            cleaned = _clean(alias)
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            aliases.append(cleaned)
    return aliases[:2]


def _jurisdiction_terms(query: str, page_candidates: Sequence[RetrievedPage]) -> list[str]:
    """Return jurisdiction-aware rewrite terms inferred from query or page hints.

    Args:
        query: Raw user query.
        page_candidates: Current scoped page candidates.

    Returns:
        list[str]: Deterministic jurisdiction-specific search terms.
    """

    haystacks = [query]
    for page in page_candidates[:3]:
        haystacks.append(getattr(page, "doc_title", ""))
        haystacks.extend(getattr(page, "law_title_aliases", []) or [])
        haystacks.extend(getattr(page, "law_titles", []) or [])

    lowered = " ".join(str(item or "") for item in haystacks).casefold()
    terms: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        cleaned = _clean(term)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            return
        seen.add(key)
        terms.append(cleaned)

    for marker, full_name in _JURISDICTION_FULL_NAMES.items():
        if marker.casefold() in lowered:
            _add(marker)
            _add(full_name)

    if "enactment notice" in lowered or "commencement" in lowered or "effective date" in lowered:
        _add("Enactment Notice")
        _add("commencement")
        _add("effective date")
    if "ruler of dubai" in lowered:
        _add("Ruler of Dubai")

    return terms


def _clean(text: str) -> str:
    """Normalize whitespace in query-rewrite helpers.

    Args:
        text: Raw text fragment.

    Returns:
        str: Whitespace-normalized text.
    """

    return _SPACING_RE.sub(" ", str(text or "")).strip()
