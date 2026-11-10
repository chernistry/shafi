"""DIFC appeal-chain detector for case-law question routing.

Scans indexed judgment chunks to build a case-level appeal graph
(SCT → CFI → CA). Expands retrieval doc_refs for appeal-related queries
so that both the original judgment and the appellate decision are retrieved.

DIFC court hierarchy (ascending):
  SCT / TCD / ARB / ENF  →  CFI  →  CA

When a CFI judgment cites an SCT case, we infer the SCT case was appealed
and record the mapping so retrieval can find the CFI document.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Canonical case ref pattern: "CFI 070/2025", "SCT 133/2025", "CA 006/2024"
_CASE_REF_RE = re.compile(
    r"\b(CA|CFI|SCT|TCD|ARB|ENF)\s+(\d{3}/\d{4})\b",
    re.IGNORECASE,
)

# Court prefix → numeric hierarchy level (higher = appellate)
_COURT_LEVEL: dict[str, int] = {
    "sct": 1,
    "tcd": 1,
    "arb": 1,
    "enf": 1,
    "cfi": 2,
    "ca": 3,
}

# Keywords that signal an appeal-related query
_APPEAL_KEYWORDS = frozenset(("appea", "affirm", "revers", "uphold", "upheld", "permission to appeal"))


def _case_level(case_ref: str) -> int:
    """Return the court hierarchy level for a canonical case reference."""
    prefix = case_ref.split()[0].lower() if case_ref else ""
    return _COURT_LEVEL.get(prefix, 0)


def _extract_case_refs(text: str) -> list[str]:
    """Extract deduplicated canonical case refs (e.g. 'CFI 070/2025') from text."""
    seen: set[str] = set()
    refs: list[str] = []
    for m in _CASE_REF_RE.finditer(text or ""):
        ref = f"{m.group(1).upper()} {m.group(2)}"
        key = ref.casefold()
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


def is_appeal_query(query: str) -> bool:
    """Return True when the query asks about an appeal relationship.

    Args:
        query: Raw user question text.

    Returns:
        True if the query contains appeal-related keywords.
    """
    ql = (query or "").lower()
    return any(kw in ql for kw in _APPEAL_KEYWORDS)


class DIFCAppealChainDetector:
    """Detects DIFC appeal chains (SCT→CFI→CA) from indexed judgment chunks.

    Build once (at server startup) by calling ``build_from_chunks()``.
    Then call ``expand_doc_refs()`` at query time to add appellate case refs.

    Usage::

        detector = DIFCAppealChainDetector()
        detector.build_from_chunks([(doc_id, doc_title, chunk_text), ...])

        # At query time:
        if is_appeal_query(query) and detector.built:
            doc_refs = detector.expand_doc_refs(doc_refs, query)
    """

    def __init__(self) -> None:
        # canonical case_ref.casefold() → appellate doc_id
        self._appeal_map: dict[str, str] = {}
        # doc_id → primary case ref from doc title
        self._doc_primary: dict[str, str] = {}
        self._built: bool = False

    @property
    def built(self) -> bool:
        """True after build_from_chunks() has been called."""
        return self._built

    def build_from_chunks(
        self,
        chunks: Iterable[tuple[str, str, str]],
    ) -> None:
        """Build the appeal graph from (doc_id, doc_title, chunk_text) triples.

        Scans each chunk for cited case refs.  When a higher-court document
        cites a lower-court case, records the appeal relationship.

        Args:
            chunks: Iterable of (doc_id, doc_title, chunk_text) triples.
        """
        doc_primary: dict[str, str] = {}
        doc_cited: dict[str, set[str]] = defaultdict(set)

        for doc_id, doc_title, chunk_text in chunks:
            doc_id = str(doc_id or "").strip()
            if not doc_id:
                continue
            if doc_id not in doc_primary:
                title_refs = _extract_case_refs(str(doc_title or ""))
                if title_refs:
                    doc_primary[doc_id] = title_refs[0]
            chunk_refs = _extract_case_refs(str(chunk_text or ""))
            doc_cited[doc_id].update(chunk_refs)

        appeal_map: dict[str, str] = {}
        for doc_id, primary in doc_primary.items():
            primary_level = _case_level(primary)
            if primary_level < 2:
                # Lower-court docs never serve as appellate evidence.
                continue
            for cited_ref in doc_cited.get(doc_id, set()):
                if cited_ref.casefold() == primary.casefold():
                    continue
                if _case_level(cited_ref) < primary_level:
                    # doc_id is the appellate decision for cited_ref.
                    key = cited_ref.casefold()
                    if key not in appeal_map:
                        appeal_map[key] = doc_id

        self._doc_primary = doc_primary
        self._appeal_map = appeal_map
        self._built = True
        logger.info(
            "DIFCAppealChainDetector: built from %d docs, %d appeal chains",
            len(doc_primary),
            len(appeal_map),
        )

    def get_appellate_case_ref(self, case_ref: str) -> str | None:
        """Return the appellate case ref for a lower-court case ref.

        Args:
            case_ref: Canonical lower-court case reference (e.g. "SCT 133/2025").

        Returns:
            Appellate case reference string, or None if not found.
        """
        doc_id = self._appeal_map.get(case_ref.casefold())
        if doc_id is None:
            return None
        return self._doc_primary.get(doc_id)

    def expand_doc_refs(
        self,
        doc_refs: list[str],
        query: str = "",
    ) -> list[str]:
        """Expand doc_refs with appellate case refs from the appeal chain.

        Only expands when ``built`` is True.  Preserves original order;
        appellate refs are appended after their corresponding source refs.

        Args:
            doc_refs: Original case references from the query classifier.
            query: Raw query text (used to check appeal keywords).

        Returns:
            Expanded list of case references including appellate cases.
        """
        if not self._built:
            return doc_refs
        if not is_appeal_query(query) and query:
            return doc_refs
        seen: set[str] = {r.casefold() for r in doc_refs}
        result = list(doc_refs)
        for ref in doc_refs:
            appellate = self.get_appellate_case_ref(ref)
            if appellate and appellate.casefold() not in seen:
                seen.add(appellate.casefold())
                result.append(appellate)
        return result
