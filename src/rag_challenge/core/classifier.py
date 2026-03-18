from __future__ import annotations

import re
from dataclasses import dataclass

from rag_challenge.config import get_settings
from rag_challenge.models import QueryComplexity

_MULTI_PART_RE = re.compile(
    r"(?:;\s*(?:and\s+)?|(?:\band\s+also\b)|(?:\badditionally\b)|(?:\bfurthermore\b)|(?:\bmoreover\b))",
    re.IGNORECASE,
)

_LEGAL_ENTITY_RE = re.compile(
    r"(?:§\s*\d+[\w().-]*"
    r"|\d+\s+U\.?S\.?C\.?\s*§?\s*\d+[\w().-]*"
    r"|Law\s+No\.?\s*\d+\s+of\s+\d{4}"
    r"|(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*\d{1,3}[/-]\d{4}"
    r"|\d+\s+[A-Z][A-Za-z.]+\s+\d+"
    r"|[A-Z][A-Za-z]+\s+v\.?\s+[A-Z][A-Za-z]+"
    r"|(?:Article|Section|Clause|Part|Schedule|Appendix)\s+\d+)",
    re.IGNORECASE,
)

_USC_RE = re.compile(r"(\d+)\s*U\.?S\.?C\.?\s*§?\s*(\d+)", re.IGNORECASE)
_LAW_NO_RE = re.compile(r"\blaw\s*no\.?\s*(\d+)\s*of\s*(\d{4})\b", re.IGNORECASE)
_DIFC_CASE_RE = re.compile(
    r"\b(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*(\d{1,4})\s*[/-]\s*(\d{4})\b", re.IGNORECASE
)
_LAW_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,10}\s+Law)\s+(\d{4})\b",
    re.IGNORECASE,
)
_REG_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,10}\s+Regulations?)\b(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_ARTICLE_SUB_RE = re.compile(r"\barticle\s+(\d+)\s*\(\s*([^)]+?)\s*\)", re.IGNORECASE)
_STRUCTURAL_REF_RE = re.compile(
    r"\b(?:Article|Section|Clause|Part|Schedule|Appendix)\s+\d+(?:\s*\([^)]+\))*",
    re.IGNORECASE,
)
_MULTI_WS_RE = re.compile(r"\s+")
_TITLE_PAGE_RE = re.compile(
    r"\b(?:title|cover)\s+page\b|\b(?:front\s+page|cover\s+sheet|first\s+page)\b",
    re.IGNORECASE,
)
_CAPTION_HEADER_RE = re.compile(r"\b(?:caption|header)\b", re.IGNORECASE)
_SECOND_PAGE_RE = re.compile(r"\b(?:second\s+page|page\s+2|2nd\s+page)\b", re.IGNORECASE)
_ORDINAL_PAGE_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\s+page\b", re.IGNORECASE)
_NUMERIC_PAGE_RE = re.compile(r"\bpage\s+(\d{1,3})\b", re.IGNORECASE)

_ACRONYMS = {
    "DIFC",
    "DFSA",
    "UAE",
    "AI",
    "AML",
    "CFT",
    "CRS",
    "ICC",
    "IC",
    "LLP",
    "PJSC",
}


@dataclass(frozen=True)
class ExplicitPageReference:
    kind: str
    phrase: str
    requested_page: int | None


class QueryClassifier:
    """Zero-latency heuristic query classifier + normalizer."""

    def __init__(self) -> None:
        self._settings = get_settings().llm

    @staticmethod
    def normalize_query(query: str) -> str:
        text = query.strip()
        text = _USC_RE.sub(r"\1 U.S.C. § \2", text)
        text = _LAW_NO_RE.sub(r"Law No. \1 of \2", text)
        text = _DIFC_CASE_RE.sub(lambda m: f"{m.group(1).upper()} {int(m.group(2)):03d}/{m.group(3)}", text)
        text = _ARTICLE_SUB_RE.sub(r"Article \1(\2)", text)
        text = _MULTI_WS_RE.sub(" ", text)
        return text

    @staticmethod
    def _normalize_law_title(raw_title: str, year: str) -> str:
        title = _MULTI_WS_RE.sub(" ", raw_title.strip())
        if not title or not year.strip():
            return ""
        stopwords = {
            "the",
            "of",
            "in",
            "under",
            "for",
            "to",
            "and",
            "or",
            "a",
            "an",
            "by",
            "on",
            "at",
            "from",
            "see",
            "compare",
        }

        tokens: list[str] = []
        for word in title.split(" "):
            clean = re.sub(r"[^A-Za-z0-9]", "", word)
            if clean:
                tokens.append(clean)
        if not tokens:
            return ""

        kept_rev: list[str] = []
        for token in reversed(tokens):
            if kept_rev and token.lower() in stopwords:
                break
            kept_rev.append(token)
        kept = list(reversed(kept_rev))
        if not kept:
            return ""

        words: list[str] = []
        for token in kept:
            if any(ch.isdigit() for ch in token):
                words.append(token)
                continue
            upper = token.upper()
            if upper in _ACRONYMS:
                words.append(upper)
            else:
                words.append(token[0].upper() + token[1:].lower())
        if not words:
            return ""
        if words[-1].lower() != "law":
            words.append("Law")
        else:
            words[-1] = "Law"
        return f"{' '.join(words)} {year.strip()}"

    @staticmethod
    def _normalize_reg_title(raw_title: str, year: str | None) -> str:
        title = _MULTI_WS_RE.sub(" ", raw_title.strip())
        if not title:
            return ""
        stopwords = {
            "the",
            "of",
            "in",
            "under",
            "for",
            "to",
            "and",
            "or",
            "a",
            "an",
            "by",
            "on",
            "at",
            "from",
            "see",
            "compare",
        }

        tokens: list[str] = []
        for word in title.split(" "):
            clean = re.sub(r"[^A-Za-z0-9]", "", word)
            if clean:
                tokens.append(clean)
        if not tokens:
            return ""

        kept_rev: list[str] = []
        for token in reversed(tokens):
            if kept_rev and token.lower() in stopwords:
                break
            kept_rev.append(token)
        kept = list(reversed(kept_rev))
        if not kept:
            return ""

        words: list[str] = []
        for token in kept:
            if any(ch.isdigit() for ch in token):
                words.append(token)
                continue
            upper = token.upper()
            if upper in _ACRONYMS:
                words.append(upper)
            else:
                words.append(token[0].upper() + token[1:].lower())
        if not words:
            return ""

        last = words[-1].lower()
        if last == "regulation":
            words[-1] = "Regulations"
        elif last != "regulations":
            words.append("Regulations")
        else:
            words[-1] = "Regulations"

        if year is not None and year.strip():
            return f"{' '.join(words)} {year.strip()}"
        return " ".join(words)

    @staticmethod
    def _normalize_article(raw: str) -> str:
        text = raw.strip()
        if not text:
            return ""
        text = re.sub(r"\barticle\b", "Article", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*\(\s*", "(", text)
        text = re.sub(r"\s*\)\s*", ")", text)
        return text.strip()

    @staticmethod
    def _normalize_structural_ref(raw: str) -> str:
        text = _MULTI_WS_RE.sub(" ", raw.strip())
        if not text:
            return ""
        match = re.match(r"(?i)(article|section|clause|part|schedule|appendix)\b", text)
        if match is None:
            return text
        label = match.group(1).capitalize()
        tail = text[match.end() :].strip()
        tail = re.sub(r"\s*\(\s*", "(", tail)
        tail = re.sub(r"\s*\)\s*", ")", tail)
        tail = re.sub(r"\s+", "", tail)
        if not tail:
            return label
        return f"{label} {tail}"

    @classmethod
    def extract_doc_refs(cls, query: str) -> list[str]:
        """Extract DIFC-style document identifiers from a query for retrieval filtering.

        Returned refs are normalized to match ingestion chunk metadata, e.g.:
        - "Law No. 12 of 2004"
        - "CFI 010/2024"
        """
        normalized = cls.normalize_query(query)
        refs: list[str] = []

        for match in _LAW_NO_RE.finditer(normalized):
            refs.append(f"Law No. {int(match.group(1))} of {match.group(2)}")

        for match in _DIFC_CASE_RE.finditer(normalized):
            refs.append(f"{match.group(1).upper()} {int(match.group(2)):03d}/{match.group(3)}")

        # Dedupe, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
        return out

    @classmethod
    def extract_query_refs(cls, query: str) -> list[str]:
        """Extract a broader set of normalized legal references from a query.

        This is intended for *analysis/guardrails* (e.g., multi-document grounding checks),
        not for retrieval filtering. It includes law titles like "Trust Law 2018".
        """
        normalized = cls.normalize_query(query)
        refs: list[str] = []

        refs.extend(cls.extract_doc_refs(normalized))
        for match in _LAW_TITLE_RE.finditer(normalized):
            normalized_title = cls._normalize_law_title(match.group(1), match.group(2))
            if normalized_title:
                refs.append(normalized_title)
        for match in _REG_TITLE_RE.finditer(normalized):
            year = match.group(2) if match.lastindex and match.lastindex >= 2 else None
            normalized_title = cls._normalize_reg_title(match.group(1), year)
            if normalized_title:
                refs.append(normalized_title)

        seen: set[str] = set()
        out: list[str] = []
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
        return out

    @classmethod
    def extract_exact_legal_refs(cls, query: str) -> list[str]:
        """Extract statute-style exact references suitable for sparse retrieval boosting.

        This intentionally excludes DIFC case references so case-law queries do not get
        extra sparse bias from citation duplication.
        """
        normalized = cls.normalize_query(query)
        refs: list[str] = []

        for match in _STRUCTURAL_REF_RE.finditer(normalized):
            structural_ref = cls._normalize_structural_ref(match.group(0))
            if structural_ref:
                refs.append(structural_ref)

        for ref in cls.extract_query_refs(normalized):
            if _DIFC_CASE_RE.fullmatch(ref):
                continue
            refs.append(ref)

        seen: set[str] = set()
        out: list[str] = []
        for ref in refs:
            key = ref.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(ref)
        return out

    @staticmethod
    def extract_explicit_page_reference(query: str) -> ExplicitPageReference | None:
        """Extract an explicit page anchor from the user query.

        Args:
            query: Raw user question text.

        Returns:
            A normalized explicit page reference when the query names a
            concrete page anchor, otherwise ``None``.
        """
        text = query.strip()
        if not text:
            return None

        title_match = _TITLE_PAGE_RE.search(text)
        if title_match is not None:
            return ExplicitPageReference(kind="title_page", phrase=title_match.group(0), requested_page=1)

        caption_match = _CAPTION_HEADER_RE.search(text)
        if caption_match is not None:
            return ExplicitPageReference(kind="caption_header", phrase=caption_match.group(0), requested_page=1)

        second_page_match = _SECOND_PAGE_RE.search(text)
        if second_page_match is not None:
            return ExplicitPageReference(kind="second_page", phrase=second_page_match.group(0), requested_page=2)

        ordinal_page_match = _ORDINAL_PAGE_RE.search(text)
        if ordinal_page_match is not None:
            requested_page = int(ordinal_page_match.group(1))
            if requested_page == 2:
                return ExplicitPageReference(kind="second_page", phrase=ordinal_page_match.group(0), requested_page=2)
            return ExplicitPageReference(
                kind="numeric_page",
                phrase=ordinal_page_match.group(0),
                requested_page=requested_page,
            )

        numeric_page_match = _NUMERIC_PAGE_RE.search(text)
        if numeric_page_match is not None:
            requested_page = int(numeric_page_match.group(1))
            return ExplicitPageReference(
                kind="numeric_page",
                phrase=numeric_page_match.group(0),
                requested_page=requested_page,
            )

        return None

    def classify(self, query: str) -> QueryComplexity:
        if len(query) > int(self._settings.complex_min_length):
            return QueryComplexity.COMPLEX

        query_lower = query.lower()
        keyword_hits = sum(1 for kw in self._settings.complex_keywords if kw.lower() in query_lower)
        if keyword_hits >= int(self._settings.complex_min_entities):
            return QueryComplexity.COMPLEX

        entities = _LEGAL_ENTITY_RE.findall(query)
        if len(entities) >= int(self._settings.complex_min_entities):
            return QueryComplexity.COMPLEX

        if _MULTI_PART_RE.search(query):
            return QueryComplexity.COMPLEX

        return QueryComplexity.SIMPLE

    def select_model(self, complexity: QueryComplexity) -> str:
        if complexity == QueryComplexity.COMPLEX:
            return self._settings.complex_model
        return self._settings.simple_model

    def select_max_tokens(self, complexity: QueryComplexity) -> int:
        if complexity == QueryComplexity.COMPLEX:
            return int(self._settings.complex_max_tokens)
        return int(self._settings.simple_max_tokens)
