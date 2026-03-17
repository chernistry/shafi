"""Pure title extraction and normalization helpers for the generator."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_challenge.llm.generator_constants import (
    BODY_LIKE_TITLE_RE,
    CITED_TITLE_PLAIN_RE,
    CITED_TITLE_RE,
    CONSOLIDATED_VERSION_RE,
    COVER_TITLE_LAW_YEAR_RE,
    DOC_SUMMARY_IS_THE_RE,
    DOC_SUMMARY_PREFIX_TITLE_RE,
    DOC_SUMMARY_TITLE_RE,
    DOC_SUMMARY_TITLED_RE,
    ENACTMENT_NOTICE_ATTACHED_TITLE_RE,
    ENACTMENT_NOTICE_COMMENCEMENT_RE,
    ENACTMENT_NOTICE_TITLE_RE,
    LEGISLATION_REF_TITLE_RE,
    QUESTION_SINGLE_LAW_TITLE_PATTERNS,
    STRUCTURED_TITLE_BAD_LEAD_RE,
    TITLE_CONTEXT_BAD_LEAD_RE,
    TITLE_LAW_NO_SUFFIX_RE,
    TITLE_LEADING_CONNECTOR_RE,
    TITLE_YEAR_RE,
    TOKEN_RE,
    UPDATED_VALUE_RE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.llm.generator_types import RecoveredTitleCandidate
    from rag_challenge.models import RankedChunk


def normalize_title_key(title: str) -> str:
    """Normalize a title for exact-family matching.

    Args:
        title: Candidate title.

    Returns:
        Normalized casefolded key.
    """
    raw = re.sub(r"\s+", " ", (title or "").strip())
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered == "enactment notice":
        return ""
    normalized = TITLE_LAW_NO_SUFFIX_RE.sub("", raw)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,;.-")
    return normalized.casefold()


def normalize_common_elements_title_key(title: str) -> str:
    """Normalize a title for common-elements matching.

    Args:
        title: Candidate title.

    Returns:
        Normalized casefolded key.
    """
    raw = re.sub(r"\s+", " ", (title or "").strip())
    if not raw:
        return ""
    normalized = TITLE_LAW_NO_SUFFIX_RE.sub("", raw)
    normalized = TITLE_YEAR_RE.sub("", normalized)
    normalized = TITLE_CONTEXT_BAD_LEAD_RE.sub("", normalized)
    normalized = TITLE_LEADING_CONNECTOR_RE.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,;.-")
    return normalized.casefold()


def needs_title_recovery(title: str) -> bool:
    """Check whether a stored document title is too weak to trust.

    Args:
        title: Candidate stored title.

    Returns:
        ``True`` when title recovery should be attempted.
    """
    raw = (title or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    if lowered.startswith("in this document underlining indicates"):
        return True
    return bool(re.fullmatch(r"[_\s-]+", raw))


def should_prefer_extracted_title(raw: str, extracted: str) -> bool:
    """Check whether an extracted title is stronger than the stored title.

    Args:
        raw: Existing stored title.
        extracted: Title extracted from page text.

    Returns:
        ``True`` when the extracted title should replace the stored one.
    """
    raw_clean = (raw or "").strip()
    extracted_clean = (extracted or "").strip()
    if not extracted_clean:
        return False
    if needs_title_recovery(raw_clean):
        return True

    raw_has_year = re.search(r"\b(19|20)\d{2}\b", raw_clean) is not None
    extracted_has_year = re.search(r"\b(19|20)\d{2}\b", extracted_clean) is not None
    raw_has_law_no = "difc law no." in raw_clean.lower()
    extracted_has_law_no = "difc law no." in extracted_clean.lower()
    if (extracted_has_year and not raw_has_year) or (extracted_has_law_no and not raw_has_law_no):
        return True

    raw_letters = re.sub(r"[^A-Za-z]+", "", raw_clean)
    if raw_letters and raw_letters.isupper() and raw_clean.casefold() != extracted_clean.casefold():
        return True

    return len(extracted_clean) > len(raw_clean) + 12


def extract_doc_title_from_text(text: str) -> str:
    """Extract a legal title from raw page text.

    Args:
        text: Raw page text.

    Returns:
        Best extracted title, if any.
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    cover_match = COVER_TITLE_LAW_YEAR_RE.search(raw)
    if cover_match:
        title = re.sub(r"\s+", " ", cover_match.group(1)).strip(" ,.;:")
        is_enactment_notice = title.casefold().startswith("enactment notice")
        if is_enactment_notice:
            title = re.sub(r"^enactment\s+notice\s+", "", title, flags=re.IGNORECASE).strip(" ,.;:")
        if title and re.fullmatch(r"[A-Z][A-Z\s/&().-]+", title):
            title = title.title()
        year = cover_match.group(2).strip()
        if is_enactment_notice:
            return title
        return f"{title} {year}".strip()

    cited_match = CITED_TITLE_RE.search(raw)
    if cited_match:
        return re.sub(r"\s+", " ", cited_match.group(1)).strip()

    cited_plain_match = CITED_TITLE_PLAIN_RE.search(raw)
    if cited_plain_match:
        return re.sub(r"\s+", " ", cited_plain_match.group(1)).strip(" ,.;:")

    legislation_ref_match = LEGISLATION_REF_TITLE_RE.search(raw)
    if legislation_ref_match:
        return re.sub(r"\s+", " ", legislation_ref_match.group(1)).strip()

    enactment_notice_match = ENACTMENT_NOTICE_ATTACHED_TITLE_RE.search(raw)
    if enactment_notice_match:
        return re.sub(r"\s+", " ", enactment_notice_match.group(1)).strip()

    enactment_match = ENACTMENT_NOTICE_TITLE_RE.search(raw)
    if enactment_match:
        candidate = re.sub(r"\s+", " ", enactment_match.group(1)).strip()
        lowered = candidate.lower()
        if not lowered.startswith(("date specified", "this law", "previous law", "registrar")) and (
            "notice in respect of this law" not in lowered
            and "date of commencement of the law" not in lowered
            and "prior to the date of commencement of the law" not in lowered
            and "shall administer this law" not in lowered
            and "administer this law" not in lowered
            and "administer the provisions of this law" not in lowered
            and "provisions of this law" not in lowered
            and "relevant authority" not in lowered
            and "competent authority" not in lowered
        ):
            return candidate

    return ""


def extract_doc_title_from_summary(summary: str) -> str:
    """Extract a legal title from document summary text.

    Args:
        summary: Document summary.

    Returns:
        Best extracted summary title, if any.
    """
    raw = (summary or "").strip()
    if not raw:
        return ""

    match = DOC_SUMMARY_TITLE_RE.search(raw)
    if match:
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:")
        return re.sub(r"\s+Enactment Notice\b", "", candidate, flags=re.IGNORECASE)

    prefix_match = DOC_SUMMARY_PREFIX_TITLE_RE.search(raw)
    if prefix_match:
        return re.sub(r"\s+", " ", prefix_match.group(1)).strip(" ,.;:")

    titled_match = DOC_SUMMARY_TITLED_RE.search(raw)
    if titled_match:
        return re.sub(r"\s+", " ", titled_match.group(1)).strip(" ,.;:")

    is_the_match = DOC_SUMMARY_IS_THE_RE.search(raw)
    if is_the_match is None:
        return ""
    candidate = re.sub(r"\s+", " ", is_the_match.group(1)).strip(" ,.;:")
    return re.sub(r"\s+Enactment Notice\b", "", candidate, flags=re.IGNORECASE)


def looks_like_legal_doc_title(title: str) -> bool:
    """Check whether a candidate looks like a real legal document title.

    Args:
        title: Candidate title.

    Returns:
        ``True`` when the title looks law-like rather than body text.
    """
    normalized = re.sub(r"\s+", " ", (title or "").strip()).strip(" ,.;:")
    if not normalized:
        return False
    lowered = normalized.casefold()
    if BODY_LIKE_TITLE_RE.search(lowered):
        return False
    words = TOKEN_RE.findall(normalized)
    if len(words) > 14 and "law no." not in lowered:
        return False
    if re.fullmatch(r"[A-Z][A-Z\s/&().-]+", normalized) and len(words) <= 8:
        return True
    return any(
        marker in lowered
        for marker in (" law", " regulations", " regulation", " rules", " rule", " code", " notice", " order")
    )


def recover_doc_title_from_chunks(
    chunks: Sequence[RankedChunk],
    *,
    prefer_citation_title: bool = False,
) -> str:
    """Recover the best document title from chunk text and metadata.

    Args:
        chunks: Chunks belonging to one document.
        prefer_citation_title: Whether extracted citation titles should outrank summary/raw titles.

    Returns:
        Best recovered document title.
    """
    candidates: list[RecoveredTitleCandidate] = []
    for chunk in chunks:
        raw_title = re.sub(r"\s+", " ", (chunk.doc_title or "").strip()).strip(" ,.;:")
        extracted_title = extract_doc_title_from_text(chunk.text or "")
        summary_title = extract_doc_title_from_summary(chunk.doc_summary or "")

        if prefer_citation_title and extracted_title:
            candidates.append((extracted_title, "extracted"))
        if summary_title:
            candidates.append((summary_title, "summary"))
        if extracted_title:
            candidates.append((extracted_title, "extracted"))
        if raw_title and not needs_title_recovery(raw_title) and normalize_title_key(raw_title):
            candidates.append((raw_title, "raw"))

    deduped: list[RecoveredTitleCandidate] = []
    seen: set[str] = set()
    for candidate, source in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip(" ,.;:")
        if not normalized or needs_title_recovery(normalized):
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((normalized, source))

    if not deduped:
        return ""

    def _family_key(title: str) -> str:
        normalized = TITLE_LAW_NO_SUFFIX_RE.sub("", title)
        normalized = TITLE_YEAR_RE.sub("", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" ,;.-")
        return normalized.casefold()

    best_info_by_family: dict[str, int] = {}
    family_has_non_raw: set[str] = set()
    for title, source in deduped:
        family_key = _family_key(title)
        info_value = (2 if "difc law no." in title.casefold() else 0) + (1 if TITLE_YEAR_RE.search(title) else 0)
        if info_value > best_info_by_family.get(family_key, -1):
            best_info_by_family[family_key] = info_value
        if source != "raw":
            family_has_non_raw.add(family_key)

    def _score(item: RecoveredTitleCandidate) -> tuple[int, int, int, int]:
        title, source = item
        source_bonus = {"raw": 120, "summary": 180, "extracted": 160}.get(source, 0)
        plausible_bonus = 220 if looks_like_legal_doc_title(title) else -220
        has_year = 1 if TITLE_YEAR_RE.search(title) else 0
        has_law_no = 1 if "difc law no." in title.lower() else 0
        info_value = has_year + (has_law_no * 2)
        family_key = _family_key(title)
        info_bonus = info_value * 220
        if source == "raw" and best_info_by_family.get(family_key, 0) > info_value:
            source_bonus -= 260
        elif source == "raw" and family_key in family_has_non_raw:
            raw_letters = re.sub(r"[^A-Za-z]+", "", title)
            if raw_letters and raw_letters.isupper():
                source_bonus -= 180
        elif source in {"summary", "extracted"} and info_value and best_info_by_family.get(family_key, 0) == info_value:
            source_bonus += 80
        compactness = -min(len(title), 120)
        return (source_bonus + plausible_bonus + info_bonus, info_value, has_law_no, compactness)

    deduped.sort(key=_score, reverse=True)
    return deduped[0][0]


def display_doc_title(chunk: RankedChunk) -> str:
    """Choose the best display title for a chunk.

    Args:
        chunk: Ranked chunk.

    Returns:
        Display title for UI or context rendering.
    """
    raw = (chunk.doc_title or "").strip()
    extracted = extract_doc_title_from_text(chunk.text or "")
    if extracted and should_prefer_extracted_title(raw, extracted):
        return extracted
    return raw or extracted or "Unknown document"


def extract_commencement_rule(text: str) -> str:
    """Extract a commencement rule clause from text.

    Args:
        text: Raw page text.

    Returns:
        Normalized commencement clause, if any.
    """
    raw = re.sub(r"\s+", " ", (text or "").strip())
    if not raw:
        return ""

    match = ENACTMENT_NOTICE_COMMENCEMENT_RE.search(raw)
    if match:
        clause = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:")
        tail = raw[match.end() : match.end() + 120].lstrip()
        if tail.startswith("("):
            closing_idx = tail.find(")")
            if closing_idx != -1:
                clause = f"{clause} {tail[: closing_idx + 1].strip()}"
        clause = re.sub(r"^This Law\s+", "", clause, flags=re.IGNORECASE)
        return clause[:1].upper() + clause[1:] if clause else ""

    general_match = re.search(
        r"\b(?:(?:this\s+law)\s+)?(?P<lemma>shall\s+come|comes?)\s+into\s+force\s+on\s+(?P<tail>[^.]+)",
        raw,
        re.IGNORECASE,
    )
    if general_match is None:
        return ""

    lemma = str(general_match.group("lemma") or "").casefold()
    tail = str(general_match.group("tail") or "").strip(" ,.;:")
    if not tail:
        return ""
    prefix = "Shall come" if lemma.startswith("shall") else "Comes"
    clause = f"{prefix} into force on {tail}"
    return clause[:1].upper() + clause[1:] if clause else ""


def extract_last_updated_support(doc_chunks: Sequence[RankedChunk]) -> tuple[str, list[str]]:
    """Extract last-updated style support from doc chunks.

    Args:
        doc_chunks: Chunks from one document.

    Returns:
        Tuple of extracted value and supporting chunk ids.
    """
    for chunk in doc_chunks:
        text_sources = [
            re.sub(r"\s+", " ", (chunk.text or "").strip()),
            re.sub(r"\s+", " ", str(chunk.doc_summary or "").strip()),
        ]
        for normalized in text_sources:
            if not normalized:
                continue
            consolidated_match = CONSOLIDATED_VERSION_RE.search(normalized)
            if consolidated_match is not None:
                return consolidated_match.group(1).strip(" ,.;:"), [chunk.chunk_id]
            updated_match = UPDATED_VALUE_RE.search(normalized)
            if updated_match is not None:
                return updated_match.group(1).strip(" ,.;:"), [chunk.chunk_id]
    return "", []


def extract_single_law_title_from_question(question: str) -> str:
    """Extract a single named law title from a question template.

    Args:
        question: User question.

    Returns:
        Extracted law title, if any.
    """
    normalized = re.sub(r"\s+", " ", (question or "").strip()).strip(" ?")
    if not normalized:
        return ""
    for pattern in QUESTION_SINGLE_LAW_TITLE_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        title = re.sub(r"\s+", " ", match.group("title")).strip(" ,.;:")
        if title:
            return title
    return ""


def normalize_commencement_rule(rule: str) -> str:
    """Normalize a commencement rule for equality checks.

    Args:
        rule: Commencement rule text.

    Returns:
        Casefolded normalized commencement rule.
    """
    normalized = re.sub(r"\s+", " ", (rule or "").strip()).rstrip(".")
    normalized = re.sub(r"^this law\s+", "", normalized, flags=re.IGNORECASE)
    return normalized.casefold()


def clean_structured_doc_label(label: str) -> str:
    """Clean a structured document label extracted from model output.

    Args:
        label: Raw label text.

    Returns:
        Cleaned label.
    """
    cleaned = re.sub(r"\s+", " ", (label or "").strip()).strip(" ,.;:")
    if not cleaned:
        return ""
    cleaned = STRUCTURED_TITLE_BAD_LEAD_RE.sub("", cleaned).strip(" ,.;:")
    cleaned = re.sub(r"\s+Enactment Notice\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:")
    return re.sub(r"^The\s+The\b", "The", cleaned, flags=re.IGNORECASE)


def clean_amendment_title_historical_year(label: str) -> str:
    """Drop historical year noise from amendment-law titles.

    Args:
        label: Raw amendment-law title.

    Returns:
        Cleaned title preserving the amendment family name.
    """
    cleaned = re.sub(r"\s+", " ", (label or "").strip()).strip(" ,.;:")
    if "amendment law" not in cleaned.casefold():
        return cleaned
    return re.sub(
        r"(?i)\b(Law)(?:\s+of)?\s+\d{4}\s+(Amendment Law)\b",
        r"\1 \2",
        cleaned,
    ).strip(" ,.;:")
