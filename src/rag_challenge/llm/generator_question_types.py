"""Pure question-family classifiers used by the generator."""

from __future__ import annotations

import re

from rag_challenge.llm.generator_constants import (
    BROAD_ENUMERATION_RE,
    COMMON_ELEMENTS_RE,
    CRIMINAL_TRAP_TERMS,
    IRAC_HINT_RE,
    TITLE_REF_RE,
)


def is_named_retention_period_question(question: str) -> bool:
    """Check whether the question targets named retention-period clauses.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the retention-period pattern.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not q:
        return False
    return (
        "retention period" in q
        or ("preserve" in q and "accounting records" in q)
        or ("records" in q and "six (6) years" in q)
        or ("records" in q and "minimum period" in q)
    )


def is_case_outcome_question(question: str) -> bool:
    """Check whether the question asks for a case outcome.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the case-outcome family.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not q:
        return False
    return any(
        phrase in q
        for phrase in (
            "what was the result of the application heard in case",
            "what was the outcome of the specific order or application",
            "looking only at the last page of the document",
            "according to the 'it is hereby ordered that' section",
            'according to the "it is hereby ordered that" section',
            "summarize the court's final ruling in case",
            "how did the court of appeal rule, and what costs were awarded",
        )
    )


def is_unsupported_criminal_trap(question: str) -> bool:
    """Check whether the question is a criminal-law trap outside corpus scope.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches unsupported criminal terminology.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    return any(term in q for term in CRIMINAL_TRAP_TERMS)


def is_consolidated_version_published_question(question: str) -> bool:
    """Check whether the question asks about consolidated-version publication.

    Args:
        question: User question.

    Returns:
        ``True`` when the question targets consolidated-version publication.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    return bool(q) and "consolidated version" in q and "published" in q


def is_remuneration_recordkeeping_question(question: str) -> bool:
    """Check whether the question asks about remuneration recordkeeping.

    Args:
        question: User question.

    Returns:
        ``True`` when the question targets the remuneration recordkeeping family.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    return bool(q) and "keep records" in q and "remuneration" in q and "article 16(1)(c)" in q


def is_broad_enumeration_question(question: str) -> bool:
    """Check whether the question is an open-ended enumeration request.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the broad-enumeration pattern.
    """
    return bool(BROAD_ENUMERATION_RE.search((question or "").strip()))


def is_common_elements_question(question: str) -> bool:
    """Check whether the question asks for common elements across laws.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the common-elements family.
    """
    return bool(COMMON_ELEMENTS_RE.search((question or "").strip()))


def is_registrar_enumeration_question(question: str) -> bool:
    """Check whether the question asks for laws administered by the Registrar.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the Registrar enumeration family.
    """
    q = (question or "").strip().lower()
    return bool(q) and is_broad_enumeration_question(question) and "administered by the registrar" in q


def is_named_reference_enumeration_question(question: str) -> bool:
    """Check whether the question enumerates references across named laws.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the named-reference enumeration family.
    """
    q = (question or "").strip().lower()
    return (
        bool(q)
        and is_broad_enumeration_question(question)
        and len(TITLE_REF_RE.findall(question or "")) >= 2
        and any(term in q for term in ("mention", "mentions", "reference", "references"))
    )


def is_company_structure_enumeration_question(question: str) -> bool:
    """Check whether the question asks for company-structure enumerations.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the company-structure family.
    """
    q = (question or "").strip().lower()
    return bool(q) and is_broad_enumeration_question(question) and (
        "company structures" in q
        or ("schedule 2" in q and "arbitration law" in q)
        or "application of the arbitration law" in q
    )


def is_ruler_enactment_enumeration_question(question: str) -> bool:
    """Check whether the question asks for enactment notices made by the ruler.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the ruler enactment family.
    """
    q = (question or "").strip().lower()
    return bool(q) and "enactment notice" in q and ("made by the ruler" in q or "ruler of dubai" in q)


def is_ruler_authority_year_enumeration_question(question: str) -> bool:
    """Check whether the question asks for ruler authority by year.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the ruler authority-year family.
    """
    q = (question or "").strip().lower()
    return (
        bool(q)
        and is_broad_enumeration_question(question)
        and "ruler of dubai" in q
        and ("enacted in" in q or "made in" in q)
        and bool(re.search(r"\b(19|20)\d{2}\b", q))
        and "enactment notice" not in q
    )


def requires_expanded_broad_enumeration_context(question: str) -> bool:
    """Check whether a broad enumeration needs expanded context budgeting.

    Args:
        question: User question.

    Returns:
        ``True`` when the question is known to need expanded evidence context.
    """
    q = (question or "").strip().lower()
    if not q:
        return False
    if "interpretative provisions" in q:
        return True
    if "difc law no. 2 of 2022" in q and "amended by" in q:
        return True
    return "enactment notice" in q and ("made by the ruler" in q or "ruler of dubai" in q)


def is_interpretative_provisions_enumeration_question(question: str) -> bool:
    """Check whether the question targets interpretative provisions.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the interpretative-provisions family.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    return bool(q) and is_broad_enumeration_question(question) and "interpretative provisions" in q


def is_amended_by_enumeration_question(question: str) -> bool:
    """Check whether the question asks for laws amended by another law.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the amended-by enumeration family.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not q or "amended by" not in q:
        return False
    return bool(
        is_broad_enumeration_question(question)
        or q.startswith("which specific ")
        or q.startswith("which difc laws ")
    )


def should_use_irac(question: str) -> bool:
    """Check whether the IRAC system prompt variant should be used.

    Args:
        question: User question.

    Returns:
        ``True`` when the question is interpretive enough for IRAC formatting.
    """
    q = (question or "").strip()
    if not q:
        return False
    lowered = q.lower()
    if "common elements" in lowered or "elements in common" in lowered or " in common" in lowered:
        return False
    return bool(IRAC_HINT_RE.search(q))


def is_named_liability_question(question: str) -> bool:
    """Check whether the question asks about partner liability.

    Args:
        question: User question.

    Returns:
        ``True`` when the question matches the named liability family.
    """
    q = re.sub(r"\s+", " ", (question or "").strip()).casefold()
    if not q or is_broad_enumeration_question(question):
        return False
    if "what kind of liability" in q or "what liability" in q:
        return True
    return ("liabil" in q or "liable" in q) and "partner" in q and "under article" in q
