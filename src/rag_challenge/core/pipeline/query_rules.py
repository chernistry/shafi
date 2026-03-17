# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false, reportUnusedFunction=false
from __future__ import annotations

import re

from .constants import (
    _AMENDMENT_TITLE_RE,
    _DIFC_CASE_ID_RE,
    _ENUMERATION_RE,
    _LAW_NO_REF_RE,
    _MULTI_CRITERIA_ENUM_HINTS,
    _TITLE_CONTEXT_BAD_LEAD_RE,
    _TITLE_GENERIC_QUESTION_LEAD_RE,
    _TITLE_LEADING_CONNECTOR_RE,
    _TITLE_PREPOSITION_BAD_LEAD_RE,
    _TITLE_QUERY_BAD_LEAD_RE,
    _TITLE_REF_BAD_LEAD_RE,
    _TITLE_REF_RE,
)


def _needs_long_free_text_answer(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    # Common enumeration/list patterns that often exceed small max_tokens caps.
    if any(
        phrase in q
        for phrase in (
            "which laws were amended",
            "which laws were made",
            "which laws",
            "list all",
            "identify all",
            "name all",
            "which articles",
            "which sections",
            "common elements",
            "elements in common",
            "in common",
        )
    ):
        return True
    return q.startswith("which ") and any(word in q for word in ("laws", "cases", "regulations", "articles"))

def _is_enumeration_query(query: str) -> bool:
    """Check if query is asking for enumeration across multiple entities."""
    return bool(_ENUMERATION_RE.search(query))

def _is_multi_criteria_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or not _is_enumeration_query(q):
        return False
    criteria_hits = sum(1 for hint in _MULTI_CRITERIA_ENUM_HINTS if hint in q)
    if criteria_hits >= 2:
        return True
    return " and " in q and criteria_hits >= 1

def _is_broad_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or not _is_enumeration_query(q):
        return False
    return q.startswith(("which laws", "which regulations", "which documents", "which cases", "list all", "identify all", "name all"))

def _is_registrar_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return _is_enumeration_query(q) and "administered by the registrar" in q

def _is_ruler_enactment_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return _is_enumeration_query(q) and ("made by the ruler" in q or "ruler of dubai" in q) and "enactment notice" in q

def _is_common_elements_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return "common elements" in q or "elements in common" in q or " in common" in q

def _is_common_judge_compare_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    return "judge" in q and (
        "in common" in q
        or "same judge" in q
        or "judges in common" in q
        or "judge who presided over both" in q
        or "presided over both" in q
        or ("did any judge" in q and "both" in q)
        or "judge involved in both" in q
        or "judge participated in both" in q
        or "judge who participated in both" in q
    )

def _is_case_outcome_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    if "result of the application" in q or "outcome of the specific order or application" in q:
        return True
    if "it is hereby ordered that" in q or "it is hereby ordered" in q:
        return True
    if "final ruling" in q or "court of appeal rule" in q:
        return True
    return ("outcome" in q or "result" in q) and (
        "application" in q or "appeal" in q or "order" in q
    )

def _is_case_issue_date_name_compare_query(query: str, *, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if answer_type.strip().lower() != "name":
        return False
    case_ref_count = len(_DIFC_CASE_ID_RE.findall(query or ""))
    if case_ref_count != 2:
        return False
    return (
        "date of issue" in q
        or "issue date" in q
        or "issued first" in q
        or "issued earlier" in q
        or ("issued" in q and "earlier" in q)
    )

def _is_interpretation_sections_common_elements_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    return _is_common_elements_query(query) and "interpretation section" in q

def _is_named_reference_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return _is_broad_enumeration_query(query) and any(term in q for term in ("mention", "mentions", "reference", "references"))

def _is_company_structure_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not _is_broad_enumeration_query(query):
        return False
    return (
        "company structures" in q
        or ("schedule 2" in q and "arbitration law" in q)
        or "application of the arbitration law" in q
    )

def _is_named_commencement_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(term in q for term in ("commencement", "come into force", "effective date", "enactment notice"))

def _is_named_multi_title_lookup_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or len(_extract_question_title_refs(query)) + len(_LAW_NO_REF_RE.findall(query or "")) < 2:
        return False
    if _is_broad_enumeration_query(query) or _is_common_elements_query(query):
        return False
    return any(
        term in q
        for term in (
            "commencement",
            "come into force",
            "effective date",
            "enactment",
            "administration",
            "administered",
            "citation title",
            "citation titles",
            "title of",
            "titles of",
            "last updated",
            "updated",
        )
    )

def _is_named_amendment_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q or _is_broad_enumeration_query(query):
        return False
    ref_count = len(_extract_question_title_refs(query)) + len(_LAW_NO_REF_RE.findall(query or ""))
    return ref_count >= 1 and "enact" in q and (
        "what law did it amend" in q or "what laws did it amend" in q
    )

def _is_account_effective_dates_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    return bool(q) and "pre-existing" in q and "new accounts" in q and "effective date" in q and "enact" in q

def _is_remuneration_recordkeeping_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    return "article 16(1)(c)" in q and "keep records" in q and "remuneration" in q

def _is_restriction_effectiveness_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    return (
        "article" in q
        and "restriction" in q
        and "transfer" in q
        and "actual knowledge" in q
        and ("effective" in q or "ineffective" in q)
    )

def _is_citation_title_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return "citation title" in q or "citation titles" in q

def _is_recall_sensitive_broad_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or not _is_broad_enumeration_query(q):
        return False
    if "interpretative provisions" in q:
        return True
    if "difc law no. 2 of 2022" in q and "amended by" in q:
        return True
    return "enactment notice" in q and ("made by the ruler" in q or "ruler of dubai" in q)

def _extract_question_title_refs(query: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    for match in _AMENDMENT_TITLE_RE.finditer(query or ""):
        ref = re.sub(r"\s+", " ", match.group(1).strip())
        ref = _TITLE_REF_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_QUERY_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", ref)
        ref = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_LEADING_CONNECTOR_RE.sub("", ref).strip(" ,.;:")
        if not ref:
            continue
        key = ref.casefold()
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)

    for title, year in _TITLE_REF_RE.findall(query or ""):
        normalized_title = _TITLE_REF_BAD_LEAD_RE.sub("", title.strip())
        normalized_title = _TITLE_QUERY_BAD_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_CONTEXT_BAD_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_LEADING_CONNECTOR_RE.sub("", normalized_title).strip(" ,.;:")
        parts = [normalized_title, year.strip()] if year else [normalized_title]
        ref = re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip(" ,.;:")
        if ref.casefold() in {"law", "difc law"}:
            continue
        if not ref:
            continue
        key = ref.casefold()
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    pruned: list[str] = []
    lowered_refs = [ref.casefold() for ref in refs]
    for idx, ref in enumerate(refs):
        lowered = lowered_refs[idx]
        if any(
            idx != other_idx
            and lowered != other_lowered
            and re.search(rf"\b{re.escape(lowered)}\b", other_lowered)
            for other_idx, other_lowered in enumerate(lowered_refs)
        ):
            continue
        pruned.append(ref)
    return pruned
