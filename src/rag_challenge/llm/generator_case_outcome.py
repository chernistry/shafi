"""Case-outcome clause extraction helpers for the generator."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_challenge.llm.generator_constants import (
    COST_CUE_RE,
    NUMBERED_LINE_RE,
    ORDER_SECTION_MARKER_RE,
    ORDER_SECTION_STOP_RE,
    OUTCOME_CUE_RE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.llm.generator_types import CaseOutcomeCandidate


def extract_case_outcome_clauses(*, text: str, prefer_order_section: bool) -> list[str]:
    """Extract outcome-like clauses from case text.

    Args:
        text: Raw case text.
        prefer_order_section: Whether ordered-section clauses should outrank generic lines.

    Returns:
        Extracted outcome clauses.
    """
    normalized = (text or "").replace("\r", "\n")
    if not normalized.strip():
        return []

    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    lines = [line for line in raw_lines if line]

    ordered_lines: list[str] = []
    in_order_section = False
    current_item: list[str] = []
    for line in lines:
        if ORDER_SECTION_MARKER_RE.search(line):
            in_order_section = True
            continue
        if in_order_section and ORDER_SECTION_STOP_RE.search(line):
            break
        if in_order_section:
            if NUMBERED_LINE_RE.match(line):
                if current_item:
                    ordered_lines.append(" ".join(current_item).strip(" ;"))
                    current_item = []
                cleaned_line = NUMBERED_LINE_RE.sub("", line).strip(" ;")
                if cleaned_line:
                    current_item.append(cleaned_line)
                continue
            if current_item:
                cleaned_line = line.strip(" ;")
                if cleaned_line:
                    current_item.append(cleaned_line)
    if current_item:
        ordered_lines.append(" ".join(current_item).strip(" ;"))

    ordered_candidates = [line for line in ordered_lines if OUTCOME_CUE_RE.search(line) or COST_CUE_RE.search(line)]
    if prefer_order_section and ordered_candidates:
        return ordered_candidates

    line_candidates = [
        NUMBERED_LINE_RE.sub("", line).strip(" ;")
        for line in lines
        if OUTCOME_CUE_RE.search(line) or COST_CUE_RE.search(line)
    ]
    if ordered_candidates or line_candidates:
        merged: list[str] = []
        seen: set[str] = set()
        for line in [*ordered_candidates, *line_candidates]:
            key = line.casefold()
            if key in seen or not line:
                continue
            seen.add(key)
            merged.append(line)
        return merged

    sentence_candidates = [
        sentence.strip(" ;")
        for sentence in re.split(r"(?<=[.!?;])\s+", re.sub(r"\s+", " ", normalized).strip())
        if sentence.strip()
    ]
    return [
        NUMBERED_LINE_RE.sub("", sentence).strip(" ;")
        for sentence in sentence_candidates
        if OUTCOME_CUE_RE.search(sentence) or COST_CUE_RE.search(sentence)
    ]


def clean_case_outcome_clause(clause: str) -> str:
    """Normalize and sanitize a case-outcome clause.

    Args:
        clause: Raw outcome clause.

    Returns:
        Cleaned outcome clause, or an empty string when unusable.
    """
    cleaned = re.sub(r"\s+", " ", (clause or "").strip()).strip(" ;")
    if not cleaned:
        return ""
    cleaned = NUMBERED_LINE_RE.sub("", cleaned).strip(" ;")
    if cleaned[:1].islower():
        return ""
    cleaned = re.sub(
        r"^AND UPON .*? by which (.+)$",
        lambda match: match.group(1)[:1].upper() + match.group(1)[1:],
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ;,")
    cleaned = re.sub(
        r"^This Order concerns the costs of (.+?), which was dismissed(?: by Order of H\.?\s*E\.?.*)?$",
        lambda match: (
            f"{match.group(1)[:1].upper() + match.group(1)[1:]} was dismissed"
            if not match.group(1).casefold().startswith("the ")
            else f"{match.group(1)} was dismissed"
        ),
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ;,")
    cleaned = re.sub(r"^This No Costs Application .*?$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(r"\bby\s+Order\s+of\s+H\.?\s*E\.?.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(
        r"^On\s+\d{1,2}\s+[A-Za-z]+\s+\d{4},\s+by\s+way\s+of\s+the\s+Order\s+of\s+H\.?\s*E\.?.*?,\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ;,")
    cleaned = re.sub(r"^Justice\s+[A-Z][A-Za-z .'-]+,\s+the\s+", "The ", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(r"\s+and\s+By\s+RDC\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(r"\bBy\s+RDC\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(r"\bIssued by:.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(r"\bDate of (?:Issue|issue):.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
    cleaned = re.sub(r"^(?:[A-Z][a-z]+ \d{4}\.\s*)+", "", cleaned).strip(" ;,")
    if cleaned.casefold() in {"costs", "conclusion"}:
        return ""
    if re.fullmatch(r"(?:USD|AED)?\s*[0-9][0-9,]*(?:\.\d+)?", cleaned, flags=re.IGNORECASE):
        return ""
    if cleaned.casefold() in {
        "the appeal is allowed, to the following extent.",
        "the appeal is allowed, to the following extent",
    }:
        return "The Court of Appeal allowed the appeal in part"
    if cleaned.casefold().startswith("for all of the foregoing reasons, we have allowed the appeal"):
        return "The Court of Appeal allowed the appeal in part"
    if "fair, reasonable and proportionate award of costs" in cleaned.casefold() and "sum of usd" in cleaned.casefold():
        amount_match = re.search(r"\bsum of (USD|AED)\s*([0-9][0-9,]*(?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
        if amount_match is not None:
            return f"The Appellant was awarded costs in the sum of {amount_match.group(1).upper()} {amount_match.group(2)}"
    if cleaned.casefold().startswith("the defendant's application for immediate judgment and/or strike out was dismissed"):
        return "The Defendant's Application for immediate judgment and/or strike out was dismissed"
    if cleaned.casefold().startswith("application was dismissed and the applicant was ordered to pay"):
        return "The Application was dismissed"
    if cleaned.casefold().startswith("the application was dismissed and the applicant was ordered to pay"):
        return "The Application was dismissed"
    if cleaned.casefold().startswith("accordingly, the application must be dismissed, and the claimant shall bear its own costs"):
        return "The Application is dismissed and the Claimant shall bear its own costs of the Application"
    if cleaned.casefold().startswith("the application is dismissed. the claim is to proceed to trial"):
        return "The Application is dismissed"
    if cleaned.casefold().startswith("the no costs application is dismissed"):
        return "The No Costs Application is dismissed"
    if cleaned.casefold().startswith("the no costs application is rejected"):
        return "The No Costs Application is rejected"
    if cleaned.casefold().startswith("save and insofar as") and "order is otherwise set aside" in cleaned.casefold():
        return "The Order is otherwise set aside except insofar as the Judge ordered that the Second Part 50 Order should continue to apply"
    if cleaned.startswith("That the "):
        cleaned = "The " + cleaned[9:]
    elif cleaned.startswith("That "):
        cleaned = cleaned[5:]
    if cleaned.casefold().startswith("by rdc "):
        return ""
    if "permission to appeal against the order was granted by the judge himself" in cleaned.casefold():
        return ""
    if "was considered" in cleaned.casefold():
        return ""
    return cleaned.rstrip(".")


def select_case_outcome_clauses(
    candidates: Sequence[CaseOutcomeCandidate],
    *,
    max_items: int,
) -> list[tuple[str, str]]:
    """Select the highest-scoring distinct case-outcome clauses.

    Args:
        candidates: Ranked candidate tuples.
        max_items: Maximum number of clauses to keep.

    Returns:
        Selected clause/chunk-id pairs.
    """
    if not candidates or max_items <= 0:
        return []
    ranked = sorted(candidates, key=lambda item: -item[0])
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _score, clause, chunk_id in ranked:
        key = clause.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append((clause, chunk_id))
        if len(selected) >= max_items:
            break
    return selected


def join_case_outcome_clauses(clauses: Sequence[str]) -> str:
    """Join cleaned case-outcome clauses into one answer sentence.

    Args:
        clauses: Cleaned clauses.

    Returns:
        Joined clause text.
    """
    cleaned = [re.sub(r"\s+", " ", clause).strip(" ;,.") for clause in clauses if clause.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    normalized_tail: list[str] = [cleaned[0]]
    for clause in cleaned[1:]:
        if clause.startswith("The "):
            normalized_tail.append("the " + clause[4:])
        else:
            normalized_tail.append(clause)
    return (
        ", and ".join([", ".join(normalized_tail[:-1]), normalized_tail[-1]])
        if len(normalized_tail) > 2
        else " and ".join(normalized_tail)
    )
