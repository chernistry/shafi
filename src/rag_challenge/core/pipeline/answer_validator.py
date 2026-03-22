"""Post-generation answer-source consistency validator.

Checks whether the extracted answer is semantically consistent with the
source chunks that produced it. Designed to catch contradictions the
deterministic coercion pipeline cannot detect.

Integration point (for Agent 1):
    In generation_logic.py, after ``coerce_strict_type_format()`` returns
    ``(answer, extracted_ok)`` and before the strict-repair flow, call::

        result = answer_validator.validate(
            question=state["query"],
            answer=answer,
            answer_type=answer_type,
            source_chunks=[c.text for c in context_chunks],
        )
        if not result.is_valid:
            # Use result.suggested_answer if available, otherwise trigger repair
            ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Negation / affirmation signal lists
# ---------------------------------------------------------------------------

_NEGATION_SIGNALS = (
    "shall not",
    "must not",
    "may not",
    "cannot",
    "can not",
    "prohibited",
    "does not",
    "do not",
    "did not",
    "is not",
    "are not",
    "was not",
    "were not",
    "no person",
    "no entity",
    "no party",
    "not permitted",
    "not allowed",
    "not required",
    "not applicable",
    "unless",
    "except",
    "without the",
    "without prior",
)

_AFFIRMATION_SIGNALS = (
    "shall",
    "must",
    "is required",
    "is permitted",
    "is allowed",
    "is obligated",
    "are required",
    "are permitted",
    "may",
    "is entitled",
    "has the right",
    "is authorized",
    "is empowered",
)

_QUESTION_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "does", "do", "did",
    "has", "have", "had", "in", "of", "to", "for", "on", "at", "by",
    "with", "from", "and", "or", "that", "this", "it", "its", "any",
    "what", "which", "who", "when", "where", "how", "under", "per",
    "as", "be", "been", "if", "not", "no", "so", "than", "there",
    "law", "difc", "uae",
})

_YEAR_AMOUNT_KEYWORDS = {
    "year": "time",
    "years": "time",
    "period": "time",
    "term": "time",
    "duration": "time",
    "months": "time",
    "days": "time",
    "amount": "money",
    "fine": "money",
    "penalty": "money",
    "damages": "money",
    "dirhams": "money",
    "aed": "money",
    "usd": "money",
    "value": "money",
    "cost": "money",
    "fee": "money",
    "price": "money",
    "compensation": "money",
    "claim value": "money",
    "claim amount": "money",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of an answer-source consistency check."""

    is_valid: bool
    """Whether the answer is consistent with the source text."""

    confidence: float = 1.0
    """Confidence in the validation verdict (0.0-1.0)."""

    reason: str = ""
    """Human-readable explanation of the validation outcome."""

    suggested_answer: str | None = None
    """If invalid, a corrected answer when determinable."""

    signals_found: list[str] = field(default_factory=lambda: [])
    """Negation/affirmation signals found in source near question terms."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_key_terms(question: str) -> list[str]:
    """Extract meaningful terms from a question for source-text search."""
    words = re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?", question.lower())
    return [w for w in words if w not in _QUESTION_STOPWORDS and len(w) > 2]


def _find_term_windows(
    source_text: str,
    terms: list[str],
    window_size: int = 50,
) -> list[str]:
    """Find text windows around occurrences of key terms in source text.

    Returns windows of ±window_size words around each term occurrence.
    """
    source_lower = source_text.lower()
    words = source_text.split()
    word_starts: list[int] = []
    pos = 0
    for w in words:
        idx = source_lower.find(w.lower(), pos)
        word_starts.append(idx if idx >= 0 else pos)
        pos = (idx if idx >= 0 else pos) + len(w)

    windows: list[str] = []
    seen_centers: set[int] = set()

    for term in terms:
        start = 0
        while True:
            idx = source_lower.find(term, start)
            if idx == -1:
                break
            # Find the word index closest to this character position
            center_word = 0
            for wi, ws in enumerate(word_starts):
                if ws <= idx:
                    center_word = wi
                else:
                    break

            # Avoid duplicate overlapping windows
            bucket = center_word // (window_size // 2)
            if bucket not in seen_centers:
                seen_centers.add(bucket)
                lo = max(0, center_word - window_size)
                hi = min(len(words), center_word + window_size)
                windows.append(" ".join(words[lo:hi]))

            start = idx + len(term)

    return windows


def _detect_signals(
    text: str,
    signal_list: tuple[str, ...],
) -> list[str]:
    """Find which signals from the list appear in the text."""
    text_lower = text.lower()
    found: list[str] = []
    for signal in signal_list:
        if signal in text_lower:
            found.append(signal)
    return found


# ---------------------------------------------------------------------------
# Validators by type
# ---------------------------------------------------------------------------


def validate_boolean(
    question: str,
    answer: str,
    source_chunks: list[str],
) -> ValidationResult:
    """Check boolean answer consistency with source signals."""
    answer_lower = answer.strip().lower()
    is_yes = answer_lower.startswith("yes")
    is_no = answer_lower.startswith("no")

    if not is_yes and not is_no:
        return ValidationResult(is_valid=True, reason="not a clear boolean answer")

    terms = _extract_key_terms(question)
    if not terms:
        return ValidationResult(is_valid=True, reason="no key terms to search")

    source_text = "\n".join(source_chunks)
    windows = _find_term_windows(source_text, terms)

    if not windows:
        return ValidationResult(is_valid=True, confidence=0.5, reason="key terms not found in source")

    combined = " ".join(windows)
    negation_signals = _detect_signals(combined, _NEGATION_SIGNALS)
    affirmation_signals = _detect_signals(combined, _AFFIRMATION_SIGNALS)

    # Filter out ambiguous "unless"/"except" — they qualify but don't negate
    strong_negations = [s for s in negation_signals if s not in ("unless", "except")]

    if is_yes and strong_negations and not affirmation_signals:
        return ValidationResult(
            is_valid=False,
            confidence=0.7,
            reason=f"answer is Yes but source has negation signals: {strong_negations}",
            suggested_answer="No",
            signals_found=negation_signals,
        )

    if is_no and affirmation_signals and not strong_negations:
        return ValidationResult(
            is_valid=False,
            confidence=0.7,
            reason=f"answer is No but source has only affirmation signals: {affirmation_signals}",
            suggested_answer="Yes",
            signals_found=affirmation_signals,
        )

    return ValidationResult(
        is_valid=True,
        reason="signals are consistent or ambiguous",
        signals_found=negation_signals + affirmation_signals,
    )


def validate_number(
    question: str,
    answer: str,
    source_chunks: list[str],
) -> ValidationResult:
    """Check number answer context-appropriateness."""
    # Extract the number from the answer
    num_match = re.search(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if not num_match:
        return ValidationResult(is_valid=True, reason="no number in answer")

    answer_num_str = num_match.group(0).replace(",", "")

    # Determine what type of number the question is asking for
    q_lower = question.lower()
    expected_category = None
    for keyword, category in _YEAR_AMOUNT_KEYWORDS.items():
        if keyword in q_lower:
            expected_category = category
            break

    if expected_category is None:
        return ValidationResult(is_valid=True, reason="question category not determined")

    # Check if the number makes sense for the category
    try:
        num_val = float(answer_num_str)
    except ValueError:
        return ValidationResult(is_valid=True, reason="could not parse number")

    if expected_category == "time":
        # Time periods are usually small numbers (1-100 years, 1-365 days, etc.)
        if num_val >= 10000:
            return ValidationResult(
                is_valid=False,
                confidence=0.6,
                reason=f"question asks for time period but answer {num_val} looks like a monetary amount",
            )
    elif expected_category == "money" and 1900 <= num_val <= 2100 and "year" not in q_lower:
        # Could be a year mistaken for an amount
        terms = _extract_key_terms(question)
        source_text = "\n".join(source_chunks)
        windows = _find_term_windows(source_text, terms, window_size=30)
        combined = " ".join(windows).lower()
        if any(yr in combined for yr in ("enacted in", "of " + answer_num_str, "year " + answer_num_str)):
            return ValidationResult(
                is_valid=False,
                confidence=0.5,
                reason=f"answer {num_val} may be a year, not a monetary amount",
            )

    return ValidationResult(is_valid=True, reason="number passes context check")


_FULL_LEGAL_TITLE_RE = re.compile(
    r"([A-Z][A-Za-z\s,]+(?:DIFC\s+)?(?:Law|Regulations?|Rules?|Order|Act|Code|Decree)\s+No\.?\s*\d+\s+of\s+\d{4})",
)


def _try_extract_full_name(source_text: str, start_idx: int) -> str | None:
    """Try to extract a full legal title from source starting at start_idx.

    Looks for patterns like "Employment Law Amendment Law DIFC Law No. 4 of 2021".
    Returns the full title if found, otherwise None.
    """
    segment = source_text[start_idx:start_idx + 200]
    m = _FULL_LEGAL_TITLE_RE.match(segment)
    if m:
        candidate = re.sub(r"\s+", " ", m.group(1).strip()).rstrip(" .;")
        if len(candidate) <= 120:
            return candidate
    # Fall back: extract up to the first sentence-ending punctuation
    end_match = re.search(r"[.!?,;:\n(]", segment[1:])
    if end_match:
        candidate = segment[:end_match.start() + 1].strip()
        if 3 < len(candidate) <= 120:
            return candidate
    return None


def validate_name(
    question: str,
    answer: str,
    source_chunks: list[str],
) -> ValidationResult:
    """Check if extracted name actually appears in source text."""
    if not answer or answer.strip().lower() in ("null", "none"):
        return ValidationResult(is_valid=True, reason="null answer")

    name = answer.strip()
    source_text = "\n".join(source_chunks)

    # Check for obvious truncation: name ending with a preposition or article
    _TRAILING_PREPOSITIONS = ("of the", "of", "in the", "in", "for the", "for", "to the", "to", "by the", "by", "and the", "and")
    name_lower_stripped = name.lower().rstrip(" .")
    for prep in _TRAILING_PREPOSITIONS:
        if name_lower_stripped.endswith(prep):
            return ValidationResult(
                is_valid=False,
                confidence=0.8,
                reason=f"name appears truncated — ends with '{prep}'",
            )

    source_lower = source_text.lower()
    name_lower = name.lower()

    # Check ALL occurrences of the name in source. If ANY occurrence is
    # "standalone" (followed by delimiter), the name is valid.
    # If every occurrence continues into a longer title, flag truncation.
    has_standalone = False
    truncation_candidate: str | None = None
    search_start = 0
    while True:
        idx = source_lower.find(name_lower, search_start)
        if idx == -1:
            break
        end_pos = idx + len(name_lower)
        after = source_text[end_pos:end_pos + 1]
        if not after or after in ".!?,;:)\n ":
            # Name appears standalone (at end of text or before delimiter/space)
            # But check further: if followed by space then uppercase, might continue
            after_word = source_text[end_pos:end_pos + 100].strip()
            if not after_word or after_word[0] in ".!?,;:)\n" or after_word[0].islower():
                has_standalone = True
                break
        # This occurrence continues into a longer phrase — possible truncation
        if truncation_candidate is None:
            truncation_candidate = _try_extract_full_name(source_text, idx)
        search_start = end_pos

    if not has_standalone and truncation_candidate and name_lower in source_lower:
        continuation = source_text[source_lower.find(name_lower) + len(name_lower):].strip()[:40]
        return ValidationResult(
            is_valid=False,
            confidence=0.5,
            reason=f"name may be truncated; source continues: ...{continuation}",
            suggested_answer=truncation_candidate,
        )

    # Exact match (case-insensitive) — name appears standalone in source
    if name_lower in source_lower:
        return ValidationResult(is_valid=True, reason="exact match in source")

    # Fuzzy: check if all significant words of the name appear in source
    name_words = [w for w in re.findall(r"[A-Za-z]+", name) if len(w) > 2]
    if not name_words:
        return ValidationResult(is_valid=True, reason="no significant words to check")

    matched_words = sum(1 for w in name_words if w.lower() in source_lower)
    match_ratio = matched_words / len(name_words)

    if match_ratio >= 0.8:
        return ValidationResult(is_valid=True, confidence=0.9, reason=f"fuzzy match {match_ratio:.0%}")

    if match_ratio < 0.5:
        return ValidationResult(
            is_valid=False,
            confidence=0.6,
            reason=f"name poorly matched in source ({match_ratio:.0%} words found)",
        )

    return ValidationResult(is_valid=True, confidence=0.7, reason=f"partial match {match_ratio:.0%}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


class AnswerValidator:
    """Validates extracted answers against their source chunks.

    Usage::

        validator = AnswerValidator()
        result = validator.validate(
            question="Is a permit required under Law No. 5?",
            answer="Yes",
            answer_type="boolean",
            source_chunks=["... shall not issue any permit without ..."],
        )
        if not result.is_valid:
            print(result.reason, result.suggested_answer)
    """

    def validate(
        self,
        question: str,
        answer: str,
        answer_type: str,
        source_chunks: list[str],
    ) -> ValidationResult:
        """Check answer consistency with source text.

        Args:
            question: The original question.
            answer: The extracted/coerced answer string.
            answer_type: One of boolean, number, date, name, names, free_text.
            source_chunks: Text of the source chunks used for answering.

        Returns:
            ValidationResult with validity flag and optional correction.
        """
        kind = answer_type.strip().lower()

        if kind == "boolean":
            return validate_boolean(question, answer, source_chunks)
        if kind == "number":
            return validate_number(question, answer, source_chunks)
        if kind in ("name", "names"):
            return validate_name(question, answer, source_chunks)

        # date and free_text — no validation needed
        return ValidationResult(is_valid=True, reason=f"no validator for type {kind}")
