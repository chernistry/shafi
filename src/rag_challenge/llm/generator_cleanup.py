"""Answer cleanup helpers for generator post-processing."""

from __future__ import annotations

import re

from rag_challenge.llm.generator_constants import (
    CITE_RE,
    COMPLETE_SENTENCE_BOUNDARY_RE,
    LIST_POSTAMBLE_RE,
    NEGATIVE_SUBCLAIM_RE,
    NUMBERED_ITEM_RE,
    SENTENCE_SPLIT_RE,
    TRAILING_NEGATIVE_RE,
)


def strip_negative_subclaims(answer: str) -> str:
    """Remove mixed positive/negative subclaims from list answers.

    Args:
        answer: Generated answer text.

    Returns:
        Cleaned answer text with trailing unsupported caveats removed when safe.
    """
    stripped = (answer or "").strip()
    if not stripped:
        return stripped
    normalized = stripped.lower().strip()
    if normalized.startswith("there is no information on this question"):
        return stripped

    has_newline = "\n" in stripped
    has_numbered = bool(re.search(r"\d+\.\s", stripped))
    has_multi_sentence = bool(re.search(r"[.!?]\s+[A-Z]", stripped))
    if not has_newline and not has_numbered and not has_multi_sentence:
        return stripped

    sentences = [segment.strip() for segment in SENTENCE_SPLIT_RE.split(stripped) if segment.strip()]
    filtered_sentences: list[str] = []
    removed_negative = False
    for sentence in sentences:
        candidate = sentence.strip()
        if not candidate:
            continue
        sentence_with_break = f"{candidate}\n"
        if NEGATIVE_SUBCLAIM_RE.search(sentence_with_break) or TRAILING_NEGATIVE_RE.search(candidate):
            removed_negative = True
            continue
        filtered_sentences.append(candidate)

    cleaned = " ".join(filtered_sentences).strip() if removed_negative else stripped
    if removed_negative and cleaned:
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(r"\(\s+cite:", "(cite:", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+\)", ")", cleaned)

    cleaned = NEGATIVE_SUBCLAIM_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        cleaned = stripped

    cleaned = TRAILING_NEGATIVE_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned or stripped


def looks_like_truncated_tail(text: str) -> bool:
    """Detect whether a trailing fragment looks abruptly truncated.

    Requires BOTH a trailing preposition AND structural signals before marking
    as truncated, to avoid corrupting valid legal text that ends with common
    prepositions (e.g. "Law No. 7 of 2019", "jointly and severally liable").

    Args:
        text: Candidate tail fragment.

    Returns:
        ``True`` when the tail strongly suggests truncation.
    """
    stripped = re.sub(r"\s+", " ", (text or "").strip())
    if not stripped:
        return False
    # Unmatched open parens are a strong truncation signal — check before
    # terminal punctuation since "Law No." inside an open paren is truncated.
    if stripped.count("(") > stripped.count(")"):
        return True
    # Properly terminated text is usually never truncated — but a dangling
    # preposition/article immediately before the terminal mark signals that
    # the sentence was cut mid-clause (e.g. "...authority in.", "...costs by.").
    if re.search(r"[.!?)]\s*$", stripped):
        last_word_match = re.search(r"\b(\w+)\s*[.!?]\s*$", stripped)
        if last_word_match:
            last_word = last_word_match.group(1).lower()
            if last_word in {
                "a", "an", "the", "and", "or", "in", "by", "to", "at", "for",
                "with", "from", "on", "under", "upon", "between", "against", "of",
            }:
                return True
        return False
    if "(cite:" in stripped.casefold() and stripped.endswith(")"):
        return False
    # "Law No." at end is only truncated if very short (likely orphaned reference
    # number rather than a complete "Law No. 7 of 2019" title).
    if re.search(r"\b(?:law\s+)?no\.\s*$", stripped, re.IGNORECASE):
        return len(stripped.split()) <= 3
    # Trailing preposition: require BOTH the preposition AND structural weakness.
    has_trailing_prep = bool(re.search(r"\b(?:of|and|the)\s*$", stripped, re.IGNORECASE))
    if not has_trailing_prep:
        return False
    word_count = len(stripped.split())
    # Short fragments ending with a preposition are likely truncated.
    if word_count <= 4:
        return True
    # Longer text is legitimate unless it lacks sentence structure entirely
    # (no period/comma anywhere) AND is still short.
    return word_count <= 6 and not re.search(r"[.,;:]", stripped)


_NUMBERED_LIST_LINE_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def cleanup_truncated_answer(answer: str) -> str:
    """Trim obviously truncated answer suffixes.

    Args:
        answer: Generated answer text.

    Returns:
        Cleaned answer text.
    """
    cleaned = (answer or "").strip()
    if not cleaned:
        return cleaned

    # Skip aggressive truncation cleanup for numbered lists — the enumerated
    # items often end with legal prepositions that look truncated but aren't.
    if _NUMBERED_LIST_LINE_RE.search(cleaned):
        # Still strip orphaned cite fragments.
        cleaned = re.sub(r"\(cite:\s*[^)]*$", "", cleaned, flags=re.IGNORECASE).rstrip()
        cleaned = re.sub(r"\(\s*[0-9a-f]{6,}$", "", cleaned, flags=re.IGNORECASE).rstrip()
        # Still drop genuinely truncated trailing bullet/numbered lines
        # (e.g. unmatched parentheses).
        lines = cleaned.splitlines()
        while lines:
            last_line = lines[-1].strip()
            if re.fullmatch(r"(?:[-*]|\d+\.)\s*", last_line):
                lines = lines[:-1]
                continue
            if re.match(r"^[-*]\s+", last_line) and looks_like_truncated_tail(last_line):
                lines = lines[:-1]
                continue
            break
        cleaned = "\n".join(lines).rstrip()
        return re.sub(r"  +", " ", cleaned).strip()

    cleaned = re.sub(r"\(cite:\s*[^)]*$", "", cleaned, flags=re.IGNORECASE).rstrip()
    cleaned = re.sub(r"\(\s*[0-9a-f]{6,}$", "", cleaned, flags=re.IGNORECASE).rstrip()

    lines = cleaned.splitlines()
    if lines:
        while lines:
            last_line = lines[-1].strip()
            if re.fullmatch(r"(?:[-*]|\d+\.)\s*", last_line):
                lines = lines[:-1]
                continue
            if re.match(r"^[-*]\s+", last_line) and looks_like_truncated_tail(last_line):
                lines = lines[:-1]
                continue
            break
        cleaned = "\n".join(lines).rstrip()

    matches = list(NUMBERED_ITEM_RE.finditer(cleaned))
    if len(matches) >= 2:
        last_start = matches[-1].start()
        last_item = cleaned[last_start:]
        last_body = re.sub(r"^\d+\.\s*", "", last_item).strip()
        if "(cite:" not in last_body.casefold() and looks_like_truncated_tail(last_body):
            cleaned = cleaned[:last_start].rstrip()

    if cleaned and not re.search(r"[.!?)]\s*$", cleaned):
        boundary_matches = list(COMPLETE_SENTENCE_BOUNDARY_RE.finditer(cleaned))
        while boundary_matches:
            last_boundary = boundary_matches[-1]
            trailing_fragment = cleaned[last_boundary.end() :].strip()
            if "(cite:" in trailing_fragment.casefold():
                break
            if trailing_fragment and not (
                TRAILING_NEGATIVE_RE.search(trailing_fragment)
                or NEGATIVE_SUBCLAIM_RE.search(f"\n{trailing_fragment}\n")
                or looks_like_truncated_tail(trailing_fragment)
            ):
                break
            cleaned = cleaned[: last_boundary.start()].rstrip()
            boundary_matches = list(COMPLETE_SENTENCE_BOUNDARY_RE.finditer(cleaned))

    return re.sub(r"  +", " ", cleaned).strip()


def cleanup_list_answer_postamble(answer: str) -> str:
    """Remove summary-like postambles after enumerated answers.

    Args:
        answer: Generated answer text.

    Returns:
        Cleaned answer text.
    """
    cleaned = (answer or "").strip()
    if not cleaned or not re.search(r"(?:^|\n)\s*1\.\s+", cleaned):
        return cleaned

    cite_matches = list(CITE_RE.finditer(cleaned))
    if not cite_matches:
        return cleaned

    trailing = cleaned[cite_matches[-1].end() :].strip()
    if not trailing or not LIST_POSTAMBLE_RE.search(trailing):
        return cleaned

    trimmed = cleaned[: cite_matches[-1].end()].rstrip(" \n.;")
    if trimmed and not re.search(r"[.!?]\s*$", trimmed):
        trimmed = f"{trimmed}."
    return trimmed.strip()


def cleanup_list_answer_preamble(answer: str) -> str:
    """Strip unsupported prose before an enumerated answer body.

    Args:
        answer: Generated answer text.

    Returns:
        Cleaned answer text.
    """
    cleaned = (answer or "").strip()
    if not cleaned:
        return cleaned

    match = re.search(r"(?:^|[\s])(?P<item>1\.\s+)", cleaned)
    if match is None:
        return cleaned
    item_start = match.start("item")
    if item_start == 0:
        return cleaned

    preamble = cleaned[:item_start].strip()
    if not preamble:
        return cleaned[item_start:].lstrip()
    if CITE_RE.search(preamble):
        return cleaned
    if preamble.lower().startswith("there is no information on this question"):
        return cleaned
    return cleaned[item_start:].lstrip()


def cleanup_final_answer(answer: str) -> str:
    """Apply the final safe cleanup pass to a generated answer.

    Args:
        answer: Generated answer text.

    Returns:
        Final cleaned answer text.
    """
    cleaned = cleanup_truncated_answer(answer)
    if not cleaned:
        return cleaned

    cleaned = cleanup_list_answer_postamble(cleaned)
    cleaned = cleanup_list_answer_preamble(cleaned)
    summary_match = re.search(r"\nSummary:\s*$|\nSummary:\s*\n", cleaned, flags=re.IGNORECASE)
    if summary_match is not None and re.search(r"(?:^|\n)1\.\s+", cleaned):
        cleaned = cleaned[: summary_match.start()].rstrip()
    cleaned = re.sub(r"(?:\n|\s)+\d+\.\s*$", "", cleaned).rstrip()
    lines = cleaned.splitlines()
    if lines:
        lines = [line for line in lines if not re.fullmatch(r"\s*(?:[-*]|\d+\.)\s*", line)]
        cleaned = "\n".join(lines).strip()
    return re.sub(r"\s{2,}", " ", cleaned).strip()

