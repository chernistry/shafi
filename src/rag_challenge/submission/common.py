from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

_FULL_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_NUMBER_CANDIDATE_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
_INLINE_CITATION_RE = re.compile(r"\s*\(cite:[^)]+\)")
_LIST_ITEM_PREFIX_RE = re.compile(r"^(?:\d+\.\s*|[-*]\s*)")
_LIST_ITEM_SEPARATOR_RE = re.compile(r"(?:^|\n)\s*(?:\d+\.\s*|[-*]\s*)")
_WHITESPACE_RE = re.compile(r"\s+")
_BOOLEAN_TRUE_RE = re.compile(r"^(?:yes|true)\b", re.IGNORECASE)
_BOOLEAN_FALSE_RE = re.compile(r"^(?:no|false)\b", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SLASH_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_TEXTUAL_DAY_FIRST_DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b")
_TEXTUAL_MONTH_FIRST_DATE_RE = re.compile(r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?](?:['\")\]]+)?$")
_DANGLING_LIST_MARKER_RE = re.compile(r"(?:^|;\s*)\d+\.$")
_NON_TERMINAL_ABBREVIATION_RE = re.compile(r"\b(?:No|Art|Sec|Mr|Ms|Mrs|Dr)\.$", re.IGNORECASE)
_FREE_TEXT_LIMIT = 280

SubmissionAnswer = bool | str | int | float | list[str] | None


@dataclass(frozen=True)
class SubmissionCase:
    case_id: str
    question: str
    answer_type: str


def as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except Exception:
            return default
    return default


def chunk_id_to_page_id(chunk_id: str) -> str:
    if ":" not in chunk_id and "_" in chunk_id:
        return chunk_id
    parts = chunk_id.split(":")
    if len(parts) < 2:
        return ""
    doc_id = parts[0].strip()
    page_raw = parts[1].strip()
    if not doc_id or not page_raw.isdigit():
        return ""
    return f"{doc_id}_{int(page_raw) + 1}"


def chunk_ids_to_page_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in ids:
        page_id = chunk_id_to_page_id(str(raw).strip())
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        out.append(page_id)
    return out


def coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [text for item in items if (text := str(item).strip())]


def select_submission_used_pages(telemetry: dict[str, object]) -> list[str]:
    used_pages = coerce_str_list(telemetry.get("used_page_ids"))
    if used_pages:
        return used_pages

    cited_pages = coerce_str_list(telemetry.get("cited_page_ids"))
    if cited_pages:
        return cited_pages

    cited_chunk_ids = coerce_str_list(telemetry.get("cited_chunk_ids"))
    if cited_chunk_ids:
        return chunk_ids_to_page_ids(cited_chunk_ids)

    return []


def parse_json_number(text: str) -> int | float | None:
    cleaned = re.sub(r"(?i)\b(?:aed|usd|eur|gbp|dirhams?|dollars?)\b", "", text)
    cleaned = cleaned.replace("\u2212", "-")
    normalized = re.sub(r"[,\s]", "", cleaned).strip().strip(".")
    if _FULL_NUMBER_RE.fullmatch(normalized):
        try:
            return int(normalized)
        except ValueError:
            try:
                return float(normalized)
            except ValueError:
                return None

    for match in _NUMBER_CANDIDATE_RE.finditer(cleaned):
        start, end = match.span()
        before = cleaned[max(0, start - 16):start]
        after = cleaned[end:min(len(cleaned), end + 2)]
        if "/" in before[-1:] or "/" in after[:1]:
            continue
        if re.search(r"(?i)(?:CA|CFI|ARB|SCT|TCD|ENF|DEC)\s*$", before):
            continue

        token = re.sub(r"[,\s]", "", match.group(0))
        if not _FULL_NUMBER_RE.fullmatch(token):
            continue
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                continue
    return None


def coerce_names(answer: str) -> list[str]:
    raw_parts = [part.strip() for part in re.split(r"[,\n;]+", answer) if part.strip()]
    parts: list[str] = []
    for part in raw_parts:
        split_once = re.split(r"\s+\band\b\s+", part, maxsplit=1, flags=re.IGNORECASE)
        if len(split_once) == 2:
            parts.extend([item.strip() for item in split_once if item.strip()])
            continue
        parts.append(part)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in parts:
        normalized = item.strip().strip(".")
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def strip_inline_citations(text: str) -> str:
    return _INLINE_CITATION_RE.sub("", text)


def _normalize_submission_text(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1+", r"\1", text)
    text = re.sub(r",\s*,", ", ", text)
    text = re.sub(r";\s*;", "; ", text)
    return text.strip()


def _has_submission_free_text_fragment(text: str) -> bool:
    raw = (text or "").strip()
    if raw.endswith("..."):
        return True
    normalized = _normalize_submission_text(text)
    if not normalized:
        return True
    if _DANGLING_LIST_MARKER_RE.search(normalized):
        return True
    if normalized.endswith("No.") or normalized.endswith("No"):
        return True
    return bool(normalized.endswith(":"))


def _extract_list_like_segments(text: str) -> list[str]:
    raw = strip_inline_citations(text).replace("\r", "\n").strip()
    if not raw:
        return []

    candidates: list[str] = []
    if "\n" in raw:
        candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        split_parts = [segment.strip() for segment in _LIST_ITEM_SEPARATOR_RE.split(raw) if segment.strip()]
        if len(split_parts) > 1:
            candidates = split_parts
        elif ";" in raw:
            semicolon_parts = [segment.strip() for segment in raw.split(";") if segment.strip()]
            if len(semicolon_parts) > 1:
                candidates = semicolon_parts

    segments: list[str] = []
    for candidate in candidates:
        if re.fullmatch(r"\d+\.\s*", candidate):
            continue
        if candidate.strip().endswith("..."):
            continue
        normalized = _normalize_submission_text(_LIST_ITEM_PREFIX_RE.sub("", candidate).strip())
        if not normalized or _has_submission_free_text_fragment(normalized):
            continue
        segments.append(normalized)
    return segments


def _pack_complete_segments(segments: list[str], *, separator: str, limit: int) -> str:
    packed: list[str] = []
    for segment in segments:
        candidate = segment if not packed else f"{separator.join(packed)}{separator}{segment}"
        if len(candidate) > limit:
            break
        packed.append(segment)
    return separator.join(packed).strip()


def _trim_fragment_tail(text: str) -> str:
    candidate = _normalize_submission_text(text)
    while candidate and _has_submission_free_text_fragment(candidate):
        if "; " in candidate:
            candidate = candidate.rsplit("; ", maxsplit=1)[0].strip()
            continue
        sentences = split_submission_sentences(candidate)
        if len(sentences) > 1:
            candidate = " ".join(sentences[:-1]).strip()
            continue
        return ""
    return candidate


def truncate_at_word_boundary(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    hard_limit = max(1, limit - 3)
    cutoff = text.rfind(" ", 0, hard_limit + 1)
    if cutoff <= 0:
        return text[:hard_limit].rstrip() + "..."
    return text[:cutoff].rstrip(" ,;:.") + "..."


def normalize_date_answer(answer: str) -> str | None:
    text = _WHITESPACE_RE.sub(" ", (answer or "").strip())
    if not text:
        return None

    candidates: list[tuple[int, str, tuple[str, ...]]] = []
    for pattern, formats in (
        (_ISO_DATE_RE, ("%Y-%m-%d",)),
        (_SLASH_DATE_RE, ("%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y")),
        (_TEXTUAL_DAY_FIRST_DATE_RE, ("%d %B %Y", "%d %b %Y")),
        (_TEXTUAL_MONTH_FIRST_DATE_RE, ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        candidates.append((match.start(), match.group(0), formats))

    if not candidates:
        return None

    _, raw, formats = min(candidates, key=lambda item: item[0])
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d")
    return None


def split_submission_sentences(text: str) -> list[str]:
    normalized = _WHITESPACE_RE.sub(" ", (text or "").strip())
    if not normalized:
        return []

    raw_parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]
    if not raw_parts:
        return []

    merged_parts: list[str] = []
    current = raw_parts[0]
    for part in raw_parts[1:]:
        if _NON_TERMINAL_ABBREVIATION_RE.search(current):
            current = f"{current} {part}".strip()
            continue
        merged_parts.append(current)
        current = part
    merged_parts.append(current)

    sentences: list[str] = []
    for idx, part in enumerate(merged_parts):
        if _SENTENCE_END_RE.search(part):
            sentences.append(part)
            continue
        if idx == 0:
            sentences.append(part)
        break
    return sentences


def count_submission_sentences(text: str) -> int:
    return len(split_submission_sentences(text))


def normalize_free_text_answer(answer: str) -> str:
    text = strip_inline_citations(answer).replace("\r", "\n").strip()
    if not text:
        return ""

    segments = _extract_list_like_segments(text)
    if segments:
        packed_segments = _pack_complete_segments(segments, separator="; ", limit=_FREE_TEXT_LIMIT)
        if packed_segments:
            return packed_segments

    if "\n" in text:
        parts: list[str] = []
        for raw_line in text.splitlines():
            line = _LIST_ITEM_PREFIX_RE.sub("", raw_line.strip())
            if line:
                parts.append(line)
        if parts:
            text = "; ".join(parts)
    else:
        split_parts = [segment.strip() for segment in _LIST_ITEM_SEPARATOR_RE.split(text) if segment.strip()]
        if len(split_parts) > 1:
            text = "; ".join(_LIST_ITEM_PREFIX_RE.sub("", part).strip() for part in split_parts if part.strip())
        else:
            text = _LIST_ITEM_PREFIX_RE.sub("", text).strip()

    text = _normalize_submission_text(text)
    sentences = split_submission_sentences(text)
    if sentences:
        capped_sentences = sentences[:3]
        while capped_sentences:
            candidate = _trim_fragment_tail(" ".join(capped_sentences).strip())
            if len(candidate) <= _FREE_TEXT_LIMIT:
                return candidate
            capped_sentences = capped_sentences[:-1]

    return _trim_fragment_tail(truncate_at_word_boundary(text, limit=_FREE_TEXT_LIMIT))


def coerce_answer_type(answer: str | None, answer_type: str) -> SubmissionAnswer:
    if answer is None:
        return None
    text = answer.strip()
    if not text:
        return None

    kind = answer_type.strip().lower()
    if kind == "boolean":
        normalized = strip_inline_citations(text).strip().strip(".")
        if _BOOLEAN_TRUE_RE.match(normalized):
            return True
        if _BOOLEAN_FALSE_RE.match(normalized):
            return False
        return text
    if kind == "number":
        numeric = parse_json_number(text)
        return text if numeric is None else numeric
    if kind == "date":
        normalized_date = normalize_date_answer(text)
        return text if normalized_date is None else normalized_date
    if kind == "name":
        cleaned = strip_inline_citations(text).strip()
        cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return None
        return cleaned
    if kind == "names":
        names = coerce_names(text)
        return names if names else [text]
    if kind == "free_text":
        normalized = normalize_free_text_answer(text)
        return normalized or None
    return text


def classify_unanswerable_answer(answer: str, answer_type: str) -> tuple[bool, bool]:
    answer_type_key = answer_type.strip().lower()
    answer_normalized = answer.strip().lower()
    strict_types = {"boolean", "number", "date", "name", "names"}
    is_unanswerable_strict = answer_type_key in strict_types and answer_normalized in {"", "null", "none"}
    is_unanswerable_free_text = answer_type_key == "free_text" and (
        answer_normalized.startswith("there is no information on this question")
        or "insufficient sources retrieved" in answer_normalized
    )
    return is_unanswerable_strict, is_unanswerable_free_text


def load_cases(path: Path) -> list[SubmissionCase]:
    data_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data_obj, list):
        raise ValueError(f"Questions JSON must be a list: {path}")
    cases: list[SubmissionCase] = []
    for item_obj in cast("list[object]", data_obj):
        if not isinstance(item_obj, dict):
            continue
        item = cast("dict[str, object]", item_obj)
        case_id = str(item.get("id") or item.get("question_id") or "").strip()
        question = str(item.get("question") or "").strip()
        answer_type = str(item.get("answer_type") or "free_text").strip() or "free_text"
        if not case_id or not question:
            continue
        cases.append(SubmissionCase(case_id=case_id, question=question, answer_type=answer_type))
    if not cases:
        raise ValueError(f"No valid cases found in: {path}")
    return cases
