from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    from rag_challenge.ingestion.parser import DocumentParser
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rag_challenge.ingestion.parser import DocumentParser

JsonDict = dict[str, Any]
CoveragePriors = dict[str, Any]

_LAW_NUMBER_RE = re.compile(
    r"\b(?:law|regulation|decision|order|rules?)\s+no\.?\s*[A-Za-z0-9./-]+\s*(?:of\s*\d{4})?\b",
    re.IGNORECASE,
)
_CITATION_TITLE_RE = re.compile(r"\b(?:may be cited as|short title)\b", re.IGNORECASE)
_CONTENTS_RE = re.compile(r"^\s*(?:#{1,6}\s*)?(?:table of contents|contents)\b", re.IGNORECASE | re.MULTILINE)
_ENACTMENT_RE = re.compile(
    r"\b(?:commencement|comes? into force|came into force|gazette|effective from|effective date|promulgated)\b",
    re.IGNORECASE,
)
_SCHEDULE_RE = re.compile(r"^\s*schedule\b", re.IGNORECASE | re.MULTILINE)
_ANNEX_RE = re.compile(r"^\s*annex(?:ure)?\b", re.IGNORECASE | re.MULTILINE)
_FORM_CHECKBOX_RE = re.compile(r"(?:\[\s?\]|\[[xX]\]|[\u2610-\u2612\u25A1\u25A3])")
_ORDER_OPERATIVE_RE = re.compile(
    r"\b(?:it is ordered|ordered that|the claim is dismissed|dismissed|granted|refused|allowed|"
    r"no order as to costs|application is denied)\b",
    re.IGNORECASE,
)
_REASONS_RE = re.compile(
    r"\b(?:reasons|background|discussion|analysis|the court finds|i find|we find|the court considers)\b",
    re.IGNORECASE,
)
_ARTICLE_REF_RE = re.compile(r"\b(?:article|section|chapter|part|schedule|annex)\s+[A-Za-z0-9()./-]+\b", re.IGNORECASE)
_GLOSSARY_RE = re.compile(r"\b(?:glossary|definition(?:s)?)\b", re.IGNORECASE)
_CONFIDENTIAL_RE = re.compile(r"\bconfidential\b", re.IGNORECASE)
_TRANSLATION_CAVEAT_RE = re.compile(
    r"(?:arabic text shall prevail|text in arabic shall prevail|in case of conflict.{0,80}arabic.{0,40}prevail)",
    re.IGNORECASE | re.DOTALL,
)
_TRACKED_CHANGES_RE = re.compile(
    r"\b(?:underlined|struck through|strikethrough|deleted|inserted|substituted|amended by)\b",
    re.IGNORECASE,
)
_AMENDMENT_RE = re.compile(
    r"\b(?:amendment|amended|repealed|substituted|replaced by|deleted and replaced|shall be amended as follows)\b",
    re.IGNORECASE,
)
_REGULATIONS_RE = re.compile(r"\b(?:regulations|rules?)\b", re.IGNORECASE)
_JUDGMENT_RE = re.compile(
    r"\b(?:judgment|claimant|defendant|respondent|appellant|before justice|background|discussion|analysis)\b",
    re.IGNORECASE,
)
_AMENDED_JUDGMENT_RE = re.compile(r"\b(?:amended judgment|corrected judgment|corrigendum)\b", re.IGNORECASE)
_CASE_REF_PATTERNS = (
    re.compile(r"\b(?:CFI|CA|ARB|SCT|CMC)\s*\d{1,4}/\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\b(?:Case|Claim)\s+No\.?\s*[A-Za-z0-9./-]+\b", re.IGNORECASE),
)
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
_ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}
_SMART_QUOTES = {"\u2018", "\u2019", "\u201c", "\u201d"}
_DASH_VARIANTS = {"\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"}
_ARABIC_PUNCTUATION = {"\u060c", "\u061b", "\u061f"}
_TITLE_LINEBREAK_RE = re.compile(
    r"(?:law|regulation|decision|order)\s+no\.?\s*\n+\s*[A-Za-z0-9./-]+|[A-Z][A-Za-z]+\s*\n+\s*No\.?\s*\d+",
    re.IGNORECASE,
)
_GLOSSARY_TABLE_ROW_RE = re.compile(r"^[A-Za-z0-9() /.-]{2,40}\s+\|\s+.+$")
_TABLE_ROW_HINT_RE = re.compile(r"\|.+\||\t| {3,}\S")
_ABBREVIATION_RE = re.compile(r"\b[A-Z]{2,6}\b")
_TOC_ENTRY_RE = re.compile(
    r"(?P<label>(?:article|section|chapter|part|schedule|annex)\s+[A-Za-z0-9()./-]+.*?)"
    r"(?:\.\.\.|\s{2,}|\t+)\s*(?P<page>\d+)\s*$",
    re.IGNORECASE,
)

_REASON_WEIGHTS: dict[str, int] = {
    "structural_page_family_risk": 8,
    "ocr_fallback_likely": 8,
    "image_heavy_low_text": 8,
    "doc_family_ambiguity": 8,
    "weak_early_metadata_confidence": 7,
    "version_or_date_dense_early_pages": 7,
    "repeated_headers_or_footers": 7,
    "toc_offset_detected": 7,
    "same_title_different_family_collision": 7,
    "tracked_changes_visual_semantics": 7,
    "underqueried_family_gap": 6,
    "exact_duplicate_cluster": 6,
    "enactment_notice_pairing": 6,
    "same_family_duplicate": 5,
    "weird_unicode_or_replacement": 5,
    "non_ascii_heavy": 5,
    "mixed_script_early_pages": 5,
    "long_structured_doc": 5,
    "article_dense_frontmatter": 5,
    "table_or_numeric_density": 5,
    "glossary_heavy_judgment": 5,
    "confidential_marker": 5,
    "cross_law_link_dependency": 4,
    "contents_without_internal_links": 3,
    "arabic_present": 2,
    "bilingual_fragments": 2,
    "page_numbering_anomaly": 2,
}


@dataclass(frozen=True, slots=True)
class PdfPageMetadata:
    page_num: int
    image_count: int
    internal_link_count: int
    link_destinations: list[str]
    external_urls: list[str]


@dataclass(frozen=True, slots=True)
class PageAnalysis:
    page_record: JsonDict
    fired_signals: list[str]
    dominant_arabic_category: str | None


@dataclass(frozen=True, slots=True)
class DocAnalysis:
    signals: JsonDict
    fired_signals: list[str]
    coarse_artifact_kind: str
    case_references: list[str]
    extras: JsonDict


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_arabic_char(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _ARABIC_RANGES)


def _is_arabic_digit(char: str) -> bool:
    codepoint = ord(char)
    return 0x0660 <= codepoint <= 0x0669 or 0x06F0 <= codepoint <= 0x06F9


def _is_extended_arabic_digit(char: str) -> bool:
    codepoint = ord(char)
    return 0x06F0 <= codepoint <= 0x06F9


def _is_latin_char(char: str) -> bool:
    return "LATIN" in unicodedata.name(char, "")


def _count_chars(text: str, predicate: Any) -> int:
    return sum(1 for char in text if predicate(char))


def _count_toc_entries(lines: list[str]) -> int:
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _ARTICLE_REF_RE.search(stripped) and re.search(r"(?:\.\.\.|\s)\d+\s*$", stripped):
            count += 1
    return count


def _glossary_definition_table(lines: list[str]) -> bool:
    glossary_rows = sum(1 for line in lines if _GLOSSARY_TABLE_ROW_RE.match(line.strip()))
    return glossary_rows >= 2


def _extract_case_references(text: str) -> list[str]:
    refs: set[str] = set()
    for pattern in _CASE_REF_PATTERNS:
        for match in pattern.findall(text):
            refs.add(re.sub(r"\s+", " ", match.strip()))
    return sorted(refs)


def _normalize_title_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = re.sub(r"^#+\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _extract_normalized_title(path: Path, page_records: list[JsonDict], full_text: str) -> str:
    law_match = _LAW_NUMBER_RE.search(full_text)
    if law_match is not None:
        return _normalize_title_key(law_match.group(0))

    for page_record in page_records[:3]:
        raw_text = page_record.get("_raw_text")
        if not isinstance(raw_text, str):
            continue
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped:
                return _normalize_title_key(stripped)
    return _normalize_title_key(path.stem)


def _normalized_text_hash(full_text: str) -> str:
    normalized = unicodedata.normalize("NFKC", full_text)
    collapsed = re.sub(r"\s+", "", normalized)
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def _parse_link_destination(dest: str) -> int | None:
    if not dest.startswith("page:"):
        return None
    page_num = dest.split(":", 1)[1]
    return int(page_num) if page_num.isdigit() else None


def _classify_doc_family_tags(
    *,
    normalized_text: str,
    page_count: int,
    page_signals: list[JsonDict],
    coarse_artifact_kind: str,
    external_link_targets: list[str],
    tracked_changes_detected: bool,
) -> list[str]:
    schedule_pages = sum(1 for page_signal in page_signals if bool(page_signal["schedule_signature"]))
    annex_pages = sum(1 for page_signal in page_signals if bool(page_signal["annex_signature"]))
    table_pages = sum(1 for page_signal in page_signals if bool(page_signal["table_heavy_signature"]))
    form_pages = sum(1 for page_signal in page_signals if bool(page_signal["form_checkbox_signature"]))
    glossary_pages = sum(
        1
        for page_signal in page_signals
        if bool(page_signal["glossary_like_signature"]) or bool(page_signal["glossary_definition_table"])
    )
    enactment_pages = sum(1 for page_signal in page_signals if bool(page_signal["enactment_notice_signature"]))
    reasons_pages = sum(1 for page_signal in page_signals if bool(page_signal["reasons_like_signature"]))
    order_pages = sum(1 for page_signal in page_signals if bool(page_signal["order_operatives_signature"]))
    one_page_enactment = any(bool(page_signal["one_page_enactment_notice"]) for page_signal in page_signals)
    has_articles = bool(_ARTICLE_REF_RE.search(normalized_text))
    has_law = bool(_LAW_NUMBER_RE.search(normalized_text)) or " law " in f" {normalized_text} "
    amendment_language = bool(_AMENDMENT_RE.search(normalized_text))
    judgment_language = bool(_JUDGMENT_RE.search(normalized_text))
    regulations_language = bool(_REGULATIONS_RE.search(normalized_text))
    translation_caveat = bool(_TRANSLATION_CAVEAT_RE.search(normalized_text))

    tags: list[str] = []
    if has_law and not amendment_language and not regulations_language and coarse_artifact_kind == "law_like":
        tags.append("consolidated_law")
    if has_law and amendment_language:
        tags.append("amendment_law")
    if one_page_enactment or (page_count <= 2 and enactment_pages > 0 and not has_articles):
        tags.append("enactment_notice")
    if regulations_language and "regulations" not in tags:
        tags.append("regulations")
    if judgment_language and reasons_pages > 0 and "judgment" not in tags:
        tags.append("judgment")
    if order_pages > 0 and (page_count <= 6 or coarse_artifact_kind == "order_like"):
        tags.append("order")
    if judgment_language and bool(_AMENDED_JUDGMENT_RE.search(normalized_text)):
        tags.append("amended_judgment")
    if annex_pages > 0 and schedule_pages == 0:
        tags.append("annex_copy")
    if glossary_pages > 0:
        tags.append("glossary_heavy")
    if table_pages >= max(1, math.ceil(max(page_count, 1) * 0.4)):
        tags.append("table_heavy")
    if schedule_pages + annex_pages >= max(2, math.ceil(max(page_count, 1) * 0.3)):
        tags.append("schedule_annex_heavy")
    if form_pages > 0:
        tags.append("form_like")
    if tracked_changes_detected and "amendment_law" in tags:
        tags.append("tracked_changes_amendment_law")
    if translation_caveat:
        tags.append("translation_caveat_doc")
    if one_page_enactment:
        tags.append("enactment_notice_one_page")
    if len(external_link_targets) >= 3:
        tags.append("cross_law_link_heavy")

    deduped_tags = list(dict.fromkeys(tags))
    if deduped_tags:
        return deduped_tags
    if coarse_artifact_kind == "law_like":
        return ["consolidated_law"]
    if coarse_artifact_kind == "order_like":
        return ["order"]
    if coarse_artifact_kind == "judgment_like":
        return ["judgment"]
    if table_pages > 0:
        return ["table_heavy"]
    if form_pages > 0:
        return ["form_like"]
    return ["judgment"] if judgment_language else ["consolidated_law"]


def _load_coverage_priors(path: Path | None) -> CoveragePriors:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Coverage priors must be a JSON object: {path}")
    return cast("CoveragePriors", payload)


def _enumerate_pdfs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())


def _extract_pdf_metadata(path: Path) -> tuple[int, list[PdfPageMetadata]]:
    try:
        import fitz  # pyright: ignore[reportMissingImports,reportMissingTypeStubs]

        fitz_any = cast("Any", fitz)
        metadata: list[PdfPageMetadata] = []
        with fitz_any.open(str(path)) as pdf_obj:
            pages = cast("list[object]", list(pdf_obj))
            for index, page_obj in enumerate(pages, start=1):
                page_any = cast("Any", page_obj)
                images_obj = page_any.get_images(full=True)
                image_rows = cast("list[object]", images_obj) if isinstance(images_obj, list) else []
                image_count = len(image_rows)
                links_obj = page_any.get_links()
                internal_link_count = 0
                link_destinations: list[str] = []
                external_urls: list[str] = []
                link_rows = cast("list[object]", links_obj) if isinstance(links_obj, list) else []
                if link_rows:
                    for raw_link in link_rows:
                        if not isinstance(raw_link, dict):
                            continue
                        link = cast("dict[str, object]", raw_link)
                        target_page = link.get("page")
                        if isinstance(target_page, int) and target_page >= 0:
                            internal_link_count += 1
                            link_destinations.append(f"page:{target_page + 1}")
                        uri_obj = link.get("uri")
                        if isinstance(uri_obj, str) and uri_obj.strip():
                            external_urls.append(uri_obj.strip())
                metadata.append(
                    PdfPageMetadata(
                        page_num=index,
                        image_count=image_count,
                        internal_link_count=internal_link_count,
                        link_destinations=link_destinations,
                        external_urls=external_urls,
                    )
                )
            return len(pages), metadata
    except Exception:
        return 0, []


def _signal_value_is_active(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value > 0
    if isinstance(value, float):
        return value > 0.0
    if isinstance(value, list):
        values = cast("list[object]", value)
        return len(values) > 0
    return False


def _coverage_bucket_score(bucket: str) -> int:
    if bucket == "zero-hit":
        return 6
    if bucket == "one-hit":
        return 3
    return 0


def _flatten_page_signal(record: JsonDict, signal_name: str) -> bool:
    for page_record in cast("list[JsonDict]", record.get("per_page") or []):
        signals = cast("JsonDict", page_record.get("signals") or {})
        if bool(signals.get(signal_name)):
            return True
    return False


def _count_page_signal(record: JsonDict, signal_name: str) -> int:
    count = 0
    for page_record in cast("list[JsonDict]", record.get("per_page") or []):
        signals = cast("JsonDict", page_record.get("signals") or {})
        if bool(signals.get(signal_name)):
            count += 1
    return count


def _extract_toc_pairs(page_text: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for raw_line in page_text.splitlines():
        match = _TOC_ENTRY_RE.search(raw_line.strip())
        if match is None:
            continue
        label = re.sub(r"\s+", " ", match.group("label").strip())
        page_raw = match.group("page").strip()
        if not page_raw.isdigit():
            continue
        pairs.append((label, int(page_raw)))
    return pairs


def _infer_toc_analysis(page_records: list[JsonDict], full_text: str) -> JsonDict:
    contents_page_records = [
        page_record for page_record in page_records if bool(cast("JsonDict", page_record["signals"])["contents_signature"])
    ]
    if not contents_page_records:
        return {
            "toc_offset_detected": False,
            "toc_offset_estimate": None,
            "toc_offset_confidence": None,
            "toc_pointer_type": "uncertain",
            "toc_targets_kind": "unknown",
        }

    linked_targets = sorted(
        {
            parsed_page
            for page_record in contents_page_records
            for destination in cast("list[str]", page_record.get("link_destinations") or [])
            for parsed_page in [_parse_link_destination(destination)]
            if parsed_page is not None
        }
    )
    if linked_targets:
        contents_text = "\n".join(str(page_record.get("_raw_text") or "") for page_record in contents_page_records)
        toc_pairs = _extract_toc_pairs(contents_text)
        numeric_pairs = [pair for pair in toc_pairs if pair[1] > 0]
        offset_estimate: int | None = None
        if numeric_pairs:
            offsets = [target - pair[1] for pair, target in zip(numeric_pairs, linked_targets, strict=False) if target > 0]
            if offsets:
                offset_estimate = max(set(offsets), key=offsets.count)
        return {
            "toc_offset_detected": bool(offset_estimate),
            "toc_offset_estimate": offset_estimate,
            "toc_offset_confidence": "high" if offset_estimate is not None else "medium",
            "toc_pointer_type": "linked",
            "toc_targets_kind": _infer_toc_targets_kind(contents_text),
        }

    contents_text = "\n".join(str(page_record.get("_raw_text") or "") for page_record in contents_page_records)
    toc_pairs = _extract_toc_pairs(contents_text)
    if not toc_pairs:
        return {
            "toc_offset_detected": False,
            "toc_offset_estimate": None,
            "toc_offset_confidence": None,
            "toc_pointer_type": "uncertain",
            "toc_targets_kind": _infer_toc_targets_kind(contents_text),
        }

    heading_hits: list[tuple[int, int]] = []
    lowered_pages = [
        re.sub(r"\s+", " ", str(page_record.get("_raw_text") or "")).strip().lower()
        for page_record in page_records
    ]
    for label, toc_page in toc_pairs[:8]:
        lowered_label = label.lower()
        for page_index, lowered_page in enumerate(lowered_pages, start=1):
            if lowered_label and lowered_label in lowered_page:
                heading_hits.append((toc_page, page_index))
                break

    if not heading_hits:
        return {
            "toc_offset_detected": False,
            "toc_offset_estimate": None,
            "toc_offset_confidence": "low",
            "toc_pointer_type": "uncertain",
            "toc_targets_kind": _infer_toc_targets_kind(contents_text),
        }

    offsets = [actual - listed for listed, actual in heading_hits]
    offset_estimate = max(set(offsets), key=offsets.count)
    pointer_type = "pdf_like" if offset_estimate == 0 else "internal_like"
    confidence = "high" if len(heading_hits) >= 2 else "low"
    return {
        "toc_offset_detected": offset_estimate != 0,
        "toc_offset_estimate": offset_estimate,
        "toc_offset_confidence": confidence,
        "toc_pointer_type": pointer_type,
        "toc_targets_kind": _infer_toc_targets_kind(contents_text),
    }


def _infer_toc_targets_kind(contents_text: str) -> str:
    lowered = contents_text.lower()
    if "schedule" in lowered or "annex" in lowered or "appendix" in lowered:
        return "toc_targets_schedule_or_annex"
    if "commencement" in lowered or "definitions" in lowered or "interpretation" in lowered:
        return "toc_targets_metadata_clause"
    if "article" in lowered or "chapter" in lowered or "part" in lowered:
        return "toc_targets_article_or_chapter"
    return "unknown"


def _normalized_url_key(url: str) -> str:
    lowered = url.lower()
    lowered = re.sub(r"^https?://", "", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def classify_arabic_page(text: str) -> str | None:
    arabic_chars = _count_chars(text, _is_arabic_char)
    arabic_digits = _count_chars(text, _is_arabic_digit)
    if arabic_chars == 0 and arabic_digits == 0:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    mixed_lines = sum(1 for line in lines if any(_is_arabic_char(char) for char in line) and any(_is_latin_char(char) for char in line))
    ornamental = _ornamental_arabic_only(text, lines)
    arabic_letters = sum(1 for char in text if _is_arabic_char(char) and not _is_arabic_digit(char) and char != "\u0640")

    if arabic_letters == 0 and arabic_digits > 0:
        return "arabic_digit_presence"
    if mixed_lines > 0:
        return "mixed_latin_arabic"
    if ornamental:
        return "ornamental_arabic"
    return "lexical_arabic"


def _ornamental_arabic_only(text: str, lines: list[str] | None = None) -> bool:
    effective_lines = lines if lines is not None else [line.strip() for line in text.splitlines() if line.strip()]
    arabic_lines = [line for line in effective_lines if any(_is_arabic_char(char) for char in line)]
    if not arabic_lines:
        return False

    total_arabic_chars = _count_chars(text, _is_arabic_char)
    total_latin_chars = _count_chars(text, _is_latin_char)
    short_isolated = all(_count_chars(line, _is_arabic_char) < 20 for line in arabic_lines)
    arabic_ratio = _safe_ratio(total_arabic_chars, max(len(text), 1))
    return (
        short_isolated
        and len(arabic_lines) <= 2
        and total_latin_chars > 0
        and total_latin_chars >= total_arabic_chars
        and total_arabic_chars < max(25, total_latin_chars // 3 + 1)
        and arabic_ratio < 0.25
    )


def analyze_page_signals(
    *,
    text: str,
    page_num: int,
    page_count: int,
    metadata: PdfPageMetadata,
    fallback_triggered: bool,
) -> PageAnalysis:
    raw_text = text or ""
    normalized = unicodedata.normalize("NFKC", raw_text)
    lines = raw_text.splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    char_count = len(raw_text)
    alphabetic_count = sum(1 for char in raw_text if char.isalpha())
    digit_count = sum(1 for char in raw_text if char.isdigit())
    whitespace_count = sum(1 for char in raw_text if char.isspace())
    punctuation_count = sum(1 for char in raw_text if unicodedata.category(char).startswith("P"))
    whitespace_ratio = _safe_ratio(whitespace_count, char_count)
    punctuation_ratio = _safe_ratio(punctuation_count, char_count)
    alpha_ratio = _safe_ratio(alphabetic_count, char_count)
    low_text_page = char_count < 40
    one_word_line_ratio = _safe_ratio(sum(1 for line in non_empty_lines if len(line.split()) == 1), len(non_empty_lines))
    many_one_word_lines = bool(non_empty_lines) and one_word_line_ratio > 0.4
    short_line_ratio = _safe_ratio(sum(1 for line in non_empty_lines if len(line) < 20), len(non_empty_lines))
    broken_lines_short_burst = len(non_empty_lines) >= 5 and short_line_ratio >= 0.6
    high_punctuation_low_alpha = punctuation_ratio > 0.3 and alpha_ratio < 0.4
    image_only_likelihood = min(
        1.0,
        (0.7 if metadata.image_count > 0 else 0.0) + (0.3 if char_count < 20 else 0.0) + (0.2 if char_count == 0 else 0.0),
    )
    layout_artifact_page = sum((high_punctuation_low_alpha, many_one_word_lines, low_text_page)) >= 2
    ocr_fallback_likelihood = 0.0
    if fallback_triggered:
        ocr_fallback_likelihood = 0.7
    if low_text_page:
        ocr_fallback_likelihood = max(ocr_fallback_likelihood, 0.5)
    if metadata.image_count > 0 and low_text_page:
        ocr_fallback_likelihood = max(ocr_fallback_likelihood, 0.9)

    toc_entry_count = _count_toc_entries(non_empty_lines)
    early_page = page_num <= 3
    tail_span = max(3, math.ceil(page_count * 0.2))
    tail_start = max(1, page_count - tail_span + 1)
    late_page = page_num >= tail_start
    first_line = non_empty_lines[0] if non_empty_lines else ""
    uppercase_first_line = bool(first_line) and first_line.isupper() and len(first_line) < 120
    article_reference_count = len(_ARTICLE_REF_RE.findall(normalized))

    title_page_signature = early_page and (
        bool(_LAW_NUMBER_RE.search(normalized))
        or bool(_CITATION_TITLE_RE.search(normalized))
        or uppercase_first_line
        or ("court" in normalized.lower() and "judgment" in normalized.lower())
    )
    contents_signature = page_num <= 5 and (bool(_CONTENTS_RE.search(normalized)) or toc_entry_count >= 4)
    enactment_notice_signature = bool(_ENACTMENT_RE.search(normalized))
    schedule_signature = bool(_SCHEDULE_RE.search(normalized))
    annex_signature = bool(_ANNEX_RE.search(normalized))
    table_heavy_signature = (
        sum(1 for line in non_empty_lines if _TABLE_ROW_HINT_RE.search(line)) >= max(2, len(non_empty_lines) // 3)
        if non_empty_lines
        else False
    )
    form_checkbox_signature = bool(_FORM_CHECKBOX_RE.search(raw_text))
    order_operatives_signature = late_page and bool(_ORDER_OPERATIVE_RE.search(normalized))
    law_number_or_citation_title_signature = early_page and (
        bool(_LAW_NUMBER_RE.search(normalized)) or bool(_CITATION_TITLE_RE.search(normalized))
    )
    high_article_reference_density = article_reference_count >= 6 or (contents_signature and article_reference_count >= 4)
    glossary_like_signature = bool(_GLOSSARY_RE.search(normalized)) or normalized.lower().count(" means ") >= 3
    reasons_like_signature = late_page and bool(_REASONS_RE.search(normalized))
    tracked_changes_visual_semantics = bool(_TRACKED_CHANGES_RE.search(normalized))
    translation_caveat_arabic_prevails = bool(_TRANSLATION_CAVEAT_RE.search(normalized))
    one_page_enactment_notice = page_count <= 2 and enactment_notice_signature and article_reference_count <= 1
    glossary_definition_table = _glossary_definition_table(non_empty_lines)
    contents_internal_link_density = (
        _safe_ratio(metadata.internal_link_count, toc_entry_count) if contents_signature and toc_entry_count > 0 else 0.0
    )

    non_ascii_ratio = _safe_ratio(sum(1 for char in raw_text if ord(char) > 127), char_count)
    zero_width_count = sum(raw_text.count(char) for char in _ZERO_WIDTH_CHARS)
    replacement_char_count = raw_text.count("\ufffd")
    nbsp_count = raw_text.count("\u00a0")
    smart_quote_count = sum(raw_text.count(char) for char in _SMART_QUOTES)
    dash_variant_count = sum(raw_text.count(char) for char in _DASH_VARIANTS)
    arabic_punctuation_count = sum(raw_text.count(char) for char in _ARABIC_PUNCTUATION)
    eastern_arabic_digit_count = sum(1 for char in raw_text if _is_arabic_digit(char))
    linebreak_inside_title_or_number = bool(_TITLE_LINEBREAK_RE.search(raw_text))
    tatweel_only_arabic_line_count = sum(
        1
        for line in non_empty_lines
        if any(_is_arabic_char(char) for char in line)
        and all((not _is_arabic_char(char)) or char == "\u0640" for char in line)
    )
    ornamental_arabic_only = _ornamental_arabic_only(raw_text, non_empty_lines)

    arabic_script_ratio = _safe_ratio(_count_chars(raw_text, _is_arabic_char), char_count)
    arabic_indic_digit_count = sum(1 for char in raw_text if 0x0660 <= ord(char) <= 0x0669)
    extended_arabic_indic_digit_count = sum(1 for char in raw_text if _is_extended_arabic_digit(char))
    mixed_arabic_latin_line_count = sum(
        1 for line in non_empty_lines if any(_is_arabic_char(char) for char in line) and any(_is_latin_char(char) for char in line)
    )
    dominant_arabic_category = classify_arabic_page(raw_text)
    likely_stamp_or_seal_overlay = dominant_arabic_category == "ornamental_arabic" and arabic_script_ratio < 0.15

    signals: JsonDict = {
        "title_page_signature": title_page_signature,
        "contents_signature": contents_signature,
        "enactment_notice_signature": enactment_notice_signature,
        "schedule_signature": schedule_signature,
        "annex_signature": annex_signature,
        "table_heavy_signature": table_heavy_signature,
        "form_checkbox_signature": form_checkbox_signature,
        "order_operatives_signature": order_operatives_signature,
        "law_number_or_citation_title_signature": law_number_or_citation_title_signature,
        "high_article_reference_density": high_article_reference_density,
        "glossary_like_signature": glossary_like_signature,
        "reasons_like_signature": reasons_like_signature,
        "tracked_changes_visual_semantics": tracked_changes_visual_semantics,
        "translation_caveat_arabic_prevails": translation_caveat_arabic_prevails,
        "one_page_enactment_notice": one_page_enactment_notice,
        "glossary_definition_table": glossary_definition_table,
        "contents_internal_link_density": round(contents_internal_link_density, 4),
        "alphabetic_count": alphabetic_count,
        "digit_count": digit_count,
        "whitespace_ratio": round(whitespace_ratio, 4),
        "punctuation_ratio": round(punctuation_ratio, 4),
        "image_count": metadata.image_count,
        "image_only_likelihood": round(image_only_likelihood, 4),
        "low_text_page": low_text_page,
        "broken_lines_short_burst": broken_lines_short_burst,
        "many_one_word_lines": many_one_word_lines,
        "layout_artifact_page": layout_artifact_page,
        "ocr_fallback_likelihood": round(ocr_fallback_likelihood, 4),
        "non_ascii_ratio": round(non_ascii_ratio, 4),
        "zero_width_count": zero_width_count,
        "replacement_char_count": replacement_char_count,
        "nbsp_count": nbsp_count,
        "smart_quote_count": smart_quote_count,
        "dash_variant_count": dash_variant_count,
        "arabic_punctuation_count": arabic_punctuation_count,
        "eastern_arabic_digit_count": eastern_arabic_digit_count,
        "linebreak_inside_title_or_number": linebreak_inside_title_or_number,
        "tatweel_only_arabic_line_count": tatweel_only_arabic_line_count,
        "ornamental_arabic_only": ornamental_arabic_only,
        "arabic_script_ratio": round(arabic_script_ratio, 4),
        "arabic_indic_digit_count": arabic_indic_digit_count,
        "extended_arabic_indic_digit_count": extended_arabic_indic_digit_count,
        "mixed_arabic_latin_line_count": mixed_arabic_latin_line_count,
        "likely_stamp_or_seal_overlay": likely_stamp_or_seal_overlay,
        "internal_link_count": metadata.internal_link_count,
    }

    fired_signals = [name for name, value in signals.items() if _signal_value_is_active(value)]

    page_record: JsonDict = {
        "page_num": page_num,
        "char_count": char_count,
        "page_internal_link_count": metadata.internal_link_count,
        "link_destinations": metadata.link_destinations,
        "external_urls": metadata.external_urls,
        "signals": signals,
        "fired_signals": sorted(fired_signals),
    }
    if dominant_arabic_category is not None:
        page_record["dominant_arabic_category"] = dominant_arabic_category

    return PageAnalysis(
        page_record=page_record,
        fired_signals=sorted(fired_signals),
        dominant_arabic_category=dominant_arabic_category,
    )


def analyze_doc_signals(
    *,
    path: Path,
    page_count: int,
    extraction_page_count: int,
    page_records: list[JsonDict],
    page_metadata: list[PdfPageMetadata],
    full_text: str,
) -> DocAnalysis:
    normalized = unicodedata.normalize("NFKC", full_text)
    page_signals = [cast("JsonDict", page_record["signals"]) for page_record in page_records]
    toc_analysis = _infer_toc_analysis(page_records, full_text)
    low_text_page_fraction = _safe_ratio(
        sum(1 for page_signal in page_signals if bool(page_signal["low_text_page"])),
        len(page_signals),
    )
    image_only_page_fraction = _safe_ratio(
        sum(1 for page_signal in page_signals if float(page_signal["image_only_likelihood"]) >= 0.8),
        len(page_signals),
    )

    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    repeated_header_footer_dominance = False
    for page_record in page_records:
        page_text_obj = page_record.get("_raw_text")
        if not isinstance(page_text_obj, str):
            continue
        non_empty_lines = [line.strip() for line in page_text_obj.splitlines() if line.strip()]
        if not non_empty_lines:
            continue
        if len(non_empty_lines[0]) <= 80:
            header_counter[non_empty_lines[0]] += 1
        if len(non_empty_lines[-1]) <= 80:
            footer_counter[non_empty_lines[-1]] += 1
    repeated_header_footer_dominance = (
        any(count >= 3 for count in header_counter.values()) or any(count >= 3 for count in footer_counter.values())
    )

    case_references = _extract_case_references(full_text)
    contents_page_records = [
        page_record for page_record in page_records if bool(cast("JsonDict", page_record["signals"])["contents_signature"])
    ]
    contents_link_densities = [
        float(cast("JsonDict", page_record["signals"])["contents_internal_link_density"])
        for page_record in contents_page_records
    ]
    contents_internal_link_density = (
        round(sum(contents_link_densities) / len(contents_link_densities), 4) if contents_link_densities else 0.0
    )
    internal_link_count = sum(page.internal_link_count for page in page_metadata)
    contents_link_count = sum(int(page_record["page_internal_link_count"]) for page_record in contents_page_records)
    external_link_targets = sorted(
        {
            url
            for page in page_metadata
            for url in page.external_urls
            if url.startswith("http://") or url.startswith("https://")
        }
    )
    toc_target_pages = sorted(
        {
            parsed_page
            for page_record in contents_page_records
            for dest in cast("list[str]", page_record["link_destinations"])
            for parsed_page in [_parse_link_destination(dest)]
            if parsed_page is not None
        }
    )

    confidential_marker_present = bool(_CONFIDENTIAL_RE.search(normalized))
    glossary_pages = sum(1 for page_signal in page_signals if bool(page_signal["glossary_like_signature"]))
    abbreviation_count = len(_ABBREVIATION_RE.findall(full_text))
    many_abbreviations_or_glossary_structure = glossary_pages >= 1 or abbreviation_count >= 15
    law_titles = {match.group(0).strip() for match in _LAW_NUMBER_RE.finditer(normalized)}
    multiple_law_titles_or_numbers = len(law_titles) > 1
    base_law_and_amendment_copresent = bool(law_titles) and bool(
        re.search(r"\b(?:amendment|amended by|as amended|replaced by)\b", normalized, re.IGNORECASE)
    )
    pdf_internal_link_graph_present = any(page.internal_link_count > 0 for page in page_metadata)
    page_number_mismatch = page_count != extraction_page_count
    tracked_changes_page_count = sum(
        1 for page_signal in page_signals if bool(page_signal["tracked_changes_visual_semantics"])
    )
    tracked_changes_detected = tracked_changes_page_count > 0
    tracked_changes_confidence: str | None
    if tracked_changes_page_count >= 3:
        tracked_changes_confidence = "high"
    elif tracked_changes_page_count > 0:
        tracked_changes_confidence = "medium"
    else:
        tracked_changes_confidence = None
    risk_note = (
        "Amendment law with visual-diff semantics. Extracted text does not distinguish insertions from deletions. "
        "LLM answers about specific amendments may be unreliable."
        if tracked_changes_detected
        else None
    )

    per_page_categories = [
        cast("str", page_record["dominant_arabic_category"])
        for page_record in page_records
        if isinstance(page_record.get("dominant_arabic_category"), str)
    ]
    arabic_categories_present = sorted(set(per_page_categories))
    arabic_on_page_1 = any(
        int(page_record["page_num"]) == 1 and isinstance(page_record.get("dominant_arabic_category"), str)
        for page_record in page_records
    )
    arabic_on_page_2 = any(
        int(page_record["page_num"]) == 2 and isinstance(page_record.get("dominant_arabic_category"), str)
        for page_record in page_records
    )

    coarse_artifact_kind = "unknown"
    if any(bool(page_signal["law_number_or_citation_title_signature"]) for page_signal in page_signals):
        coarse_artifact_kind = "law_like"
    elif any(bool(page_signal["order_operatives_signature"]) for page_signal in page_signals):
        coarse_artifact_kind = "order_like"
    elif re.search(r"\b(?:judgment|claimant|defendant|respondent|appellant)\b", normalized, re.IGNORECASE):
        coarse_artifact_kind = "judgment_like"

    normalized_title = _extract_normalized_title(path, page_records, full_text)
    doc_family_tags = _classify_doc_family_tags(
        normalized_text=normalized,
        page_count=page_count,
        page_signals=page_signals,
        coarse_artifact_kind=coarse_artifact_kind,
        external_link_targets=external_link_targets,
        tracked_changes_detected=tracked_changes_detected,
    )

    signals: JsonDict = {
        "page_number_mismatch": page_number_mismatch,
        "low_text_page_fraction": round(low_text_page_fraction, 4),
        "image_only_page_fraction": round(image_only_page_fraction, 4),
        "repeated_header_footer_dominance": repeated_header_footer_dominance,
        "confidential_marker_present": confidential_marker_present,
        "many_abbreviations_or_glossary_structure": many_abbreviations_or_glossary_structure,
        "multiple_law_titles_or_numbers": multiple_law_titles_or_numbers,
        "base_law_and_amendment_copresent": base_law_and_amendment_copresent,
        "pdf_internal_link_graph_present": pdf_internal_link_graph_present,
        "contents_internal_link_density": contents_internal_link_density,
        "same_case_multi_artifact_hint": False,
        "non_ascii_ratio": round(_safe_ratio(sum(1 for char in full_text if ord(char) > 127), len(full_text)), 4),
        "zero_width_count": sum(full_text.count(char) for char in _ZERO_WIDTH_CHARS),
        "replacement_char_count": full_text.count("\ufffd"),
        "nbsp_count": full_text.count("\u00a0"),
        "smart_quote_count": sum(full_text.count(char) for char in _SMART_QUOTES),
        "dash_variant_count": sum(full_text.count(char) for char in _DASH_VARIANTS),
        "arabic_punctuation_count": sum(full_text.count(char) for char in _ARABIC_PUNCTUATION),
        "eastern_arabic_digit_count": sum(1 for char in full_text if _is_arabic_digit(char)),
        "linebreak_inside_title_or_number": bool(_TITLE_LINEBREAK_RE.search(full_text)),
        "tatweel_only_arabic_line_count": sum(
            1
            for line in full_text.splitlines()
            if line.strip()
            and any(_is_arabic_char(char) for char in line)
            and all((not _is_arabic_char(char)) or char == "\u0640" for char in line)
        ),
        "ornamental_arabic_only": _ornamental_arabic_only(full_text),
        "arabic_script_ratio": round(_safe_ratio(_count_chars(full_text, _is_arabic_char), len(full_text)), 4),
        "arabic_indic_digit_count": sum(1 for char in full_text if 0x0660 <= ord(char) <= 0x0669),
        "extended_arabic_indic_digit_count": sum(1 for char in full_text if _is_extended_arabic_digit(char)),
        "mixed_arabic_latin_line_count": sum(
            1
            for line in full_text.splitlines()
            if any(_is_arabic_char(char) for char in line) and any(_is_latin_char(char) for char in line)
        ),
        "arabic_on_page_1": arabic_on_page_1,
        "arabic_on_page_2": arabic_on_page_2,
        "likely_stamp_or_seal_overlay": any(
            bool(cast("JsonDict", page_record["signals"])["likely_stamp_or_seal_overlay"]) for page_record in page_records
        ),
        "arabic_categories_present": arabic_categories_present,
        "case_references": case_references,
        "coarse_artifact_kind": coarse_artifact_kind,
        "tracked_changes_detected": tracked_changes_detected,
        "tracked_changes_page_count": tracked_changes_page_count,
    }

    fired_signals = [name for name, value in signals.items() if _signal_value_is_active(value)]

    return DocAnalysis(
        signals=signals,
        fired_signals=sorted(fired_signals),
        coarse_artifact_kind=coarse_artifact_kind,
        case_references=case_references,
        extras={
            "doc_family_tags": doc_family_tags,
            "normalized_title": normalized_title,
            "internal_link_count": internal_link_count,
            "contents_link_count": contents_link_count,
            "external_link_targets": external_link_targets,
            "toc_target_pages": toc_target_pages,
            "toc_offset_detected": bool(toc_analysis["toc_offset_detected"]),
            "toc_offset_estimate": toc_analysis["toc_offset_estimate"],
            "toc_offset_confidence": toc_analysis["toc_offset_confidence"],
            "toc_pointer_type": toc_analysis["toc_pointer_type"],
            "toc_targets_kind": toc_analysis["toc_targets_kind"],
            "collision_doc_ids": [],
            "cross_law_link_doc_ids": [],
            "exact_duplicate_cluster_id": None,
            "exact_duplicate_cluster_members": [],
            "duplicate_same_family_doc_ids": [],
            "same_case_family_doc_ids": [],
            "tracked_changes_detected": tracked_changes_detected,
            "tracked_changes_page_count": tracked_changes_page_count,
            "tracked_changes_confidence": tracked_changes_confidence,
            "risk_note": risk_note,
            "family_query_coverage_bucket": "unscored",
            "public_coverage_gap_score": 0,
        },
    )


def _apply_same_case_multi_artifact_hint(records: list[JsonDict]) -> None:
    refs_to_kinds: dict[str, set[str]] = defaultdict(set)
    refs_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        signals = cast("JsonDict", record["signals"])
        case_refs = cast("list[str]", signals.get("case_references") or [])
        coarse_kind = str(signals.get("coarse_artifact_kind") or "unknown")
        for case_ref in case_refs:
            refs_to_kinds[case_ref].add(coarse_kind)
            refs_to_indices[case_ref].append(index)

    for case_ref, kinds in refs_to_kinds.items():
        concrete_kinds = {kind for kind in kinds if kind != "unknown"}
        if len(concrete_kinds) < 2:
            continue
        for index in refs_to_indices[case_ref]:
            record = records[index]
            signals = cast("JsonDict", record["signals"])
            signals["same_case_multi_artifact_hint"] = True
            sibling_hashes = sorted(
                records[sibling_index]["sha256"]
                for sibling_index in refs_to_indices[case_ref]
                if sibling_index != index
            )
            record["same_case_family_doc_ids"] = sibling_hashes
            family_tags = cast("list[str]", record["doc_family_tags"])
            if (
                any(tag in family_tags for tag in ("order", "judgment", "amended_judgment"))
                and "same_case_order_or_judgment_candidate" not in family_tags
            ):
                family_tags.append("same_case_order_or_judgment_candidate")
            fired_signals = cast("list[str]", record["_fired_signals"])
            if "same_case_multi_artifact_hint" not in fired_signals:
                fired_signals.append("same_case_multi_artifact_hint")


def _apply_same_title_same_family_candidates(records: list[JsonDict]) -> None:
    title_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        normalized_title = str(record.get("normalized_title") or "").strip()
        if normalized_title:
            title_groups[normalized_title].append(index)

    for indices in title_groups.values():
        if len(indices) < 2:
            continue
        family_intersection = set(cast("list[str]", records[indices[0]]["doc_family_tags"]))
        for index in indices[1:]:
            family_intersection &= set(cast("list[str]", records[index]["doc_family_tags"]))
        if not family_intersection:
            continue
        for index in indices:
            record = records[index]
            sibling_hashes = sorted(records[sibling_index]["sha256"] for sibling_index in indices if sibling_index != index)
            record["duplicate_same_family_doc_ids"] = sibling_hashes
            family_tags = cast("list[str]", record["doc_family_tags"])
            if "duplicate_same_family_candidate" not in family_tags:
                family_tags.append("duplicate_same_family_candidate")


def _apply_exact_duplicate_clusters(records: list[JsonDict]) -> None:
    cluster_members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        normalized_hash = str(record.get("_normalized_text_hash") or "").strip()
        if normalized_hash:
            cluster_members[normalized_hash].append(index)

    cluster_counter = 1
    for indices in cluster_members.values():
        if len(indices) < 2:
            continue
        cluster_id = f"cluster_{cluster_counter:03d}"
        member_hashes = sorted(records[index]["sha256"] for index in indices)
        cluster_counter += 1
        for index in indices:
            record = records[index]
            record["exact_duplicate_cluster_id"] = cluster_id
            record["exact_duplicate_cluster_members"] = member_hashes


def _apply_collision_and_link_hints(records: list[JsonDict]) -> None:
    title_groups: dict[str, list[int]] = defaultdict(list)
    law_number_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        normalized_title = str(record.get("normalized_title") or "").strip()
        if normalized_title:
            title_groups[normalized_title].append(index)
        signals = cast("JsonDict", record.get("signals") or {})
        law_refs = cast("list[str]", signals.get("case_references") or [])
        for law_ref in law_refs:
            law_number_groups[law_ref].append(index)

    for indices in title_groups.values():
        if len(indices) < 2:
            continue
        family_sets = [set(cast("list[str]", records[index].get("doc_family_tags") or [])) for index in indices]
        has_family_divergence = len({tuple(sorted(family_set)) for family_set in family_sets}) > 1
        enactment_pairing = any("enactment_notice" in family_set for family_set in family_sets) and any(
            "consolidated_law" in family_set or "amendment_law" in family_set for family_set in family_sets
        )
        for index in indices:
            record = records[index]
            sibling_hashes = sorted(records[sibling_index]["sha256"] for sibling_index in indices if sibling_index != index)
            if has_family_divergence:
                record["collision_doc_ids"] = sibling_hashes
            if enactment_pairing:
                family_tags = cast("list[str]", record["doc_family_tags"])
                if "enactment_notice_pair_candidate" not in family_tags:
                    family_tags.append("enactment_notice_pair_candidate")

    title_keys = {
        str(record["sha256"]): str(record.get("normalized_title") or "")
        for record in records
        if str(record.get("normalized_title") or "")
    }
    for record in records:
        url_matches: set[str] = set()
        normalized_urls = [_normalized_url_key(url) for url in cast("list[str]", record.get("external_link_targets") or [])]
        if not normalized_urls:
            continue
        for other in records:
            if other["sha256"] == record["sha256"]:
                continue
            title_key = title_keys.get(str(other["sha256"]), "")
            if not title_key:
                continue
            title_tokens = [token for token in title_key.split() if len(token) >= 3]
            if not title_tokens:
                continue
            if any(all(token in url for token in title_tokens[: min(3, len(title_tokens))]) for url in normalized_urls):
                url_matches.add(str(other["sha256"]))
        record["cross_law_link_doc_ids"] = sorted(url_matches)


def _apply_coverage_priors(records: list[JsonDict], coverage_priors: CoveragePriors) -> None:
    family_buckets_obj = coverage_priors.get("family_buckets")
    family_scores_obj = coverage_priors.get("family_scores")
    family_buckets = cast("dict[str, str]", family_buckets_obj) if isinstance(family_buckets_obj, dict) else {}
    family_scores = cast("dict[str, int]", family_scores_obj) if isinstance(family_scores_obj, dict) else {}

    for record in records:
        doc_family_tags = cast("list[str]", record.get("doc_family_tags") or [])
        bucket = "unscored"
        score = 0
        for tag in doc_family_tags:
            candidate_bucket = str(family_buckets.get(tag) or "")
            if candidate_bucket == "zero-hit":
                bucket = candidate_bucket
                score = max(score, int(family_scores.get(tag) or _coverage_bucket_score(candidate_bucket)))
                break
            if candidate_bucket == "one-hit" and bucket in {"unscored", "exercised"}:
                bucket = candidate_bucket
                score = max(score, int(family_scores.get(tag) or _coverage_bucket_score(candidate_bucket)))
            if candidate_bucket == "exercised" and bucket == "unscored":
                bucket = candidate_bucket
        record["family_query_coverage_bucket"] = bucket
        record["public_coverage_gap_score"] = score


def _score_record(record: JsonDict) -> tuple[int, list[str]]:
    reasons: list[str] = []
    signals = cast("JsonDict", record.get("signals") or {})
    doc_family_tags = set(cast("list[str]", record.get("doc_family_tags") or []))
    if (
        _flatten_page_signal(record, "title_page_signature")
        or _flatten_page_signal(record, "contents_signature")
        or _flatten_page_signal(record, "schedule_signature")
        or _flatten_page_signal(record, "annex_signature")
        or _flatten_page_signal(record, "table_heavy_signature")
        or _flatten_page_signal(record, "form_checkbox_signature")
    ):
        reasons.append("structural_page_family_risk")
    if bool(record.get("fallback_triggered")) or float(max(
        (cast("JsonDict", page["signals"]).get("ocr_fallback_likelihood") or 0.0)
        for page in cast("list[JsonDict]", record.get("per_page") or [{"signals": {}}])
    )) >= 0.7:
        reasons.append("ocr_fallback_likely")
    if float(record.get("image_only_page_fraction") or signals.get("image_only_page_fraction") or 0.0) >= 0.2:
        reasons.append("image_heavy_low_text")
    if bool(signals.get("base_law_and_amendment_copresent")):
        reasons.append("doc_family_ambiguity")
    early_metadata = sum(
        1
        for page_record in cast("list[JsonDict]", record.get("per_page") or [])[:3]
        if bool(cast("JsonDict", page_record["signals"]).get("title_page_signature"))
        or bool(cast("JsonDict", page_record["signals"]).get("contents_signature"))
    )
    early_law_confident = any(
        bool(cast("JsonDict", page_record["signals"]).get("law_number_or_citation_title_signature"))
        for page_record in cast("list[JsonDict]", record.get("per_page") or [])[:3]
    )
    if early_metadata >= 2 and not early_law_confident:
        reasons.append("weak_early_metadata_confidence")
    if bool(signals.get("repeated_header_footer_dominance")):
        reasons.append("repeated_headers_or_footers")
    if bool(record.get("toc_offset_detected")):
        reasons.append("toc_offset_detected")
    if cast("list[str]", record.get("collision_doc_ids") or []):
        reasons.append("same_title_different_family_collision")
    if "enactment_notice_pair_candidate" in doc_family_tags:
        reasons.append("enactment_notice_pairing")
    if cast("list[str]", record.get("duplicate_same_family_doc_ids") or []):
        reasons.append("same_family_duplicate")
    if bool(record.get("tracked_changes_detected")):
        reasons.append("tracked_changes_visual_semantics")
    if int(record.get("public_coverage_gap_score") or 0) > 0:
        reasons.append("underqueried_family_gap")
    if record.get("exact_duplicate_cluster_id"):
        reasons.append("exact_duplicate_cluster")
    if cast("list[str]", record.get("cross_law_link_doc_ids") or []):
        reasons.append("cross_law_link_dependency")
    if _flatten_page_signal(record, "contents_signature") and float(signals.get("contents_internal_link_density") or 0.0) == 0.0:
        reasons.append("contents_without_internal_links")
    if any(int(signals.get(name) or 0) > 0 for name in ("zero_width_count", "replacement_char_count", "nbsp_count", "smart_quote_count", "dash_variant_count")):
        reasons.append("weird_unicode_or_replacement")
    if float(signals.get("non_ascii_ratio") or 0.0) >= 0.05:
        reasons.append("non_ascii_heavy")
    if bool(signals.get("arabic_on_page_1")) or bool(signals.get("arabic_on_page_2")):
        reasons.append("mixed_script_early_pages")
    if int(record.get("page_count") or 0) >= 20 and "structural_page_family_risk" in reasons:
        reasons.append("long_structured_doc")
    if _count_page_signal(record, "high_article_reference_density") >= 1:
        reasons.append("article_dense_frontmatter")
    if "table_heavy" in doc_family_tags or _flatten_page_signal(record, "table_heavy_signature"):
        reasons.append("table_or_numeric_density")
    if "glossary_heavy" in doc_family_tags and "judgment" in doc_family_tags:
        reasons.append("glossary_heavy_judgment")
    if bool(signals.get("confidential_marker_present")):
        reasons.append("confidential_marker")
    if float(signals.get("arabic_script_ratio") or 0.0) > 0.0:
        reasons.append("arabic_present")
    if int(signals.get("mixed_arabic_latin_line_count") or 0) > 0:
        reasons.append("bilingual_fragments")
    if bool(signals.get("linebreak_inside_title_or_number")) or str(record.get("toc_pointer_type") or "") == "uncertain":
        reasons.append("page_numbering_anomaly")
    unique_reasons = sorted(set(reasons), key=lambda item: (-_REASON_WEIGHTS[item], item))
    suspicion_score = sum(_REASON_WEIGHTS[reason] for reason in unique_reasons)
    return suspicion_score, unique_reasons


def scan_pdf_corpus(
    *,
    input_dir: Path,
    mode: str,
    coverage_priors: CoveragePriors,
) -> list[JsonDict]:
    parser = DocumentParser()
    records: list[JsonDict] = []
    for pdf_path in _enumerate_pdfs(input_dir):
        page_count, page_metadata = _extract_pdf_metadata(pdf_path)
        extraction = parser.extract_pdf_pages_for_scan(pdf_path)
        per_page: list[JsonDict] = []
        for index, page_text in enumerate(extraction.pages, start=1):
            metadata = (
                page_metadata[index - 1]
                if index - 1 < len(page_metadata)
                else PdfPageMetadata(
                    page_num=index,
                    image_count=0,
                    internal_link_count=0,
                    link_destinations=[],
                    external_urls=[],
                )
            )
            page_analysis = analyze_page_signals(
                text=page_text,
                page_num=index,
                page_count=max(page_count, len(extraction.pages)),
                metadata=metadata,
                fallback_triggered=extraction.fallback_triggered,
            )
            page_record = dict(page_analysis.page_record)
            page_record["_raw_text"] = page_text
            per_page.append(page_record)

        full_text = "\n\n".join(page for page in extraction.pages if page.strip()).strip()
        doc_analysis = analyze_doc_signals(
            path=pdf_path,
            page_count=page_count,
            extraction_page_count=len(extraction.pages),
            page_records=per_page,
            page_metadata=page_metadata,
            full_text=full_text,
        )

        fired_signals = sorted(doc_analysis.fired_signals + [name for page in per_page for name in page["fired_signals"]])
        record: JsonDict = {
            "doc_id": pdf_path.stem,
            "filename": pdf_path.name,
            "sha256": _file_sha256(pdf_path),
            "mode": mode,
            "page_count": page_count,
            "extracted_page_count": len(extraction.pages),
            "page_number_mismatch": page_count != len(extraction.pages),
            "parser_mode": extraction.parser_mode,
            "fallback_triggered": extraction.fallback_triggered,
            "signal_count": 0,
            "signals": doc_analysis.signals,
            "per_page": per_page,
            "coverage_priors": coverage_priors,
            "suspicion_score": 0,
            "reason_tags": [],
            **doc_analysis.extras,
            "_fired_signals": fired_signals,
            "_normalized_text_hash": _normalized_text_hash(full_text),
        }
        records.append(record)

    _apply_exact_duplicate_clusters(records)
    _apply_same_case_multi_artifact_hint(records)
    _apply_same_title_same_family_candidates(records)
    _apply_collision_and_link_hints(records)
    _apply_coverage_priors(records, coverage_priors)

    for record in records:
        suspicion_score, reason_tags = _score_record(record)
        record["suspicion_score"] = suspicion_score
        record["reason_tags"] = reason_tags
        record["signal_count"] = len(cast("list[str]", record["_fired_signals"]))
        cleaned_pages: list[JsonDict] = []
        for page_record in cast("list[JsonDict]", record["per_page"]):
            page_copy = dict(page_record)
            page_copy.pop("_raw_text", None)
            cleaned_pages.append(page_copy)
        record["per_page"] = cleaned_pages
        record.pop("_fired_signals", None)
        record.pop("_normalized_text_hash", None)
    records.sort(
        key=lambda record: (
            -int(record["suspicion_score"]),
            -int(record["signal_count"]),
            str(record["filename"]),
        )
    )
    return records


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_text_dumps(output_dir: Path, records: list[JsonDict], parser: DocumentParser, input_dir: Path) -> None:
    for record in records:
        pdf_path = input_dir / str(record["filename"])
        if not pdf_path.exists():
            continue
        extraction = parser.extract_pdf_pages_for_scan(pdf_path)
        doc_dir = output_dir / str(record["doc_id"])
        doc_dir.mkdir(parents=True, exist_ok=True)
        for page_num, page_text in enumerate(extraction.pages, start=1):
            (doc_dir / f"page_{page_num}.txt").write_text(page_text, encoding="utf-8")


def _write_screenshots(output_dir: Path, records: list[JsonDict], input_dir: Path) -> None:
    try:
        import fitz  # pyright: ignore[reportMissingImports,reportMissingTypeStubs]

        fitz_any = cast("Any", fitz)
    except Exception:
        return

    for record in records:
        pdf_path = input_dir / str(record["filename"])
        if not pdf_path.exists():
            continue
        suspicious_pages = [
            int(page_record["page_num"])
            for page_record in cast("list[JsonDict]", record.get("per_page") or [])
            if cast("list[str]", page_record.get("fired_signals") or [])
        ]
        if not suspicious_pages:
            continue
        doc_dir = output_dir / str(record["doc_id"])
        doc_dir.mkdir(parents=True, exist_ok=True)
        with fitz_any.open(str(pdf_path)) as pdf_obj:
            for page_num in suspicious_pages:
                if page_num < 1 or page_num > len(pdf_obj):
                    continue
                pixmap = pdf_obj[page_num - 1].get_pixmap()
                pixmap.save(str(doc_dir / f"page_{page_num}.png"))


def build_summary_markdown(records: list[JsonDict]) -> str:
    docs_scanned = len(records)
    pages_scanned = sum(int(record["page_count"]) for record in records)
    docs_with_signals = sum(1 for record in records if int(record["signal_count"]) > 0)
    docs_with_parser_fallback = sum(1 for record in records if bool(record["fallback_triggered"]))

    signal_counter: Counter[str] = Counter()
    for record in records:
        signals = cast("JsonDict", record["signals"])
        for name, value in signals.items():
            if _signal_value_is_active(value):
                signal_counter[name] += 1
        for page_record in cast("list[JsonDict]", record["per_page"]):
            for signal_name in cast("list[str]", page_record["fired_signals"]):
                signal_counter[signal_name] += 1

    top_docs = sorted(records, key=lambda record: (-int(record["suspicion_score"]), str(record["filename"])))[:20]
    lines = [
        "# Private Doc Anomaly Scan Summary",
        "",
        f"- Docs scanned: {docs_scanned}",
        f"- Pages scanned: {pages_scanned}",
        f"- Docs with any signals: {docs_with_signals}",
        f"- Docs with parser fallback: {docs_with_parser_fallback}",
        f"- Max suspicion score: {max((int(record['suspicion_score']) for record in records), default=0)}",
        "",
        "## Top Docs By Suspicion Score",
        "",
        "| doc_id | filename | suspicion_score | signal_count | parser_mode |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    if top_docs:
        for record in top_docs:
            lines.append(
                f"| {record['doc_id']} | {record['filename']} | {record['suspicion_score']} | "
                f"{record['signal_count']} | {record['parser_mode']} |"
            )
    else:
        lines.append("| - | - | 0 | 0 | - |")

    lines.extend(
        [
            "",
            "## Top Signal Frequencies",
            "",
            "| signal | count |",
            "| --- | ---: |",
        ]
    )
    if signal_counter:
        for signal_name, count in signal_counter.most_common(20):
            lines.append(f"| {signal_name} | {count} |")
    else:
        lines.append("| - | 0 |")

    duplicate_clusters: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        cluster_id = record.get("exact_duplicate_cluster_id")
        if isinstance(cluster_id, str) and cluster_id:
            duplicate_clusters[cluster_id].append(record)

    lines.extend(
        [
            "",
            "## Exact Duplicate Clusters",
            "",
            "| cluster_id | member_count | filenames | normalized_title |",
            "| --- | ---: | --- | --- |",
        ]
    )
    if duplicate_clusters:
        for cluster_id, members in sorted(duplicate_clusters.items()):
            filenames = ", ".join(str(member["filename"]) for member in members)
            normalized_title = str(members[0].get("normalized_title") or "")
            lines.append(f"| {cluster_id} | {len(members)} | {filenames} | {normalized_title} |")
    else:
        lines.append("| - | 0 | - | - |")

    lines.extend(
        [
            "",
            "Weighted suspicion scoring is heuristic and intended for private-day triage only.",
            "",
        ]
    )
    return "\n".join(lines)


def build_top20_report_markdown(records: list[JsonDict]) -> str:
    lines = [
        "# Top 20 Suspicious Documents",
        "",
        "| rank | doc_id | filename | score | top_reasons |",
        "| ---: | --- | --- | ---: | --- |",
    ]
    top_docs = sorted(records, key=lambda record: (-int(record["suspicion_score"]), str(record["filename"])))[:20]
    if not top_docs:
        lines.append("| 1 | - | - | 0 | - |")
        return "\n".join(lines) + "\n"

    for rank, record in enumerate(top_docs, start=1):
        top_reasons = ", ".join(cast("list[str]", record.get("reason_tags") or [])[:3]) or "-"
        lines.append(
            f"| {rank} | {record['doc_id']} | {record['filename']} | {record['suspicion_score']} | {top_reasons} |"
        )

    lines.extend(["", "## Page Highlights", ""])
    for rank, record in enumerate(top_docs, start=1):
        lines.append(f"### {rank}. {record['filename']} ({record['doc_id']})")
        lines.append("")
        lines.append(f"- Score: {record['suspicion_score']}")
        lines.append(f"- Reasons: {', '.join(cast('list[str]', record.get('reason_tags') or [])) or '-'}")
        lines.append(f"- Doc families: {', '.join(cast('list[str]', record.get('doc_family_tags') or [])) or '-'}")
        page_lines: list[str] = []
        for page_record in cast("list[JsonDict]", record.get("per_page") or []):
            fired = cast("list[str]", page_record.get("fired_signals") or [])
            if not fired:
                continue
            page_lines.append(f"page {page_record['page_num']}: {', '.join(fired[:6])}")
        lines.append(f"- Flagged pages: {'; '.join(page_lines) if page_lines else '-'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan a PDF corpus for private-day anomaly signals.")
    parser.add_argument("input_dir", help="Directory containing PDF files to scan.")
    parser.add_argument("--output-dir", required=True, help="Directory to write scan_results.jsonl and summary.md")
    parser.add_argument(
        "--mode",
        default="raw-pdf-corpus",
        choices=("raw-pdf-corpus", "post-ingest-advisory"),
        help="Advisory label describing how the scan is being used.",
    )
    parser.add_argument("--coverage-priors-json", help="Optional JSON file with public question-surface coverage priors.")
    parser.add_argument("--dump-text", action="store_true", help="Dump per-page extracted text for scanned PDFs.")
    parser.add_argument(
        "--dump-screenshots",
        action="store_true",
        help="Render suspicious PDF pages as PNG screenshots for manual review.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    coverage_priors: CoveragePriors = (
        _load_coverage_priors(Path(args.coverage_priors_json)) if args.coverage_priors_json else {}
    )
    records = scan_pdf_corpus(
        input_dir=input_dir,
        mode=str(args.mode),
        coverage_priors=coverage_priors,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "scan_results.jsonl", records)
    (output_dir / "summary.md").write_text(build_summary_markdown(records), encoding="utf-8")
    (output_dir / "top20_report.md").write_text(build_top20_report_markdown(records), encoding="utf-8")
    if args.dump_text:
        _write_text_dumps(output_dir, records, DocumentParser(), input_dir)
    if args.dump_screenshots:
        _write_screenshots(output_dir, records, input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
