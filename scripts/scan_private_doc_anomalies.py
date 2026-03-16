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


def _load_coverage_priors(path: Path | None) -> Any:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
                image_count = len(images_obj) if isinstance(images_obj, list) else 0
                links_obj = page_any.get_links()
                internal_link_count = 0
                link_destinations: list[str] = []
                external_urls: list[str] = []
                if isinstance(links_obj, list):
                    for raw_link in cast("list[object]", links_obj):
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
        return len(value) > 0
    return False


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
    contents_link_densities = [
        float(cast("JsonDict", page_record["signals"])["contents_internal_link_density"])
        for page_record in page_records
        if bool(cast("JsonDict", page_record["signals"])["contents_signature"])
    ]
    contents_internal_link_density = (
        round(sum(contents_link_densities) / len(contents_link_densities), 4) if contents_link_densities else 0.0
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
    }

    fired_signals = [name for name, value in signals.items() if _signal_value_is_active(value)]

    return DocAnalysis(
        signals=signals,
        fired_signals=sorted(fired_signals),
        coarse_artifact_kind=coarse_artifact_kind,
        case_references=case_references,
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
            fired_signals = cast("list[str]", record["_fired_signals"])
            if "same_case_multi_artifact_hint" not in fired_signals:
                fired_signals.append("same_case_multi_artifact_hint")


def _stub_score(_record: JsonDict) -> tuple[int, list[str]]:
    return 0, []


def scan_pdf_corpus(
    *,
    input_dir: Path,
    mode: str,
    coverage_priors: Any,
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
            "_fired_signals": fired_signals,
        }
        records.append(record)

    _apply_same_case_multi_artifact_hint(records)

    for record in records:
        suspicion_score, reason_tags = _stub_score(record)
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

    top_docs = sorted(records, key=lambda record: (-int(record["signal_count"]), str(record["filename"])))[:20]
    lines = [
        "# Private Doc Anomaly Scan Summary",
        "",
        f"- Docs scanned: {docs_scanned}",
        f"- Pages scanned: {pages_scanned}",
        f"- Docs with any signals: {docs_with_signals}",
        f"- Docs with parser fallback: {docs_with_parser_fallback}",
        "",
        "## Top Docs By Signal Count",
        "",
        "| doc_id | filename | signal_count | parser_mode | fallback_triggered |",
        "| --- | --- | ---: | --- | --- |",
    ]
    if top_docs:
        for record in top_docs:
            lines.append(
                f"| {record['doc_id']} | {record['filename']} | {record['signal_count']} | "
                f"{record['parser_mode']} | {record['fallback_triggered']} |"
            )
    else:
        lines.append("| - | - | 0 | - | - |")

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

    lines.extend(
        [
            "",
            "Weighted suspicion scoring deferred to ticket 339.",
            "",
        ]
    )
    return "\n".join(lines)


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

    coverage_priors = _load_coverage_priors(Path(args.coverage_priors_json)) if args.coverage_priors_json else {}
    records = scan_pdf_corpus(
        input_dir=input_dir,
        mode=str(args.mode),
        coverage_priors=coverage_priors,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "scan_results.jsonl", records)
    (output_dir / "summary.md").write_text(build_summary_markdown(records), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
