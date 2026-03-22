from __future__ import annotations

import re
import unicodedata

_ARABIC_INDIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u2060\uFEFF]")
_DASH_TRANSLATION = {
    ord("\u2010"): "-",
    ord("\u2011"): "-",
    ord("\u2012"): "-",
    ord("\u2013"): "-",
    ord("\u2014"): "-",
    ord("\u2212"): "-",
    ord("\u2044"): "/",
    ord("\u00A0"): " ",
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201C"): '"',
    ord("\u201D"): '"',
    ord("\u060C"): ",",
    ord("\u061B"): ";",
    ord("\u066B"): ".",
    ord("\u066C"): ",",
}


def normalize_legal_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_ARABIC_INDIC_DIGITS)
    normalized = normalized.translate(_DASH_TRANSLATION)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
