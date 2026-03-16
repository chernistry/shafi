from __future__ import annotations

from rag_challenge.core.normalize_legal_text import normalize_legal_text


def test_normalize_legal_text_normalizes_unicode_and_digits() -> None:
    raw = "Law No.\u00a05 of ٢٠١٨ \u2014 \u2018quoted\u2019 \u200btext"
    normalized = normalize_legal_text(raw)
    assert normalized == "Law No. 5 of 2018 - 'quoted' text"
