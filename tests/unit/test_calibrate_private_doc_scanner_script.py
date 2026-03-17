from __future__ import annotations

from pathlib import Path

import fitz

from scripts import calibrate_private_doc_scanner as mod


def _write_pdf(path: Path, pages: list[str]) -> None:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 520, 760), text, fontsize=12)
    pdf.save(path)
    pdf.close()


def test_build_scanner_baseline_includes_signal_stats_and_fixture_candidates(tmp_path: Path) -> None:
    _write_pdf(
        tmp_path / "law_doc.pdf",
        [
            "LAW NO. 1 OF 2024\nThis Law may be cited as the Example Law.",
            "CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4\nArticle 3 ... 6\nArticle 4 ... 8",
        ],
    )
    _write_pdf(
        tmp_path / "tracked_doc.pdf",
        [
            "LAW NO. 2 OF 2024\nAmendment Law\nThe words are deleted and replaced by the following.\nThe underlined words are inserted.",
        ],
    )

    report = mod.build_scanner_baseline(docs_dir=tmp_path)

    assert report["docs_scanned"] == 2
    assert "score_distribution" in report
    assert "signal_baselines" in report
    assert "fixture_candidates" in report
    assert report["signal_baselines"]["doc_signals"]["boolean"]["tracked_changes_detected"]["active_count"] == 1
    assert report["signal_baselines"]["page_signals"]["numeric"]["contents_internal_link_density"]["count"] >= 1
    assert report["fixture_candidates"]["tracked_changes_docs"][0]["doc_id"] == "tracked_doc"


def test_fixture_candidates_detect_contents_linked_docs_from_nested_signals() -> None:
    candidates = mod._fixture_candidates(
        [
            {
                "doc_id": "linked_doc",
                "filename": "linked_doc.pdf",
                "suspicion_score": 12,
                "reason_tags": ["contents_without_internal_links"],
                "signals": {"contents_internal_link_density": 1.25},
                "contents_link_count": 0,
            }
        ]
    )

    assert candidates["contents_linked_docs"] == [
        {
            "doc_id": "linked_doc",
            "filename": "linked_doc.pdf",
            "score": 12,
            "reason_tags": ["contents_without_internal_links"],
        }
    ]
