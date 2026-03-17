from __future__ import annotations

import json
from typing import TYPE_CHECKING

import fitz
from scripts import run_scanner_ticket_341_stress as mod

if TYPE_CHECKING:
    from pathlib import Path


def _write_pdf(path: Path, pages: list[str]) -> None:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(48, 48, 565, 790), text, fontsize=11)
    pdf.save(path)
    pdf.close()


def test_run_ticket_341_stress_validation_writes_reports(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    law_pdf = docs_dir / "law.pdf"
    judgment_pdf = docs_dir / "judgment.pdf"
    enactment_pdf = docs_dir / "enactment.pdf"

    _write_pdf(
        law_pdf,
        [
            "GENERAL PARTNERSHIP LAW\nLAW NO. 11 OF 2004\nThis Law may be cited as the Example Law.\nSchedule 1 applies.",
            "CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4\nArticle 3 ... 6\nArticle 4 ... 8",
        ],
    )
    _write_pdf(
        judgment_pdf,
        [
            "IN THE DIFC COURTS\nJudgment\nClaimant v Defendant\nBackground\nDiscussion",
            "IT IS HEREBY ORDERED THAT the claim is dismissed.\nNo order as to costs.",
        ],
    )
    _write_pdf(
        enactment_pdf,
        [
            "ENACTMENT NOTICE\nDIFC Law No. 1 of 2024\nThis Law shall come into force on the 5th business day after enactment.",
        ],
    )

    output_dir = tmp_path / "out"
    report = mod.run_ticket_341_stress_validation(
        docs_dir=docs_dir,
        output_dir=output_dir,
        law_pdf=law_pdf,
        judgment_pdf=judgment_pdf,
        enactment_pdf=enactment_pdf,
    )

    assert report["variant_count"] == 7
    assert report["overall_pass"] is True
    assert (output_dir / "scan_results.jsonl").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "top20_report.md").exists()
    assert (output_dir / "stress_validation_report.json").exists()
    assert (output_dir / "stress_validation_report.md").exists()

    payload = json.loads((output_dir / "stress_validation_report.json").read_text(encoding="utf-8"))
    by_id = {row["doc_id"]: row for row in payload["variants"]}
    assert by_id["unicode_weirdness"]["passed"] is True
    assert by_id["tracked_changes_notice"]["passed"] is True
    assert "Ticket 341 Scanner Stress Validation" in (output_dir / "stress_validation_report.md").read_text(encoding="utf-8")
