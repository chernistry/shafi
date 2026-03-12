from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import fitz

if TYPE_CHECKING:
    from pathlib import Path


def test_audit_ocr_page_boundaries_flags_multi_page_low_text_pdf(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    scaffold_path = tmp_path / "scaffold.json"
    seed_qids_path = tmp_path / "seed_qids.txt"
    out_path = tmp_path / "ocr_audit.md"
    json_path = tmp_path / "ocr_audit.json"

    pdf = fitz.open()
    for text in ("x", "y", "z"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(docs_dir / "risk-doc.pdf")
    pdf.close()

    scaffold_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q-risk",
                        "failure_class": "support_undercoverage",
                        "required_page_anchor": {"kind": "explicit_page", "pages": [2]},
                        "retrieved_chunk_pages": [{"doc_id": "risk-doc", "page_numbers": [1, 2]}],
                        "minimal_required_support_pages": ["risk-doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seed_qids_path.write_text("q-risk\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_ocr_page_boundaries.py",
            "--docs-dir",
            str(docs_dir),
            "--scaffold",
            str(scaffold_path),
            "--seed-qids-file",
            str(seed_qids_path),
            "--out",
            str(out_path),
            "--json-out",
            str(json_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    assert "- OCR single-page risk count: `1`" in report
    assert "- Anchor-sensitive risk count: `1`" in report
    assert "## risk-doc" in report

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["ocr_single_page_risk_count"] == 1
    assert payload["risk_docs"][0]["doc_id"] == "risk-doc"
    assert payload["risk_docs"][0]["anchor_sensitive"] is True
