from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import fitz

if TYPE_CHECKING:
    from pathlib import Path


def _write_pdf(path: Path, pages: list[str]) -> None:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 520, 760), text, fontsize=12)
    pdf.save(path)
    pdf.close()


def test_audit_question_surface_gap_builds_family_report(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_pdf(docs_dir / "law_doc.pdf", ["LAW NO. 1 OF 2024\nThis Law may be cited as the Example Law."])
    _write_pdf(docs_dir / "order_doc.pdf", ["IN THE DIFC COURTS\nIT IS ORDERED THAT the claim is dismissed."])

    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {"id": "q1", "question": "What does the law say?", "answer_type": "free_text"},
                {"id": "q2", "question": "What was ordered?", "answer_type": "free_text"},
            ]
        ),
        encoding="utf-8",
    )
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q1", "retrieved_chunk_pages": [{"doc_id": "law_doc", "page_numbers": [1]}]},
                    {"question_id": "q2", "retrieved_chunk_pages": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "gap.json"
    out_md = tmp_path / "gap.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_question_surface_gap.py",
            "--docs-dir",
            str(docs_dir),
            "--questions",
            str(questions),
            "--scaffold",
            str(scaffold),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["family_buckets"]["consolidated_law"] == "one-hit"
    assert "order" in payload["family_buckets"]
    assert payload["family_buckets"]["order"] == "zero-hit"
    assert "Public Question-Surface Gap Report" in out_md.read_text(encoding="utf-8")
