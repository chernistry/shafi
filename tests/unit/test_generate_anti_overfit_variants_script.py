from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_anti_overfit_variants_outputs_metadata(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Under Article 42 of the Companies Law, DIFC Law No. 5 of 2018, what happened in CFI 010/2024?",
                    "answer_type": "free_text",
                }
            ]
        ),
        encoding="utf-8",
    )
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {"records": [{"question_id": "q1", "retrieved_chunk_pages": [{"doc_id": "law-doc", "page_numbers": [1]}]}]}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "variants.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_anti_overfit_variants.py",
            "--questions",
            str(questions),
            "--scaffold",
            str(scaffold),
            "--out",
            str(out),
            "--limit",
            "5",
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload
    variant_types = {row["variant_type"] for row in payload}
    assert payload[0]["original_question_id"] == "q1"
    assert payload[0]["expected_gold_doc_ids"] == ["law-doc"]
    assert "variant_type" in payload[0]
    assert "law_number_only_framing" in variant_types
    assert "indirect_case_framing" in variant_types


def test_generate_anti_overfit_variants_preserves_distinct_case_placeholders(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "q_multi",
                    "question": "Did CFI 010/2024 and CA 002/2025 reach the same outcome under Article 10 of DIFC Law No. 3 of 2022?",
                    "answer_type": "free_text",
                }
            ]
        ),
        encoding="utf-8",
    )
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {"records": [{"question_id": "q_multi", "retrieved_chunk_pages": [{"doc_id": "case-doc", "page_numbers": [1]}]}]}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "variants.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_anti_overfit_variants.py",
            "--questions",
            str(questions),
            "--scaffold",
            str(scaffold),
            "--out",
            str(out),
            "--limit",
            "5",
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    indirect_case_rows = [row for row in payload if row["variant_type"] == "indirect_case_framing"]

    assert len(indirect_case_rows) == 1
    assert "the first relevant DIFC court case" in indirect_case_rows[0]["question"]
    assert "the second relevant DIFC court case" in indirect_case_rows[0]["question"]
