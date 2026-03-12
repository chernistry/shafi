from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_select_question_family_qids_party_title_metadata(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    out = tmp_path / "qids.txt"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "a",
                    "question": "Who were the claimants in case CFI 010/2024?",
                    "answer_type": "names",
                },
                {
                    "id": "b",
                    "question": "What was the outcome in case CFI 010/2024?",
                    "answer_type": "free_text",
                },
                {
                    "id": "c",
                    "question": "Do cases CA 004/2025 and SCT 295/2025 involve any of the same legal entities or individuals as parties?",
                    "answer_type": "boolean",
                },
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/select_question_family_qids.py",
            "--questions",
            str(questions),
            "--family",
            "party_title_metadata",
            "--out",
            str(out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.read_text(encoding="utf-8").splitlines() == ["a", "c"]
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["family"] == "party_title_metadata"
    assert payload["count"] == 2


def test_select_question_family_qids_explicit_page_reference(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    out = tmp_path / "qids.txt"
    questions.write_text(
        json.dumps(
            [
                {"id": "a", "question": "According to page 2 of the judgment, what happened?", "answer_type": "boolean"},
                {"id": "b", "question": "Who were the claimants in case CFI 010/2024?", "answer_type": "names"},
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/select_question_family_qids.py",
            "--questions",
            str(questions),
            "--family",
            "explicit_page_reference",
            "--out",
            str(out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.read_text(encoding="utf-8").splitlines() == ["a"]
