from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _raw(qid: str, question: str, answer_type: str, used_pages: list[str], answer: str = "false") -> dict[str, object]:
    return {
        "case": {"case_id": qid, "question": question, "answer_type": answer_type},
        "answer_text": answer,
        "telemetry": {
            "used_page_ids": used_pages,
            "context_page_ids": used_pages,
            "retrieved_page_ids": used_pages,
        },
    }


def test_audit_explicit_page_reference_candidates_reports_page_rescue(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "According to page 2 of the judgment, what is the claim number?",
                    "answer_type": "name",
                }
            ]
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            [
                _raw(
                    "q1",
                    "According to page 2 of the judgment, what is the claim number?",
                    "name",
                    ["doc_1"],
                    "ENF 316/2023",
                )
            ]
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                _raw(
                    "q1",
                    "According to page 2 of the judgment, what is the claim number?",
                    "name",
                    ["doc_2"],
                    "ENF 316/2023",
                )
            ]
        ),
        encoding="utf-8",
    )
    scaffold = tmp_path / "scaffold.json"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "route_family": "model",
                        "support_shape_class": "generic",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_explicit_page_reference_candidates.py",
            "--questions",
            str(questions),
            "--baseline-raw-results",
            str(baseline),
            "--baseline-label",
            "baseline",
            "--scaffold",
            str(scaffold),
            "--source",
            f"candidate={source}",
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ],
        check=True,
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert record["target_page"] == 2
    assert record["baseline_used_page_hits"] == 0
    assert record["source_signals"][0]["used_page_hits"] == 1
    assert record["recommendation"] == "PROMISING"
