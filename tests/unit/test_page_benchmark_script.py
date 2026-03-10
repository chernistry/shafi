from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_score_page_benchmark_script_reports_f_beta_and_orphans(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    eval_path = tmp_path / "eval.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc_1", "doc_2"],
                        "gold_items": ["Alpha", "Beta"],
                        "wrong_document_risk": True,
                        "items": [
                            {
                                "id": "item-1",
                                "text": "Alpha only.",
                                "gold_page_ids": ["doc_1", "doc_2"],
                                "slots": [
                                    {"name": "title", "gold_page_ids": ["doc_1"]},
                                    {"name": "updated", "gold_page_ids": ["doc_2"]},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "answer": "Alpha only.",
                        "telemetry": {"used_page_ids": ["doc_1", "doc_3"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_page_benchmark.py",
            "--eval",
            str(eval_path),
            "--benchmark",
            str(benchmark_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    assert "## All Cases" in result.stdout
    assert "## Trusted Tier" in result.stdout
    assert "Page-level F_beta(2.5)" in result.stdout
    assert "orphan=['doc_3']" in result.stdout
    assert "trust_tier=trusted" in result.stdout
    assert "gold_origin=manual_override" in result.stdout
    assert "item_coverage=0.5000" in result.stdout
    assert "Mean slot recall: 0.5000" in result.stdout
    assert "Evidence-family full-coverage rate: 0.0000" in result.stdout
    assert "Overprune violations: 1" in result.stdout
    assert "Wrong-document tagged case rate: 1.0000" in result.stdout
