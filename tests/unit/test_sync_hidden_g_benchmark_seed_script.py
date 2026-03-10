from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_sync_hidden_g_benchmark_seed_script_extends_fixture_with_eval_cases(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    eval_path = tmp_path / "eval.json"
    out_path = tmp_path / "merged.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "name": "seed_v1",
                "source_eval": "old.json",
                "description": "old",
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "audit_note": "manual",
                        "gold_page_ids": ["doc_1"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": True,
                    }
                ],
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
                        "telemetry": {"used_page_ids": ["doc_1"]},
                    },
                    {
                        "question_id": "q2",
                        "telemetry": {"used_page_ids": ["doc_2", "doc_3"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/sync_hidden_g_benchmark_seed.py",
            "--benchmark",
            str(benchmark_path),
            "--eval",
            str(eval_path),
            "--out",
            str(out_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert merged["source_eval"] == "eval.json"
    assert [case["question_id"] for case in merged["cases"]] == ["q1", "q2"]
    assert merged["cases"][0]["wrong_document_risk"] is True
    assert merged["cases"][0]["trust_tier"] == "trusted"
    assert merged["cases"][0]["gold_origin"] == "manual_override"
    assert merged["cases"][1]["gold_page_ids"] == ["doc_2", "doc_3"]
    assert merged["cases"][1]["trust_tier"] == "suspect"
    assert merged["cases"][1]["gold_origin"] == "seeded_eval"
