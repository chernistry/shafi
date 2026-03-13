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


def test_sync_hidden_g_benchmark_seed_script_adds_trusted_scaffold_cases_without_eval(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    scaffold_path = tmp_path / "scaffold.json"
    qids_path = tmp_path / "qids.txt"
    out_path = tmp_path / "merged.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "name": "seed_v1",
                "source_eval": "old.json",
                "description": "old",
                "cases": [
                    {
                        "question_id": "q-existing",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "audit_note": "manual",
                        "gold_page_ids": ["doc_1"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scaffold_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q-add",
                        "question": "According to the title page, what is the law number?",
                        "manual_verdict": "correct",
                        "failure_class": "",
                        "minimal_required_support_pages": ["doc_2"],
                    },
                    {
                        "question_id": "q-skip",
                        "question": "Non-anchor case",
                        "manual_verdict": "correct",
                        "failure_class": "",
                        "minimal_required_support_pages": ["doc_3"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    qids_path.write_text("q-add\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/sync_hidden_g_benchmark_seed.py",
            "--benchmark",
            str(benchmark_path),
            "--scaffold",
            str(scaffold_path),
            "--qids-file",
            str(qids_path),
            "--out",
            str(out_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert merged["source_eval"] == "old.json"
    assert [case["question_id"] for case in merged["cases"]] == ["q-existing", "q-add"]
    added = merged["cases"][1]
    assert added["trust_tier"] == "trusted"
    assert added["gold_origin"] == "manual_override"
    assert added["gold_page_ids"] == ["doc_2"]
    assert "title_page" in added["audit_note"]


def test_sync_hidden_g_benchmark_seed_script_promotes_existing_seeded_eval_case_from_scaffold(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    scaffold_path = tmp_path / "scaffold.json"
    out_path = tmp_path / "merged.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "name": "seed_v1",
                "source_eval": "old.json",
                "description": "old",
                "cases": [
                    {
                        "question_id": "q-existing",
                        "trust_tier": "suspect",
                        "gold_origin": "seeded_eval",
                        "audit_note": "seed",
                        "gold_page_ids": ["doc_4"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    scaffold_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q-existing",
                        "question": "According to Article 10, what is the limit?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "minimal_required_support_pages": ["doc_6"],
                    }
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
            "--scaffold",
            str(scaffold_path),
            "--out",
            str(out_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert [case["question_id"] for case in merged["cases"]] == ["q-existing"]
    promoted = merged["cases"][0]
    assert promoted["trust_tier"] == "trusted"
    assert promoted["gold_origin"] == "manual_override"
    assert promoted["gold_page_ids"] == ["doc_6"]
    assert "scaffold" in promoted["audit_note"].casefold()
