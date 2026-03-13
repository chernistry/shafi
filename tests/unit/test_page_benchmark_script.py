from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = "/Users/sasha/IdeaProjects/.codex-worktrees/rag_challenge-main"


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
        cwd=REPO_ROOT,
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


def test_build_page_trace_ledger_can_export_bounded_miss_pack(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    scaffold_path = tmp_path / "scaffold.json"
    raw_results_path = tmp_path / "raw_results.json"
    ledger_json = tmp_path / "ledger.json"
    ledger_md = tmp_path / "ledger.md"
    miss_pack_json = tmp_path / "miss_pack.json"
    miss_pack_md = tmp_path / "miss_pack.md"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q_explicit", "gold_page_ids": ["docA_2"], "trust_tier": "trusted"},
                    {"question_id": "q_title", "gold_page_ids": ["docB_1"], "trust_tier": "trusted"},
                    {"question_id": "q_same", "gold_page_ids": ["docC_3"], "trust_tier": "trusted"},
                    {"question_id": "q_multi", "gold_page_ids": ["docD_1", "docE_2"], "trust_tier": "trusted", "wrong_document_risk": True},
                    {"question_id": "q_ocr", "gold_page_ids": ["docF_1"], "trust_tier": "trusted", "ocr_risk": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    scaffold_path.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q_explicit", "question": "What is on page 2?", "route_family": "page_sensitive"},
                    {"question_id": "q_title", "question": "What is on the title page?", "route_family": "metadata"},
                    {"question_id": "q_same", "question": "Who is the claimant?", "route_family": "metadata"},
                    {"question_id": "q_multi", "question": "Compare the parties in both cases.", "route_family": "comparison"},
                    {"question_id": "q_ocr", "question": "What is the scanned cover page?", "route_family": "ocr", "ocr_risk": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_results_path.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q_explicit"},
                    "telemetry": {
                        "retrieved_page_ids": ["docA_1"],
                        "context_page_ids": ["docA_1"],
                        "used_page_ids": ["docA_1"],
                    },
                },
                {
                    "case": {"case_id": "q_title"},
                    "telemetry": {
                        "retrieved_page_ids": ["docB_2"],
                        "context_page_ids": ["docB_2"],
                        "used_page_ids": ["docB_2"],
                    },
                },
                {
                    "case": {"case_id": "q_same"},
                    "telemetry": {
                        "retrieved_page_ids": ["docC_2"],
                        "context_page_ids": ["docC_2"],
                        "used_page_ids": ["docC_2"],
                    },
                },
                {
                    "case": {"case_id": "q_multi"},
                    "telemetry": {
                        "retrieved_page_ids": ["docD_3"],
                        "context_page_ids": ["docD_3"],
                        "used_page_ids": ["docD_3"],
                    },
                },
                {
                    "case": {"case_id": "q_ocr"},
                    "telemetry": {
                        "retrieved_page_ids": ["docF_2"],
                        "context_page_ids": ["docF_2"],
                        "used_page_ids": ["docF_2"],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_page_trace_ledger.py",
            "--raw-results",
            str(raw_results_path),
            "--benchmark",
            str(benchmark_path),
            "--scaffold",
            str(scaffold_path),
            "--out-json",
            str(ledger_json),
            "--out-md",
            str(ledger_md),
            "--miss-pack-json",
            str(miss_pack_json),
            "--miss-pack-md",
            str(miss_pack_md),
            "--miss-pack-max-per-family",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    miss_pack = json.loads(miss_pack_json.read_text(encoding="utf-8"))
    families = {case["miss_family"] for case in miss_pack["cases"]}
    assert families == {"explicit_page", "title_page", "same_doc", "mixed_doc", "ocr_risk"}
    assert miss_pack["summary"]["selected_case_count"] == 5


def test_score_page_benchmark_respects_include_qids_file(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    eval_path = tmp_path / "eval.json"
    include_qids = tmp_path / "include_qids.txt"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "gold_page_ids": ["doc_1"]},
                    {"question_id": "q2", "gold_page_ids": ["doc_2"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "answer": "A", "telemetry": {"used_page_ids": ["doc_1"]}},
                    {"question_id": "q2", "answer": "B", "telemetry": {"used_page_ids": ["doc_9"]}},
                ]
            }
        ),
        encoding="utf-8",
    )
    include_qids.write_text("q1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_page_benchmark.py",
            "--eval",
            str(eval_path),
            "--benchmark",
            str(benchmark_path),
            "--include-qids-file",
            str(include_qids),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "- Cases: 1" in result.stdout
    assert "q2" not in result.stdout
