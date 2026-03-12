from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _submission(answer: str, pages: list[int]) -> dict[str, object]:
    return {
        "answers": [
            {
                "question_id": "q1",
                "answer": answer,
                "telemetry": {
                    "answer_type": "boolean",
                    "retrieval": {
                        "retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": pages}],
                    },
                },
            }
        ]
    }


def _raw(answer: str, used_pages: list[str]) -> list[dict[str, object]]:
    return [
        {
            "case": {
                "case_id": "q1",
                "question": "Do both cases share the same party?",
                "answer_type": "boolean",
            },
            "answer_text": answer,
            "telemetry": {
                "ttft_ms": 100,
                "retrieved_page_ids": used_pages,
                "context_page_ids": used_pages,
                "used_page_ids": used_pages,
                "retrieved_chunk_ids": [],
                "context_chunk_ids": [],
                "cited_chunk_ids": [],
            },
        }
    ]


def _preflight(p95: int = 2) -> dict[str, object]:
    return {
        "phase": "warmup",
        "questions_count": 1,
        "answer_type_counts": {"boolean": 1},
        "page_count_distribution": {"min": 1, "p50": p95, "p95": p95, "max": p95},
        "code_archive_sha256": "archive",
        "questions_sha256": "questions",
        "documents_zip_sha256": "docs",
        "pdf_count": 1,
        "phase_collection_name": "collection",
        "qdrant_point_count": 1,
        "truth_audit_workbook_path": "workbook",
    }


def test_search_portfolio_support_combos_builds_ranked_report(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    baseline_submission.write_text(json.dumps(_submission("No", [2])), encoding="utf-8")
    baseline_raw = tmp_path / "baseline_raw.json"
    baseline_raw.write_text(json.dumps(_raw("No", ["doc_2"])), encoding="utf-8")
    baseline_preflight = tmp_path / "baseline_preflight.json"
    baseline_preflight.write_text(json.dumps(_preflight(2)), encoding="utf-8")

    source_submission = tmp_path / "source_submission.json"
    source_submission.write_text(json.dumps(_submission("No", [1])), encoding="utf-8")
    source_raw = tmp_path / "source_raw.json"
    source_raw.write_text(json.dumps(_raw("No", ["doc_1"])), encoding="utf-8")
    source_preflight = tmp_path / "source_preflight.json"
    source_preflight.write_text(json.dumps(_preflight(2)), encoding="utf-8")

    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_page_ids": ["doc_1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{"id": "q1", "question": "Do both cases share the same party?", "answer_type": "boolean"}]),
        encoding="utf-8",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text(
        json.dumps(
            [
                {
                    "qid": "q1",
                    "label": "page1",
                    "submission_path": str(source_submission),
                    "raw_results_path": str(source_raw),
                    "preflight_path": str(source_preflight),
                    "notes": "page1 support",
                }
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            "scripts/search_portfolio_support_combos.py",
            "--baseline-label",
            "baseline",
            "--baseline-submission",
            str(baseline_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--baseline-preflight",
            str(baseline_preflight),
            "--portfolio-json",
            str(portfolio),
            "--benchmark",
            str(benchmark),
            "--questions",
            str(questions),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
            "--judge-top-k",
            "0",
        ],
        check=True,
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
    )

    payload = json.loads((out_dir / "portfolio_support_combo_search.json").read_text(encoding="utf-8"))
    assert payload["results"][0]["qids"] == ["q1"]
    assert payload["results"][0]["recommendation"] == "PROMISING"
    assert payload["results"][0]["benchmark_trusted_candidate"] > payload["results"][0]["benchmark_trusted_baseline"]
