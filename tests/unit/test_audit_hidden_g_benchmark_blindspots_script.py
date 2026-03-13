from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.audit_hidden_g_benchmark_blindspots import build_blindspots

if TYPE_CHECKING:
    from pathlib import Path


def _write_scaffold(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"summary": {}, "records": records}, indent=2), encoding="utf-8")


def _write_benchmark(path: Path, qids: list[str]) -> None:
    path.write_text(
        json.dumps({"name": "bench", "source_eval": "seed", "description": "", "cases": [{"question_id": qid} for qid in qids]}, indent=2),
        encoding="utf-8",
    )


def _write_raw_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_build_blindspots_surfaces_missing_anchor_case_not_in_benchmark(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    benchmark = tmp_path / "benchmark.json"
    raw_results = tmp_path / "raw_results.json"
    _write_scaffold(
        scaffold,
        [
            {
                "question_id": "qid-1",
                "question": "According to the title page, what is the law number?",
                "manual_verdict": "correct",
                "failure_class": "support_undercoverage",
                "minimal_required_support_pages": ["doc_1"],
            }
        ],
    )
    _write_benchmark(benchmark, [])
    _write_raw_results(
        raw_results,
        [{"case": {"case_id": "qid-1"}, "telemetry": {"used_page_ids": ["doc_4"], "context_page_ids": [], "retrieved_page_ids": []}}],
    )

    blindspots = build_blindspots(
        scaffold_path=scaffold,
        benchmark_path=benchmark,
        current_raw_results_path=raw_results,
    )

    assert len(blindspots) == 1
    assert blindspots[0].question_id == "qid-1"
    assert blindspots[0].in_benchmark is False
    assert blindspots[0].current_has_gold is False


def test_build_blindspots_keeps_benchmark_presence_flag(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    benchmark = tmp_path / "benchmark.json"
    raw_results = tmp_path / "raw_results.json"
    _write_scaffold(
        scaffold,
        [
            {
                "question_id": "qid-1",
                "question": "Based on the second page, was the application granted?",
                "manual_verdict": "correct",
                "failure_class": "support_undercoverage",
                "minimal_required_support_pages": ["doc_2"],
            }
        ],
    )
    _write_benchmark(benchmark, ["qid-1"])
    _write_raw_results(
        raw_results,
        [{"case": {"case_id": "qid-1"}, "telemetry": {"used_page_ids": ["doc_2"], "context_page_ids": [], "retrieved_page_ids": []}}],
    )

    blindspots = build_blindspots(
        scaffold_path=scaffold,
        benchmark_path=benchmark,
        current_raw_results_path=raw_results,
    )

    assert len(blindspots) == 1
    assert blindspots[0].in_benchmark is True
    assert blindspots[0].current_has_gold is True
