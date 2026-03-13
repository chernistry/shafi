from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts import private_phase_dashboard as mod

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_summarize_run_flags_null_empty_page_and_high_page_cases(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    questions = tmp_path / "questions.json"
    truth = tmp_path / "truth.json"
    _write_json(
        run,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "free_text"},
                "answer_text": "There is no information on this question in the provided documents.",
                "telemetry": {"model_llm": "gpt-4o-mini", "ttft_ms": 150, "used_page_ids": []},
            },
            {
                "case": {"case_id": "q2", "question": "Q2?", "answer_type": "boolean"},
                "answer_text": "Yes",
                "telemetry": {"model_llm": "strict-extractor", "ttft_ms": 20, "used_page_ids": ["doc_1", "doc_2", "doc_3"]},
            },
        ],
    )
    _write_json(questions, [{"id": "q1", "question": "Q1?", "answer_type": "free_text"}, {"id": "q2", "question": "Q2?", "answer_type": "boolean"}])
    _write_json(
        truth,
        {
            "records": [
                {"question_id": "q1", "route_family": "model"},
                {"question_id": "q2", "route_family": "strict"},
            ]
        },
    )

    summary = mod._summarize_run(
        label="run_a",
        rows=mod._load_json_list(run),
        questions=mod._load_questions(questions),
        truth_audit=mod._load_truth_audit(truth),
        high_page_threshold=2,
    )

    assert summary["null_answer_count"] == 1
    assert summary["empty_used_page_count"] == 1
    assert summary["high_page_count_case_count"] == 1
    assert summary["route_counts"] == {"model": 1, "strict": 1}
    assert summary["ttft_p95_ms"] == 143.5


def test_main_writes_diff_report(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a.json"
    run_b = tmp_path / "run_b.json"
    out_json = tmp_path / "dashboard.json"
    out_md = tmp_path / "dashboard.md"

    _write_json(
        run_a,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "free_text"},
                "answer_text": "Answer A",
                "telemetry": {"model_llm": "gpt-4o-mini", "ttft_ms": 100, "used_page_ids": ["doc_1"]},
            }
        ],
    )
    _write_json(
        run_b,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "free_text"},
                "answer_text": "There is no information on this question in the provided documents.",
                "telemetry": {"model_llm": "strict-extractor", "ttft_ms": 300, "used_page_ids": []},
            }
        ],
    )

    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "private_phase_dashboard.py",
            "--run-a",
            str(run_a),
            "--run-b",
            str(run_b),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
        mod.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["diff"]["route_drift_count"] == 1
    assert payload["diff"]["model_drift_count"] == 1
    assert payload["diff"]["null_answer_state_changed_count"] == 1
    markdown = out_md.read_text(encoding="utf-8")
    assert "Private-Phase Telemetry Dashboard" in markdown
    assert "## Diff" in markdown
