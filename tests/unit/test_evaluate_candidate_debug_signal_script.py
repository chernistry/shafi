from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts import evaluate_candidate_debug_signal as mod

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_select_qids_uses_answer_and_page_drift() -> None:
    baseline = {
        "q1": mod.CandidateCase("q1", "Q1", "boolean", "Yes", {"used_page_ids": ["doc_1"]}, 10.0),
        "q2": mod.CandidateCase("q2", "Q2", "name", "A", {"used_page_ids": ["doc_2"]}, 12.0),
    }
    candidate = {
        "q1": mod.CandidateCase("q1", "Q1", "boolean", "Yes", {"used_page_ids": ["doc_9"]}, 10.0),
        "q2": mod.CandidateCase("q2", "Q2", "name", "B", {"used_page_ids": ["doc_2"]}, 12.0),
    }
    assert mod._select_qids(
        baseline_cases=baseline,
        candidate_cases=candidate,
        scope="changed",
        include_qids=set(),
    ) == ["q1", "q2"]


@pytest.mark.asyncio
async def test_async_main_writes_candidate_debug_artifacts(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    baseline_raw = tmp_path / "baseline.json"
    candidate_raw = tmp_path / "candidate.json"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    out_dir = tmp_path / "out"

    _write_json(
        questions,
        [
            {"id": "q1", "question": "Q1?", "answer_type": "boolean"},
            {"id": "q2", "question": "Q2?", "answer_type": "name"},
        ],
    )
    _write_json(
        baseline_raw,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "boolean"},
                "answer_text": "Yes",
                "telemetry": {"ttft_ms": 10, "used_page_ids": ["doc_1"], "context_chunk_ids": [], "cited_chunk_ids": []},
            },
            {
                "case": {"case_id": "q2", "question": "Q2?", "answer_type": "name"},
                "answer_text": "Alice",
                "telemetry": {"ttft_ms": 20, "used_page_ids": ["doc_2"], "context_chunk_ids": [], "cited_chunk_ids": []},
            },
        ],
    )
    _write_json(
        candidate_raw,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "boolean"},
                "answer_text": "Yes",
                "telemetry": {"ttft_ms": 10, "used_page_ids": ["doc_9"], "context_chunk_ids": [], "cited_chunk_ids": []},
            },
            {
                "case": {"case_id": "q2", "question": "Q2?", "answer_type": "name"},
                "answer_text": "Alice",
                "telemetry": {"ttft_ms": 20, "used_page_ids": ["doc_2"], "context_chunk_ids": [], "cited_chunk_ids": []},
            },
        ],
    )

    namespace = type(
        "Args",
        (),
        {
            "baseline_label": "baseline_x",
            "baseline_raw_results": baseline_raw,
            "candidate_label": "candidate_y",
            "candidate_raw_results": candidate_raw,
            "questions": questions,
            "docs_dir": docs_dir,
            "out_dir": out_dir,
            "case_scope": "changed",
            "judge_scope": "none",
            "include_qids_file": None,
        },
    )()

    await mod._async_main(namespace)

    compare_json = out_dir / "candidate_debug_compare_candidate_y_vs_baseline_x.json"
    compare_md = out_dir / "candidate_debug_compare_candidate_y_vs_baseline_x.md"
    eval_baseline = out_dir / "eval_candidate_debug_baseline_x.json"
    eval_candidate = out_dir / "eval_candidate_debug_candidate_y.json"
    assert compare_json.exists()
    assert compare_md.exists()
    assert eval_baseline.exists()
    assert eval_candidate.exists()

    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    assert payload["selected_qids"] == ["q1"]
    baseline_payload = json.loads(eval_baseline.read_text(encoding="utf-8"))
    assert baseline_payload["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"
    assert baseline_payload["summary"]["total_cases"] == 1
