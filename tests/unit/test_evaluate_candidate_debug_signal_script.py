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


def test_build_answer_to_page_attribution_signal_detects_real_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_settings",
        lambda: type("Settings", (), {"judge": type("Judge", (), {"sources_max_pages": 12})()})(),
    )
    cases = {
        "q1": mod.CandidateCase(
            "q1",
            "Which claim number?",
            "name",
            "ENF 316/2023",
            {"used_page_ids": ["doc_bad_1", "doc_bad_3", "doc_good_2", "doc_good_4"]},
            10.0,
        ),
        "q2": mod.CandidateCase(
            "q2",
            "Who was the claimant?",
            "name",
            "Alice Holdings Ltd",
            {"used_page_ids": ["doc_fp_7", "doc_gold_3"]},
            12.0,
        ),
    }
    gold_pages = {
        "q1": ["doc_good_2", "doc_good_4"],
        "q2": ["doc_gold_3"],
    }
    page_text = {
        "doc_good_2": "The originating matter was ENF 316/2023 before the appeal.",
        "doc_good_4": "CA 009/2024 originated from claim ENF 316/2023 in the lower matter.",
        "doc_bad_1": "Generic procedural background with no claim number.",
        "doc_bad_3": "General appeal standard and procedural background only.",
        "doc_gold_3": "Claimant Alice Holdings Ltd sought damages in this matter.",
        "doc_fp_7": "The defendant denied liability and raised a jurisdiction challenge.",
    }

    payload = mod._build_answer_to_page_attribution_signal(
        cases_by_qid=cases,
        selected_qids=["q1", "q2"],
        gold_pages_by_qid=gold_pages,
        page_text_for=lambda page_id: page_text.get(page_id),
    )

    assert payload is not None
    assert payload.verdict == "real signal"
    assert payload.evaluated_cases == 2
    assert payload.pairwise_comparisons == 5
    assert payload.gold_beats_false_positive_rate == 1.0
    assert payload.mean_gold_overlap > payload.mean_false_positive_overlap
    assert payload.rows[0].signal_source == "answer"


def test_build_answer_to_page_attribution_signal_uses_question_fallback_for_boolean() -> None:
    cases = {
        "q1": mod.CandidateCase(
            "q1",
            "Do cases CA 004/2025 and SCT 295/2025 involve any of the same parties?",
            "boolean",
            "No",
            {"used_page_ids": ["ca_2", "ca_3", "sct_7"]},
            10.0,
        )
    }
    gold_pages = {"q1": ["ca_1", "sct_1"]}
    page_text = {
        "ca_1": "Case No CA 004/2025 between MR ORAN and OAKEN on the title page.",
        "sct_1": "Claim No SCT 295/2025 between OLEXA and ODON on the title page.",
        "ca_2": "Procedural timetable and costs order.",
        "ca_3": "Interest and service directions only.",
        "sct_7": "Release agreement analysis and compensation discussion only.",
    }

    payload = mod._build_answer_to_page_attribution_signal(
        cases_by_qid=cases,
        selected_qids=["q1"],
        gold_pages_by_qid=gold_pages,
        page_text_for=lambda page_id: page_text.get(page_id),
    )

    assert payload is not None
    assert payload.verdict == "real signal"
    assert payload.evaluated_cases == 1
    assert payload.pairwise_comparisons == 6
    assert payload.rows[0].signal_source == "question+answer"
    assert "004/2025" in payload.rows[0].signal_terms
    assert "295/2025" in payload.rows[0].signal_terms


@pytest.mark.asyncio
async def test_async_main_writes_attribution_falsifier_when_benchmark_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions = tmp_path / "questions.json"
    baseline_raw = tmp_path / "baseline.json"
    candidate_raw = tmp_path / "candidate.json"
    benchmark = tmp_path / "benchmark.json"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    out_dir = tmp_path / "out"

    _write_json(
        questions,
        [{"id": "q1", "question": "Q1?", "answer_type": "name"}],
    )
    _write_json(
        baseline_raw,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "name"},
                "answer_text": "ENF 316/2023",
                "telemetry": {"ttft_ms": 10, "used_page_ids": ["doc_bad_1"], "context_chunk_ids": [], "cited_chunk_ids": []},
            },
        ],
    )
    _write_json(
        candidate_raw,
        [
            {
                "case": {"case_id": "q1", "question": "Q1?", "answer_type": "name"},
                "answer_text": "ENF 316/2023",
                "telemetry": {"ttft_ms": 10, "used_page_ids": ["doc_bad_1", "doc_good_2"], "context_chunk_ids": [], "cited_chunk_ids": []},
            },
        ],
    )
    _write_json(
        benchmark,
        {
            "cases": [
                {
                    "question_id": "q1",
                    "trust_tier": "trusted",
                    "gold_page_ids": ["doc_good_2"],
                }
            ]
        },
    )

    class _FakePdfPageTextProvider:
        def __init__(self, docs_dir, *, max_chars_per_page) -> None:  # type: ignore[no-untyped-def]
            del docs_dir, max_chars_per_page

        def get_page_text(self, *, doc_id: str, page: int) -> str | None:
            mapping = {
                ("doc_bad", 1): "Generic background only.",
                ("doc_good", 2): "The originating matter was ENF 316/2023 before the appeal.",
            }
            return mapping.get((doc_id, page))

        def close(self) -> None:
            return None

    monkeypatch.setattr(mod, "PdfPageTextProvider", _FakePdfPageTextProvider)
    monkeypatch.setattr(mod, "get_settings", lambda: type("Settings", (), {"judge": type("Judge", (), {"enabled": False, "sources_max_chars_per_page": 20000, "sources_max_pages": 12})()})())

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
            "case_scope": "all",
            "judge_scope": "none",
            "include_qids_file": None,
            "page_benchmark": benchmark,
        },
    )()

    await mod._async_main(namespace)

    payload = json.loads((out_dir / "candidate_debug_compare_candidate_y_vs_baseline_x.json").read_text(encoding="utf-8"))
    assert payload["baseline_answer_to_page_signal"]["verdict"] == "noise"
    assert payload["baseline_answer_to_page_signal"]["evaluated_cases"] == 1
    assert payload["candidate_answer_to_page_signal"]["verdict"] == "noise"
    assert payload["candidate_answer_to_page_signal"]["evaluated_cases"] == 1
    assert payload["candidate_answer_to_page_signal"]["cases"][0]["signal_source"] == "answer"


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
