from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.audit_exactness_candidate import _build_exactness_rows, _incorrect_scaffold_qids, _render_markdown

if TYPE_CHECKING:
    from pathlib import Path


def test_build_exactness_rows_marks_expected_matches() -> None:
    baseline = {
        "q1": {"question_id": "q1", "answer": "Wrong LLC"},
        "q2": {"question_id": "q2", "answer": "Old"},
    }
    candidate = {
        "q1": {"question_id": "q1", "answer": "Architeriors Interior Design (L.L.C)"},
        "q2": {"question_id": "q2", "answer": "Still wrong"},
    }
    scaffold = {
        "q1": {
            "question_id": "q1",
            "manual_verdict": "incorrect",
            "expected_answer": ["Architeriors Interior Design (L.L.C)"],
            "manual_exactness_labels": ["suffix_risk"],
            "failure_class": "wrong_strict_extraction",
        },
        "q2": {
            "question_id": "q2",
            "manual_verdict": "incorrect",
            "expected_answer": "Expected",
            "manual_exactness_labels": ["platform_exact_risk"],
            "failure_class": "weak_path_fallback",
        },
    }

    rows = _build_exactness_rows(
        changed_qids=["q1", "q2"],
        baseline_submission=baseline,
        candidate_submission=candidate,
        scaffold_records=scaffold,
    )

    assert rows[0].candidate_matches_expected is True
    assert rows[0].baseline_matches_expected is False
    assert rows[1].candidate_matches_expected is False
    assert rows[1].expected_answers == ["Expected"]


def test_render_markdown_includes_changed_case_table(tmp_path: Path) -> None:
    rows = _build_exactness_rows(
        changed_qids=["q1"],
        baseline_submission={"q1": {"question_id": "q1", "answer": "Wrong LLC"}},
        candidate_submission={"q1": {"question_id": "q1", "answer": "Architeriors Interior Design (L.L.C)"}},
        scaffold_records={
            "q1": {
                "question_id": "q1",
                "manual_verdict": "incorrect",
                "expected_answer": ["Architeriors Interior Design (L.L.C)"],
                "manual_exactness_labels": ["suffix_risk"],
                "failure_class": "wrong_strict_extraction",
            }
        },
    )
    debug_eval = {
        "summary": {
            "citation_coverage": 1.0,
            "answer_type_format_compliance": 1.0,
            "judge": {"pass_rate": 1.0, "avg_grounding": 5.0},
        },
        "cases": [
            {
                "question_id": "q1",
                "judge": {"verdict": "PASS", "scores": {"grounding": 5}},
            }
        ],
    }

    markdown = _render_markdown(
        label="candidate",
        baseline_label="baseline",
        rows=rows,
        debug_eval=debug_eval,
    )

    assert "`q1`" in markdown
    assert "`PASS`" in markdown
    assert "Architeriors Interior Design (L.L.C)" in markdown


def test_payload_shape_for_rows_is_json_serializable(tmp_path: Path) -> None:
    payload = {
        "baseline_label": "baseline",
        "candidate_label": "candidate",
        "answer_changed_qids": ["q1"],
        "resolved_incorrect_qids": ["q1"],
        "still_mismatched_incorrect_qids": [],
        "rows": [
            {
                "question_id": "q1",
                "manual_verdict": "incorrect",
                "expected_answers": ["Expected"],
                "baseline_answer": "Wrong",
                "candidate_answer": "Expected",
                "baseline_matches_expected": False,
                "candidate_matches_expected": True,
                "labels": ["suffix_risk"],
                "failure_class": "wrong_strict_extraction",
            }
        ],
        "debug_eval_path": None,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    out = tmp_path / "payload.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = json.loads(out.read_text(encoding="utf-8"))
    assert restored["resolved_incorrect_qids"] == ["q1"]


def test_incorrect_scaffold_qids_selects_all_incorrect_cases_present_in_both_submissions() -> None:
    baseline = {
        "q1": {"question_id": "q1", "answer": "Wrong"},
        "q2": {"question_id": "q2", "answer": "Still wrong"},
        "q3": {"question_id": "q3", "answer": "Correct"},
    }
    candidate = {
        "q1": {"question_id": "q1", "answer": "Expected"},
        "q2": {"question_id": "q2", "answer": "Still wrong"},
        "q3": {"question_id": "q3", "answer": "Correct"},
    }
    scaffold = {
        "q1": {"question_id": "q1", "manual_verdict": "incorrect"},
        "q2": {"question_id": "q2", "manual_verdict": "incorrect"},
        "q3": {"question_id": "q3", "manual_verdict": "correct"},
        "q4": {"question_id": "q4", "manual_verdict": "incorrect"},
    }

    assert _incorrect_scaffold_qids(baseline, candidate, scaffold) == ["q1", "q2"]
