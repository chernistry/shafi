from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_exactness_review_queue_script_prioritizes_page_specific_model_cases(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    out = tmp_path / "queue.md"
    out_json = tmp_path / "queue.json"
    out_qids = tmp_path / "queue.qids"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q-page",
                        "answer_type": "number",
                        "route_family": "model",
                        "question": "According to page 2, what is the claim number?",
                        "current_answer_text": "ENF 316/2023",
                        "expected_answer": "",
                        "manual_verdict": "",
                        "failure_class": "support_undercoverage",
                        "manual_exactness_labels": ["platform_exact_risk", "page_specific_exact_risk"],
                        "exactness_review_flags": ["no_exact_span_candidates"],
                        "support_shape_flags": ["metadata_title_anchor_maybe_missing"],
                        "required_page_anchor": {"page": 2},
                    },
                    {
                        "question_id": "q-name",
                        "answer_type": "names",
                        "route_family": "strict",
                        "question": "Who are the parties?",
                        "current_answer_text": "Alpha",
                        "expected_answer": "",
                        "manual_verdict": "correct",
                        "failure_class": "",
                        "manual_exactness_labels": [],
                        "exactness_review_flags": [],
                        "support_shape_flags": [],
                        "required_page_anchor": {},
                    },
                    {
                        "question_id": "q-free",
                        "answer_type": "free_text",
                        "route_family": "model",
                        "question": "Explain the reasoning.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_exactness_review_queue.py",
            "--scaffold",
            str(scaffold),
            "--limit",
            "10",
            "--out",
            str(out),
            "--out-json",
            str(out_json),
            "--out-qids-file",
            str(out_qids),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "Exactness Review Queue" in report
    assert "## q-page" in report
    assert "risk_score:" in report
    assert "platform_exact_risk, page_specific_exact_risk" in report
    assert "reasons:" in report
    assert "## q-name" in report
    assert "q-free" not in report
    assert payload["candidate_count"] == 2
    assert payload["candidates"][0]["question_id"] == "q-page"
    assert "required_page_anchor" in payload["candidates"][0]["reasons"]
    assert out_qids.read_text(encoding="utf-8").splitlines() == ["q-page", "q-name"]


def test_build_exactness_review_queue_script_downranks_reviewed_semantic_correct_cases(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    out = tmp_path / "queue.md"
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q-unresolved",
                        "answer_type": "name",
                        "route_family": "model",
                        "question": "According to page 2, what is the claim number?",
                        "current_answer_text": "ENF 316/2023",
                        "expected_answer": "",
                        "manual_verdict": "incorrect",
                        "failure_class": "weak_path_fallback",
                        "manual_exactness_labels": ["platform_exact_risk", "page_specific_exact_risk"],
                        "exactness_review_flags": ["required_page_anchor_missing_in_candidates"],
                        "support_shape_flags": [],
                        "required_page_anchor": {"page": 2},
                    },
                    {
                        "question_id": "q-reviewed",
                        "answer_type": "name",
                        "route_family": "model",
                        "question": "Who is the defendant?",
                        "current_answer_text": "ONORA",
                        "expected_answer": "ONORA",
                        "manual_verdict": "correct",
                        "failure_class": "",
                        "manual_exactness_labels": ["semantic_correct"],
                        "exactness_review_flags": [],
                        "support_shape_flags": [],
                        "required_page_anchor": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_exactness_review_queue.py",
            "--scaffold",
            str(scaffold),
            "--limit",
            "10",
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    report = out.read_text(encoding="utf-8")
    assert report.index("## q-unresolved") < report.index("## q-reviewed")
