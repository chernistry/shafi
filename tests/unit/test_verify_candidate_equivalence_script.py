from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _submission_payload(*, q1_answer: object, q2_answer: object, q1_pages: list[dict[str, object]], q2_pages: list[dict[str, object]], q1_used: list[str], q2_used: list[str]) -> dict[str, object]:
    return {
        "answers": [
            {
                "question_id": "q1",
                "answer": q1_answer,
                "telemetry": {
                    "used_page_ids": q1_used,
                    "retrieval": {"retrieved_chunk_pages": q1_pages},
                },
            },
            {
                "question_id": "q2",
                "answer": q2_answer,
                "telemetry": {
                    "used_page_ids": q2_used,
                    "retrieval": {"retrieved_chunk_pages": q2_pages},
                },
            },
        ]
    }


def test_verify_candidate_equivalence_reports_safe_and_unsafe_baselines(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline_safe = tmp_path / "baseline_safe.json"
    baseline_unsafe = tmp_path / "baseline_unsafe.json"
    report = tmp_path / "report.md"
    json_out = tmp_path / "report.json"

    baseline_safe.write_text(
        json.dumps(
            _submission_payload(
                q1_answer=["old dotted"],
                q2_answer="alpha",
                q1_pages=[{"doc_id": "doc1", "page_numbers": [1]}],
                q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
                q1_used=["doc1_1"],
                q2_used=["doc2_2"],
            )
        ),
        encoding="utf-8",
    )
    baseline_unsafe.write_text(
        json.dumps(
            _submission_payload(
                q1_answer=["other baseline"],
                q2_answer="alpha",
                q1_pages=[{"doc_id": "doc1", "page_numbers": [3]}],
                q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
                q1_used=["doc1_3"],
                q2_used=["doc2_2"],
            )
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            _submission_payload(
                q1_answer=["new dotted"],
                q2_answer="alpha",
                q1_pages=[{"doc_id": "doc1", "page_numbers": [1]}],
                q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
                q1_used=["doc1_1"],
                q2_used=["doc2_2"],
            )
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_equivalence.py",
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline_safe),
            "--baseline",
            str(baseline_unsafe),
            "--allowed-answer-id",
            "q1",
            "--practical-public-state",
            "public_0.74156_state",
            "--out",
            str(report),
            "--json-out",
            str(json_out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report_text = report.read_text(encoding="utf-8")
    assert "Candidate Lineage And Equivalence Report" in report_text
    assert "public_0.74156_state" in report_text
    assert "baseline_safe.json" in report_text
    assert "- verdict: `lineage_safe_exactness_candidate`" in report_text
    assert "baseline_unsafe.json" in report_text
    assert "- verdict: `lineage_unsafe_candidate`" in report_text
    assert "- page_drift_count: `1`" in report_text

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    assert summary["safe_baselines"] == [str(baseline_safe)]
    unsafe_comparison = next(item for item in summary["comparisons"] if item["baseline_path"] == str(baseline_unsafe))
    assert unsafe_comparison["page_drift_ids"] == ["q1"]


def test_verify_candidate_equivalence_flags_missing_allowed_answer_ids(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    json_out = tmp_path / "report.json"

    payload = _submission_payload(
        q1_answer=["same"],
        q2_answer="same",
        q1_pages=[{"doc_id": "doc1", "page_numbers": [1]}],
        q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
        q1_used=["doc1_1"],
        q2_used=["doc2_2"],
    )
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_equivalence.py",
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--allowed-answer-id",
            "q1",
            "--json-out",
            str(json_out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    comparison = summary["comparisons"][0]
    assert comparison["missing_allowed_answer_ids"] == ["q1"]
    assert comparison["lineage_safe"] is False
