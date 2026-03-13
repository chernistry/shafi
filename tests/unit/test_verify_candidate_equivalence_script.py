from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

REPO_ROOT = "/Users/sasha/IdeaProjects/.codex-worktrees/rag_challenge-main"


def _submission_payload(
    *,
    q1_answer: object,
    q2_answer: object,
    q1_pages: list[dict[str, object]],
    q2_pages: list[dict[str, object]],
    q1_used: list[str],
    q2_used: list[str],
) -> dict[str, object]:
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


def test_verify_candidate_equivalence_reports_champion_and_safe_baselines(tmp_path: Path) -> None:
    champion = tmp_path / "champion.json"
    baseline_safe = tmp_path / "baseline_safe.json"
    baseline_unsafe = tmp_path / "baseline_unsafe.json"
    report = tmp_path / "report.md"
    json_out = tmp_path / "report.json"

    champion_payload = _submission_payload(
        q1_answer=["old dotted"],
        q2_answer="alpha",
        q1_pages=[{"doc_id": "doc1", "page_numbers": [1]}],
        q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
        q1_used=["doc1_1"],
        q2_used=["doc2_2"],
    )
    champion.write_text(json.dumps(champion_payload), encoding="utf-8")
    baseline_safe.write_text(json.dumps(champion_payload), encoding="utf-8")
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

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_equivalence.py",
            "--champion-label",
            "v6_context_seed",
            "--champion-submission",
            str(champion),
            "--baseline",
            str(champion),
            "--baseline",
            str(baseline_safe),
            "--baseline",
            str(baseline_unsafe),
            "--out-md",
            str(report),
            "--out-json",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report_text = report.read_text(encoding="utf-8")
    assert "Champion Equivalence Report" in report_text
    assert "v6_context_seed" in report_text
    assert "baseline_safe" in report_text
    assert "not_equivalent_to_champion" in report_text

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    assert summary["practical_champion_label"] == "v6_context_seed"
    assert str(champion) in summary["safe_baselines"]
    assert str(baseline_safe) in summary["safe_baselines"]
    assert str(baseline_unsafe) not in summary["safe_baselines"]


def test_verify_candidate_equivalence_flags_page_and_answer_drift(tmp_path: Path) -> None:
    champion = tmp_path / "champion.json"
    baseline = tmp_path / "baseline.json"
    json_out = tmp_path / "report.json"
    report = tmp_path / "report.md"

    champion.write_text(
        json.dumps(
            _submission_payload(
                q1_answer="same",
                q2_answer="same",
                q1_pages=[{"doc_id": "doc1", "page_numbers": [1]}],
                q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
                q1_used=["doc1_1"],
                q2_used=["doc2_2"],
            )
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            _submission_payload(
                q1_answer="changed",
                q2_answer="same",
                q1_pages=[{"doc_id": "doc1", "page_numbers": [9]}],
                q2_pages=[{"doc_id": "doc2", "page_numbers": [2]}],
                q1_used=["doc1_9"],
                q2_used=["doc2_2"],
            )
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_equivalence.py",
            "--champion-label",
            "v6_context_seed",
            "--champion-submission",
            str(champion),
            "--baseline",
            str(baseline),
            "--out-md",
            str(report),
            "--out-json",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(json_out.read_text(encoding="utf-8"))
    comparison = summary["comparisons"][0]
    assert comparison["equivalent_to_champion"] is False
    assert comparison["answer_changed_qids"] == ["q1"]
    assert comparison["page_changed_qids"] == ["q1"]
