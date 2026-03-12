from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _raw_case(*, qid: str, used: list[str], context: list[str] | None = None) -> dict[str, object]:
    return {
        "case": {"case_id": qid, "question": qid, "answer_type": "free_text"},
        "answer_text": "answer",
        "telemetry": {
            "used_page_ids": used,
            "context_page_ids": context or used,
        },
    }


def test_analyze_benchmark_delta_reports_case_level_improvements_and_regressions(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    report_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc_1", "doc_2"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                    },
                    {
                        "question_id": "q2",
                        "trust_tier": "suspect",
                        "gold_origin": "seeded_eval",
                        "gold_page_ids": ["doc_9"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                _raw_case(qid="q1", used=["doc_1"]),
                _raw_case(qid="q2", used=["doc_9"]),
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                _raw_case(qid="q1", used=["doc_1", "doc_2"]),
                _raw_case(qid="q2", used=[]),
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_benchmark_delta.py",
            "--baseline-label",
            "baseline",
            "--candidate-label",
            "candidate",
            "--baseline-raw-results",
            str(baseline_raw),
            "--candidate-raw-results",
            str(candidate_raw),
            "--benchmark",
            str(benchmark_path),
            "--out",
            str(report_path),
            "--json-out",
            str(json_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "trusted: cases=`1`, baseline=`0.5370`, candidate=`1.0000`, delta=`+0.4630`, improved=`1`, regressed=`0`, unchanged=`0`" in report
    assert "suspect: cases=`1`, baseline=`1.0000`, candidate=`0.0000`, delta=`-1.0000`, improved=`0`, regressed=`1`, unchanged=`0`" in report
    assert "## Trusted Improvements" in report
    assert "`q1`: delta=+0.4630" in report
    assert "## Top Regressions" in report
    assert "`q2`: delta=-1.0000" in report

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["trusted"]["improved"] == 1
    assert payload["summary"]["suspect"]["regressed"] == 1
    assert payload["deltas"][0]["question_id"] == "q1"
