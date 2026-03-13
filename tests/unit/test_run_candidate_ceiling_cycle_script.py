from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.analyze_leaderboard import LeaderboardRow
from scripts.run_candidate_ceiling_cycle import _candidate_score_estimates, _combined_score, _load_manifest

if TYPE_CHECKING:
    from pathlib import Path


def test_load_manifest_resolves_candidate_specs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    submission = tmp_path / "candidate_submission.json"
    raw_results = tmp_path / "candidate_raw_results.json"
    submission.write_text("{}", encoding="utf-8")
    raw_results.write_text("[]", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "label": "cand-a",
                        "submission": str(submission),
                        "raw_results": str(raw_results),
                        "allowed_answer_qids": ["q1"],
                        "allowed_page_qids": ["q2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = _load_manifest(manifest, root=tmp_path)
    assert len(rows) == 1
    assert rows[0].label == "cand-a"
    assert rows[0].submission == submission.resolve()
    assert rows[0].raw_results == raw_results.resolve()
    assert rows[0].allowed_answer_qids == ["q1"]
    assert rows[0].allowed_page_qids == ["q2"]


def test_combined_score_prefers_lineage_and_exactness_when_hidden_g_ties() -> None:
    base = {
        "recommendation": "PROMISING",
        "lineage_ok": True,
        "strict_total_estimate": 0.7600,
        "upper_total_estimate": 0.7800,
        "hidden_g_trusted_delta": 0.0425,
        "hidden_g_all_delta": 0.1993 - 0.1787,
        "judge_pass_delta": 1.0,
        "judge_grounding_delta": 5.0,
        "page_drift": 4,
        "answer_drift": 2,
        "resolved_incorrect_count": 2,
        "label": "plus-dotted",
    }
    weaker = dict(base)
    weaker["resolved_incorrect_count"] = 0
    weaker["label"] = "support-only"

    assert _combined_score(base) > _combined_score(weaker)


def test_candidate_score_estimates_use_public_realized_qids_when_provided() -> None:
    row = {
        "hidden_g_trusted_delta": 0.04,
        "hidden_g_all_delta": 0.02,
        "resolved_incorrect_qids": ["q_dotted", "q_new"],
    }
    subject_summary = {
        "total": 0.74156,
        "s": 0.888,
        "g": 0.800729,
        "t": 0.996,
        "f": 1.0471,
    }
    leaderboard_rows = [
        LeaderboardRow(
            rank=1,
            team_name="Leader",
            total=0.86,
            det=1.0,
            asst=0.66,
            g=0.92,
            t=0.996,
            f=1.05,
            latency_ms=80,
            submissions=6,
            last_submission="2026-03-12T14:02:54.255238",
        ),
        LeaderboardRow(
            rank=8,
            team_name="Tzur Labs",
            total=0.74156,
            det=0.971429,
            asst=0.693333,
            g=0.800729,
            t=0.996,
            f=1.0471,
            latency_ms=347,
            submissions=9,
            last_submission="2026-03-12T14:56:17.082289",
        ),
    ]

    no_public_history = _candidate_score_estimates(
        row=row,
        subject_summary=subject_summary,
        leaderboard_rows=leaderboard_rows,
        team_name="Tzur Labs",
        public_realized_exactness_qids=None,
    )
    with_public_history = _candidate_score_estimates(
        row=row,
        subject_summary=subject_summary,
        leaderboard_rows=leaderboard_rows,
        team_name="Tzur Labs",
        public_realized_exactness_qids={"q_dotted"},
    )

    assert no_public_history["strict_resolved_incorrect_count"] == 0
    assert no_public_history["upper_resolved_incorrect_count"] == 2
    assert with_public_history["strict_resolved_incorrect_count"] == 1
    assert with_public_history["upper_resolved_incorrect_count"] == 1
    assert with_public_history["strict_total_estimate"] < no_public_history["upper_total_estimate"]
