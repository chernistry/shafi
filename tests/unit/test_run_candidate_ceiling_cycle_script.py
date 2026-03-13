from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_leaderboard import LeaderboardRow
from scripts.impact_router import route_changed_files
from scripts.run_candidate_ceiling_cycle import (
    CandidateSpec,
    _apply_impact_router,
    _candidate_score_estimates,
    _combined_score,
    _load_manifest,
)


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
                        "changed_files": ["src/rag_challenge/core/strict_answerer.py"],
                        "completed_packs": ["strict_answerer_pack"],
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
    assert rows[0].changed_files == ["src/rag_challenge/core/strict_answerer.py"]
    assert rows[0].completed_packs == ["strict_answerer_pack"]


def test_combined_score_prefers_lineage_and_exactness_when_hidden_g_ties() -> None:
    base = {
        "recommendation": "PROMISING",
        "lineage_ok": True,
        "paranoid_total_estimate": 0.7550,
        "strict_total_estimate": 0.7600,
        "upper_total_estimate": 0.7800,
        "blindspot_improved_case_count": 3,
        "blindspot_support_undercoverage_case_count": 2,
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
    weaker["blindspot_improved_case_count"] = 0
    weaker["blindspot_support_undercoverage_case_count"] = 0
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
    assert with_public_history["paranoid_total_estimate"] < with_public_history["strict_total_estimate"]
    assert with_public_history["paranoid_rank_estimate"] >= with_public_history["strict_rank_estimate"]


def test_combined_score_prefers_higher_paranoid_total_when_other_metrics_tie() -> None:
    safer = {
        "recommendation": "PROMISING",
        "lineage_ok": True,
        "paranoid_total_estimate": 0.7520,
        "strict_total_estimate": 0.7600,
        "upper_total_estimate": 0.7800,
        "blindspot_improved_case_count": 3,
        "blindspot_support_undercoverage_case_count": 2,
        "hidden_g_trusted_delta": 0.0425,
        "hidden_g_all_delta": 0.0206,
        "judge_pass_delta": 1.0,
        "judge_grounding_delta": 5.0,
        "page_drift": 3,
        "answer_drift": 1,
        "resolved_incorrect_count": 2,
        "label": "safer",
    }
    riskier = dict(safer)
    riskier["paranoid_total_estimate"] = 0.7440
    riskier["label"] = "riskier"

    assert _combined_score(safer) > _combined_score(riskier)


def test_impact_router_routes_major_change_classes() -> None:
    strict = route_changed_files(["src/rag_challenge/core/strict_answerer.py"], completed_packs=["strict_answerer_pack"])
    retrieval = route_changed_files(["src/rag_challenge/core/retriever.py"], completed_packs=["page_localization_pack"])
    reingest = route_changed_files(["src/rag_challenge/ingestion/parser.py"], completed_packs=["reingest_ocr_pack"])
    free_text = route_changed_files(["src/rag_challenge/prompts/llm/generator_system_complex.md"], completed_packs=["free_text_pack"])
    config = route_changed_files(["src/rag_challenge/config/settings.py"], completed_packs=["full_regression_pack"])

    assert strict["required_packs"] == ["strict_answerer_pack"]
    assert retrieval["required_packs"] == ["page_localization_pack"]
    assert reingest["required_packs"] == ["reingest_ocr_pack"]
    assert free_text["required_packs"] == ["free_text_pack"]
    assert config["required_packs"] == ["full_regression_pack"]


def test_apply_impact_router_blocks_candidate_when_required_pack_missing() -> None:
    row = {"label": "cand-a", "recommendation": "PROMISING"}
    _apply_impact_router(
        CandidateSpec(
            label="cand-a",
            submission=Path("/tmp/submission.json"),
            raw_results=Path("/tmp/raw_results.json"),
            preflight=None,
            candidate_scaffold=None,
            allowed_answer_qids=[],
            allowed_page_qids=[],
            changed_files=["src/rag_challenge/core/retriever.py"],
            completed_packs=[],
        ),
        row,
    )

    assert row["impact_router_blocked"] is True
    assert row["required_packs"] == ["page_localization_pack"]
    assert row["missing_packs"] == ["page_localization_pack"]
    assert row["recommendation"] == "BLOCKED_MISSING_IMPACT_PACK"
