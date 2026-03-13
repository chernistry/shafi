from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_run_production_mimic_eval_script_writes_ranked_output(tmp_path: Path) -> None:
    leaderboard = tmp_path / "leaderboard.csv"
    candidate_cycle = tmp_path / "cycle.json"
    exactness = tmp_path / "exactness.json"
    equivalence = tmp_path / "equivalence.json"
    history = tmp_path / "history.json"
    out_json = tmp_path / "production_mimic.json"
    out_md = tmp_path / "production_mimic.md"

    leaderboard.write_text(
        '"Rank","Team name","Total score","Det","Asst","G","T","F","Latency","Submissions","Last submission"\n'
        '"1","Leader","0.860000","1","0.70","0.920000","0.996","1.05","100","6","2026-03-12T10:00:00"\n'
        '"8","Tzur Labs","0.741560","0.971429","0.693333","0.800729","0.996","1.0471","347","9","2026-03-12T14:56:17"\n',
        encoding="utf-8",
    )
    candidate_cycle.write_text(
        json.dumps(
            {
                "ranked_candidates": [
                    {
                        "label": "triad_f331_e0798_plus_dotted",
                        "branch_class": "combined_small_diff_ceiling",
                        "strict_total_estimate": 0.7800,
                        "upper_total_estimate": 0.8000,
                        "paranoid_total_estimate": 0.7700,
                        "hidden_g_trusted_delta": 0.0425,
                        "lineage_ok": True,
                        "page_drift": 4,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    exactness.write_text(
        json.dumps(
            {
                "resolved_incorrect_qids": ["43f77", "f950"],
                "still_mismatched_incorrect_qids": [],
            }
        ),
        encoding="utf-8",
    )
    equivalence.write_text(json.dumps({"safe_baselines": ["/tmp/submission_v6_context_seed.json"]}), encoding="utf-8")
    history.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "external_total": 0.72,
                        "strict_total_estimate": 0.74,
                        "paranoid_total_estimate": 0.73,
                        "platform_like_total_estimate": 0.735,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_production_mimic_eval.py",
            "--leaderboard",
            str(leaderboard),
            "--team",
            "Tzur Labs",
            "--candidate-cycle-json",
            str(candidate_cycle),
            "--candidate-label",
            "triad_f331_e0798_plus_dotted",
            "--exactness-json",
            str(exactness),
            "--equivalence-json",
            str(equivalence),
            "--history-json",
            str(history),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    report = payload["production_mimic"]
    markdown = out_md.read_text(encoding="utf-8")

    assert report["lineage_confidence"] == "high"
    assert report["platform_like_rank_estimate"] >= 1
    assert report["strict_rank_estimate"] >= 1
    assert report["paranoid_rank_estimate"] >= 1
    assert "Production-Mimic Local Eval" in markdown
    assert "triad_f331_e0798_plus_dotted" in markdown
