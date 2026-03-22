from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.search_exactness_rider_subsets import _candidate_qids, _combined_score, _subset_label

if TYPE_CHECKING:
    from pathlib import Path


def _submission(*, answers: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "architecture_summary": {},
        "answers": [
            {
                "question_id": qid,
                "answer": answer,
                "telemetry": {"retrieval": {"retrieved_chunk_pages": []}},
            }
            for qid, answer in answers
        ],
    }


def test_candidate_qids_uses_incorrect_scaffold_and_answer_diffs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    rider = tmp_path / "rider.json"
    scaffold = tmp_path / "scaffold.json"

    baseline.write_text(
        json.dumps(_submission(answers=[("43f77abc", "LLC"), ("f950xyz", "BSC"), ("33060ok", "Confirmation Statement")])),
        encoding="utf-8",
    )
    rider.write_text(
        json.dumps(
            _submission(
                answers=[
                    ("43f77abc", "L.L.C"),
                    ("f950xyz", "B.S.C. (C)"),
                    ("33060ok", "Confirmation Statement"),
                ]
            )
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "43f77abc", "manual_verdict": "incorrect"},
                    {"question_id": "f950xyz", "manual_verdict": "incorrect"},
                    {"question_id": "33060ok", "manual_verdict": "correct"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _candidate_qids(
        baseline_submission=baseline,
        rider_source_submission=rider,
        scaffold=scaffold,
    ) == ["43f77abc", "f950xyz"]


def test_combined_score_prefers_hidden_g_then_exactness_then_lower_drift() -> None:
    base = {
        "label": "plus-dotted",
        "lineage_ok": True,
        "recommendation": "PROMISING",
        "hidden_g_trusted_delta": 0.0425,
        "hidden_g_all_delta": 0.0206,
        "resolved_incorrect_count": 2,
        "judge_pass_delta": 0.0,
        "judge_grounding_delta": 0.0,
        "answer_drift": 2,
        "page_drift": 4,
    }
    weaker_hidden_g = dict(base)
    weaker_hidden_g["label"] = "5046-only"
    weaker_hidden_g["hidden_g_all_delta"] = 0.0

    assert _combined_score(base) > _combined_score(weaker_hidden_g)

    more_drift = dict(base)
    more_drift["label"] = "plus-dotted-dd736"
    more_drift["page_drift"] = 5

    assert _combined_score(base) > _combined_score(more_drift)


def test_subset_label_uses_qid_prefixes() -> None:
    assert _subset_label("triad_f331_e0798", ["43f77ed8xxxx", "f950917fyyyy"]) == "triad_f331_e0798_plus_43f77e_f95091"
