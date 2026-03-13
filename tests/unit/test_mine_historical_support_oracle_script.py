from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _write_submission(path: Path, *, qid: str, answer: str, doc_id: str, pages: list[int]) -> None:
    payload = {
        "answers": [
            {
                "question_id": qid,
                "answer": answer,
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": doc_id, "page_numbers": pages}
                        ]
                    }
                },
            }
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_mine_historical_support_oracle_finds_better_gold_pages(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    baseline = tmp_path / "submission_v_base.json"
    donor = tmp_path / "submission_v_donor.json"
    worse = tmp_path / "submission_v_worse.json"
    out_json = tmp_path / "oracle.json"
    out_md = tmp_path / "oracle.md"

    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "According to page 2, what happened?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "minimal_required_support_pages": ["docA_2"],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_submission(baseline, qid="q1", answer="same", doc_id="docA", pages=[1])
    _write_submission(donor, qid="q1", answer="same", doc_id="docA", pages=[2])
    _write_submission(worse, qid="q1", answer="diff", doc_id="docA", pages=[1, 3])

    subprocess.run(
        [
            "python",
            "scripts/mine_historical_support_oracle.py",
            "--scaffold",
            str(scaffold),
            "--baseline-submission",
            str(baseline),
            "--submissions-dir",
            str(tmp_path),
            "--manual-verdict",
            "correct",
            "--failure-class",
            "support_undercoverage",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["opportunity_count"] == 1
    opp = payload["opportunities"][0]
    assert opp["question_id"] == "q1"
    assert opp["source_submission"] == donor.name
    assert opp["answer_same_as_baseline"] is True
    assert opp["gold_hit_gain"] == 1
    assert opp["exact_gold_recovered"] is True


def test_mine_historical_support_oracle_sorts_answer_same_before_different_when_gain_ties(tmp_path: Path) -> None:
    scaffold = tmp_path / "scaffold.json"
    baseline = tmp_path / "submission_v_base.json"
    donor_same = tmp_path / "submission_v_same.json"
    donor_diff = tmp_path / "submission_v_diff.json"
    out_json = tmp_path / "oracle.json"
    out_md = tmp_path / "oracle.md"

    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "question": "According to page 2, what happened?",
                        "manual_verdict": "correct",
                        "failure_class": "support_undercoverage",
                        "minimal_required_support_pages": ["docA_2"],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_submission(baseline, qid="q1", answer="same", doc_id="docA", pages=[1])
    _write_submission(donor_same, qid="q1", answer="same", doc_id="docA", pages=[2, 4])
    _write_submission(donor_diff, qid="q1", answer="different", doc_id="docA", pages=[2, 5])

    subprocess.run(
        [
            "python",
            "scripts/mine_historical_support_oracle.py",
            "--scaffold",
            str(scaffold),
            "--baseline-submission",
            str(baseline),
            "--submissions-dir",
            str(tmp_path),
            "--manual-verdict",
            "correct",
            "--failure-class",
            "support_undercoverage",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["opportunity_count"] == 2
    assert payload["opportunities"][0]["source_submission"] == donor_same.name
