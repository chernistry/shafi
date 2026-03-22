from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _submission(*, answer_qids: dict[str, object], page_qids: dict[str, list[int]]) -> dict[str, object]:
    answers: list[dict[str, object]] = []
    for qid, answer in answer_qids.items():
        answers.append(
            {
                "question_id": qid,
                "answer": answer,
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": f"{qid}_doc", "page_numbers": page_qids[qid]},
                        ]
                    }
                },
            }
        )
    return {"answers": answers}


def test_verify_candidate_lineage_accepts_allowed_qids(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    baseline.write_text(
        json.dumps(_submission(answer_qids={"q1": "A", "q2": "B"}, page_qids={"q1": [1], "q2": [1]})),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_submission(answer_qids={"q1": "A", "q2": "C"}, page_qids={"q1": [2], "q2": [1]})),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_lineage.py",
            "--baseline-submission",
            str(baseline),
            "--candidate-submission",
            str(candidate),
            "--allowed-answer-qid",
            "q2",
            "--allowed-page-qid",
            "q1",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lineage_ok"] is True
    assert payload["answer_changed_qids"] == ["q2"]
    assert payload["page_changed_qids"] == ["q1"]


def test_verify_candidate_lineage_flags_unexpected_qids(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    baseline.write_text(
        json.dumps(_submission(answer_qids={"q1": "A"}, page_qids={"q1": [1]})),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_submission(answer_qids={"q1": "B"}, page_qids={"q1": [2]})),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_lineage.py",
            "--baseline-submission",
            str(baseline),
            "--candidate-submission",
            str(candidate),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lineage_ok"] is False
    assert payload["unexpected_answer_qids"] == ["q1"]
    assert payload["unexpected_page_qids"] == ["q1"]


def test_verify_candidate_lineage_blocks_unsafe_baseline_when_champion_report_present(tmp_path: Path) -> None:
    champion = tmp_path / "champion.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    equivalence_json = tmp_path / "equivalence.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"

    champion.write_text(
        json.dumps(_submission(answer_qids={"q1": "A"}, page_qids={"q1": [1]})),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(_submission(answer_qids={"q1": "A"}, page_qids={"q1": [2]})),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_submission(answer_qids={"q1": "A"}, page_qids={"q1": [2]})),
        encoding="utf-8",
    )
    equivalence_json.write_text(
        json.dumps(
            {
                "practical_champion_label": "v6_context_seed",
                "practical_champion_submission": str(champion),
                "practical_champion_sha256": "ignored-for-test",
                "safe_baselines": [str(champion)],
                "safe_baseline_sha256": [],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/verify_candidate_lineage.py",
            "--baseline-submission",
            str(baseline),
            "--candidate-submission",
            str(candidate),
            "--champion-equivalence-json",
            str(equivalence_json),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["baseline_lineage_safe"] is False
    assert payload["lineage_ok"] is False
    assert payload["practical_champion_label"] == "v6_context_seed"
    assert "not proven equivalent" in payload["lineage_ambiguity_reason"]
