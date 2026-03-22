from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_calibrated_replay_candidate.py"
SPEC = importlib.util.spec_from_file_location("build_calibrated_replay_candidate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_artifact_bundle_requires_complete_triplet(tmp_path: Path) -> None:
    phase_dir = tmp_path / "platform_runs" / "warmup"
    phase_dir.mkdir(parents=True)
    (phase_dir / "submission_demo.json").write_text("{}", encoding="utf-8")
    (phase_dir / "raw_results_demo.json").write_text("[]", encoding="utf-8")

    try:
        MODULE.artifact_bundle(root=tmp_path, phase="warmup", suffix="demo")
    except FileNotFoundError as exc:
        assert "preflight_summary_demo.json" in str(exc)
    else:
        raise AssertionError("Expected missing preflight artifact to fail")


def test_build_replay_command_uses_expected_artifact_paths(tmp_path: Path) -> None:
    answer_bundle = {
        "submission": tmp_path / "submission_answer.json",
        "raw_results": tmp_path / "raw_answer.json",
        "preflight": tmp_path / "preflight_answer.json",
    }
    page_bundle = {
        "submission": tmp_path / "submission_page.json",
        "raw_results": tmp_path / "raw_page.json",
        "preflight": tmp_path / "preflight_page.json",
    }
    command = MODULE.build_replay_command(
        answer_bundle=answer_bundle,
        page_bundle=page_bundle,
        out_dir=tmp_path / "out",
        reviewed_all=tmp_path / "reviewed_all.json",
        reviewed_high=tmp_path / "reviewed_high.json",
        page_source_pages_default="all",
        answer_qids=["q1"],
        answer_qids_file=None,
        page_qids=["q2"],
        page_qids_file=None,
    )

    command_text = " ".join(str(part) for part in command)
    assert "run_answer_stable_grounding_replay.py" in command_text
    assert "--answer-source-submission" in command
    assert str(answer_bundle["submission"]) in command
    assert str(page_bundle["raw_results"]) in command
    assert "--page-source-answer-qid" in command
    assert "--page-source-page-qid" in command


def test_default_out_dir_nests_under_phase_replay_candidates(tmp_path: Path) -> None:
    out_dir = MODULE.default_out_dir(
        root=tmp_path,
        phase="final",
        answer_source_suffix="baseline",
        page_source_suffix="challenger",
    )
    assert out_dir == tmp_path / "platform_runs" / "final" / "replay_candidates" / "replay_final_answers_baseline__pages_challenger"
