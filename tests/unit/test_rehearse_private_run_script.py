from __future__ import annotations

from pathlib import Path

from scripts import analyze_answer_drift as drift_mod
from scripts import rehearse_private_run as mod


def test_rewrite_host_qdrant_url_rewrites_docker_hostname() -> None:
    rewritten, reason = mod._rewrite_host_qdrant_url("http://qdrant:6333")
    assert rewritten == "http://localhost:6333"
    assert reason == "rewrote_qdrant_hostname_for_host_shell"


def test_build_env_sets_placeholder_eval_key_and_rewrites_qdrant() -> None:
    env, notes = mod._build_env({"QDRANT_URL": "http://qdrant:6333"}, None)
    assert env["QDRANT_URL"] == "http://localhost:6333"
    assert env["EVAL_API_KEY"] == "local-rehearsal-placeholder"
    assert "rewrote_qdrant_hostname_for_host_shell" in notes
    assert "set_placeholder_eval_api_key_for_local_rehearsal" in notes


def test_build_env_defaults_qdrant_to_localhost_for_host_rehearsal() -> None:
    env, notes = mod._build_env({}, None)
    assert env["QDRANT_URL"] == "http://localhost:6333"
    assert "defaulted_qdrant_url_to_localhost_for_host_rehearsal" in notes


def test_build_equivalence_canary_detects_answer_and_page_drift() -> None:
    baseline = [
        {
            "case": {"case_id": "q1"},
            "answer_text": "null",
            "telemetry": {"model_llm": "strict-extractor", "used_page_ids": []},
        }
    ]
    candidate = [
        {
            "case": {"case_id": "q1"},
            "answer_text": "Yes",
            "telemetry": {"model_llm": "strict-extractor", "used_page_ids": ["doc_1"]},
        }
    ]

    canary = mod._build_equivalence_canary(baseline_rows=baseline, candidate_rows=candidate)

    assert canary["answer_drift_count"] == 1
    assert canary["page_drift_count"] == 1
    assert canary["model_drift_count"] == 0
    assert canary["stable"] is False


def test_copy_artifacts_copies_expected_files(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    source_dir = tmp_path / "warmup"
    source_dir.mkdir(parents=True)
    monkeypatch.setattr(mod, "WARMUP_DIR", source_dir)
    for path in mod._artifact_paths("sample").values():
        path.write_text("{}", encoding="utf-8")

    copied = mod._copy_artifacts(suffix="sample", out_dir=tmp_path / "out")

    assert set(copied) == {
        "raw_results",
        "submission",
        "preflight_summary",
        "truth_audit",
        "truth_audit_workbook",
        "code_archive",
        "code_archive_audit",
    }
    assert all((tmp_path / "out" / Path(path).name).exists() for path in copied.values())


def test_runtime_recommendation_requires_higher_concurrency_evidence() -> None:
    assert (
        drift_mod._runtime_recommendation(
            baseline_concurrency=1,
            candidate_concurrency=1,
            answer_drift_count=0,
            page_drift_count=0,
            model_drift_count=0,
            missing_case_count=0,
        )
        == "query_concurrency=1_stable_only"
    )
    assert (
        drift_mod._runtime_recommendation(
            baseline_concurrency=1,
            candidate_concurrency=2,
            answer_drift_count=0,
            page_drift_count=0,
            model_drift_count=0,
            missing_case_count=0,
        )
        == "query_concurrency>1_allowed"
    )
    assert (
        drift_mod._runtime_recommendation(
            baseline_concurrency=1,
            candidate_concurrency=2,
            answer_drift_count=1,
            page_drift_count=0,
            model_drift_count=0,
            missing_case_count=0,
        )
        == "query_concurrency=1"
    )
