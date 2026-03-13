from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_run_manifest_script_writes_deterministic_fingerprint(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    parser_file = repo_root / "src" / "rag_challenge" / "ingestion" / "parser.py"
    pipeline_file = repo_root / "src" / "rag_challenge" / "core" / "pipeline.py"
    touched_file = repo_root / "src" / "rag_challenge" / "core" / "retriever.py"
    raw_results = repo_root / "artifacts" / "raw_results.json"
    preflight = repo_root / "artifacts" / "preflight.json"
    out_json = tmp_path / "run_manifest.json"
    out_md = tmp_path / "run_manifest.md"

    parser_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)
    touched_file.parent.mkdir(parents=True, exist_ok=True)
    raw_results.parent.mkdir(parents=True, exist_ok=True)

    parser_file.write_text("def parse() -> str:\n    return 'parser'\n", encoding="utf-8")
    pipeline_file.write_text("def run() -> str:\n    return 'pipeline'\n", encoding="utf-8")
    touched_file.write_text("def retrieve() -> str:\n    return 'retriever'\n", encoding="utf-8")
    raw_results.write_text(
        json.dumps(
            [
                {"telemetry": {"route": "strict", "model_name": "gpt-4o-mini"}},
                {"telemetry": {"route": "strict", "model_name": "gpt-4o-mini"}},
                {"telemetry": {"route": "free_text", "model_llm": "claude-3-5-sonnet-latest"}},
            ]
        ),
        encoding="utf-8",
    )
    preflight.write_text(
        json.dumps(
            {
                "submission_sha256": "submission123",
                "code_archive_sha256": "code456",
                "questions_sha256": "questions789",
                "documents_zip_sha256": "documents999",
                "phase_collection_name": "legal-rag-phase2",
                "qdrant_point_count": 4242,
                "pdf_count": 312,
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["SAFE_TEST_ENV"] = "manifest-safe-value"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_run_manifest.py"),
            "--candidate-label",
            "triad_f331_e0798_plus_dotted",
            "--repo-root",
            str(repo_root),
            "--git-sha",
            "abcdef1234567890",
            "--preflight-json",
            str(preflight),
            "--raw-results-json",
            str(raw_results),
            "--touched-file",
            str(touched_file.relative_to(repo_root)),
            "--core-file",
            str(parser_file.relative_to(repo_root)),
            "--core-file",
            str(pipeline_file.relative_to(repo_root)),
            "--model-version",
            "llm.strict_model=gpt-4o-mini",
            "--model-version",
            "rerank.primary_model=zerank-2",
            "--env-key",
            "SAFE_TEST_ENV",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    manifest = payload["run_manifest"]
    markdown = out_md.read_text(encoding="utf-8")

    assert manifest["candidate_label"] == "triad_f331_e0798_plus_dotted"
    assert manifest["fingerprint"]
    assert manifest["git"]["sha"] == "abcdef1234567890"
    assert manifest["touched_file_hashes"] == [
        {
            "path": "src/rag_challenge/core/retriever.py",
            "exists": True,
            "sha256": manifest["touched_file_hashes"][0]["sha256"],
            "size_bytes": touched_file.stat().st_size,
        }
    ]
    assert manifest["core_pipeline_hashes"][0]["path"] == "src/rag_challenge/core/pipeline.py"
    assert manifest["models"]["declared_versions"] == {
        "llm.strict_model": "gpt-4o-mini",
        "rerank.primary_model": "zerank-2",
    }
    assert manifest["models"]["observed_models"] == [
        {"name": "gpt-4o-mini", "count": 2},
        {"name": "claude-3-5-sonnet-latest", "count": 1},
    ]
    assert manifest["qdrant"]["collection_name"] == "legal-rag-phase2"
    assert manifest["qdrant"]["point_count"] == 4242
    assert manifest["environment"]["selected_env"] == [
        {
            "name": "SAFE_TEST_ENV",
            "is_set": True,
            "value_sha256": manifest["environment"]["selected_env"][0]["value_sha256"],
        }
    ]
    assert "Run Manifest" in markdown
    assert "triad_f331_e0798_plus_dotted" in markdown
