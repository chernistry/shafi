from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_legalbenchrag_mini_bootstrap_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests" / "fixtures" / "legalbenchrag_mini_bootstrap"
    out_json = tmp_path / "bootstrap.json"
    out_md = tmp_path / "bootstrap.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_legalbenchrag_mini_bootstrap.py",
            "--benchmark-json",
            str(fixture_root / "benchmarks" / "legalbenchrag_mini.json"),
            "--corpus-dir",
            str(fixture_root / "corpus"),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--max-tests",
            "2",
            "--chunk-size",
            "80",
            "--chunk-overlap",
            "0",
            "--top-k",
            "1",
        ],
        check=True,
        cwd=repo_root,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["case_count"] == 2
    assert payload["doc_count"] == 2
    assert payload["avg_recall"] == 1.0
    assert payload["avg_precision"] > 0.75
    assert len(payload["cases"]) == 2
    assert "LegalBench-RAG Mini Bootstrap" in out_md.read_text(encoding="utf-8")
