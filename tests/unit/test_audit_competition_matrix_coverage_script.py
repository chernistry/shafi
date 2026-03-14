from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("audit_competition_matrix_coverage")


def test_build_audit_detects_missing_and_duplicates() -> None:
    module = _load_module()
    payload = {
        "rows": [
            {
                "label": "candidate_a",
                "status": "candidate",
                "platform_like_total_estimate": 0.75,
                "strict_total_estimate": 0.76,
                "paranoid_total_estimate": 0.74,
                "notes": "",
            },
            {
                "label": "candidate_b",
                "status": "rejected",
                "platform_like_total_estimate": 0.75,
                "strict_total_estimate": 0.76,
                "paranoid_total_estimate": 0.74,
                "notes": "",
            },
            {
                "label": "candidate_c",
                "status": "candidate",
                "platform_like_total_estimate": None,
                "strict_total_estimate": None,
                "paranoid_total_estimate": None,
                "notes": "",
            },
            {
                "label": "candidate_d",
                "status": "candidate",
                "platform_like_total_estimate": None,
                "strict_total_estimate": None,
                "paranoid_total_estimate": None,
                "notes": "[estimates=unsupported_local_envelope]",
            },
        ]
    }
    audit = module.build_audit(payload)
    assert audit["missing_supported_estimates"] == ["candidate_c"]
    assert audit["missing_unsupported_estimates"] == ["candidate_d"]
    duplicates = audit["duplicate_estimate_groups"]
    assert len(duplicates) == 1
    assert duplicates[0]["labels"] == ["candidate_a", "candidate_b"]


def test_script_writes_markdown_and_json(tmp_path: Path) -> None:
    matrix_json = tmp_path / "matrix.json"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"
    matrix_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "candidate_a",
                        "status": "candidate",
                        "platform_like_total_estimate": 0.75,
                        "strict_total_estimate": 0.76,
                        "paranoid_total_estimate": 0.74,
                        "notes": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_competition_matrix_coverage.py",
            "--matrix-json",
            str(matrix_json),
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
    audit = json.loads(out_json.read_text(encoding="utf-8"))
    assert audit["candidate_like_rows"] == 1
    assert "Competition Matrix Coverage Audit" in out_md.read_text(encoding="utf-8")
