from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_audit_named_ref_public_pack_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pack_json = tmp_path / "public_pack.json"
    out_json = tmp_path / "audit.json"
    out_md = tmp_path / "audit.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_public_legal_synthetic_pack.py",
            "--out-json",
            str(pack_json),
        ],
        check=True,
        cwd=repo_root,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_named_ref_public_pack.py",
            "--pack-json",
            str(pack_json),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--chunk-size",
            "120",
            "--chunk-overlap",
            "20",
            "--top-k-per-query",
            "1",
        ],
        check=True,
        cwd=repo_root,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["runnable_case_count"] == 5
    assert payload["skipped_case_count"] == 4
    assert payload["overall_verdict"] in {"ACTIONABLE_MISS_FOUND", "TRANSFER_CONFIDENCE_ONLY"}
    assert len(payload["cases"]) == 5
    assert "Named-Ref Public Pack Audit" in out_md.read_text(encoding="utf-8")
