from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_public_legal_synthetic_pack_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_json = tmp_path / "public_pack.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_public_legal_synthetic_pack.py",
            "--out-json",
            str(out_json),
        ],
        check=True,
        cwd=repo_root,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["ticket"] == 80
    assert payload["summary"]["case_count"] == 9
    assert payload["summary"]["family_counts"] == {
        "compare": 1,
        "exact_reference": 2,
        "explicit_provision": 2,
        "ocr_risk": 2,
        "unsupported": 2,
    }
    assert payload["required_families"] == [
        "compare",
        "exact_reference",
        "explicit_provision",
        "ocr_risk",
        "unsupported",
    ]
