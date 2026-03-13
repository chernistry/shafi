from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

from scripts.build_investigated_qid_universe import build_investigated_qid_universe

if TYPE_CHECKING:
    from pathlib import Path


def test_build_investigated_qid_universe_unions_answer_and_page_qids(tmp_path: Path) -> None:
    manifest_a = tmp_path / "manifest_a.json"
    manifest_b = tmp_path / "manifest_b.json"
    manifest_a.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "label": "a",
                        "allowed_answer_qids": ["q1"],
                        "allowed_page_qids": ["q2", "q3"],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_b.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "label": "b",
                        "allowed_answer_qids": ["q3", "q4"],
                        "allowed_page_qids": ["q5"],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    qids = build_investigated_qid_universe([manifest_a, manifest_b])
    assert qids == ["q1", "q2", "q3", "q4", "q5"]


def test_build_investigated_qid_universe_accepts_extra_qids() -> None:
    qids = build_investigated_qid_universe([], extra_qids={"q9", "q8"})
    assert qids == ["q8", "q9"]


def test_build_investigated_qid_universe_script_writes_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "label": "a",
                        "allowed_answer_qids": ["q1"],
                        "allowed_page_qids": ["q2"],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "qids.txt"
    json_out = tmp_path / "qids.json"
    extra = tmp_path / "extra.txt"
    extra.write_text("q3\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_investigated_qid_universe.py",
            "--manifest-json",
            str(manifest),
            "--extra-qids-file",
            str(extra),
            "--out",
            str(out),
            "--json-out",
            str(json_out),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.read_text(encoding="utf-8") == "q1\nq2\nq3\n"
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["count"] == 3
    assert payload["extra_qids_file"] == str(extra)
