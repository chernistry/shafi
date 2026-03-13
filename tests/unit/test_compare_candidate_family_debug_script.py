from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_candidate_family_debug import _parse_candidate, _render_markdown


def test_parse_candidate_resolves_label_and_path(tmp_path: Path) -> None:
    root = tmp_path
    path = root / "raw.json"
    path.write_text("[]", encoding="utf-8")

    candidate = _parse_candidate(f"demo={path}", root=root)

    assert candidate.label == "demo"
    assert candidate.raw_results == path.resolve()


def test_render_markdown_orders_rows_by_delta() -> None:
    rows = [
        {
            "label": "core",
            "judge_pass_rate": 0.5,
            "judge_pass_delta": 0.5,
            "judge_grounding": 3.0,
            "judge_grounding_delta": 3.0,
            "judge_accuracy_delta": 2.0,
            "citation_delta": 0.0,
            "format_delta": 0.0,
            "ttft_p50_delta_ms": 0.0,
        },
        {
            "label": "best",
            "judge_pass_rate": 1.0,
            "judge_pass_delta": 1.0,
            "judge_grounding": 5.0,
            "judge_grounding_delta": 5.0,
            "judge_accuracy_delta": 4.0,
            "citation_delta": 0.0,
            "format_delta": 0.0,
            "ttft_p50_delta_ms": 1.0,
        },
    ]

    markdown = _render_markdown(
        family_label="comparison_party_title_metadata",
        include_qids_file=Path("/tmp/family.txt"),
        rows=rows,
    )

    lines = markdown.splitlines()
    ranked_line = next(line for line in lines if "`best`" in line)
    assert ranked_line.startswith("| 1 |")


def test_payload_shape_is_json_serializable(tmp_path: Path) -> None:
    payload = {
        "family_label": "comparison",
        "include_qids_file": "/tmp/family.txt",
        "baseline_label": "v6_context_seed",
        "case_scope": "changed",
        "judge_scope": "all",
        "ranked_candidates": [
            {
                "label": "best",
                "candidate_raw_results": "/tmp/raw.json",
                "judge_pass_rate": 1.0,
                "judge_pass_delta": 1.0,
                "judge_grounding": 5.0,
                "judge_grounding_delta": 5.0,
                "judge_accuracy_delta": 4.0,
                "citation_delta": 0.0,
                "format_delta": 0.0,
                "ttft_p50_delta_ms": 0.0,
                "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            }
        ],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }

    out = tmp_path / "payload.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = json.loads(out.read_text(encoding="utf-8"))
    assert restored["ranked_candidates"][0]["label"] == "best"
