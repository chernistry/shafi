from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.import_reviewed_labels import import_reviewed_labels

if TYPE_CHECKING:
    from pathlib import Path


def _build_reviewed_row(question_id: str, *, confidence: str, pages: list[str]) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": f"Question {question_id}",
        "answer_type": "name",
        "golden_answer": f"Answer {question_id}",
        "golden_page_ids": pages,
        "confidence": confidence,
        "label_status": "correct",
        "audit_note": f"Audit note {question_id}",
        "current_label_problem": "",
    }


def test_import_reviewed_labels_derives_confidence_slices(tmp_path: Path) -> None:
    reviewed_golden = tmp_path / "corrected_golden_labels_v3.json"
    reviewed_page_benchmark = tmp_path / "corrected_page_benchmark_v3.json"
    audit_report = tmp_path / "label_audit_report.md"
    output_dir = tmp_path / "reviewed"

    rows = [
        _build_reviewed_row("q-high", confidence="high", pages=["doc_1"]),
        _build_reviewed_row("q-medium", confidence="medium", pages=["doc_2"]),
        _build_reviewed_row("q-low", confidence="low", pages=[]),
    ]
    reviewed_golden.write_text(json.dumps(rows), encoding="utf-8")
    reviewed_page_benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q-high", "gold_page_ids": ["doc_1"], "trust_tier": "trusted"},
                    {"question_id": "q-medium", "gold_page_ids": ["doc_2"], "trust_tier": "trusted"},
                    {"question_id": "q-low", "gold_page_ids": [], "trust_tier": "trusted"},
                ]
            }
        ),
        encoding="utf-8",
    )
    audit_report.write_text("# audit\n", encoding="utf-8")

    manifest = import_reviewed_labels(
        reviewed_golden_path=reviewed_golden,
        reviewed_page_benchmark_path=reviewed_page_benchmark,
        audit_report_path=audit_report,
        output_dir=output_dir,
        expected_count=3,
    )

    assert manifest["slice_counts"] == {
        "reviewed_all_100": 3,
        "reviewed_high_confidence_81": 1,
        "reviewed_medium_plus_high_95": 2,
    }
    assert manifest["confidence_counts"] == {"high": 1, "medium": 1, "low": 1}

    high_rows = json.loads((output_dir / "reviewed_high_confidence_81.json").read_text(encoding="utf-8"))
    assert [row["question_id"] for row in high_rows] == ["q-high"]
    assert high_rows[0]["label_weight"] == 1.0

    medium_plus_high = json.loads((output_dir / "reviewed_medium_plus_high_95.json").read_text(encoding="utf-8"))
    assert [row["question_id"] for row in medium_plus_high] == ["q-high", "q-medium"]
    assert medium_plus_high[1]["label_weight"] == 0.5

    high_benchmark = json.loads(
        (output_dir / "reviewed_page_benchmark_high_confidence_81.json").read_text(encoding="utf-8")
    )
    assert high_benchmark["cases"] == [{"question_id": "q-high", "gold_page_ids": ["doc_1"], "trust_tier": "trusted"}]


def test_import_reviewed_labels_rejects_id_mismatch(tmp_path: Path) -> None:
    reviewed_golden = tmp_path / "corrected_golden_labels_v3.json"
    reviewed_page_benchmark = tmp_path / "corrected_page_benchmark_v3.json"
    audit_report = tmp_path / "label_audit_report.md"

    reviewed_golden.write_text(
        json.dumps([_build_reviewed_row("q1", confidence="high", pages=["doc_1"])]), encoding="utf-8"
    )
    reviewed_page_benchmark.write_text(
        json.dumps({"cases": [{"question_id": "q2", "gold_page_ids": ["doc_1"], "trust_tier": "trusted"}]}),
        encoding="utf-8",
    )
    audit_report.write_text("# audit\n", encoding="utf-8")

    try:
        import_reviewed_labels(
            reviewed_golden_path=reviewed_golden,
            reviewed_page_benchmark_path=reviewed_page_benchmark,
            audit_report_path=audit_report,
            output_dir=tmp_path / "reviewed",
            expected_count=1,
        )
    except ValueError as exc:
        assert "same ID set" in str(exc)
    else:
        raise AssertionError("Expected ID-set mismatch to raise ValueError")
