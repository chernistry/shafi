from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rag_challenge.ml.grounding_dataset import export_grounding_ml_dataset

if TYPE_CHECKING:
    from pathlib import Path


def _write_raw_results(path: Path, *, question_id: str) -> None:
    payload = [
        {
            "case": {
                "case_id": question_id,
                "question": "Who were the claimants?",
                "answer_type": "names",
            },
            "answer_text": "Alice",
            "telemetry": {
                "question_id": question_id,
                "used_page_ids": ["doc_1"],
                "retrieved_page_ids": ["doc_1", "doc_2"],
                "context_page_ids": ["doc_1"],
                "cited_page_ids": ["doc_1"],
                "retrieved_chunk_ids": ["doc:0:0"],
                "context_chunk_ids": ["doc:0:0"],
                "cited_chunk_ids": ["doc:0:0"],
                "chunk_snippets": {"doc:0:0": "The claimants were Alice and Bob."},
                "doc_refs": ["doc"],
            },
            "total_ms": 10,
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_grounding_ml_dataset_preserves_reviewed_metadata(tmp_path: Path) -> None:
    legacy_raw = tmp_path / "legacy.json"
    sidecar_raw = tmp_path / "sidecar.json"
    golden = tmp_path / "golden.json"
    benchmark = tmp_path / "benchmark.json"
    reviewed_dir = tmp_path / "reviewed"
    reviewed_dir.mkdir()
    reviewed = reviewed_dir / "reviewed_all_100.json"
    reviewed_high = reviewed_dir / "reviewed_high_confidence_81.json"
    reviewed_medium_plus_high = reviewed_dir / "reviewed_medium_plus_high_95.json"
    reviewed_benchmark_all = reviewed_dir / "reviewed_page_benchmark_all_100.json"
    reviewed_benchmark_high = reviewed_dir / "reviewed_page_benchmark_high_confidence_81.json"
    reviewed_benchmark_medium = reviewed_dir / "reviewed_page_benchmark_medium_plus_high_95.json"
    import_manifest = reviewed_dir / "import_manifest.json"
    output_dir = tmp_path / "out"

    _write_raw_results(legacy_raw, question_id="qid-1")
    _write_raw_results(sidecar_raw, question_id="qid-1")
    golden.write_text(
        json.dumps(
            [
                {
                    "question_id": "qid-1",
                    "question": "Who were the claimants?",
                    "answer_type": "names",
                    "golden_answer": "Alice",
                    "golden_page_ids": ["doc_1"],
                    "confidence": "low",
                }
            ]
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps({"cases": [{"question_id": "qid-1", "gold_page_ids": ["doc_1"], "trust_tier": "reviewed"}]}),
        encoding="utf-8",
    )
    reviewed.write_text(
        json.dumps(
            [
                {
                    "question_id": "qid-1",
                    "question": "Who were the claimants?",
                    "answer_type": "names",
                    "golden_answer": "Alice",
                    "golden_page_ids": ["doc_1"],
                    "confidence": "medium",
                    "label_status": "partly_correct",
                    "audit_note": "The current answer is right but the old page was off by one.",
                    "current_label_problem": "wrong page",
                    "trust_tier": "medium",
                    "label_weight": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    reviewed_high.write_text(reviewed.read_text(encoding="utf-8"), encoding="utf-8")
    reviewed_medium_plus_high.write_text(reviewed.read_text(encoding="utf-8"), encoding="utf-8")
    reviewed_benchmark_all.write_text(benchmark.read_text(encoding="utf-8"), encoding="utf-8")
    reviewed_benchmark_high.write_text(benchmark.read_text(encoding="utf-8"), encoding="utf-8")
    reviewed_benchmark_medium.write_text(benchmark.read_text(encoding="utf-8"), encoding="utf-8")
    import_manifest.write_text(
        json.dumps(
            {
                "slice_counts": {
                    "reviewed_all_100": 100,
                    "reviewed_high_confidence_81": 81,
                    "reviewed_medium_plus_high_95": 95,
                },
                "confidence_counts": {
                    "high": 81,
                    "medium": 14,
                    "low": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = export_grounding_ml_dataset(
        legacy_raw_results_path=legacy_raw,
        sidecar_raw_results_path=sidecar_raw,
        golden_labels_path=golden,
        page_benchmark_path=benchmark,
        suspect_labels_path=None,
        reviewed_labels_path=reviewed,
        output_dir=output_dir,
        split_seed=1,
        dev_ratio=0.5,
    )

    assert manifest.label_source_counts["reviewed"] == 1
    assert manifest.reviewed_slice_counts == {
        "reviewed_all_100": 100,
        "reviewed_high_confidence_81": 81,
        "reviewed_medium_plus_high_95": 95,
    }
    assert manifest.label_confidence_counts == {"high": 81, "low": 5, "medium": 14}
    assert manifest.source_paths["reviewed_all_100"] == str(reviewed)
    assert manifest.source_paths["reviewed_high_confidence_81"] == str(reviewed_high)
    assert manifest.source_paths["active_reviewed_page_benchmark"] == str(benchmark)
    row = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["label_source"] == "reviewed"
    assert row["label_confidence"] == "medium"
    assert row["label_status"] == "partly_correct"
    assert row["label_weight"] == 0.5
    assert row["label_note_present"] is True
