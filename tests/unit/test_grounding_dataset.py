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
    reviewed = tmp_path / "reviewed.json"
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
    row = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["label_source"] == "reviewed"
    assert row["label_confidence"] == "medium"
    assert row["label_status"] == "partly_correct"
    assert row["label_weight"] == 0.5
    assert row["label_note_present"] is True
