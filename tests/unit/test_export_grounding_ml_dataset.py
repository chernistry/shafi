from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rag_challenge.ml.grounding_dataset import export_grounding_ml_dataset, split_grounding_ml_rows

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_split_grounding_ml_rows_is_deterministic(tmp_path: Path) -> None:
    from rag_challenge.ml.grounding_dataset import GroundingMlRow, PageRetrievalFeatureRecord, SupportFactFeatureRecord

    rows = [
        GroundingMlRow(
            question_id=f"qid-{index}",
            question=f"Question {index}",
            answer_type="name",
            label_source="soft_ai_gold",
            scope_mode="single_field_single_doc",
            support_fact_features=SupportFactFeatureRecord(),
            page_retrieval_features=PageRetrievalFeatureRecord(),
        )
        for index in range(10)
    ]

    train_a, dev_a = split_grounding_ml_rows(rows, split_seed=601, dev_ratio=0.2)
    train_b, dev_b = split_grounding_ml_rows(rows, split_seed=601, dev_ratio=0.2)

    assert [row.question_id for row in train_a] == [row.question_id for row in train_b]
    assert [row.question_id for row in dev_a] == [row.question_id for row in dev_b]


def test_export_grounding_ml_dataset_writes_expected_files(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.json"
    sidecar_path = tmp_path / "sidecar.json"
    golden_path = tmp_path / "golden.json"
    benchmark_path = tmp_path / "benchmark.json"
    suspect_path = tmp_path / "suspect.json"
    output_dir = tmp_path / "out"

    raw_payload = [
        {
            "answer_text": "annual return",
            "case": {
                "case_id": "qid-1",
                "question": "According to Article 16 of the law, what must be filed?",
                "answer_type": "name",
            },
            "telemetry": {
                "doc_refs": ["Operating Law 2018"],
                "retrieved_page_ids": ["law_16"],
                "context_page_ids": ["law_16"],
                "cited_page_ids": ["law_16"],
                "used_page_ids": ["law_16"],
                "retrieved_chunk_ids": ["law:15:0:a"],
                "context_chunk_ids": ["law:15:0:a"],
                "cited_chunk_ids": ["law:15:0:a"],
                "chunk_snippets": {"law:15:0:a": "Article 16 requires the filing of the annual return."},
            },
            "total_ms": 100,
        }
    ]
    _write_json(legacy_path, raw_payload)
    _write_json(sidecar_path, raw_payload)
    _write_json(
        golden_path,
        [
            {
                "question_id": "qid-1",
                "question": "According to Article 16 of the law, what must be filed?",
                "answer_type": "name",
                "golden_answer": "annual return",
                "golden_page_ids": ["law_16"],
            }
        ],
    )
    _write_json(benchmark_path, {"cases": [{"question_id": "qid-1", "gold_page_ids": ["law_16"], "trust_tier": "trusted"}]})
    _write_json(suspect_path, {"case_count": 0, "rows": []})

    manifest = export_grounding_ml_dataset(
        legacy_raw_results_path=legacy_path,
        sidecar_raw_results_path=sidecar_path,
        golden_labels_path=golden_path,
        page_benchmark_path=benchmark_path,
        suspect_labels_path=suspect_path,
        output_dir=output_dir,
        split_seed=601,
        dev_ratio=0.5,
    )

    assert manifest.row_count == 1
    train_rows = [json.loads(line) for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    dev_rows = [json.loads(line) for line in (output_dir / "dev.jsonl").read_text(encoding="utf-8").splitlines() if line]
    exported_rows = train_rows + dev_rows
    assert len(exported_rows) == 1
    row = exported_rows[0]
    assert row["label_source"] == "soft_ai_gold"
    assert row["hard_anchor_strings"] == ["Article 16"]
    assert row["page_candidates"][0]["page_id"] == "law_16"
    assert "legacy_retrieved" in row["page_candidates"][0]["candidate_sources"]
    assert "sidecar_used" in row["page_candidates"][0]["candidate_sources"]
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "dev.jsonl").exists()
    assert (output_dir / "feature_inventory.md").exists()
    assert (output_dir / "failure_canaries.json").exists()
    assert (output_dir / "export_manifest.json").exists()
