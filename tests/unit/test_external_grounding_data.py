from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from shafi.ml.external_grounding_data import (
    export_normalized_external_grounding_data,
    normalize_contractnli_record,
    normalize_cuad_record,
    normalize_ledgar_record,
    normalize_obliqa_record,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_normalize_obliqa_record_maps_support_scope() -> None:
    row = normalize_obliqa_record(
        {
            "QuestionID": "q-obliqa",
            "Question": "What must the firm do?",
            "Group": "obligation",
            "Passages": [
                {"DocumentID": 12, "PassageID": "COB_1.1", "Passage": "A firm must keep records."},
            ],
        },
        split="train",
    )

    assert row.source_dataset == "obliqa"
    assert row.label_type == "support_scope"
    assert row.scope_label == "single_field_single_doc"
    assert row.support_label == "supported"
    assert "COB_1.1" in row.text


def test_normalize_cuad_record_extracts_role_label() -> None:
    row = normalize_cuad_record(
        {
            "id": "cuad-1",
            "title": "Agreement",
            "context": "This agreement is effective on January 1.",
            "question": 'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer.',
            "answers": {"text": ["January 1"], "answer_start": [27]},
        },
        split="train",
    )

    assert row.source_dataset == "cuad"
    assert row.label_type == "role_label"
    assert row.role_label == "effective_date"
    assert row.support_label == "supported"


def test_normalize_contractnli_record_maps_support_label() -> None:
    row = normalize_contractnli_record(
        {
            "sentence1": "The supplier must indemnify the buyer.",
            "sentence2": "The contract requires indemnification.",
            "label": 1,
            "gold_label": "entailment",
        },
        split="train",
        index=1,
        label_names=["contradiction", "entailment", "neutral"],
    )

    assert row.source_dataset == "contractnli"
    assert row.label_type == "support_label"
    assert row.support_label == "entailment"
    assert row.scope_label == "pair_entailment"


def test_normalize_ledgar_record_maps_role_label() -> None:
    row = normalize_ledgar_record(
        {"text": "This agreement is governed by New York law.", "label": 47},
        split="train",
        index=1,
        label_names=["Adjustments"] * 47 + ["Governing Laws"],
    )

    assert row.source_dataset == "ledgar"
    assert row.label_type == "role_label"
    assert row.role_label == "governing_laws"
    assert row.support_label == "role_supervision"


def test_export_normalized_external_grounding_data_writes_manifest(tmp_path: Path) -> None:
    obliqa_root = tmp_path / "obliqa"
    obliqa_dir = obliqa_root / "ObliQA"
    obliqa_dir.mkdir(parents=True)
    (obliqa_dir / "ObliQA_train.json").write_text(
        json.dumps(
            [
                {
                    "QuestionID": "q-obliqa",
                    "Question": "What must the firm do?",
                    "Passages": [{"DocumentID": 1, "PassageID": "A", "Passage": "A firm must keep records."}],
                }
            ]
        ),
        encoding="utf-8",
    )

    cuad_root = tmp_path / "cuad"
    materialized_dir = cuad_root / "materialized"
    materialized_dir.mkdir(parents=True)
    (materialized_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "id": "cuad-1",
                "title": "Agreement",
                "context": "This agreement is effective on January 1.",
                "question": 'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer.',
                "answers": {"text": ["January 1"], "answer_start": [27]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    contract_root = tmp_path / "contractnli"
    contract_data = contract_root / "data"
    contract_data.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "sentence1": "The supplier must indemnify the buyer.",
                "sentence2": "The contract requires indemnification.",
                "label": 1,
                "gold_label": "entailment",
            }
        ]
    ).to_parquet(contract_data / "train-00000-of-00001.parquet", index=False)
    (contract_root / "README.md").write_text(
        "---\n"
        "dataset_info:\n"
        "  features:\n"
        "  - name: label\n"
        "    dtype:\n"
        "      class_label:\n"
        "        names:\n"
        "          '0': contradiction\n"
        "          '1': entailment\n"
        "          '2': neutral\n"
        "---\n",
        encoding="utf-8",
    )

    ledgar_root = tmp_path / "ledgar"
    ledgar_data = ledgar_root / "data"
    ledgar_data.mkdir(parents=True)
    pd.DataFrame([{"text": "This agreement is governed by New York law.", "label": 1}]).to_parquet(
        ledgar_data / "train-00000-of-00001.parquet",
        index=False,
    )
    (ledgar_root / "README.md").write_text(
        "---\n"
        "dataset_info:\n"
        "  features:\n"
        "  - name: label\n"
        "    dtype:\n"
        "      class_label:\n"
        "        names:\n"
        "          '0': Adjustments\n"
        "          '1': Governing Laws\n"
        "---\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "normalized"
    manifest = export_normalized_external_grounding_data(
        obliqa_root=obliqa_root,
        cuad_root=cuad_root,
        contractnli_root=contract_root,
        ledgar_root=ledgar_root,
        output_dir=output_dir,
        max_rows_per_dataset=1,
    )

    assert manifest.total_rows == 4
    assert manifest.row_counts_by_dataset == {
        "contractnli": 1,
        "cuad": 1,
        "ledgar": 1,
        "obliqa": 1,
    }
    assert (output_dir / "normalized_rows.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
