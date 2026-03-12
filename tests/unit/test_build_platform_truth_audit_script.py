from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.build_platform_truth_audit import build_truth_audit_scaffold

if TYPE_CHECKING:
    from pathlib import Path


def test_build_truth_audit_scaffold_adds_route_and_support_shape_metadata(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    submission_path = tmp_path / "submission.json"

    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-compare",
                    "question": "Which case has an earlier Date of Issue: CFI 010/2024 or CFI 016/2025?",
                    "answer_type": "name",
                },
                {
                    "id": "q-costs",
                    "question": "How did the Court of Appeal rule, and what costs were awarded?",
                    "answer_type": "free_text",
                },
            ]
        ),
        encoding="utf-8",
    )
    submission_path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q-compare",
                        "answer": "CFI 010/2024",
                        "telemetry": {
                            "model_name": "strict-extractor",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "cfi-010-2024", "page_numbers": [1]},
                                ]
                            },
                        },
                    },
                    {
                        "question_id": "q-costs",
                        "answer": "The Permission to Appeal Application is refused. The Appellant was awarded costs in the sum of USD 40,000.",
                        "telemetry": {
                            "model_name": "structured-extractor",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "ca-009-2024", "page_numbers": [5, 6]},
                                ]
                            },
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    scaffold = build_truth_audit_scaffold(
        questions_path=questions_path,
        submission_path=submission_path,
    )

    assert scaffold["summary"]["route_family_counts"] == {"strict": 1, "structured": 1}
    assert scaffold["summary"]["support_shape_class_counts"] == {"comparison": 1, "outcome_plus_costs": 1}

    deterministic_case = scaffold["deterministic_cases"][0]
    assert deterministic_case["route_family"] == "strict"
    assert deterministic_case["support_shape_class"] == "comparison"
    assert deterministic_case["support_doc_count"] == 1
    assert deterministic_case["support_page_count"] == 1
    assert deterministic_case["support_shape_flags"] == ["comparison_missing_side"]
    assert deterministic_case["question_refs"] == ["CFI 010/2024", "CFI 016/2025"]
    assert deterministic_case["failure_class"] == ""
    assert deterministic_case["minimal_required_support_pages"] == []

    free_text_case = scaffold["free_text_cases"][0]
    assert free_text_case["route_family"] == "structured"
    assert free_text_case["support_shape_class"] == "outcome_plus_costs"
    assert free_text_case["support_doc_count"] == 1
    assert free_text_case["support_page_count"] == 2
    assert free_text_case["support_shape_flags"] == []
