from __future__ import annotations

import json
from typing import TYPE_CHECKING

import fitz

if TYPE_CHECKING:
    from pathlib import Path

from scripts.build_platform_truth_audit import build_truth_audit_scaffold, render_truth_audit_workbook


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
    assert scaffold["summary"]["support_shape_flag_counts"] == {"comparison_missing_side": 1}
    assert len(scaffold["records"]) == 2

    deterministic_case = scaffold["deterministic_cases"][0]
    assert deterministic_case["route_family"] == "strict"
    assert deterministic_case["support_shape_class"] == "comparison"
    assert deterministic_case["support_shape_requirements"] == {}
    assert deterministic_case["support_doc_count"] == 1
    assert deterministic_case["support_page_count"] == 1
    assert deterministic_case["support_shape_flags"] == ["comparison_missing_side"]
    assert deterministic_case["question_refs"] == ["CFI 010/2024", "CFI 016/2025"]
    assert deterministic_case["required_page_anchor"] == {}
    assert deterministic_case["exact_span_candidates"] == []
    assert deterministic_case["failure_class"] == ""
    assert deterministic_case["minimal_required_support_pages"] == []
    assert deterministic_case["audit_priority"] == 20
    assert "review_packet" in deterministic_case

    free_text_case = scaffold["free_text_cases"][0]
    assert free_text_case["route_family"] == "structured"
    assert free_text_case["support_shape_class"] == "outcome_plus_costs"
    assert free_text_case["support_shape_requirements"] == {}
    assert free_text_case["support_doc_count"] == 1
    assert free_text_case["support_page_count"] == 2
    assert free_text_case["support_shape_flags"] == []
    assert free_text_case["audit_priority"] == 50


def test_build_truth_audit_scaffold_extracts_required_page_anchor_and_exact_spans(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    submission_path = tmp_path / "submission.json"
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-page-2",
                    "question": "According to page 2 of case CFI 010/2024, who is the defendant?",
                    "answer_type": "name",
                }
            ]
        ),
        encoding="utf-8",
    )
    submission_path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q-page-2",
                        "answer": "ONORA",
                        "telemetry": {
                            "model_name": "strict-extractor",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "cfi-010-2024", "page_numbers": [2]},
                                ]
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pdf = fitz.open()
    page1 = pdf.new_page()
    page1.insert_text((72, 72), "CFI 010/2024")
    page2 = pdf.new_page()
    page2.insert_text((72, 72), "The defendant is ONORA.")
    pdf.save(docs_dir / "cfi-010-2024.pdf")
    pdf.close()

    scaffold = build_truth_audit_scaffold(
        questions_path=questions_path,
        submission_path=submission_path,
        docs_dir=docs_dir,
    )

    record = scaffold["deterministic_cases"][0]
    assert record["required_page_anchor"] == {"kind": "explicit_page", "pages": [2]}
    assert record["audit_priority"] == 25
    assert record["exact_span_candidates"][0]["page"] == 2
    assert record["exact_span_candidates"][0]["text"] == "The defendant is ONORA."
    assert record["review_packet"]["required_page_anchor"] == {"kind": "explicit_page", "pages": [2]}


def test_build_truth_audit_scaffold_only_flags_multi_atom_named_metadata_when_undercovered(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    submission_path = tmp_path / "submission.json"

    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-law-number",
                    "question": "According to the title page of the Common Reporting Standard Law, what is its official DIFC Law number?",
                    "answer_type": "number",
                },
                {
                    "id": "q-title-update",
                    "question": "What is the title of the Foundations Law 2018 and when was its consolidated version last updated?",
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
                        "question_id": "q-law-number",
                        "answer": 2,
                        "telemetry": {
                            "model_name": "strict-extractor",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "law-2", "page_numbers": [1]},
                                ]
                            },
                        },
                    },
                    {
                        "question_id": "q-title-update",
                        "answer": "The title is Foundations Law 2018 and the consolidated version was updated on 1 July 2024.",
                        "telemetry": {
                            "model_name": "structured-extractor",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "foundations-law", "page_numbers": [4]},
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

    deterministic_case = scaffold["deterministic_cases"][0]
    assert deterministic_case["support_shape_class"] == "named_metadata"
    assert deterministic_case["support_shape_requirements"]["requires_multi_page"] is False
    assert deterministic_case["support_shape_flags"] == []

    free_text_case = scaffold["free_text_cases"][0]
    assert free_text_case["support_shape_class"] == "named_metadata"
    assert free_text_case["support_shape_requirements"]["requires_multi_page"] is True
    assert free_text_case["support_shape_flags"] == ["metadata_multi_atom_maybe_undercovered"]


def test_build_truth_audit_scaffold_preserves_manual_fields_and_adds_pdf_previews(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    submission_path = tmp_path / "submission.json"
    existing_scaffold_path = tmp_path / "truth_audit_scaffold.json"
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-law",
                    "question": "According to Article 2 of the DIFC Personal Property Law 2005, who made this Law?",
                    "answer_type": "free_text",
                }
            ]
        ),
        encoding="utf-8",
    )
    submission_path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q-law",
                        "answer": "This Law was made by the Ruler of Dubai.",
                        "telemetry": {
                            "model_name": "gpt-4.1-mini",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "law-doc", "page_numbers": [1]},
                                ]
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    existing_scaffold_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q-law",
                        "manual_verdict": "correct",
                        "expected_answer": "Ruler of Dubai",
                        "minimal_required_support_pages": ["law-doc_1"],
                        "manual_exactness_labels": ["semantic_correct", "page_specific_exact_risk"],
                        "failure_class": "support_undercoverage",
                        "notes": "keep this note",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "DIFC Personal Property Law 2005\nThis Law was made by the Ruler of Dubai.")
    pdf.save(docs_dir / "law-doc.pdf")
    pdf.close()

    scaffold = build_truth_audit_scaffold(
        questions_path=questions_path,
        submission_path=submission_path,
        docs_dir=docs_dir,
        existing_scaffold_path=existing_scaffold_path,
    )

    record = scaffold["records"][0]
    assert record["manual_verdict"] == "correct"
    assert record["expected_answer"] == "Ruler of Dubai"
    assert record["minimal_required_support_pages"] == ["law-doc_1"]
    assert record["manual_exactness_labels"] == ["semantic_correct", "page_specific_exact_risk"]
    assert record["failure_class"] == "support_undercoverage"
    assert record["notes"] == "keep this note"
    assert record["support_page_previews"][0]["doc_title"].startswith("DIFC Personal Property Law 2005")
    assert "Ruler of Dubai" in record["support_page_previews"][0]["snippet"]
    assert record["review_packet"]["support_page_previews"][0]["page"] == 1
    assert record["review_packet"]["manual_exactness_labels"] == ["semantic_correct", "page_specific_exact_risk"]
    assert scaffold["summary"]["manual_verdict_counts"]["free_text_complete"] == 1
    assert scaffold["summary"]["manual_exactness_label_counts"] == {
        "semantic_correct": 1,
        "page_specific_exact_risk": 1,
    }


def test_render_truth_audit_workbook_includes_manual_fields_and_previews(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.json"
    submission_path = tmp_path / "submission.json"

    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-1",
                    "question": "Who is the defendant in case ARB 034/2025?",
                    "answer_type": "name",
                }
            ]
        ),
        encoding="utf-8",
    )
    submission_path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q-1",
                        "answer": "ONORA",
                        "telemetry": {
                            "model_name": "gpt-4.1-mini",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "arb-034", "page_numbers": [1, 2]},
                                ]
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    scaffold = build_truth_audit_scaffold(
        questions_path=questions_path,
        submission_path=submission_path,
    )
    workbook = render_truth_audit_workbook(scaffold)

    assert "# Truth Audit Workbook" in workbook
    assert "### q-1" in workbook
    assert "- route_family: `model`" in workbook
    assert "- manual_verdict: `(blank)`" in workbook
    assert "- manual_exactness_labels: (none)" in workbook
    assert "- support_page_previews:" in workbook
