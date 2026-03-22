from __future__ import annotations

from pathlib import Path

import pytest

from shafi.submission.replay import (
    build_counterfactual_preflight,
    compare_submission_drift,
    merge_answer_stable_records,
    validate_replay_contract,
)


def _submission_record(
    *, qid: str, answer: object, page_id: str, model_name: str = "baseline-model"
) -> dict[str, object]:
    return {
        "question_id": qid,
        "answer": answer,
        "telemetry": {
            "model_name": model_name,
            "retrieval": {
                "retrieved_chunk_pages": [
                    {"doc_id": page_id.rsplit("_", 1)[0], "page_numbers": [int(page_id.rsplit("_", 1)[1])]}
                ]
            },
        },
    }


def _raw_record(*, qid: str, answer_text: str, used_page_id: str, context_page_id: str) -> dict[str, object]:
    return {
        "case": {"case_id": qid, "question": qid, "answer_type": "name"},
        "answer_text": answer_text,
        "telemetry": {
            "retrieved_chunk_ids": [f"{used_page_id}:chunk"],
            "retrieved_page_ids": [used_page_id],
            "context_chunk_ids": [f"{context_page_id}:chunk"],
            "context_page_ids": [context_page_id],
            "used_chunk_ids": [f"{used_page_id}:chunk"],
            "used_page_ids": [used_page_id],
        },
        "total_ms": 10,
    }


def test_merge_answer_stable_records_freezes_answers_and_swaps_page_projection() -> None:
    answer_submission = {"answers": [_submission_record(qid="q1", answer="A", page_id="doca_1")]}
    answer_raw = [_raw_record(qid="q1", answer_text="A", used_page_id="doca_1", context_page_id="doca_1")]
    page_submission = {
        "answers": [_submission_record(qid="q1", answer="B", page_id="docb_2", model_name="candidate-model")]
    }
    page_raw = [_raw_record(qid="q1", answer_text="B", used_page_id="docb_2", context_page_id="docb_2")]

    merged_submission, merged_raw, report = merge_answer_stable_records(
        answer_source_submission=answer_submission,
        answer_source_raw_results=answer_raw,
        page_source_submission=page_submission,
        page_source_raw_results=page_raw,
        allowlisted_qids=set(),
        page_allowlisted_qids=set(),
        page_source_pages_default="all",
    )

    merged_answer = merged_submission["answers"][0]
    assert merged_answer["answer"] == "A"
    assert merged_answer["telemetry"]["answer_type"] == "name"
    assert merged_answer["telemetry"]["retrieval"]["retrieved_chunk_pages"][0]["doc_id"] == "docb"
    assert merged_raw[0]["answer_text"] == "A"
    assert merged_raw[0]["telemetry"]["used_page_ids"] == ["docb_2"]
    assert report["answer_changed_count_vs_answer_source"] == 0
    assert report["page_projection_changed_count_vs_answer_source"] == 1


def test_compare_submission_drift_reports_answer_and_page_changes() -> None:
    baseline_submission = {"answers": [_submission_record(qid="q1", answer="A", page_id="doca_1")]}
    candidate_submission = {"answers": [_submission_record(qid="q1", answer="B", page_id="docb_2")]}
    baseline_raw = [_raw_record(qid="q1", answer_text="A", used_page_id="doca_1", context_page_id="doca_1")]
    candidate_raw = [_raw_record(qid="q1", answer_text="B", used_page_id="docb_2", context_page_id="docb_2")]

    drift = compare_submission_drift(
        baseline_submission=baseline_submission,
        candidate_submission=candidate_submission,
        baseline_raw_results=baseline_raw,
        candidate_raw_results=candidate_raw,
    )

    assert drift.answer_changed_qids == ["q1"]
    assert drift.page_changed_qids == ["q1"]
    assert drift.used_page_count_deltas == {"q1": 0}


def test_validate_replay_contract_fails_loudly_on_missing_used_page_ids() -> None:
    answer_submission = {"answers": [_submission_record(qid="q1", answer="A", page_id="doca_1")]}
    answer_raw = [_raw_record(qid="q1", answer_text="A", used_page_id="doca_1", context_page_id="doca_1")]
    page_submission = {"answers": [_submission_record(qid="q1", answer="A", page_id="doca_1")]}
    page_raw = [
        {
            "case": {"case_id": "q1", "question": "q1", "answer_type": "name"},
            "answer_text": "A",
            "telemetry": {"context_page_ids": ["doca_1"]},
            "total_ms": 10,
        }
    ]

    with pytest.raises(ValueError, match="missing used/context page IDs"):
        validate_replay_contract(
            answer_source_submission=answer_submission,
            answer_source_raw_results=answer_raw,
            page_source_submission=page_submission,
            page_source_raw_results=page_raw,
        )


def test_build_counterfactual_preflight_preserves_answer_types_and_fingerprint() -> None:
    merged_payload = {
        "answers": [
            _submission_record(qid="q1", answer="A", page_id="doca_1"),
            _submission_record(qid="q2", answer=False, page_id="docb_2"),
        ]
    }
    merged_payload["answers"][0]["telemetry"]["answer_type"] = "name"
    merged_payload["answers"][1]["telemetry"]["answer_type"] = "boolean"

    preflight = build_counterfactual_preflight(
        merged_payload=merged_payload,
        answer_source_preflight={
            "phase": "warmup",
            "score_settings_sha256": "answer-sha",
            "score_settings_fingerprint": {"source": "answer"},
            "questions_sha256": "questions",
            "documents_zip_sha256": "docs",
            "pdf_count": 30,
        },
        page_source_preflight={
            "phase_collection_name": "candidate_collection",
            "qdrant_point_count": 123,
            "truth_audit_workbook_path": "audit.md",
            "code_archive_sha256": "archive-sha",
            "score_settings_sha256": "page-sha",
            "score_settings_fingerprint": {"source": "page"},
        },
        answer_source_submission=Path("/tmp/answer.json"),
        page_source_submission=Path("/tmp/page.json"),
        allowlisted_qids=set(),
        page_allowlisted_qids=set(),
    )

    assert preflight["answer_type_counts"] == {"name": 1, "boolean": 1}
    assert preflight["score_settings_sha256"] == "page-sha"
    assert preflight["score_settings_fingerprint"] == {"source": "page"}
    projection = preflight["counterfactual_projection"]
    assert projection["answer_source_code_archive_sha256"] == ""
    assert projection["page_source_code_archive_sha256"] == "archive-sha"
    assert projection["answer_source_score_settings_sha256"] == "answer-sha"
    assert projection["page_source_score_settings_sha256"] == "page-sha"
