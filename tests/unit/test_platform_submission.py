from __future__ import annotations

import argparse
import asyncio
import json
import os
import zipfile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import httpx
import pytest

from shafi.submission.common import SubmissionCase
from shafi.submission.platform import (
    ArchiveAllowlist,
    PlatformCaseResult,
    PlatformEvaluationClient,
    PlatformPaths,
    _async_main,
    _build_all_context_pages_challenger_payload,
    _build_anchor_page_challenger_payload,
    _build_equivalence_canary,
    _build_preflight_summary,
    _build_results_anomaly_report,
    _build_support_shape_report,
    _build_truth_audit_report,
    _check_existing_artifact_preflight,
    _create_code_archive,
    _extract_http_error_message,
    _is_resources_not_published_error,
    _phase_collection_override,
    _project_platform_answer,
    _repair_anomalous_results,
    _resolve_query_concurrency,
    _resolve_source_truth_audit_path,
    _result_anomaly_flags,
    _run_case_direct,
    _run_questions,
    _scan_text_for_secrets,
    _submit_existing_artifacts,
    _validate_platform_args,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_project_platform_answer_uses_nested_platform_shape() -> None:
    result = PlatformCaseResult(
        case=SubmissionCase(case_id="q-1", question="Q?", answer_type="boolean"),
        answer_text="Yes",
        telemetry={
            "ttft_ms": 123,
            "time_per_output_token_ms": 45,
            "total_ms": 999,
            "used_page_ids": ["abc123_3", "abc123_5", "def456_2"],
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "model_llm": "gpt-4.1-mini",
        },
        total_ms=1001,
    )

    payload = _project_platform_answer(result)

    assert payload["question_id"] == "q-1"
    assert payload["answer"] is True
    telemetry = payload["telemetry"]
    assert isinstance(telemetry, dict)
    retrieval = telemetry["retrieval"]
    assert isinstance(retrieval, dict)
    refs = retrieval["retrieved_chunk_pages"]
    assert refs == [
        {"doc_id": "abc123", "page_numbers": [3, 5]},
        {"doc_id": "def456", "page_numbers": [2]},
    ]


def test_project_platform_answer_normalizes_date_and_caps_free_text_sentences() -> None:
    date_result = PlatformCaseResult(
        case=SubmissionCase(case_id="q-date", question="Q?", answer_type="date"),
        answer_text="March 11, 2025",
        telemetry={
            "ttft_ms": 100,
            "time_per_output_token_ms": 10,
            "total_ms": 150,
            "used_page_ids": ["doca_1"],
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "model_llm": "gpt-4.1-mini",
        },
        total_ms=150,
    )
    free_text_result = PlatformCaseResult(
        case=SubmissionCase(case_id="q-free", question="Q?", answer_type="free_text"),
        answer_text="Alpha. Beta. Gamma. Delta.",
        telemetry={
            "ttft_ms": 100,
            "time_per_output_token_ms": 10,
            "total_ms": 150,
            "used_page_ids": ["docb_2"],
            "prompt_tokens": 5,
            "completion_tokens": 8,
            "model_llm": "gpt-4.1",
        },
        total_ms=150,
    )

    date_payload = _project_platform_answer(date_result)
    free_text_payload = _project_platform_answer(free_text_result)

    assert date_payload["answer"] == "2025-03-11"
    assert free_text_payload["answer"] == "Alpha. Beta. Gamma."


def test_project_platform_answer_keeps_specific_free_text_clauses() -> None:
    consolidated_result = PlatformCaseResult(
        case=SubmissionCase(case_id="q-cons", question="Q?", answer_type="free_text"),
        answer_text=(
            "The consolidated version of Law on the Application of Civil and Commercial Laws in the DIFC 2004 "
            "was published in November 2024 (cite: ff746f7b583490a80ba104361c0a82a1ebbf7ed9097cd03dc49d744cb5057761:0:0:9a3fdb82)."
        ),
        telemetry={
            "ttft_ms": 100,
            "time_per_output_token_ms": 10,
            "total_ms": 150,
            "used_page_ids": ["ff746f7b583490a80ba104361c0a82a1ebbf7ed9097cd03dc49d744cb5057761_1"],
            "prompt_tokens": 5,
            "completion_tokens": 8,
            "model_llm": "structured-extractor",
        },
        total_ms=150,
    )
    foundations_result = PlatformCaseResult(
        case=SubmissionCase(case_id="q-found", question="Q?", answer_type="free_text"),
        answer_text=(
            "Registrar administers Foundations Law, DIFC Law No. 3 of 2018 and any Regulations made under it "
            "(cite: 22442c5ee999e2519c68de908be511875a84f2b810ed540c2dcfcbcc65031434:3:0:c7e85219)"
        ),
        telemetry={
            "ttft_ms": 100,
            "time_per_output_token_ms": 10,
            "total_ms": 150,
            "used_page_ids": ["22442c5ee999e2519c68de908be511875a84f2b810ed540c2dcfcbcc65031434_4"],
            "prompt_tokens": 5,
            "completion_tokens": 8,
            "model_llm": "structured-extractor",
        },
        total_ms=150,
    )

    consolidated_payload = _project_platform_answer(consolidated_result)
    foundations_payload = _project_platform_answer(foundations_result)

    assert consolidated_payload["answer"] == (
        "The consolidated version of Law on the Application of Civil and Commercial Laws in the DIFC 2004 "
        "was published in November 2024."
    )
    assert foundations_payload["answer"] == (
        "Registrar administers Foundations Law, DIFC Law No. 3 of 2018 and any Regulations made under it"
    )


def test_project_platform_answer_canonicalizes_free_text_unanswerable_and_clears_refs() -> None:
    result = PlatformCaseResult(
        case=SubmissionCase(
            case_id="q-unanswerable",
            question="Who administers the Foundations Law?",
            answer_type="free_text",
        ),
        answer_text="There is no information on this question.",
        telemetry={
            "used_page_ids": ["doca_4"],
            "retrieved_page_ids": ["doca_4"],
            "doc_refs": ["Foundations Law 2018"],
            "model_llm": "gpt-4.1-mini",
        },
        total_ms=10,
    )

    payload = _project_platform_answer(result)

    assert payload["answer"] == "There is no information on this question in the provided documents."
    telemetry = payload["telemetry"]
    assert isinstance(telemetry, dict)
    retrieval = telemetry["retrieval"]
    assert isinstance(retrieval, dict)
    assert retrieval["retrieved_chunk_pages"] == []


def test_create_code_archive_only_includes_allowlisted_files(tmp_path: Path) -> None:
    root = tmp_path
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / ".DS_Store").write_text("junk\n", encoding="utf-8")
    (root / "README.md").write_text("# Readme\n", encoding="utf-8")
    (root / ".env").write_text("EVAL_API_KEY=secret\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "eval.json").write_text("{}", encoding="utf-8")

    archive_path = root / "platform_runs" / "warmup" / "code_archive.zip"
    allowlist = ArchiveAllowlist(
        include=["src", "README.md"],
        exclude_globs=[".env", "data/**", "platform_runs/**"],
    )

    report = _create_code_archive(root, archive_path, allowlist)

    assert report["issues"] == []
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        with pytest.raises(KeyError):
            zip_handle.getinfo(".env")
        with pytest.raises(KeyError):
            zip_handle.getinfo("src/.DS_Store")
        assert sorted(zip_handle.namelist()) == ["README.md", "src/pkg/__init__.py"]


def test_create_code_archive_rejects_secret_like_content(tmp_path: Path) -> None:
    root = tmp_path
    (root / "src").mkdir()
    (root / "src" / "secret.py").write_text(
        'EVAL_API_KEY="mcs_KSX40No6Poxlu5oyDDIlxaPN7sLUhHs7GjCexDPpZFA"\n',
        encoding="utf-8",
    )

    archive_path = root / "platform_runs" / "warmup" / "code_archive.zip"
    allowlist = ArchiveAllowlist(
        include=["src"],
        exclude_globs=["platform_runs/**"],
    )

    with pytest.raises(ValueError, match="Code archive audit failed"):
        _create_code_archive(root, archive_path, allowlist)


def test_phase_collection_override_scopes_shadow_collection() -> None:
    original_collection = os.environ.get("QDRANT_COLLECTION")
    original_page = os.environ.get("QDRANT_PAGE_COLLECTION")
    original_shadow = os.environ.get("QDRANT_SHADOW_COLLECTION")
    original_segment = os.environ.get("QDRANT_SEGMENT_COLLECTION")
    original_bridge = os.environ.get("QDRANT_BRIDGE_FACT_COLLECTION")
    original_support = os.environ.get("QDRANT_SUPPORT_FACT_COLLECTION")

    with _phase_collection_override("legal_chunks_platform_warmup"):
        assert os.environ["QDRANT_COLLECTION"] == "legal_chunks_platform_warmup"
        assert os.environ["QDRANT_PAGE_COLLECTION"] == "legal_chunks_platform_warmup_pages"
        assert os.environ["QDRANT_SHADOW_COLLECTION"] == "legal_chunks_platform_warmup_shadow"
        assert os.environ["QDRANT_SEGMENT_COLLECTION"] == "legal_chunks_platform_warmup_segments"
        assert os.environ["QDRANT_BRIDGE_FACT_COLLECTION"] == "legal_chunks_platform_warmup_bridge_facts"
        assert os.environ["QDRANT_SUPPORT_FACT_COLLECTION"] == "legal_chunks_platform_warmup_support_facts"

    assert os.environ.get("QDRANT_COLLECTION") == original_collection
    assert os.environ.get("QDRANT_PAGE_COLLECTION") == original_page
    assert os.environ.get("QDRANT_SHADOW_COLLECTION") == original_shadow
    assert os.environ.get("QDRANT_SEGMENT_COLLECTION") == original_segment
    assert os.environ.get("QDRANT_BRIDGE_FACT_COLLECTION") == original_bridge
    assert os.environ.get("QDRANT_SUPPORT_FACT_COLLECTION") == original_support


def test_extract_http_error_message_prefers_nested_error_message() -> None:
    request = httpx.Request("GET", "https://platform.agentic-challenge.ai/api/v1/questions")
    response = httpx.Response(
        403,
        request=request,
        json={"error": {"code": "403", "message": "Questions and documents are not published yet", "details": {}}},
    )
    exc = httpx.HTTPStatusError("forbidden", request=request, response=response)

    assert _extract_http_error_message(exc) == "Questions and documents are not published yet"
    assert _is_resources_not_published_error(exc) is True


def test_extract_http_error_message_handles_generic_403_without_publication_message() -> None:
    request = httpx.Request("GET", "https://platform.agentic-challenge.ai/api/v1/questions")
    response = httpx.Response(
        403,
        request=request,
        text=json.dumps({"error": {"code": "403", "message": "Team is suspended", "details": {}}}),
        headers={"content-type": "application/json"},
    )
    exc = httpx.HTTPStatusError("forbidden", request=request, response=response)

    assert _extract_http_error_message(exc) == "Team is suspended"
    assert _is_resources_not_published_error(exc) is False


def test_scan_text_for_secrets_ignores_empty_env_placeholders() -> None:
    content = "EVAL_API_KEY=\nLLM_API_KEY=\nQDRANT_API_KEY=\n"

    assert _scan_text_for_secrets(content) is False


def test_scan_text_for_secrets_detects_real_assigned_key() -> None:
    content = 'EVAL_API_KEY="mcs_VcPLTnn4XOca3HYbu8KPVBHsZxiThHnlC6Nl5z9SG10"\n'

    assert _scan_text_for_secrets(content) is True


@pytest.mark.asyncio
async def test_submit_existing_artifacts_submits_without_polling(tmp_path: Path) -> None:
    submission_path = tmp_path / "submission.json"
    archive_path = tmp_path / "code_archive.zip"
    submission_path.write_text("{}", encoding="utf-8")
    archive_path.write_bytes(b"PK\x03\x04")

    client = AsyncMock()
    client.submit_submission.return_value = {"uuid": "sub-1", "status": "queued"}

    result = await _submit_existing_artifacts(
        client,
        submission_path=submission_path,
        code_archive_path=archive_path,
        poll=False,
        poll_interval_s=1.0,
        poll_timeout_s=30.0,
        force=True,
    )

    assert result == {"uuid": "sub-1", "status": "queued"}
    client.submit_submission.assert_awaited_once_with(submission_path, archive_path)
    client.get_submission_status.assert_not_called()


@pytest.mark.asyncio
async def test_submit_existing_artifacts_raises_for_missing_files(tmp_path: Path) -> None:
    client = AsyncMock()

    with pytest.raises(FileNotFoundError, match="Submission JSON not found"):
        await _submit_existing_artifacts(
            client,
            submission_path=tmp_path / "missing-submission.json",
            code_archive_path=tmp_path / "missing-archive.zip",
            poll=False,
            poll_interval_s=1.0,
            poll_timeout_s=30.0,
            force=True,
        )


def test_check_existing_artifact_preflight_blocks_red_artifact(tmp_path: Path) -> None:
    import json

    submission_path = tmp_path / "submission.json"
    submission_path.write_text("{}", encoding="utf-8")
    preflight_path = tmp_path / "preflight_summary.json"
    preflight_path.write_text(
        json.dumps(
            {
                "support_shape_report": {"blocking_case_count": 2},
                "anomaly_report": {"anomaly_count": 3},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Preflight checks failed"):
        _check_existing_artifact_preflight(submission_path, force=False)


def test_check_existing_artifact_preflight_allows_force(tmp_path: Path) -> None:
    import json

    submission_path = tmp_path / "submission.json"
    submission_path.write_text("{}", encoding="utf-8")
    preflight_path = tmp_path / "preflight_summary.json"
    preflight_path.write_text(
        json.dumps(
            {
                "support_shape_report": {"blocking_case_count": 2},
                "anomaly_report": {"anomaly_count": 3},
            }
        ),
        encoding="utf-8",
    )

    _check_existing_artifact_preflight(submission_path, force=True)


def test_check_existing_artifact_preflight_passes_clean(tmp_path: Path) -> None:
    import json

    submission_path = tmp_path / "submission.json"
    submission_path.write_text("{}", encoding="utf-8")
    preflight_path = tmp_path / "preflight_summary.json"
    preflight_path.write_text(
        json.dumps(
            {
                "support_shape_report": {"blocking_case_count": 0},
                "anomaly_report": {"anomaly_count": 0},
            }
        ),
        encoding="utf-8",
    )

    _check_existing_artifact_preflight(submission_path, force=False)


@pytest.mark.asyncio
async def test_run_questions_uses_isolated_runtime_per_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtimes = []

    class DummyRuntime:
        def __init__(self, name: str) -> None:
            self.name = name
            self.close = AsyncMock()

    runtime_names = ["worker-a", "worker-b"]

    async def fake_runtime_factory() -> DummyRuntime:
        runtime = DummyRuntime(runtime_names[len(runtimes)])
        runtimes.append(runtime)
        return runtime

    seen: list[tuple[str, str]] = []

    async def fake_run_case_direct(
        case: SubmissionCase,
        runtime: DummyRuntime,
        *,
        fail_fast: bool,
    ) -> PlatformCaseResult:
        seen.append((case.case_id, runtime.name))
        await asyncio.sleep(0)
        return PlatformCaseResult(
            case=case,
            answer_text="True",
            telemetry={},
            total_ms=1,
        )

    monkeypatch.setattr("shafi.submission.platform._run_case_direct", fake_run_case_direct)

    cases = [
        SubmissionCase(case_id="q-1", question="Q1?", answer_type="boolean"),
        SubmissionCase(case_id="q-2", question="Q2?", answer_type="boolean"),
        SubmissionCase(case_id="q-3", question="Q3?", answer_type="boolean"),
    ]

    results = await _run_questions(
        cases,
        concurrency=2,
        fail_fast=False,
        runtime_factory=fake_runtime_factory,
    )

    assert [result.case.case_id for result in results] == ["q-1", "q-2", "q-3"]
    assert len(runtimes) == 2
    for runtime in runtimes:
        runtime.close.assert_awaited_once()
    assert {runtime_name for _, runtime_name in seen} == {"worker-a", "worker-b"}


def test_build_preflight_summary_reports_counts_and_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    phase_dir = tmp_path / "platform_runs" / "warmup"
    docs_dir = phase_dir / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "documents.zip").write_bytes(b"zip-bytes")
    (docs_dir / "doc-1.pdf").write_bytes(b"%PDF-1.4")
    (docs_dir / "doc-2.pdf").write_bytes(b"%PDF-1.4")

    submission_path = phase_dir / "submission.json"
    submission_path.write_text("{}", encoding="utf-8")
    archive_path = phase_dir / "code_archive.zip"
    archive_path.write_bytes(b"PK\x03\x04")
    questions_path = phase_dir / "questions.json"
    questions_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "shafi.submission.platform.build_score_settings_fingerprint",
        lambda _settings: {
            "sha256": "abc123" * 10 + "ab",
            "settings": {
                "pipeline": {"enable_grounding_sidecar": False},
                "ingestion": {"ingest_version": "v3_anchor_retrieval"},
            },
        },
    )

    paths = PlatformPaths(
        phase_dir=phase_dir,
        docs_dir=docs_dir,
        questions_path=questions_path,
        submission_path=submission_path,
        raw_results_path=phase_dir / "raw_results.json",
        code_archive_path=archive_path,
        audit_report_path=phase_dir / "audit.json",
        status_path=phase_dir / "status.json",
        preflight_summary_path=phase_dir / "preflight_summary.json",
        canary_path=phase_dir / "canary.json",
        truth_audit_path=phase_dir / "truth_audit_scaffold.json",
        truth_audit_workbook_path=phase_dir / "truth_audit_workbook.md",
    )

    results = [
        PlatformCaseResult(
            case=SubmissionCase(case_id="q-1", question="Q1?", answer_type="boolean"),
            answer_text="Yes",
            telemetry={"model_llm": "gpt-4.1-mini"},
            total_ms=10,
        ),
        PlatformCaseResult(
            case=SubmissionCase(case_id="q-2", question="Q2?", answer_type="free_text"),
            answer_text="Alpha. Beta.",
            telemetry={"model_llm": "gpt-4.1"},
            total_ms=20,
        ),
    ]
    payload = {
        "architecture_summary": "summary",
        "answers": [
            {
                "question_id": "q-1",
                "answer": True,
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "doca", "page_numbers": [1, 2]}]},
                    "model_name": "gpt-4.1-mini",
                },
            },
            {
                "question_id": "q-2",
                "answer": "Alpha. Beta.",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": []},
                    "model_name": "",
                },
            },
        ],
    }

    summary = _build_preflight_summary(
        paths=paths,
        collection_name="legal_chunks_warmup",
        payload=payload,
        results=results,
        point_count=42,
        anomaly_report={"anomaly_case_ids": ["q-2"], "anomaly_count": 1},
        canary_report={"answer_drift_count": 0},
        support_shape_report={"blocking_case_count": 0},
        truth_audit_report={"deterministic_incomplete_count": 70},
    )

    assert summary["phase"] == "warmup"
    assert summary["questions_count"] == 2
    assert summary["answer_type_counts"] == {"boolean": 1, "free_text": 1}
    assert summary["null_answer_counts_by_type"] == {}
    assert summary["empty_retrieved_chunk_pages_counts_by_type"] == {"free_text": 1}
    assert summary["page_count_distribution"] == {"min": 0, "p50": 0, "p95": 0, "max": 2}
    assert summary["free_text_char_distribution"] == {"min": 12, "p50": 12, "p95": 12, "max": 12}
    assert summary["free_text_sentence_distribution"] == {"min": 2, "p50": 2, "p95": 2, "max": 2}
    assert summary["model_name_empty_count"] == 1
    assert summary["pdf_count"] == 2
    assert summary["anomaly_report"] == {"anomaly_case_ids": ["q-2"], "anomaly_count": 1}
    assert summary["equivalence_canary"] == {"answer_drift_count": 0}
    assert summary["support_shape_report"] == {"blocking_case_count": 0}
    assert summary["truth_audit_report"] == {"deterministic_incomplete_count": 70}
    assert summary["truth_audit_workbook_path"] == str(phase_dir / "truth_audit_workbook.md")
    assert summary["raw_results_path"] == str(phase_dir / "raw_results.json")
    assert summary["phase_collection_name"] == "legal_chunks_warmup"
    assert summary["qdrant_point_count"] == 42
    assert summary["score_settings_sha256"] == "abc123" * 10 + "ab"
    assert summary["score_settings_fingerprint"] == {
        "pipeline": {"enable_grounding_sidecar": False},
        "ingestion": {"ingest_version": "v3_anchor_retrieval"},
    }
    assert isinstance(summary["submission_sha256"], str) and summary["submission_sha256"]
    assert isinstance(summary["code_archive_sha256"], str) and summary["code_archive_sha256"]
    assert isinstance(summary["questions_sha256"], str) and summary["questions_sha256"]
    assert isinstance(summary["documents_zip_sha256"], str) and summary["documents_zip_sha256"]


def test_build_anchor_page_challenger_payload_adds_title_and_page_two_anchors() -> None:
    questions_by_id = {
        "q-title": SubmissionCase(
            case_id="q-title",
            question="According to the title page of case CFI 010/2024, who is the defendant?",
            answer_type="name",
        ),
        "q-page-2": SubmissionCase(
            case_id="q-page-2",
            question="According to page 2 of case DEC 001/2025, what is the order number?",
            answer_type="name",
        ),
        "q-compare": SubmissionCase(
            case_id="q-compare",
            question="Which case has an earlier Date of Issue: CFI 010/2024 or CFI 016/2025?",
            answer_type="name",
        ),
    }
    source_payload = {
        "architecture_summary": "summary",
        "answers": [
            {
                "question_id": "q-title",
                "answer": "ONORA",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "cfi-010-2024", "page_numbers": [3]}]},
                    "model_name": "strict-extractor",
                },
            },
            {
                "question_id": "q-page-2",
                "answer": "DEC 001/2025",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "dec-001-2025", "page_numbers": [5]}]},
                    "model_name": "strict-extractor",
                },
            },
            {
                "question_id": "q-compare",
                "answer": "CFI 010/2024",
                "telemetry": {
                    "retrieval": {
                        "retrieved_chunk_pages": [
                            {"doc_id": "cfi-010-2024", "page_numbers": [8]},
                            {"doc_id": "cfi-016-2025", "page_numbers": [14]},
                        ]
                    },
                    "model_name": "strict-extractor",
                },
            },
        ],
    }

    challenger = _build_anchor_page_challenger_payload(
        source_payload=source_payload,
        questions_by_id=questions_by_id,
    )
    answers = {answer["question_id"]: answer for answer in challenger["answers"]}

    assert answers["q-title"]["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [
        {"doc_id": "cfi-010-2024", "page_numbers": [1, 3]},
    ]
    assert answers["q-page-2"]["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [
        {"doc_id": "dec-001-2025", "page_numbers": [2, 5]},
    ]
    assert answers["q-compare"]["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [
        {"doc_id": "cfi-010-2024", "page_numbers": [1, 8]},
        {"doc_id": "cfi-016-2025", "page_numbers": [1, 14]},
    ]


def test_build_all_context_pages_challenger_payload_uses_context_page_ids() -> None:
    source_payload = {
        "architecture_summary": "summary",
        "answers": [
            {
                "question_id": "q-1",
                "answer": "ONORA",
                "telemetry": {
                    "retrieval": {"retrieved_chunk_pages": [{"doc_id": "arb-034", "page_numbers": [2]}]},
                    "model_name": "strict-extractor",
                },
            }
        ],
    }
    source_results = [
        PlatformCaseResult(
            case=SubmissionCase(
                case_id="q-1", question="Who is the defendant in case ARB 034/2025?", answer_type="name"
            ),
            answer_text="ONORA",
            telemetry={"context_page_ids": ["arb-034_1", "arb-034_2", "arb-034_4"]},
            total_ms=10,
        )
    ]

    challenger = _build_all_context_pages_challenger_payload(
        source_payload=source_payload,
        source_results=source_results,
    )
    answers = {answer["question_id"]: answer for answer in challenger["answers"]}

    assert answers["q-1"]["telemetry"]["retrieval"]["retrieved_chunk_pages"] == [
        {"doc_id": "arb-034", "page_numbers": [1, 2, 4]},
    ]


def test_resolve_source_truth_audit_path_infers_suffixed_scaffold(tmp_path: Path) -> None:
    source_submission_path = tmp_path / "submission_v3_recovered.json"
    source_submission_path.write_text("{}", encoding="utf-8")
    inferred_truth_audit_path = tmp_path / "truth_audit_scaffold_v3_recovered.json"
    inferred_truth_audit_path.write_text("{}", encoding="utf-8")

    resolved = _resolve_source_truth_audit_path(source_submission_path, None)

    assert resolved == inferred_truth_audit_path


def test_resolve_source_truth_audit_path_fails_when_inferred_scaffold_missing(tmp_path: Path) -> None:
    source_submission_path = tmp_path / "submission_v3_recovered.json"
    source_submission_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Source truth audit scaffold not found"):
        _resolve_source_truth_audit_path(source_submission_path, None)


def test_validate_platform_args_rejects_support_only_submit_combo() -> None:
    args = argparse.Namespace(
        support_only_challenger="anchor-page-restitution",
        submit=True,
    )

    with pytest.raises(ValueError, match="cannot be combined with --submit"):
        _validate_platform_args(args)


@pytest.mark.asyncio
async def test_async_main_support_only_challenger_preserves_source_truth_audit_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_dir = tmp_path / "warmup"
    paths = PlatformPaths(
        phase_dir=phase_dir,
        docs_dir=phase_dir / "documents",
        questions_path=phase_dir / "questions.json",
        submission_path=phase_dir / "submission.json",
        raw_results_path=phase_dir / "raw_results.json",
        code_archive_path=phase_dir / "code_archive.zip",
        audit_report_path=phase_dir / "code_archive_audit.json",
        status_path=phase_dir / "submission_status.json",
        preflight_summary_path=phase_dir / "preflight_summary.json",
        canary_path=phase_dir / "equivalence_canary.json",
        truth_audit_path=phase_dir / "truth_audit_scaffold.json",
        truth_audit_workbook_path=phase_dir / "truth_audit_workbook.md",
    )
    source_submission_path = tmp_path / "submission_v4.json"
    source_questions_path = tmp_path / "questions.json"
    source_truth_audit_path = tmp_path / "truth_audit_scaffold_v4.json"

    source_submission_path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q-title",
                        "answer": "ONORA",
                        "telemetry": {
                            "model_name": "strict-extractor",
                            "retrieval": {
                                "retrieved_chunk_pages": [
                                    {"doc_id": "cfi-010-2024", "page_numbers": [3]},
                                ]
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source_questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "q-title",
                    "question": "According to the title page of case CFI 010/2024, who is the defendant?",
                    "answer_type": "name",
                }
            ]
        ),
        encoding="utf-8",
    )
    source_truth_audit_path.write_text(
        json.dumps(
            {
                "summary": {
                    "manual_verdict_counts": {
                        "deterministic_complete": 1,
                        "deterministic_incomplete": 0,
                        "free_text_complete": 0,
                        "free_text_incomplete": 0,
                    }
                },
                "records": [
                    {
                        "question_id": "q-title",
                        "manual_verdict": "correct",
                        "expected_answer": "ONORA",
                        "minimal_required_support_pages": ["cfi-010-2024_1"],
                        "manual_exactness_labels": ["semantic_correct", "page_specific_exact_risk"],
                        "failure_class": "support_undercoverage",
                        "notes": "keep this note",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    settings = SimpleNamespace(
        app=SimpleNamespace(log_level="INFO", log_format="text"),
        platform=SimpleNamespace(archive_allowlist_path="unused", phase="warmup"),
    )
    monkeypatch.setattr("shafi.submission.platform.get_settings", lambda: settings)
    monkeypatch.setattr(
        "shafi.submission.platform.build_score_settings_fingerprint",
        lambda _settings: {"sha256": "0" * 64, "settings": {"platform": {"phase": "warmup"}}},
    )
    monkeypatch.setattr("shafi.submission.platform.setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("shafi.submission.platform._resolve_phase_paths", lambda: paths)
    monkeypatch.setattr("shafi.submission.platform._phase_collection_name", lambda: "legal_chunks_warmup")
    monkeypatch.setattr(
        "shafi.submission.platform._load_archive_allowlist",
        lambda _path: ArchiveAllowlist(include=["src"], exclude_globs=[]),
    )
    monkeypatch.setattr(
        "shafi.submission.platform._write_archive_artifacts",
        lambda _root, _paths, _allowlist: None,
    )

    args = argparse.Namespace(
        archive_only=False,
        support_only_challenger="anchor-page-restitution",
        source_submission_path=str(source_submission_path),
        source_questions_path=str(source_questions_path),
        source_truth_audit_path=str(source_truth_audit_path),
        source_raw_results_path=None,
        artifact_suffix="challenger",
        refresh_downloads=False,
        skip_ingest=False,
        submit=False,
        submit_existing=False,
        submission_path=None,
        code_archive_path=None,
        query_concurrency=None,
        equivalence_canary_concurrency=None,
        poll=False,
        fail_fast=False,
    )

    exit_code = await _async_main(args)

    assert exit_code == 0
    generated_scaffold = json.loads((phase_dir / "truth_audit_scaffold_challenger.json").read_text(encoding="utf-8"))
    record = generated_scaffold["records"][0]
    assert record["manual_verdict"] == "correct"
    assert record["expected_answer"] == "ONORA"
    assert record["minimal_required_support_pages"] == ["cfi-010-2024_1"]
    assert record["manual_exactness_labels"] == ["semantic_correct", "page_specific_exact_risk"]
    assert record["failure_class"] == "support_undercoverage"
    assert record["notes"] == "keep this note"

    preflight_summary = json.loads((phase_dir / "preflight_summary_challenger.json").read_text(encoding="utf-8"))
    assert preflight_summary["support_only_challenger"] == {
        "mode": "anchor-page-restitution",
        "source_submission_path": str(source_submission_path),
        "source_truth_audit_path": str(source_truth_audit_path),
        "source_submission_sha256": preflight_summary["support_only_challenger"]["source_submission_sha256"],
        "same_answers_sha256": preflight_summary["support_only_challenger"]["same_answers_sha256"],
        "answer_changed_count": 0,
        "page_changed_count": 1,
        "source_manual_verdict_counts": {
            "deterministic_complete": 1,
            "deterministic_incomplete": 0,
            "free_text_complete": 0,
            "free_text_incomplete": 0,
        },
    }
    assert preflight_summary["support_only_challenger"]["same_answers_sha256"]


def test_result_anomaly_flags_detect_specific_unsupported_with_support_pages() -> None:
    result = PlatformCaseResult(
        case=SubmissionCase(
            case_id="q-foundations",
            question="Who administers the Foundations Law?",
            answer_type="free_text",
        ),
        answer_text="There is no information on this question in the provided documents.",
        telemetry={
            "used_page_ids": ["doca_4"],
            "retrieved_page_ids": ["doca_4"],
            "doc_refs": ["Foundations Law 2018"],
        },
        total_ms=10,
    )

    flags = _result_anomaly_flags(result)

    assert flags == [
        "specific_question_unsupported",
        "unsupported_with_support_pages",
        "unsupported_with_retrieved_pages",
        "unsupported_with_doc_refs",
    ]


def test_result_anomaly_flags_detect_projection_induced_unsupported() -> None:
    result = PlatformCaseResult(
        case=SubmissionCase(
            case_id="q-published",
            question="When was the consolidated version of the Law on the Application of Civil and Commercial Laws in the DIFC published?",
            answer_type="free_text",
        ),
        answer_text="",
        telemetry={
            "used_page_ids": ["doca_1", "doca_2", "doca_3"],
            "retrieved_page_ids": ["doca_1", "doca_2", "doca_3"],
        },
        total_ms=10,
    )

    flags = _result_anomaly_flags(result)

    assert flags == [
        "specific_question_unsupported",
        "unsupported_with_support_pages",
        "unsupported_with_retrieved_pages",
    ]


def test_result_anomaly_flags_detect_answerable_support_pages_lost_in_projection() -> None:
    result = PlatformCaseResult(
        case=SubmissionCase(
            case_id="q-boolean",
            question="Does Article 28 apply?",
            answer_type="boolean",
        ),
        answer_text="Yes",
        telemetry={
            "used_page_ids": ["bad-page-id"],
            "model_llm": "strict-extractor",
        },
        total_ms=10,
    )

    flags = _result_anomaly_flags(result)

    assert flags == ["answerable_support_pages_lost_in_projection"]


def test_build_results_anomaly_report_collects_case_ids() -> None:
    clean = PlatformCaseResult(
        case=SubmissionCase(case_id="q-ok", question="What is the law number?", answer_type="number"),
        answer_text="7",
        telemetry={"used_page_ids": ["doc_1"]},
        total_ms=10,
    )
    anomalous = PlatformCaseResult(
        case=SubmissionCase(case_id="q-bad", question="Who administers the Foundations Law?", answer_type="free_text"),
        answer_text="There is no information on this question in the provided documents.",
        telemetry={"used_page_ids": ["doc_4"]},
        total_ms=10,
    )

    report = _build_results_anomaly_report([clean, anomalous])

    assert report["anomaly_case_ids"] == ["q-bad"]
    assert report["anomaly_count"] == 1


def test_build_equivalence_canary_reports_answer_and_page_drift() -> None:
    base = [
        PlatformCaseResult(
            case=SubmissionCase(case_id="q-1", question="Q1?", answer_type="boolean"),
            answer_text="Yes",
            telemetry={"used_page_ids": ["doca_1"], "model_llm": "strict-extractor"},
            total_ms=10,
        )
    ]
    candidate = [
        PlatformCaseResult(
            case=SubmissionCase(case_id="q-1", question="Q1?", answer_type="boolean"),
            answer_text="No",
            telemetry={"used_page_ids": ["doca_2"], "model_llm": "gpt-4.1-mini"},
            total_ms=10,
        )
    ]

    report = _build_equivalence_canary(
        baseline_results=base,
        candidate_results=candidate,
        baseline_concurrency=1,
        candidate_concurrency=2,
    )

    assert report["answer_drift_case_ids"] == ["q-1"]
    assert report["model_drift_case_ids"] == ["q-1"]
    assert report["page_drift_case_ids"] == ["q-1"]


def test_build_support_shape_report_collects_blocking_flags() -> None:
    scaffold = {
        "records": [
            {
                "question_id": "q-ok",
                "support_shape_flags": [],
            },
            {
                "question_id": "q-compare",
                "support_shape_flags": ["comparison_missing_side"],
            },
            {
                "question_id": "q-meta",
                "support_shape_flags": ["metadata_multi_atom_maybe_undercovered"],
            },
            {
                "question_id": "q-info",
                "support_shape_flags": ["unsupported_with_support_pages"],
            },
            {
                "question_id": "q-outcome",
                "support_shape_flags": ["case_outcome_disposition_maybe_missing"],
            },
        ]
    }

    report = _build_support_shape_report(scaffold)

    assert report["flagged_case_ids"] == ["q-compare", "q-meta", "q-info", "q-outcome"]
    assert report["blocking_case_ids"] == ["q-compare", "q-meta", "q-outcome"]
    assert report["flag_counts"] == {
        "comparison_missing_side": 1,
        "metadata_multi_atom_maybe_undercovered": 1,
        "unsupported_with_support_pages": 1,
        "case_outcome_disposition_maybe_missing": 1,
    }


def test_build_truth_audit_report_counts_incomplete_deterministic_cases() -> None:
    scaffold = {
        "records": [
            {
                "question_id": "q-det-complete",
                "answer_type": "boolean",
                "manual_verdict": "correct",
            },
            {
                "question_id": "q-det-missing",
                "answer_type": "name",
                "manual_verdict": "",
            },
            {
                "question_id": "q-free-missing",
                "answer_type": "free_text",
                "manual_verdict": "",
            },
        ]
    }

    report = _build_truth_audit_report(scaffold)

    assert report == {
        "deterministic_complete_count": 1,
        "deterministic_incomplete_count": 1,
        "deterministic_incomplete_case_ids": ["q-det-missing"],
        "free_text_complete_count": 0,
        "free_text_incomplete_count": 1,
        "free_text_incomplete_case_ids": ["q-free-missing"],
    }


def test_resolve_query_concurrency_defaults_to_safe_mode() -> None:
    assert _resolve_query_concurrency(None) == 1
    assert _resolve_query_concurrency(1) == 1
    assert _resolve_query_concurrency(3) == 3


@pytest.mark.asyncio
async def test_repair_anomalous_results_replaces_fixed_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRuntime:
        async def close(self) -> None:
            return None

    async def fake_runtime_factory() -> DummyRuntime:
        return DummyRuntime()

    async def fake_run_case_direct(
        case: SubmissionCase,
        runtime: DummyRuntime,
        *,
        fail_fast: bool,
    ) -> PlatformCaseResult:
        assert case.case_id == "q-bad"
        return PlatformCaseResult(
            case=case,
            answer_text="The Foundations Law is administered by the Registrar.",
            telemetry={"used_page_ids": ["doca_4"]},
            total_ms=11,
        )

    monkeypatch.setattr("shafi.submission.platform._run_case_direct", fake_run_case_direct)

    anomalous = PlatformCaseResult(
        case=SubmissionCase(case_id="q-bad", question="Who administers the Foundations Law?", answer_type="free_text"),
        answer_text="There is no information on this question in the provided documents.",
        telemetry={"used_page_ids": ["doca_4"]},
        total_ms=10,
    )

    repaired, report = await _repair_anomalous_results(
        [anomalous],
        fail_fast=False,
        runtime_factory=fake_runtime_factory,
    )

    assert repaired[0].answer_text == "The Foundations Law is administered by the Registrar."
    assert report["repaired_case_ids"] == ["q-bad"]


@pytest.mark.asyncio
async def test_run_case_direct_marks_first_token_for_fallback_answer() -> None:
    class DummyPipeline:
        async def astream(self, *_args, **_kwargs):
            if False:
                yield {}
            raise RuntimeError("boom")

    runtime = SimpleNamespace(pipeline=DummyPipeline())
    result = await _run_case_direct(
        SubmissionCase(case_id="q-1", question="Q?", answer_type="boolean"),
        runtime,
        fail_fast=False,
    )

    assert result.answer_text == "null"
    assert result.telemetry["ttft_ms"] <= result.telemetry["total_ms"]


@pytest.mark.asyncio
async def test_platform_client_retries_rate_limited_question_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = PlatformEvaluationClient(api_key="test-key", base_url="https://platform.example/api/v1")
    await client.close()

    request = httpx.Request("GET", "https://platform.example/api/v1/questions")
    first = httpx.Response(429, request=request, headers={"Retry-After": "0"})
    second = httpx.Response(200, request=request, json=[{"id": "q-1", "question": "Q?", "answer_type": "boolean"}])
    get_mock = AsyncMock(side_effect=[first, second])
    sleep_mock = AsyncMock()
    client._client = AsyncMock()
    client._client.get = get_mock
    monkeypatch.setattr("shafi.submission.platform.asyncio.sleep", sleep_mock)

    target_path = tmp_path / "questions.json"
    downloaded = await client.download_questions(target_path)

    assert downloaded == target_path
    assert json.loads(target_path.read_text(encoding="utf-8")) == [
        {"id": "q-1", "question": "Q?", "answer_type": "boolean"}
    ]
    assert get_mock.await_count == 2
    sleep_mock.assert_awaited_once()
