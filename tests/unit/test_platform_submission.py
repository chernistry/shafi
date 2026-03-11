from __future__ import annotations

import asyncio
import json
import zipfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import httpx
import pytest

from rag_challenge.submission.common import SubmissionCase
from rag_challenge.submission.platform import (
    ArchiveAllowlist,
    PlatformCaseResult,
    PlatformEvaluationClient,
    PlatformPaths,
    _build_preflight_summary,
    _create_code_archive,
    _extract_http_error_message,
    _is_resources_not_published_error,
    _project_platform_answer,
    _run_questions,
    _scan_text_for_secrets,
    _submit_existing_artifacts,
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
        'EVAL_API_KEY="mcs_VcPLTnn4XOca3HYbu8KPVBHsZxiThHnlC6Nl5z9SG10"\n',
        encoding="utf-8",
    )

    archive_path = root / "platform_runs" / "warmup" / "code_archive.zip"
    allowlist = ArchiveAllowlist(
        include=["src"],
        exclude_globs=["platform_runs/**"],
    )

    with pytest.raises(ValueError, match="Code archive audit failed"):
        _create_code_archive(root, archive_path, allowlist)


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
        )


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

    monkeypatch.setattr("rag_challenge.submission.platform._run_case_direct", fake_run_case_direct)

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


def test_build_preflight_summary_reports_counts_and_hashes(tmp_path: Path) -> None:
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

    paths = PlatformPaths(
        phase_dir=phase_dir,
        docs_dir=docs_dir,
        questions_path=questions_path,
        submission_path=submission_path,
        code_archive_path=archive_path,
        audit_report_path=phase_dir / "audit.json",
        status_path=phase_dir / "status.json",
        preflight_summary_path=phase_dir / "preflight_summary.json",
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
    assert summary["phase_collection_name"] == "legal_chunks_warmup"
    assert summary["qdrant_point_count"] == 42
    assert isinstance(summary["submission_sha256"], str) and summary["submission_sha256"]
    assert isinstance(summary["code_archive_sha256"], str) and summary["code_archive_sha256"]
    assert isinstance(summary["questions_sha256"], str) and summary["questions_sha256"]
    assert isinstance(summary["documents_zip_sha256"], str) and summary["documents_zip_sha256"]


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
    monkeypatch.setattr("rag_challenge.submission.platform.asyncio.sleep", sleep_mock)

    target_path = tmp_path / "questions.json"
    downloaded = await client.download_questions(target_path)

    assert downloaded == target_path
    assert json.loads(target_path.read_text(encoding="utf-8")) == [
        {"id": "q-1", "question": "Q?", "answer_type": "boolean"}
    ]
    assert get_mock.await_count == 2
    sleep_mock.assert_awaited_once()
