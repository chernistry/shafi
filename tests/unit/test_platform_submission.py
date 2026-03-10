from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

import httpx
import pytest

from rag_challenge.submission.common import SubmissionCase
from rag_challenge.submission.platform import (
    ArchiveAllowlist,
    PlatformCaseResult,
    _create_code_archive,
    _extract_http_error_message,
    _is_resources_not_published_error,
    _project_platform_answer,
    _scan_text_for_secrets,
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
