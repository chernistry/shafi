from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx

from rag_challenge.config import get_settings
from rag_challenge.config.logging import setup_logging
from rag_challenge.core.classifier import QueryClassifier
from rag_challenge.core.embedding import EmbeddingClient
from rag_challenge.core.pipeline import RAGPipelineBuilder
from rag_challenge.core.qdrant import QdrantStore
from rag_challenge.core.reranker import RerankerClient
from rag_challenge.core.retriever import HybridRetriever
from rag_challenge.core.verifier import AnswerVerifier
from rag_challenge.ingestion.pipeline import IngestionPipeline
from rag_challenge.llm.generator import RAGGenerator
from rag_challenge.llm.provider import LLMProvider
from rag_challenge.submission.common import (
    SubmissionAnswer,
    SubmissionCase,
    as_int,
    classify_unanswerable_answer,
    coerce_answer_type,
    count_submission_sentences,
    load_cases,
    normalize_date_answer,
    select_submission_used_pages,
)
from rag_challenge.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable

_DEFAULT_UNANSWERABLE_FREE_TEXT = "There is no information on this question in the provided documents."
_PAGE_ID_RE = re.compile(r"^(?P<doc_id>.+)_(?P<page>\d+)$")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bmcs_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\biuak_v1_[A-Za-z0-9_\-]{24,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)
_NONEMPTY_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?m)^[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[ \t]*=[ \t]*(?!$|\"\"$|''$|<your-|your-|xxxx|xxxxx).+$"
)
_FORBIDDEN_ARCHIVE_PARTS = {
    ".git",
    ".venv",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    ".rapidocr_models",
    ".benchmarks",
    ".cursor",
    ".sdd",
    "data",
    "dataset",
    "starter_kit",
    "platform_runs",
}
_MAX_CODE_ARCHIVE_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class ArchiveAllowlist:
    include: list[str]
    exclude_globs: list[str]


@dataclass(frozen=True)
class PlatformPaths:
    phase_dir: Path
    docs_dir: Path
    questions_path: Path
    submission_path: Path
    code_archive_path: Path
    audit_report_path: Path
    status_path: Path
    preflight_summary_path: Path


@dataclass(frozen=True)
class PlatformCaseResult:
    case: SubmissionCase
    answer_text: str
    telemetry: dict[str, object]
    total_ms: int


@dataclass
class PipelineRuntime:
    pipeline_builder: RAGPipelineBuilder
    pipeline: Any
    embedder: EmbeddingClient
    store: QdrantStore
    reranker: RerankerClient
    llm: LLMProvider
    verifier: AnswerVerifier | None

    async def close(self) -> None:
        await self.embedder.close()
        await self.store.close()
        await self.reranker.close()
        await self.llm.close()


class PlatformEvaluationClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True,
        )

    @classmethod
    def from_settings(cls) -> PlatformEvaluationClient:
        settings = get_settings().platform
        api_key = settings.api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("EVAL_API_KEY is not configured")
        return cls(api_key=api_key, base_url=settings.base_url)

    async def close(self) -> None:
        await self._client.aclose()

    async def download_questions(self, target_path: Path) -> Path:
        response = await self._client.get(f"{self._base_url}/questions")
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(response.json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_path

    async def download_documents(self, target_dir: Path) -> Path:
        response = await self._client.get(f"{self._base_url}/documents")
        response.raise_for_status()
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / "documents.zip"
        archive_path.write_bytes(response.content)
        with zipfile.ZipFile(archive_path, "r") as zip_handle:
            zip_handle.extractall(target_dir)
        return target_dir

    async def submit_submission(self, submission_path: Path, code_archive_path: Path) -> dict[str, object]:
        with submission_path.open("rb") as submission_handle, code_archive_path.open("rb") as archive_handle:
            response = await self._client.post(
                f"{self._base_url}/submissions",
                files={
                    "file": (submission_path.name, submission_handle, "application/json"),
                    "code_archive": (code_archive_path.name, archive_handle, "application/zip"),
                },
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Submission response must be an object")
        return cast("dict[str, object]", payload)

    async def get_submission_status(self, submission_uuid: str) -> dict[str, object]:
        response = await self._client.get(f"{self._base_url}/submissions/{submission_uuid}/status")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Status response must be an object")
        return cast("dict[str, object]", payload)


def _extract_http_error_message(exc: httpx.HTTPStatusError) -> str:
    try:
        payload = exc.response.json()
    except ValueError:
        return exc.response.text.strip()
    if isinstance(payload, dict):
        payload_dict = cast("dict[str, object]", payload)
        error_obj = payload_dict.get("error")
        if isinstance(error_obj, dict):
            error_dict = cast("dict[str, object]", error_obj)
            message = error_dict.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        detail = payload_dict.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
    return exc.response.text.strip()


def _is_resources_not_published_error(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 403:
        return False
    message = _extract_http_error_message(exc).lower()
    return "not published yet" in message or "questions and documents are not published yet" in message


def _resolve_phase_paths() -> PlatformPaths:
    settings = get_settings().platform
    phase_dir = Path(settings.work_dir) / settings.phase
    return PlatformPaths(
        phase_dir=phase_dir,
        docs_dir=phase_dir / settings.documents_dirname,
        questions_path=phase_dir / settings.questions_filename,
        submission_path=phase_dir / settings.submission_filename,
        code_archive_path=phase_dir / settings.code_archive_filename,
        audit_report_path=phase_dir / "code_archive_audit.json",
        status_path=phase_dir / "submission_status.json",
        preflight_summary_path=phase_dir / "preflight_summary.json",
    )


def _phase_collection_name() -> str:
    settings = get_settings()
    return f"{settings.platform.collection_prefix}_{settings.platform.phase}"


@contextmanager
def _phase_collection_override(collection_name: str):
    previous = os.environ.get("QDRANT_COLLECTION")
    os.environ["QDRANT_COLLECTION"] = collection_name
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("QDRANT_COLLECTION", None)
        else:
            os.environ["QDRANT_COLLECTION"] = previous
        get_settings.cache_clear()


async def _build_pipeline_runtime() -> PipelineRuntime:
    settings = get_settings()
    embedder = EmbeddingClient()
    store = QdrantStore()
    reranker = RerankerClient()
    llm = LLMProvider()
    verifier = AnswerVerifier(llm) if settings.verifier.enabled else None

    retriever = HybridRetriever(store=store, embedder=embedder)
    generator = RAGGenerator(llm=llm)
    classifier = QueryClassifier()
    pipeline_builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
        verifier=verifier,
    )
    return PipelineRuntime(
        pipeline_builder=pipeline_builder,
        pipeline=pipeline_builder.compile(),
        embedder=embedder,
        store=store,
        reranker=reranker,
        llm=llm,
        verifier=verifier,
    )


def _default_answer_text(answer_type: str) -> str:
    if answer_type.strip().lower() == "free_text":
        return _DEFAULT_UNANSWERABLE_FREE_TEXT
    return "null"


async def _run_case_direct(
    case: SubmissionCase,
    runtime: PipelineRuntime,
    *,
    fail_fast: bool,
) -> PlatformCaseResult:
    collector = TelemetryCollector(
        request_id=case.case_id,
        question_id=case.case_id,
        answer_type=case.answer_type,
    )
    tokens: list[str] = []
    answer_final = ""
    telemetry_payload: dict[str, object] | None = None
    t0 = time.perf_counter()

    try:
        async for event in runtime.pipeline.astream(
            {
                "query": case.question,
                "request_id": case.case_id,
                "question_id": case.case_id,
                "answer_type": case.answer_type,
                "collector": collector,
            },
            stream_mode="custom",
        ):
            event_type = str(event.get("type") or "")
            if event_type == "token":
                tokens.append(str(event.get("text", "")))
            elif event_type == "answer_final":
                answer_final = str(event.get("text", "")).strip()
            elif event_type == "telemetry":
                payload_obj = event.get("payload")
                if isinstance(payload_obj, dict):
                    telemetry_payload = cast("dict[str, object]", payload_obj)
    except Exception:
        if fail_fast:
            raise
        logger.exception("Pipeline execution failed for question_id=%s", case.case_id)
        answer_final = _default_answer_text(case.answer_type)
        telemetry_payload = collector.finalize().model_dump()

    total_ms = int((time.perf_counter() - t0) * 1000.0)
    if not answer_final:
        answer_final = "".join(tokens).strip() or _default_answer_text(case.answer_type)
    if telemetry_payload is None:
        telemetry_payload = collector.finalize().model_dump()

    return PlatformCaseResult(
        case=case,
        answer_text=answer_final,
        telemetry=telemetry_payload,
        total_ms=total_ms,
    )


def _page_ids_to_retrieval_refs(page_ids: list[str]) -> list[dict[str, object]]:
    by_doc: dict[str, set[int]] = {}
    for page_id in page_ids:
        match = _PAGE_ID_RE.match(page_id.strip())
        if match is None:
            continue
        doc_id = match.group("doc_id").strip()
        page_raw = match.group("page").strip()
        if not doc_id or not page_raw.isdigit():
            continue
        by_doc.setdefault(doc_id, set()).add(int(page_raw))
    return [
        {"doc_id": doc_id, "page_numbers": sorted(page_numbers)}
        for doc_id, page_numbers in sorted(by_doc.items())
    ]


def _validate_projected_answer(answer: SubmissionAnswer, answer_type: str) -> None:
    answer_type_key = answer_type.strip().lower()
    if answer is None:
        return
    if answer_type_key == "boolean" and not isinstance(answer, bool):
        raise ValueError(f"Boolean answer must project to JSON boolean or null, got {answer!r}")
    if answer_type_key == "number" and not isinstance(answer, int | float):
        raise ValueError(f"Number answer must project to JSON number or null, got {answer!r}")
    if answer_type_key == "name" and not isinstance(answer, str):
        raise ValueError(f"Name answer must project to string or null, got {answer!r}")
    if answer_type_key == "names":
        if not isinstance(answer, list):
            raise ValueError(f"Names answer must project to list[str] or null, got {answer!r}")
        items = cast("list[object]", answer)
        if any(not isinstance(item, str) for item in items):
            raise ValueError(f"Names answer must project to list[str] or null, got {answer!r}")
    if answer_type_key == "date":
        if not isinstance(answer, str):
            raise ValueError(f"Date answer must project to ISO string or null, got {answer!r}")
        if normalize_date_answer(answer) != answer:
            raise ValueError(f"Date answer must be ISO YYYY-MM-DD or null, got {answer!r}")
    if answer_type_key == "free_text":
        if not isinstance(answer, str):
            raise ValueError(f"Free-text answer must project to string, got {answer!r}")
        if len(answer) > 280:
            raise ValueError(f"Free-text answer exceeds 280 chars: {len(answer)}")
        sentence_count = count_submission_sentences(answer)
        if sentence_count < 1 or sentence_count > 3:
            raise ValueError(f"Free-text answer must contain 1-3 sentences, got {sentence_count}")


def _project_platform_answer(result: PlatformCaseResult) -> dict[str, object]:
    answer_type = result.case.answer_type
    answer_text = result.answer_text
    telemetry = result.telemetry
    is_unanswerable_strict, is_unanswerable_free_text = classify_unanswerable_answer(answer_text, answer_type)

    answer_out: SubmissionAnswer = None if is_unanswerable_strict else coerce_answer_type(answer_text, answer_type)
    if answer_out is None and not is_unanswerable_strict and answer_type.strip().lower() == "free_text":
        answer_out = _DEFAULT_UNANSWERABLE_FREE_TEXT
    _validate_projected_answer(answer_out, answer_type)

    used_pages = select_submission_used_pages(telemetry)
    if is_unanswerable_strict or is_unanswerable_free_text:
        used_pages = []

    return {
        "question_id": result.case.case_id,
        "answer": answer_out,
        "telemetry": {
            "timing": {
                "ttft_ms": as_int(telemetry.get("ttft_ms"), 0),
                "tpot_ms": as_int(telemetry.get("time_per_output_token_ms"), 0),
                "total_time_ms": as_int(telemetry.get("total_ms"), result.total_ms),
            },
            "retrieval": {
                "retrieved_chunk_pages": _page_ids_to_retrieval_refs(used_pages),
            },
            "usage": {
                "input_tokens": as_int(telemetry.get("prompt_tokens"), 0),
                "output_tokens": as_int(telemetry.get("completion_tokens"), 0),
            },
            "model_name": str(telemetry.get("model_llm") or ""),
        },
    }


def _build_platform_submission_payload(results: list[PlatformCaseResult]) -> dict[str, object]:
    settings = get_settings().platform
    return {
        "architecture_summary": settings.architecture_summary,
        "answers": [_project_platform_answer(result) for result in results],
    }


def _resolve_archive_allowlist_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_archive_allowlist(path: Path) -> ArchiveAllowlist:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Archive allowlist must be a JSON object: {path}")
    payload = cast("dict[str, object]", payload_obj)
    include_obj = payload.get("include")
    exclude_obj = payload.get("exclude_globs")
    if not isinstance(include_obj, list) or not isinstance(exclude_obj, list):
        raise ValueError(f"Archive allowlist must contain include/exclude_globs lists: {path}")
    include = [str(item).strip() for item in cast("list[object]", include_obj) if str(item).strip()]
    exclude_globs = [str(item).strip() for item in cast("list[object]", exclude_obj) if str(item).strip()]
    if not include:
        raise ValueError(f"Archive allowlist include[] must not be empty: {path}")
    return ArchiveAllowlist(include=include, exclude_globs=exclude_globs)


def _iter_allowlisted_files(root: Path, allowlist: ArchiveAllowlist) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for include_item in allowlist.include:
        target = root / include_item
        if not target.exists():
            continue
        if target.is_file():
            rel_path = target.relative_to(root)
            if _should_exclude(rel_path, allowlist.exclude_globs):
                continue
            if target not in seen:
                seen.add(target)
                files.append(target)
            continue
        for file_path in target.rglob("*"):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(root)
            if _should_exclude(rel_path, allowlist.exclude_globs):
                continue
            if file_path not in seen:
                seen.add(file_path)
                files.append(file_path)
    return sorted(files)


def _should_exclude(rel_path: Path, exclude_globs: Iterable[str]) -> bool:
    rel_text = rel_path.as_posix()
    if any(part in _FORBIDDEN_ARCHIVE_PARTS for part in rel_path.parts):
        return True
    if rel_path.name == ".DS_Store":
        return True
    if "__pycache__" in rel_path.parts or rel_path.suffix in {".pyc", ".pyo"}:
        return True
    return any(rel_path.match(pattern) or rel_text == pattern for pattern in exclude_globs)


def _scan_text_for_secrets(content: str) -> bool:
    if any(pattern.search(content) for pattern in _SECRET_VALUE_PATTERNS):
        return True
    return _NONEMPTY_SECRET_ASSIGNMENT_RE.search(content) is not None


def _audit_code_archive(root: Path, archived_files: list[Path], archive_path: Path) -> dict[str, object]:
    issues: list[str] = []
    archived_rel_paths = [path.relative_to(root).as_posix() for path in archived_files]
    for rel_path in archived_rel_paths:
        rel_parts = Path(rel_path).parts
        if any(part in _FORBIDDEN_ARCHIVE_PARTS for part in rel_parts):
            issues.append(f"forbidden_path:{rel_path}")
    for file_path in archived_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _scan_text_for_secrets(content):
            issues.append(f"secret_like_content:{file_path.relative_to(root).as_posix()}")
    archive_size = archive_path.stat().st_size if archive_path.exists() else 0
    if archive_size > _MAX_CODE_ARCHIVE_BYTES:
        issues.append(f"archive_too_large:{archive_size}")
    return {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "files": archived_rel_paths,
        "issues": issues,
    }


def _create_code_archive(root: Path, archive_path: Path, allowlist: ArchiveAllowlist) -> dict[str, object]:
    files = _iter_allowlisted_files(root, allowlist)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in files:
            zip_handle.write(file_path, file_path.relative_to(root))
    report = _audit_code_archive(root, files, archive_path)
    if cast("list[str]", report["issues"]):
        issues_text = ", ".join(cast("list[str]", report["issues"]))
        raise ValueError(f"Code archive audit failed: {issues_text}")
    return report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _percentile_int(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return int(ordered[idx])


async def _qdrant_collection_point_count(collection_name: str) -> int | None:
    client = QdrantStore()
    try:
        response = await client.client.count(
            collection_name=collection_name,
            count_filter=None,
            exact=True,
        )
        return int(response.count)
    except Exception:
        logger.warning("Failed to fetch Qdrant point count for %s", collection_name, exc_info=True)
        return None
    finally:
        await client.close()


def _build_preflight_summary(
    *,
    paths: PlatformPaths,
    collection_name: str,
    payload: dict[str, object],
    results: list[PlatformCaseResult],
    point_count: int | None,
) -> dict[str, object]:
    answers_obj = payload.get("answers")
    answers = cast("list[dict[str, object]]", answers_obj) if isinstance(answers_obj, list) else []
    answer_type_counts: dict[str, int] = {}
    null_answer_counts_by_type: dict[str, int] = {}
    empty_pages_counts_by_type: dict[str, int] = {}
    page_counts: list[int] = []
    free_text_char_counts: list[int] = []
    free_text_sentence_counts: list[int] = []
    model_name_empty_count = 0

    for result, answer_payload in zip(results, answers, strict=False):
        answer_type = result.case.answer_type.strip().lower() or "free_text"
        answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1

        answer_value = answer_payload.get("answer")
        if answer_value is None:
            null_answer_counts_by_type[answer_type] = null_answer_counts_by_type.get(answer_type, 0) + 1

        telemetry_obj = answer_payload.get("telemetry")
        telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        retrieval_obj = telemetry.get("retrieval")
        retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
        refs_obj = retrieval.get("retrieved_chunk_pages")
        refs = cast("list[dict[str, object]]", refs_obj) if isinstance(refs_obj, list) else []
        page_count = 0
        for ref in refs:
            page_numbers_obj = ref.get("page_numbers")
            if isinstance(page_numbers_obj, list):
                page_numbers = cast("list[object]", page_numbers_obj)
                page_count += len([item for item in page_numbers if isinstance(item, int | float)])
        page_counts.append(page_count)
        if page_count == 0:
            empty_pages_counts_by_type[answer_type] = empty_pages_counts_by_type.get(answer_type, 0) + 1

        usage_model_name = str(telemetry.get("model_name") or "").strip()
        if not usage_model_name:
            model_name_empty_count += 1

        if answer_type == "free_text" and isinstance(answer_value, str):
            free_text_char_counts.append(len(answer_value))
            free_text_sentence_counts.append(count_submission_sentences(answer_value))

    documents_zip = paths.docs_dir / "documents.zip"
    pdf_count = sum(1 for _ in paths.docs_dir.rglob("*.pdf")) if paths.docs_dir.exists() else 0
    summary: dict[str, object] = {
        "phase": get_settings().platform.phase,
        "questions_count": len(results),
        "answer_type_counts": answer_type_counts,
        "null_answer_counts_by_type": null_answer_counts_by_type,
        "empty_retrieved_chunk_pages_counts_by_type": empty_pages_counts_by_type,
        "page_count_distribution": {
            "min": min(page_counts, default=0),
            "p50": _percentile_int(page_counts, 0.50),
            "p95": _percentile_int(page_counts, 0.95),
            "max": max(page_counts, default=0),
        },
        "free_text_char_distribution": {
            "min": min(free_text_char_counts, default=0),
            "p50": _percentile_int(free_text_char_counts, 0.50),
            "p95": _percentile_int(free_text_char_counts, 0.95),
            "max": max(free_text_char_counts, default=0),
        },
        "free_text_sentence_distribution": {
            "min": min(free_text_sentence_counts, default=0),
            "p50": _percentile_int(free_text_sentence_counts, 0.50),
            "p95": _percentile_int(free_text_sentence_counts, 0.95),
            "max": max(free_text_sentence_counts, default=0),
        },
        "model_name_empty_count": model_name_empty_count,
        "submission_sha256": _sha256_file(paths.submission_path) if paths.submission_path.exists() else "",
        "code_archive_sha256": _sha256_file(paths.code_archive_path) if paths.code_archive_path.exists() else "",
        "questions_sha256": _sha256_file(paths.questions_path) if paths.questions_path.exists() else "",
        "documents_zip_sha256": _sha256_file(documents_zip) if documents_zip.exists() else "",
        "pdf_count": pdf_count,
        "phase_collection_name": collection_name,
        "qdrant_point_count": point_count,
    }
    return summary


async def _download_phase_documents(client: PlatformEvaluationClient, paths: PlatformPaths, *, refresh: bool) -> None:
    has_pdf = any(path.suffix.lower() == ".pdf" for path in paths.docs_dir.rglob("*.pdf")) if paths.docs_dir.exists() else False
    if refresh or not has_pdf:
        await client.download_documents(paths.docs_dir)


async def _download_phase_questions(client: PlatformEvaluationClient, paths: PlatformPaths, *, refresh: bool) -> None:
    if refresh or not paths.questions_path.exists():
        await client.download_questions(paths.questions_path)


def _write_archive_artifacts(root: Path, paths: PlatformPaths, allowlist: ArchiveAllowlist) -> None:
    paths.phase_dir.mkdir(parents=True, exist_ok=True)
    audit_report = _create_code_archive(root, paths.code_archive_path, allowlist)
    paths.audit_report_path.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _ingest_phase_documents(doc_dir: Path) -> None:
    pipeline = IngestionPipeline()
    try:
        await pipeline.run(doc_dir)
    finally:
        await pipeline.close()


async def _run_questions(
    cases: list[SubmissionCase],
    runtime: PipelineRuntime,
    *,
    concurrency: int,
    fail_fast: bool,
) -> list[PlatformCaseResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[PlatformCaseResult | None] = [None] * len(cases)

    async def worker(index: int, case: SubmissionCase) -> None:
        async with semaphore:
            results[index] = await _run_case_direct(case, runtime, fail_fast=fail_fast)

    await asyncio.gather(*[worker(index, case) for index, case in enumerate(cases)])
    return [cast("PlatformCaseResult", result) for result in results]


async def _poll_submission_status(
    client: PlatformEvaluationClient,
    submission_uuid: str,
    *,
    poll_interval_s: float,
    poll_timeout_s: float,
) -> dict[str, object]:
    start = time.perf_counter()
    while True:
        payload = await client.get_submission_status(submission_uuid)
        status = str(payload.get("status") or "").strip().lower()
        if status in {"completed", "error"}:
            return payload
        if (time.perf_counter() - start) >= poll_timeout_s:
            return payload
        await asyncio.sleep(max(1.0, poll_interval_s))


async def _submit_existing_artifacts(
    client: PlatformEvaluationClient,
    *,
    submission_path: Path,
    code_archive_path: Path,
    poll: bool,
    poll_interval_s: float,
    poll_timeout_s: float,
) -> dict[str, object]:
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission JSON not found: {submission_path}")
    if not code_archive_path.exists():
        raise FileNotFoundError(f"Code archive not found: {code_archive_path}")
    submit_response = await client.submit_submission(submission_path, code_archive_path)
    if not poll:
        return submit_response
    submission_uuid = str(submit_response.get("uuid") or "").strip()
    if not submission_uuid:
        return submit_response
    return await _poll_submission_status(
        client,
        submission_uuid,
        poll_interval_s=poll_interval_s,
        poll_timeout_s=poll_timeout_s,
    )


async def _async_main(args: argparse.Namespace) -> int:
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_format)
    paths = _resolve_phase_paths()
    collection_name = _phase_collection_name()
    archive_allowlist = _load_archive_allowlist(
        _resolve_archive_allowlist_path(settings.platform.archive_allowlist_path)
    )

    if bool(args.archive_only):
        _write_archive_artifacts(Path.cwd(), paths, archive_allowlist)
        logger.info("Prepared curated code archive without touching platform resources")
        logger.info("code_archive=%s", paths.code_archive_path)
        logger.info("audit_report=%s", paths.audit_report_path)
        return 0

    client = PlatformEvaluationClient.from_settings()
    try:
        if bool(args.submit_existing):
            submission_path = Path(args.submission_path) if args.submission_path else paths.submission_path
            code_archive_path = Path(args.code_archive_path) if args.code_archive_path else paths.code_archive_path
            final_status = await _submit_existing_artifacts(
                client,
                submission_path=submission_path,
                code_archive_path=code_archive_path,
                poll=bool(args.poll),
                poll_interval_s=float(settings.platform.poll_interval_s),
                poll_timeout_s=float(settings.platform.poll_timeout_s),
            )
            paths.phase_dir.mkdir(parents=True, exist_ok=True)
            paths.status_path.write_text(
                json.dumps(final_status, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("status_path=%s", paths.status_path)
            return 0

        try:
            await _download_phase_documents(client, paths, refresh=bool(args.refresh_downloads))
        except httpx.HTTPStatusError as exc:
            if _is_resources_not_published_error(exc):
                logger.error(
                    "Platform resources for phase '%s' are not published yet. "
                    "Use --archive-only for preflight archive checks and retry downloads after publication.",
                    settings.platform.phase,
                )
                return 2
            raise

        with _phase_collection_override(collection_name):
            if not bool(args.skip_ingest):
                await _ingest_phase_documents(paths.docs_dir)

            try:
                await _download_phase_questions(client, paths, refresh=bool(args.refresh_downloads))
            except httpx.HTTPStatusError as exc:
                if _is_resources_not_published_error(exc):
                    logger.error(
                        "Platform questions for phase '%s' are not published yet. "
                        "Documents may already be available and ingested; retry question download after publication.",
                        settings.platform.phase,
                    )
                    return 2
                raise

            cases = load_cases(paths.questions_path)
            runtime = await _build_pipeline_runtime()
            try:
                results = await _run_questions(
                    cases,
                    runtime,
                    concurrency=int(settings.platform.query_concurrency),
                    fail_fast=bool(args.fail_fast),
                )
            finally:
                await runtime.close()

        payload = _build_platform_submission_payload(results)
        paths.phase_dir.mkdir(parents=True, exist_ok=True)
        paths.submission_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_archive_artifacts(Path.cwd(), paths, archive_allowlist)
        point_count = await _qdrant_collection_point_count(collection_name)
        preflight_summary = _build_preflight_summary(
            paths=paths,
            collection_name=collection_name,
            payload=payload,
            results=results,
            point_count=point_count,
        )
        paths.preflight_summary_path.write_text(
            json.dumps(preflight_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not bool(args.submit):
            logger.info("Prepared platform submission artifacts without uploading")
            logger.info("submission_json=%s", paths.submission_path)
            logger.info("code_archive=%s", paths.code_archive_path)
            logger.info("audit_report=%s", paths.audit_report_path)
            logger.info("preflight_summary=%s", paths.preflight_summary_path)
            return 0

        submit_response = await client.submit_submission(paths.submission_path, paths.code_archive_path)
        final_status = submit_response
        if bool(args.poll):
            submission_uuid = str(submit_response.get("uuid") or "").strip()
            if submission_uuid:
                final_status = await _poll_submission_status(
                    client,
                    submission_uuid,
                    poll_interval_s=float(settings.platform.poll_interval_s),
                    poll_timeout_s=float(settings.platform.poll_timeout_s),
                )
        paths.status_path.write_text(
            json.dumps(final_status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("status_path=%s", paths.status_path)
        return 0
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a platform-native submission package for the competition platform.",
    )
    parser.add_argument("--archive-only", action="store_true", help="Build and audit only the curated code archive.")
    parser.add_argument("--refresh-downloads", action="store_true", help="Redownload questions/documents for the active phase.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse the existing phase-specific Qdrant collection.")
    parser.add_argument("--submit", action="store_true", help="Upload submission.json and code_archive.zip to the platform.")
    parser.add_argument("--submit-existing", action="store_true", help="Upload an existing submission.json and code archive without downloading, ingesting, or querying.")
    parser.add_argument("--submission-path", help="Path to an existing submission.json for --submit-existing.")
    parser.add_argument("--code-archive-path", help="Path to an existing code archive for --submit-existing.")
    parser.add_argument("--poll", action="store_true", help="Poll submission status until completion or timeout.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first question execution failure.")
    args = parser.parse_args(argv)

    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
