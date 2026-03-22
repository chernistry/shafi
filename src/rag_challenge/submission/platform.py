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
from scripts.build_platform_truth_audit import build_truth_audit_scaffold, render_truth_audit_workbook

from rag_challenge.config import get_settings
from rag_challenge.config.logging import setup_logging
from rag_challenge.config.settings import build_score_settings_fingerprint
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
    coerce_str_list,
    count_submission_sentences,
    load_cases,
    normalize_date_answer,
    select_submission_used_pages,
)
from rag_challenge.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

_DEFAULT_UNANSWERABLE_FREE_TEXT = "There is no information on this question in the provided documents."
_PLATFORM_RATE_LIMIT_MAX_ATTEMPTS = 4
_PLATFORM_RATE_LIMIT_BASE_DELAY_S = 5.0
_PLATFORM_RATE_LIMIT_MAX_DELAY_S = 60.0
_PAGE_ID_RE = re.compile(r"^(?P<doc_id>.+)_(?P<page>\d+)$")
_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*\d{1,4}\s*[/-]\s*\d{4}\b", re.IGNORECASE)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bmcs_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\biuak_v1_[A-Za-z0-9_\-]{24,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)
_NONEMPTY_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?m)^[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)[ \t]*=[ \t]*(?!$|\"\"$|''$|<your-|your-|xxxx|xxxxx).+$"
)
_SOURCE_SUBMISSION_FILENAME_RE = re.compile(r"^submission(?P<suffix>.*)\.json$")
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
    raw_results_path: Path
    code_archive_path: Path
    audit_report_path: Path
    status_path: Path
    preflight_summary_path: Path
    canary_path: Path
    truth_audit_path: Path
    truth_audit_workbook_path: Path


@dataclass(frozen=True)
class PlatformCaseResult:
    case: SubmissionCase
    answer_text: str
    telemetry: dict[str, object]
    total_ms: int


type PipelineRuntimeFactory = Callable[[], Awaitable[PipelineRuntime]]


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

    @staticmethod
    def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
        raw_retry_after = response.headers.get("Retry-After", "").strip()
        if raw_retry_after:
            try:
                retry_after = float(raw_retry_after)
            except ValueError:
                retry_after = 0.0
            else:
                return max(0.0, min(_PLATFORM_RATE_LIMIT_MAX_DELAY_S, retry_after))
        return min(
            _PLATFORM_RATE_LIMIT_MAX_DELAY_S,
            _PLATFORM_RATE_LIMIT_BASE_DELAY_S * (2 ** max(0, attempt - 1)),
        )

    async def _get_with_rate_limit_retry(self, url: str) -> httpx.Response:
        last_response: httpx.Response | None = None
        for attempt in range(1, _PLATFORM_RATE_LIMIT_MAX_ATTEMPTS + 1):
            response = await self._client.get(url)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            last_response = response
            if attempt >= _PLATFORM_RATE_LIMIT_MAX_ATTEMPTS:
                break
            delay_s = self._retry_delay_seconds(response, attempt)
            logger.warning(
                "Platform API rate-limited on GET %s; retrying in %.1fs (attempt %d/%d)",
                url,
                delay_s,
                attempt,
                _PLATFORM_RATE_LIMIT_MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay_s)
        if last_response is None:
            raise RuntimeError(f"Platform request failed without a response: GET {url}")
        last_response.raise_for_status()
        return last_response

    async def _post_files_with_rate_limit_retry(
        self,
        url: str,
        *,
        files: dict[str, tuple[str, Any, str]],
    ) -> httpx.Response:
        last_response: httpx.Response | None = None
        for attempt in range(1, _PLATFORM_RATE_LIMIT_MAX_ATTEMPTS + 1):
            response = await self._client.post(url, files=files)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            last_response = response
            if attempt >= _PLATFORM_RATE_LIMIT_MAX_ATTEMPTS:
                break
            delay_s = self._retry_delay_seconds(response, attempt)
            logger.warning(
                "Platform API rate-limited on POST %s; retrying in %.1fs (attempt %d/%d)",
                url,
                delay_s,
                attempt,
                _PLATFORM_RATE_LIMIT_MAX_ATTEMPTS,
            )
            await asyncio.sleep(delay_s)
        if last_response is None:
            raise RuntimeError(f"Platform request failed without a response: POST {url}")
        last_response.raise_for_status()
        return last_response

    async def download_questions(self, target_path: Path) -> Path:
        response = await self._get_with_rate_limit_retry(f"{self._base_url}/questions")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(response.json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target_path

    async def download_documents(self, target_dir: Path) -> Path:
        response = await self._get_with_rate_limit_retry(f"{self._base_url}/documents")
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / "documents.zip"
        archive_path.write_bytes(response.content)
        with zipfile.ZipFile(archive_path, "r") as zip_handle:
            zip_handle.extractall(target_dir)
        return target_dir

    async def submit_submission(self, submission_path: Path, code_archive_path: Path) -> dict[str, object]:
        with submission_path.open("rb") as submission_handle, code_archive_path.open("rb") as archive_handle:
            response = await self._post_files_with_rate_limit_retry(
                f"{self._base_url}/submissions",
                files={
                    "file": (submission_path.name, submission_handle, "application/json"),
                    "code_archive": (code_archive_path.name, archive_handle, "application/zip"),
                },
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Submission response must be an object")
        return cast("dict[str, object]", payload)

    async def get_submission_status(self, submission_uuid: str) -> dict[str, object]:
        response = await self._get_with_rate_limit_retry(f"{self._base_url}/submissions/{submission_uuid}/status")
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
        raw_results_path=phase_dir / "raw_results.json",
        code_archive_path=phase_dir / settings.code_archive_filename,
        audit_report_path=phase_dir / "code_archive_audit.json",
        status_path=phase_dir / "submission_status.json",
        preflight_summary_path=phase_dir / "preflight_summary.json",
        canary_path=phase_dir / "equivalence_canary.json",
        truth_audit_path=phase_dir / "truth_audit_scaffold.json",
        truth_audit_workbook_path=phase_dir / "truth_audit_workbook.md",
    )


def _with_artifact_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _suffix_platform_paths(paths: PlatformPaths, suffix: str) -> PlatformPaths:
    if not suffix:
        return paths
    return PlatformPaths(
        phase_dir=paths.phase_dir,
        docs_dir=paths.docs_dir,
        questions_path=paths.questions_path,
        submission_path=_with_artifact_suffix(paths.submission_path, suffix),
        raw_results_path=_with_artifact_suffix(paths.raw_results_path, suffix),
        code_archive_path=_with_artifact_suffix(paths.code_archive_path, suffix),
        audit_report_path=_with_artifact_suffix(paths.audit_report_path, suffix),
        status_path=_with_artifact_suffix(paths.status_path, suffix),
        preflight_summary_path=_with_artifact_suffix(paths.preflight_summary_path, suffix),
        canary_path=_with_artifact_suffix(paths.canary_path, suffix),
        truth_audit_path=_with_artifact_suffix(paths.truth_audit_path, suffix),
        truth_audit_workbook_path=_with_artifact_suffix(paths.truth_audit_workbook_path, suffix),
    )


def _phase_collection_name() -> str:
    settings = get_settings()
    return f"{settings.platform.collection_prefix}_{settings.platform.phase}"


def _resolve_query_concurrency(override: int | None) -> int:
    return max(1, int(override)) if override is not None else 1


@contextmanager
def _phase_collection_override(collection_name: str):
    previous = os.environ.get("QDRANT_COLLECTION")
    previous_page = os.environ.get("QDRANT_PAGE_COLLECTION")
    previous_shadow = os.environ.get("QDRANT_SHADOW_COLLECTION")
    previous_segment = os.environ.get("QDRANT_SEGMENT_COLLECTION")
    previous_bridge = os.environ.get("QDRANT_BRIDGE_FACT_COLLECTION")
    previous_support = os.environ.get("QDRANT_SUPPORT_FACT_COLLECTION")
    os.environ["QDRANT_COLLECTION"] = collection_name
    os.environ["QDRANT_PAGE_COLLECTION"] = f"{collection_name}_pages"
    os.environ["QDRANT_SHADOW_COLLECTION"] = f"{collection_name}_shadow"
    os.environ["QDRANT_SEGMENT_COLLECTION"] = f"{collection_name}_segments"
    os.environ["QDRANT_BRIDGE_FACT_COLLECTION"] = f"{collection_name}_bridge_facts"
    os.environ["QDRANT_SUPPORT_FACT_COLLECTION"] = f"{collection_name}_support_facts"
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("QDRANT_COLLECTION", None)
        else:
            os.environ["QDRANT_COLLECTION"] = previous
        if previous_page is None:
            os.environ.pop("QDRANT_PAGE_COLLECTION", None)
        else:
            os.environ["QDRANT_PAGE_COLLECTION"] = previous_page
        if previous_shadow is None:
            os.environ.pop("QDRANT_SHADOW_COLLECTION", None)
        else:
            os.environ["QDRANT_SHADOW_COLLECTION"] = previous_shadow
        if previous_segment is None:
            os.environ.pop("QDRANT_SEGMENT_COLLECTION", None)
        else:
            os.environ["QDRANT_SEGMENT_COLLECTION"] = previous_segment
        if previous_bridge is None:
            os.environ.pop("QDRANT_BRIDGE_FACT_COLLECTION", None)
        else:
            os.environ["QDRANT_BRIDGE_FACT_COLLECTION"] = previous_bridge
        if previous_support is None:
            os.environ.pop("QDRANT_SUPPORT_FACT_COLLECTION", None)
        else:
            os.environ["QDRANT_SUPPORT_FACT_COLLECTION"] = previous_support
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
        collector.mark_first_token()
        telemetry_payload = collector.finalize().model_dump()

    total_ms = int((time.perf_counter() - t0) * 1000.0)
    if not answer_final:
        answer_final = "".join(tokens).strip() or _default_answer_text(case.answer_type)
    if telemetry_payload is None:
        if answer_final.strip():
            collector.mark_first_token()
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


def _flatten_retrieval_refs(refs: list[dict[str, object]]) -> list[str]:
    flattened: list[str] = []
    for ref in refs:
        doc_id = str(ref.get("doc_id") or "").strip()
        page_numbers_obj = ref.get("page_numbers")
        if not doc_id or not isinstance(page_numbers_obj, list):
            continue
        page_numbers = cast("list[object]", page_numbers_obj)
        for page in page_numbers:
            if isinstance(page, int | float):
                flattened.append(f"{doc_id}_{int(page)}")
    return flattened


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

    answer_out: SubmissionAnswer
    if is_unanswerable_strict:
        answer_out = None
    elif is_unanswerable_free_text:
        answer_out = _DEFAULT_UNANSWERABLE_FREE_TEXT
    else:
        answer_out = coerce_answer_type(answer_text, answer_type)
        if answer_out is None and answer_type.strip().lower() == "free_text":
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


def _serialize_platform_case_result(result: PlatformCaseResult) -> dict[str, object]:
    return {
        "case": {
            "case_id": result.case.case_id,
            "question": result.case.question,
            "answer_type": result.case.answer_type,
        },
        "answer_text": result.answer_text,
        "telemetry": result.telemetry,
        "total_ms": result.total_ms,
    }


def _deserialize_platform_case_result(payload: dict[str, object]) -> PlatformCaseResult:
    case_obj = payload.get("case")
    case_payload = cast("dict[str, object]", case_obj) if isinstance(case_obj, dict) else {}
    return PlatformCaseResult(
        case=SubmissionCase(
            case_id=str(case_payload.get("case_id") or ""),
            question=str(case_payload.get("question") or ""),
            answer_type=str(case_payload.get("answer_type") or "free_text"),
        ),
        answer_text=str(payload.get("answer_text") or ""),
        telemetry=cast("dict[str, object]", payload.get("telemetry") or {}),
        total_ms=as_int(payload.get("total_ms"), 0),
    )


def _projected_answer_signature(result: PlatformCaseResult) -> tuple[SubmissionAnswer, list[str]]:
    payload = _project_platform_answer(result)
    telemetry_obj = payload.get("telemetry")
    telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    retrieval_obj = telemetry.get("retrieval")
    retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
    refs_obj = retrieval.get("retrieved_chunk_pages")
    refs = cast("list[dict[str, object]]", refs_obj) if isinstance(refs_obj, list) else []
    return cast("SubmissionAnswer", payload.get("answer")), _flatten_retrieval_refs(refs)


def _is_specific_question(question: str, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not q:
        return False
    if _CASE_REF_RE.search(q):
        return True
    if any(token in q for token in ("article ", "section ", "schedule ", "page ", "law no.", "law no ", "regulations")):
        return True
    if any(
        phrase in q
        for phrase in (
            "who administers",
            "when was the consolidated version",
            "who is the defendant",
            "what is the law number",
            "what was the result of the application",
            "what was the outcome of the specific order",
            "how did the court of appeal rule",
            "what is the effective date",
        )
    ):
        return True
    return answer_type.strip().lower() in {"boolean", "date", "name", "names", "number"}


def _result_anomaly_flags(result: PlatformCaseResult) -> list[str]:
    flags: list[str] = []
    answer_type = result.case.answer_type.strip().lower()
    projected_answer, projected_pages = _projected_answer_signature(result)
    answer_text = projected_answer if isinstance(projected_answer, str) else ("null" if projected_answer is None else str(projected_answer))
    telemetry = result.telemetry
    is_unanswerable_strict, is_unanswerable_free_text = classify_unanswerable_answer(answer_text, answer_type)
    raw_support_surfaces_present = any(
        (
            coerce_str_list(telemetry.get("used_page_ids")),
            coerce_str_list(telemetry.get("cited_page_ids")),
            coerce_str_list(telemetry.get("cited_chunk_ids")),
        )
    )
    if not (is_unanswerable_strict or is_unanswerable_free_text) and raw_support_surfaces_present and not projected_pages:
        flags.append("answerable_support_pages_lost_in_projection")
    if not (is_unanswerable_strict or is_unanswerable_free_text):
        return flags

    if _is_specific_question(result.case.question, answer_type):
        flags.append("specific_question_unsupported")

    raw_used_pages = select_submission_used_pages(telemetry)
    if projected_pages or raw_used_pages:
        flags.append("unsupported_with_support_pages")

    retrieved_pages = coerce_str_list(telemetry.get("retrieved_page_ids"))
    if retrieved_pages:
        flags.append("unsupported_with_retrieved_pages")

    doc_refs = coerce_str_list(telemetry.get("doc_refs"))
    if doc_refs:
        flags.append("unsupported_with_doc_refs")

    return flags


def _build_results_anomaly_report(results: list[PlatformCaseResult]) -> dict[str, object]:
    anomaly_case_ids: list[str] = []
    anomaly_flags_by_case: dict[str, list[str]] = {}
    for result in results:
        flags = _result_anomaly_flags(result)
        if not flags:
            continue
        anomaly_case_ids.append(result.case.case_id)
        anomaly_flags_by_case[result.case.case_id] = flags
    return {
        "anomaly_case_ids": anomaly_case_ids,
        "anomaly_flags_by_case": anomaly_flags_by_case,
        "anomaly_count": len(anomaly_case_ids),
    }


async def _repair_anomalous_results(
    results: list[PlatformCaseResult],
    *,
    fail_fast: bool,
    runtime_factory: PipelineRuntimeFactory = _build_pipeline_runtime,
) -> tuple[list[PlatformCaseResult], dict[str, object]]:
    updated = list(results)
    repaired_case_ids: list[str] = []
    unchanged_case_ids: list[str] = []
    skipped_case_ids: list[str] = []

    for index, result in enumerate(results):
        flags = _result_anomaly_flags(result)
        if not flags:
            continue

        try:
            runtime = await runtime_factory()
            try:
                rerun = await _run_case_direct(result.case, runtime, fail_fast=fail_fast)
            finally:
                await runtime.close()
        except Exception:
            logger.exception("Anomaly rerun failed for question_id=%s", result.case.case_id)
            if fail_fast:
                raise
            skipped_case_ids.append(result.case.case_id)
            continue

        rerun_flags = _result_anomaly_flags(rerun)
        old_answer, old_pages = _projected_answer_signature(result)
        new_answer, new_pages = _projected_answer_signature(rerun)
        old_answer_text = old_answer if isinstance(old_answer, str) else ("null" if old_answer is None else str(old_answer))
        new_answer_text = new_answer if isinstance(new_answer, str) else ("null" if new_answer is None else str(new_answer))
        improved = (not rerun_flags and bool(new_answer_text.strip())) or (
            new_answer_text != old_answer_text and new_pages != old_pages
        )

        if improved:
            updated[index] = rerun
            repaired_case_ids.append(result.case.case_id)
            logger.info(
                "Anomaly rerun replaced result for question_id=%s; flags=%s -> %s",
                result.case.case_id,
                flags,
                rerun_flags,
            )
        else:
            unchanged_case_ids.append(result.case.case_id)

    return updated, {
        "repaired_case_ids": repaired_case_ids,
        "unchanged_case_ids": unchanged_case_ids,
        "skipped_case_ids": skipped_case_ids,
    }


def _build_equivalence_canary(
    *,
    baseline_results: list[PlatformCaseResult],
    candidate_results: list[PlatformCaseResult],
    baseline_concurrency: int,
    candidate_concurrency: int,
) -> dict[str, object]:
    baseline_by_id = {result.case.case_id: result for result in baseline_results}
    candidate_by_id = {result.case.case_id: result for result in candidate_results}

    answer_drift: list[str] = []
    model_drift: list[str] = []
    page_drift: list[str] = []
    missing_case_ids: list[str] = []

    for case_id, baseline in baseline_by_id.items():
        candidate = candidate_by_id.get(case_id)
        if candidate is None:
            missing_case_ids.append(case_id)
            continue
        if baseline.answer_text.strip() != candidate.answer_text.strip():
            answer_drift.append(case_id)
        if str(baseline.telemetry.get("model_llm") or "").strip() != str(candidate.telemetry.get("model_llm") or "").strip():
            model_drift.append(case_id)
        if select_submission_used_pages(baseline.telemetry) != select_submission_used_pages(candidate.telemetry):
            page_drift.append(case_id)

    return {
        "baseline_concurrency": baseline_concurrency,
        "candidate_concurrency": candidate_concurrency,
        "total_cases": len(baseline_results),
        "answer_drift_case_ids": answer_drift,
        "model_drift_case_ids": model_drift,
        "page_drift_case_ids": page_drift,
        "missing_case_ids": missing_case_ids,
        "answer_drift_count": len(answer_drift),
        "model_drift_count": len(model_drift),
        "page_drift_count": len(page_drift),
    }


def _build_platform_submission_payload(results: list[PlatformCaseResult]) -> dict[str, object]:
    settings = get_settings().platform
    return {
        "architecture_summary": settings.architecture_summary,
        "answers": [_project_platform_answer(result) for result in results],
    }


def _load_questions_by_id(path: Path) -> dict[str, SubmissionCase]:
    return {case.case_id: case for case in load_cases(path)}


def _load_platform_submission_payload(path: Path) -> dict[str, object]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Submission payload must be an object: {path}")
    payload = cast("dict[str, object]", payload_obj)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Submission payload must contain answers[]: {path}")
    return payload


def _load_raw_results(path: Path) -> list[PlatformCaseResult]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, list):
        raise ValueError(f"Raw results payload must be a list: {path}")
    results: list[PlatformCaseResult] = []
    for item in cast("list[object]", payload_obj):
        if not isinstance(item, dict):
            continue
        results.append(_deserialize_platform_case_result(cast("dict[str, object]", item)))
    return results


def _answer_payload_by_question_id(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    answers_obj = payload.get("answers")
    answers = cast("list[dict[str, object]]", answers_obj) if isinstance(answers_obj, list) else []
    return {
        str(answer.get("question_id") or "").strip(): answer
        for answer in answers
        if str(answer.get("question_id") or "").strip()
    }


def _validate_platform_args(args: argparse.Namespace) -> None:
    if args.support_only_challenger and bool(args.submit):
        raise ValueError(
            "--support-only-challenger cannot be combined with --submit; "
            "inspect the artifact first, then use --submit-existing."
        )


def _load_truth_audit_scaffold(path: Path) -> dict[str, object]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"Truth audit scaffold must be an object: {path}")
    return cast("dict[str, object]", payload_obj)


def _infer_source_truth_audit_path(source_submission_path: Path) -> Path:
    match = _SOURCE_SUBMISSION_FILENAME_RE.fullmatch(source_submission_path.name)
    if match is None:
        raise ValueError(
            "Cannot infer source truth audit scaffold path from submission filename: "
            f"{source_submission_path}"
        )
    suffix = match.group("suffix")
    return source_submission_path.with_name(f"truth_audit_scaffold{suffix}.json")


def _resolve_source_truth_audit_path(
    source_submission_path: Path,
    raw_source_truth_audit_path: str | None,
) -> Path:
    path = Path(raw_source_truth_audit_path) if raw_source_truth_audit_path else _infer_source_truth_audit_path(source_submission_path)
    if not path.exists():
        raise FileNotFoundError(f"Source truth audit scaffold not found: {path}")
    return path


def _canonical_answer_value(answer_value: object) -> str:
    return json.dumps(answer_value, ensure_ascii=False, sort_keys=True)


def _normalized_answer_map(payload: dict[str, object]) -> dict[str, str]:
    answers_obj = payload.get("answers")
    answers = cast("list[dict[str, object]]", answers_obj) if isinstance(answers_obj, list) else []
    normalized: dict[str, str] = {}
    for answer in answers:
        question_id = str(answer.get("question_id") or "").strip()
        if not question_id:
            continue
        normalized[question_id] = _canonical_answer_value(answer.get("answer"))
    return normalized


def _normalize_retrieved_chunk_pages(refs_obj: object) -> tuple[tuple[str, tuple[int, ...]], ...]:
    refs = cast("list[dict[str, object]]", refs_obj) if isinstance(refs_obj, list) else []
    normalized: list[tuple[str, tuple[int, ...]]] = []
    for ref in refs:
        doc_id = str(ref.get("doc_id") or "").strip()
        page_numbers_obj = ref.get("page_numbers")
        if not doc_id or not isinstance(page_numbers_obj, list):
            continue
        page_numbers = sorted(
            {
                int(page)
                for page in cast("list[object]", page_numbers_obj)
                if isinstance(page, int | float)
            }
        )
        normalized.append((doc_id, tuple(page_numbers)))
    return tuple(sorted(normalized))


def _normalized_retrieval_map(payload: dict[str, object]) -> dict[str, tuple[tuple[str, tuple[int, ...]], ...]]:
    answers_obj = payload.get("answers")
    answers = cast("list[dict[str, object]]", answers_obj) if isinstance(answers_obj, list) else []
    normalized: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {}
    for answer in answers:
        question_id = str(answer.get("question_id") or "").strip()
        if not question_id:
            continue
        telemetry_obj = answer.get("telemetry")
        telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        retrieval_obj = telemetry.get("retrieval")
        retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
        normalized[question_id] = _normalize_retrieved_chunk_pages(retrieval.get("retrieved_chunk_pages"))
    return normalized


def _answer_pairs_sha256(payload: dict[str, object]) -> str:
    normalized_answers = _normalized_answer_map(payload)
    digest_payload = [[question_id, answer] for question_id, answer in sorted(normalized_answers.items())]
    return hashlib.sha256(
        json.dumps(digest_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _count_changed_answers(source_payload: dict[str, object], challenger_payload: dict[str, object]) -> int:
    source_answers = _normalized_answer_map(source_payload)
    challenger_answers = _normalized_answer_map(challenger_payload)
    return sum(
        1
        for question_id in sorted(set(source_answers) | set(challenger_answers))
        if source_answers.get(question_id) != challenger_answers.get(question_id)
    )


def _count_changed_pages(source_payload: dict[str, object], challenger_payload: dict[str, object]) -> int:
    source_pages = _normalized_retrieval_map(source_payload)
    challenger_pages = _normalized_retrieval_map(challenger_payload)
    return sum(
        1
        for question_id in sorted(set(source_pages) | set(challenger_pages))
        if source_pages.get(question_id) != challenger_pages.get(question_id)
    )


def _source_manual_verdict_counts(scaffold: dict[str, object]) -> dict[str, int]:
    summary_obj = scaffold.get("summary")
    summary = cast("dict[str, object]", summary_obj) if isinstance(summary_obj, dict) else {}
    counts_obj = summary.get("manual_verdict_counts")
    if isinstance(counts_obj, dict):
        counts = cast("dict[str, object]", counts_obj)
        return {
            "deterministic_complete": as_int(counts.get("deterministic_complete")),
            "deterministic_incomplete": as_int(counts.get("deterministic_incomplete")),
            "free_text_complete": as_int(counts.get("free_text_complete")),
            "free_text_incomplete": as_int(counts.get("free_text_incomplete")),
        }

    report = _build_truth_audit_report(scaffold)
    return {
        "deterministic_complete": as_int(report.get("deterministic_complete_count")),
        "deterministic_incomplete": as_int(report.get("deterministic_incomplete_count")),
        "free_text_complete": as_int(report.get("free_text_complete_count")),
        "free_text_incomplete": as_int(report.get("free_text_incomplete_count")),
    }


def _build_support_only_challenger_report(
    *,
    mode: str,
    source_payload: dict[str, object],
    challenger_payload: dict[str, object],
    source_submission_path: Path,
    source_truth_audit_path: Path,
    source_truth_audit_scaffold: dict[str, object],
) -> dict[str, object]:
    answer_changed_count = _count_changed_answers(source_payload, challenger_payload)
    shared_answer_hash = _answer_pairs_sha256(source_payload)
    return {
        "mode": mode,
        "source_submission_path": str(source_submission_path),
        "source_truth_audit_path": str(source_truth_audit_path),
        "source_submission_sha256": _sha256_file(source_submission_path),
        "same_answers_sha256": shared_answer_hash if answer_changed_count == 0 else "",
        "answer_changed_count": answer_changed_count,
        "page_changed_count": _count_changed_pages(source_payload, challenger_payload),
        "source_manual_verdict_counts": _source_manual_verdict_counts(source_truth_audit_scaffold),
    }


def _question_anchor_pages(question: str) -> list[int]:
    normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if not normalized:
        return []
    if "title page" in normalized or "cover page" in normalized or "title/cover page" in normalized:
        return [1]
    if "first page" in normalized:
        return [1]
    if "second page" in normalized:
        return [2]
    if match := re.search(r"\bpage\s+(\d+)\b", normalized):
        return [int(match.group(1))]
    return []


def _is_anchor_comparison_question(question: str, answer_type: str) -> bool:
    normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
    if answer_type.strip().lower() not in {"boolean", "name", "names", "date", "number"}:
        return False
    if len(_CASE_REF_RE.findall(question or "")) < 2:
        return False
    return any(term in normalized for term in ("party", "claimant", "defendant", "judge", "presided", "date of issue", "issued"))


def _anchor_page_restitution_refs(
    *,
    refs: list[dict[str, object]],
    question: str,
    answer_type: str,
) -> list[dict[str, object]]:
    anchor_pages = set(_question_anchor_pages(question))
    if _is_anchor_comparison_question(question, answer_type):
        anchor_pages.add(1)
    if not anchor_pages:
        return refs

    by_doc: dict[str, set[int]] = {}
    for ref in refs:
        doc_id = str(ref.get("doc_id") or "").strip()
        page_numbers_obj = ref.get("page_numbers")
        if not doc_id or not isinstance(page_numbers_obj, list):
            continue
        page_numbers = {
            int(page)
            for page in cast("list[object]", page_numbers_obj)
            if isinstance(page, int | float)
        }
        if not page_numbers:
            continue
        by_doc.setdefault(doc_id, set()).update(page_numbers)

    for page in anchor_pages:
        for doc_id in list(by_doc.keys()):
            by_doc[doc_id].add(page)

    return [
        {"doc_id": doc_id, "page_numbers": sorted(page_numbers)}
        for doc_id, page_numbers in sorted(by_doc.items())
    ]


def _rebuild_results_from_submission_payload(
    *,
    questions_by_id: dict[str, SubmissionCase],
    payload: dict[str, object],
) -> list[PlatformCaseResult]:
    answers_obj = payload.get("answers")
    answers = cast("list[dict[str, object]]", answers_obj) if isinstance(answers_obj, list) else []
    results: list[PlatformCaseResult] = []
    for answer in answers:
        question_id = str(answer.get("question_id") or "").strip()
        case = questions_by_id.get(question_id)
        if case is None:
            continue
        telemetry_obj = answer.get("telemetry")
        telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        retrieval_obj = telemetry.get("retrieval")
        retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
        refs_obj = retrieval.get("retrieved_chunk_pages")
        refs = cast("list[dict[str, object]]", refs_obj) if isinstance(refs_obj, list) else []
        answer_value = cast("SubmissionAnswer", answer.get("answer"))
        answer_text = "null" if answer_value is None else (answer_value if isinstance(answer_value, str) else json.dumps(answer_value, ensure_ascii=False))
        results.append(
            PlatformCaseResult(
                case=case,
                answer_text=answer_text,
                telemetry={
                    "model_llm": str(telemetry.get("model_name") or ""),
                    "used_page_ids": _flatten_retrieval_refs(refs),
                },
                total_ms=0,
            )
        )
    return results


def _build_anchor_page_challenger_payload(
    *,
    source_payload: dict[str, object],
    questions_by_id: dict[str, SubmissionCase],
) -> dict[str, object]:
    challenger = json.loads(json.dumps(source_payload))
    answers_by_id = _answer_payload_by_question_id(challenger)
    for question_id, answer in answers_by_id.items():
        case = questions_by_id.get(question_id)
        if case is None:
            continue
        telemetry_obj = answer.get("telemetry")
        telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        retrieval_obj = telemetry.get("retrieval")
        retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
        refs_obj = retrieval.get("retrieved_chunk_pages")
        refs = cast("list[dict[str, object]]", refs_obj) if isinstance(refs_obj, list) else []
        retrieval["retrieved_chunk_pages"] = _anchor_page_restitution_refs(
            refs=refs,
            question=case.question,
            answer_type=case.answer_type,
        )
        telemetry["retrieval"] = retrieval
        answer["telemetry"] = telemetry
    return challenger


def _build_all_context_pages_challenger_payload(
    *,
    source_payload: dict[str, object],
    source_results: list[PlatformCaseResult],
) -> dict[str, object]:
    challenger = json.loads(json.dumps(source_payload))
    answers_by_id = _answer_payload_by_question_id(challenger)
    results_by_id = {result.case.case_id: result for result in source_results}
    for question_id, answer in answers_by_id.items():
        result = results_by_id.get(question_id)
        if result is None:
            continue
        context_page_ids = coerce_str_list(result.telemetry.get("context_page_ids"))
        if not context_page_ids:
            continue
        telemetry_obj = answer.get("telemetry")
        telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        retrieval_obj = telemetry.get("retrieval")
        retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
        retrieval["retrieved_chunk_pages"] = _page_ids_to_retrieval_refs(context_page_ids)
        telemetry["retrieval"] = retrieval
        answer["telemetry"] = telemetry
    return challenger


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
    anomaly_report: dict[str, object] | None = None,
    canary_report: dict[str, object] | None = None,
    support_shape_report: dict[str, object] | None = None,
    truth_audit_report: dict[str, object] | None = None,
    support_only_challenger_report: dict[str, object] | None = None,
) -> dict[str, object]:
    settings_fingerprint = build_score_settings_fingerprint(get_settings())
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
        "score_settings_sha256": settings_fingerprint["sha256"],
        "score_settings_fingerprint": settings_fingerprint["settings"],
        "truth_audit_workbook_path": str(paths.truth_audit_workbook_path),
        "raw_results_path": str(paths.raw_results_path),
    }
    if anomaly_report:
        summary["anomaly_report"] = anomaly_report
    if canary_report:
        summary["equivalence_canary"] = canary_report
    if support_shape_report:
        summary["support_shape_report"] = support_shape_report
    if truth_audit_report:
        summary["truth_audit_report"] = truth_audit_report
    if support_only_challenger_report:
        summary["support_only_challenger"] = support_only_challenger_report
    return summary


def _build_support_shape_report(scaffold: dict[str, object]) -> dict[str, object]:
    records_obj = scaffold.get("records")
    records = cast("list[dict[str, object]]", records_obj) if isinstance(records_obj, list) else []

    blocking_flags = {
        "comparison_missing_side",
        "multi_slot_support_maybe_undercovered",
        "metadata_multi_atom_maybe_undercovered",
        "case_outcome_disposition_maybe_missing",
    }
    flagged_case_ids: list[str] = []
    blocking_case_ids: list[str] = []
    flags_by_case: dict[str, list[str]] = {}
    flag_counts: dict[str, int] = {}

    for record in records:
        case_id = str(record.get("question_id") or "").strip()
        flags = [str(flag).strip() for flag in cast("list[object]", record.get("support_shape_flags") or []) if str(flag).strip()]
        if not case_id or not flags:
            continue
        flagged_case_ids.append(case_id)
        flags_by_case[case_id] = flags
        if any(flag in blocking_flags for flag in flags):
            blocking_case_ids.append(case_id)
        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "flagged_case_ids": flagged_case_ids,
        "blocking_case_ids": blocking_case_ids,
        "flagged_case_count": len(flagged_case_ids),
        "blocking_case_count": len(blocking_case_ids),
        "flags_by_case": flags_by_case,
        "flag_counts": flag_counts,
    }


def _build_truth_audit_report(scaffold: dict[str, object]) -> dict[str, object]:
    records_obj = scaffold.get("records")
    records = cast("list[dict[str, object]]", records_obj) if isinstance(records_obj, list) else []

    deterministic_incomplete_case_ids: list[str] = []
    free_text_incomplete_case_ids: list[str] = []
    deterministic_complete = 0
    free_text_complete = 0

    for record in records:
        answer_type = str(record.get("answer_type") or "").strip().lower()
        case_id = str(record.get("question_id") or "").strip()
        verdict = str(record.get("manual_verdict") or "").strip()
        if answer_type == "free_text":
            if verdict:
                free_text_complete += 1
            elif case_id:
                free_text_incomplete_case_ids.append(case_id)
        else:
            if verdict:
                deterministic_complete += 1
            elif case_id:
                deterministic_incomplete_case_ids.append(case_id)

    return {
        "deterministic_complete_count": deterministic_complete,
        "deterministic_incomplete_count": len(deterministic_incomplete_case_ids),
        "deterministic_incomplete_case_ids": deterministic_incomplete_case_ids,
        "free_text_complete_count": free_text_complete,
        "free_text_incomplete_count": len(free_text_incomplete_case_ids),
        "free_text_incomplete_case_ids": free_text_incomplete_case_ids,
    }


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
    *,
    concurrency: int,
    fail_fast: bool,
    runtime_factory: PipelineRuntimeFactory = _build_pipeline_runtime,
) -> list[PlatformCaseResult]:
    if not cases:
        return []

    worker_count = max(1, min(concurrency, len(cases)))
    results: list[PlatformCaseResult | None] = [None] * len(cases)
    queue: asyncio.Queue[tuple[int, SubmissionCase] | None] = asyncio.Queue()

    for index, case in enumerate(cases):
        queue.put_nowait((index, case))
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def worker() -> None:
        runtime = await runtime_factory()
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                index, case = item
                results[index] = await _run_case_direct(case, runtime, fail_fast=fail_fast)
        finally:
            await runtime.close()

    await asyncio.gather(*[worker() for _ in range(worker_count)])
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


def _check_existing_artifact_preflight(
    submission_path: Path,
    *,
    force: bool,
) -> None:
    """Guard for --submit-existing: reject red-preflight artifacts unless --force-submit-existing is set."""
    preflight_candidates = [
        submission_path.parent / submission_path.name.replace("submission", "preflight_summary"),
        submission_path.parent / "preflight_summary.json",
    ]
    preflight_path: Path | None = None
    for p in preflight_candidates:
        if p.exists():
            preflight_path = p
            break

    if preflight_path is None:
        if not force:
            raise RuntimeError(
                f"No preflight_summary found near {submission_path}. "
                "Use --force-submit-existing to bypass this check."
            )
        logger.warning("No preflight_summary found; proceeding because --force-submit-existing is set.")
        return

    try:
        preflight_obj = json.loads(preflight_path.read_text(encoding="utf-8"))
    except Exception as err:
        if not force:
            raise RuntimeError(
                f"Cannot parse preflight_summary at {preflight_path}. "
                "Use --force-submit-existing to bypass."
            ) from err
        logger.warning("Cannot parse preflight_summary; proceeding because --force-submit-existing is set.")
        return
    if not isinstance(preflight_obj, dict):
        if not force:
            raise RuntimeError(
                f"preflight_summary at {preflight_path} is not a JSON object. "
                "Use --force-submit-existing to bypass this check."
            )
        logger.warning("preflight_summary is not a JSON object; proceeding because --force-submit-existing is set.")
        return
    preflight = cast("dict[str, object]", preflight_obj)

    issues: list[str] = []
    support_shape_obj = preflight.get("support_shape_report")
    support_shape = cast("dict[str, object]", support_shape_obj) if isinstance(support_shape_obj, dict) else {}
    blocking = as_int(support_shape.get("blocking_case_count"))
    if blocking > 0:
        issues.append(f"support_shape blocking_case_count={blocking}")

    anomaly_obj = preflight.get("anomaly_report")
    anomaly = cast("dict[str, object]", anomaly_obj) if isinstance(anomaly_obj, dict) else {}
    anomaly_count = as_int(anomaly.get("anomaly_count"))
    if anomaly_count > 0:
        issues.append(f"anomaly_count={anomaly_count}")

    if issues:
        summary = "; ".join(issues)
        if not force:
            raise RuntimeError(
                f"Preflight checks failed for existing artifact: {summary}. "
                "Use --force-submit-existing to bypass."
            )
        logger.warning(
            "Preflight issues detected but proceeding because --force-submit-existing is set: %s",
            summary,
        )


async def _submit_existing_artifacts(
    client: PlatformEvaluationClient,
    *,
    submission_path: Path,
    code_archive_path: Path,
    poll: bool,
    poll_interval_s: float,
    poll_timeout_s: float,
    force: bool = False,
) -> dict[str, object]:
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission JSON not found: {submission_path}")
    if not code_archive_path.exists():
        raise FileNotFoundError(f"Code archive not found: {code_archive_path}")
    _check_existing_artifact_preflight(submission_path, force=force)
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
    _validate_platform_args(args)
    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_format)
    paths = _suffix_platform_paths(_resolve_phase_paths(), str(args.artifact_suffix or "").strip())
    collection_name = _phase_collection_name()
    query_concurrency = _resolve_query_concurrency(args.query_concurrency)
    canary_concurrency = int(args.equivalence_canary_concurrency or 0)
    archive_allowlist = _load_archive_allowlist(
        _resolve_archive_allowlist_path(settings.platform.archive_allowlist_path)
    )

    if bool(args.archive_only):
        _write_archive_artifacts(Path.cwd(), paths, archive_allowlist)
        logger.info("Prepared curated code archive without touching platform resources")
        logger.info("code_archive=%s", paths.code_archive_path)
        logger.info("audit_report=%s", paths.audit_report_path)
        return 0

    if args.support_only_challenger:
        source_submission_path = Path(args.source_submission_path) if args.source_submission_path else paths.submission_path
        source_questions_path = Path(args.source_questions_path) if args.source_questions_path else paths.questions_path
        source_raw_results_path = Path(args.source_raw_results_path) if args.source_raw_results_path else paths.raw_results_path
        if not source_submission_path.exists():
            raise FileNotFoundError(f"Source submission JSON not found: {source_submission_path}")
        if not source_questions_path.exists():
            raise FileNotFoundError(f"Source questions JSON not found: {source_questions_path}")
        source_truth_audit_path = _resolve_source_truth_audit_path(source_submission_path, args.source_truth_audit_path)

        source_payload = _load_platform_submission_payload(source_submission_path)
        source_truth_audit_scaffold = _load_truth_audit_scaffold(source_truth_audit_path)
        questions_by_id = _load_questions_by_id(source_questions_path)
        if args.support_only_challenger == "anchor-page-restitution":
            payload = _build_anchor_page_challenger_payload(
                source_payload=source_payload,
                questions_by_id=questions_by_id,
            )
            results = _rebuild_results_from_submission_payload(
                questions_by_id=questions_by_id,
                payload=payload,
            )
        elif args.support_only_challenger == "all-context-pages":
            if not source_raw_results_path.exists():
                raise FileNotFoundError(
                    "Source raw results JSON not found for support-only challenger 'all-context-pages': "
                    f"{source_raw_results_path}"
                )
            source_results = _load_raw_results(source_raw_results_path)
            payload = _build_all_context_pages_challenger_payload(
                source_payload=source_payload,
                source_results=source_results,
            )
            results = _rebuild_results_from_submission_payload(
                questions_by_id=questions_by_id,
                payload=payload,
            )
        else:
            raise ValueError(f"Unsupported support-only challenger: {args.support_only_challenger}")

        paths.phase_dir.mkdir(parents=True, exist_ok=True)
        paths.submission_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.raw_results_path.write_text(
            json.dumps([_serialize_platform_case_result(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        truth_audit_scaffold = build_truth_audit_scaffold(
            questions_path=source_questions_path,
            submission_path=paths.submission_path,
            docs_dir=paths.docs_dir if paths.docs_dir.exists() else None,
            existing_scaffold_path=source_truth_audit_path,
        )
        paths.truth_audit_path.write_text(
            json.dumps(truth_audit_scaffold, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.truth_audit_workbook_path.write_text(
            render_truth_audit_workbook(truth_audit_scaffold),
            encoding="utf-8",
        )
        _write_archive_artifacts(Path.cwd(), paths, archive_allowlist)
        support_shape_report = _build_support_shape_report(truth_audit_scaffold)
        truth_audit_report = _build_truth_audit_report(truth_audit_scaffold)
        support_only_challenger_report = _build_support_only_challenger_report(
            mode=args.support_only_challenger,
            source_payload=source_payload,
            challenger_payload=payload,
            source_submission_path=source_submission_path,
            source_truth_audit_path=source_truth_audit_path,
            source_truth_audit_scaffold=source_truth_audit_scaffold,
        )
        preflight_summary = _build_preflight_summary(
            paths=paths,
            collection_name=collection_name,
            payload=payload,
            results=results,
            point_count=None,
            anomaly_report=_build_results_anomaly_report(results),
            canary_report=None,
            support_shape_report=support_shape_report,
            truth_audit_report=truth_audit_report,
            support_only_challenger_report=support_only_challenger_report,
        )
        paths.preflight_summary_path.write_text(
            json.dumps(preflight_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Prepared support-only challenger artifact")
        logger.info("submission_json=%s", paths.submission_path)
        logger.info("preflight_summary=%s", paths.preflight_summary_path)
        logger.info("truth_audit=%s", paths.truth_audit_path)
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
                force=bool(getattr(args, "force_submit_existing", False)),
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
            results = await _run_questions(
                cases,
                concurrency=query_concurrency,
                fail_fast=bool(args.fail_fast),
            )
            results, anomaly_repairs = await _repair_anomalous_results(
                results,
                fail_fast=bool(args.fail_fast),
            )

            canary_report: dict[str, object] | None = None
            if canary_concurrency > 0 and canary_concurrency != query_concurrency:
                canary_results = await _run_questions(
                    cases,
                    concurrency=canary_concurrency,
                    fail_fast=bool(args.fail_fast),
                )
                canary_report = _build_equivalence_canary(
                    baseline_results=results,
                    candidate_results=canary_results,
                    baseline_concurrency=query_concurrency,
                    candidate_concurrency=canary_concurrency,
                )
                paths.phase_dir.mkdir(parents=True, exist_ok=True)
                paths.canary_path.write_text(
                    json.dumps(canary_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                canary_report = None

        payload = _build_platform_submission_payload(results)
        paths.phase_dir.mkdir(parents=True, exist_ok=True)
        paths.raw_results_path.write_text(
            json.dumps([_serialize_platform_case_result(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.submission_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        truth_audit_scaffold = build_truth_audit_scaffold(
            questions_path=paths.questions_path,
            submission_path=paths.submission_path,
            docs_dir=paths.docs_dir,
            existing_scaffold_path=paths.truth_audit_path,
        )
        paths.truth_audit_path.write_text(
            json.dumps(truth_audit_scaffold, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths.truth_audit_workbook_path.write_text(
            render_truth_audit_workbook(truth_audit_scaffold),
            encoding="utf-8",
        )
        _write_archive_artifacts(Path.cwd(), paths, archive_allowlist)
        point_count = await _qdrant_collection_point_count(collection_name)
        anomaly_report = _build_results_anomaly_report(results)
        anomaly_report["repair_report"] = anomaly_repairs
        support_shape_report = _build_support_shape_report(truth_audit_scaffold)
        truth_audit_report = _build_truth_audit_report(truth_audit_scaffold)
        preflight_summary = _build_preflight_summary(
            paths=paths,
            collection_name=collection_name,
            payload=payload,
            results=results,
            point_count=point_count,
            anomaly_report=anomaly_report,
            canary_report=canary_report,
            support_shape_report=support_shape_report,
            truth_audit_report=truth_audit_report,
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
            logger.info("score_settings_sha256=%s", preflight_summary.get("score_settings_sha256"))
            logger.info("truth_audit=%s", paths.truth_audit_path)
            logger.info("truth_audit_workbook=%s", paths.truth_audit_workbook_path)
            if canary_report is not None:
                logger.info("equivalence_canary=%s", paths.canary_path)
            return 0

        if as_int(support_shape_report.get("blocking_case_count")) > 0:
            logger.error(
                "Blocking support-shape violations detected in artifact: %s",
                ", ".join(cast("list[str]", support_shape_report.get("blocking_case_ids") or [])),
            )
            logger.error("Inspect truth audit and preflight summary before submitting.")
            return 3
        if as_int(truth_audit_report.get("deterministic_incomplete_count")) > 0:
            logger.error(
                "Deterministic truth audit incomplete for %d cases; refusing submit until audit is filled.",
                as_int(truth_audit_report.get("deterministic_incomplete_count")),
            )
            logger.error("Inspect truth audit scaffold and complete manual_verdict / failure_class fields first.")
            return 3

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
    parser.add_argument(
        "--support-only-challenger",
        choices=("anchor-page-restitution", "all-context-pages"),
        help="Build a support-only challenger artifact from an existing submission without changing answers.",
    )
    parser.add_argument("--source-submission-path", help="Source platform submission JSON for --support-only-challenger.")
    parser.add_argument("--source-questions-path", help="Source questions JSON for --support-only-challenger.")
    parser.add_argument(
        "--source-truth-audit-path",
        help="Source truth_audit_scaffold.json for --support-only-challenger. "
        "If omitted, infer it from --source-submission-path.",
    )
    parser.add_argument("--source-raw-results-path", help="Source raw_results.json for support-only challengers that need context_page_ids.")
    parser.add_argument("--artifact-suffix", help="Optional suffix for generated artifact filenames.")
    parser.add_argument("--refresh-downloads", action="store_true", help="Redownload questions/documents for the active phase.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse the existing phase-specific Qdrant collection.")
    parser.add_argument("--submit", action="store_true", help="Upload submission.json and code_archive.zip to the platform.")
    parser.add_argument("--submit-existing", action="store_true", help="Upload an existing submission.json and code archive without downloading, ingesting, or querying.")
    parser.add_argument(
        "--force-submit-existing",
        action="store_true",
        help="Bypass preflight checks for --submit-existing. Required when the artifact has known anomalies or blocking support-shape issues.",
    )
    parser.add_argument("--submission-path", help="Path to an existing submission.json for --submit-existing.")
    parser.add_argument("--code-archive-path", help="Path to an existing code archive for --submit-existing.")
    parser.add_argument("--query-concurrency", type=int, help="Override question execution concurrency for artifact generation.")
    parser.add_argument("--equivalence-canary-concurrency", type=int, help="Optional secondary concurrency to compare against the primary artifact build.")
    parser.add_argument("--poll", action="store_true", help="Poll submission status until completion or timeout.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first question execution failure.")
    args = parser.parse_args(argv)

    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
