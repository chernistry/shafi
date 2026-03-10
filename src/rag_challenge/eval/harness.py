from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from rag_challenge.config import get_settings
from rag_challenge.config.logging import setup_logging
from rag_challenge.eval.golden import GoldenCase, load_golden_dataset
from rag_challenge.eval.judge import JudgeClient, JudgeOutcome
from rag_challenge.eval.metrics import AnswerTypeFormatCompliance, CitationCoverage
from rag_challenge.eval.sources import (
    PdfPageTextProvider,
    QdrantPageTextFallback,
    build_sources_text,
    select_used_pages,
)

logger = logging.getLogger(__name__)


def _float_list_factory() -> list[float]:
    return []


def _str_list_factory() -> list[str]:
    return []


def _list_by_type_factory() -> dict[str, list[float]]:
    return {}


def _sum_by_type_factory() -> dict[str, float]:
    return {}


def _count_by_type_factory() -> dict[str, int]:
    return {}


def _stage_values_factory() -> dict[str, list[float]]:
    return {
        "classify_ms": [],
        "embed_ms": [],
        "qdrant_ms": [],
        "rerank_ms": [],
        "llm_ms": [],
        "verify_ms": [],
    }


def _cases_factory() -> list[dict[str, object]]:
    return []


@dataclass
class EvalResult:
    total_cases: int = 0
    gold_annotated_cases: int = 0
    answer_type_cases: int = 0
    recall_at_80_sum: float = 0.0
    recall_at_k_sum: float = 0.0
    citation_coverage_sum: float = 0.0
    g_score_sum: float = 0.0
    g_score_cases: int = 0
    answer_type_format_compliance_sum: float = 0.0
    ttft_values: list[float] = field(default_factory=_float_list_factory)
    failures: list[str] = field(default_factory=_str_list_factory)
    doc_ref_cases: int = 0
    doc_ref_measured_cases: int = 0
    doc_ref_hit_rate_sum: float = 0.0
    ttft_by_answer_type: dict[str, list[float]] = field(default_factory=_list_by_type_factory)
    format_compliance_sum_by_answer_type: dict[str, float] = field(default_factory=_sum_by_type_factory)
    format_compliance_count_by_answer_type: dict[str, int] = field(default_factory=_count_by_type_factory)
    citation_coverage_sum_by_answer_type: dict[str, float] = field(default_factory=_sum_by_type_factory)
    citation_coverage_count_by_answer_type: dict[str, int] = field(default_factory=_count_by_type_factory)
    g_score_sum_by_answer_type: dict[str, float] = field(default_factory=_sum_by_type_factory)
    g_score_count_by_answer_type: dict[str, int] = field(default_factory=_count_by_type_factory)
    stage_values_ms: dict[str, list[float]] = field(default_factory=_stage_values_factory)
    citation_hallucination_cases: int = 0
    citation_hallucination_rate_sum: float = 0.0
    judge_cases: int = 0
    judge_pass_cases: int = 0
    judge_failures: int = 0
    judge_accuracy_sum: int = 0
    judge_grounding_sum: int = 0
    judge_clarity_sum: int = 0
    judge_uncertainty_sum: int = 0
    cases: list[dict[str, object]] = field(default_factory=_cases_factory)

    @property
    def recall_at_80(self) -> float | None:
        if self.gold_annotated_cases <= 0:
            return None
        return self.recall_at_80_sum / self.gold_annotated_cases

    @property
    def recall_at_k(self) -> float | None:
        if self.gold_annotated_cases <= 0:
            return None
        return self.recall_at_k_sum / self.gold_annotated_cases

    @property
    def citation_coverage(self) -> float:
        return self.citation_coverage_sum / max(1, self.total_cases)

    @property
    def answer_type_format_compliance(self) -> float | None:
        if self.answer_type_cases <= 0:
            return None
        return self.answer_type_format_compliance_sum / self.answer_type_cases

    @property
    def grounding_g_score_beta_2_5(self) -> float | None:
        if self.g_score_cases <= 0:
            return None
        return self.g_score_sum / self.g_score_cases

    @property
    def ttft_p50(self) -> float:
        return _percentile(self.ttft_values, 0.50)

    @property
    def ttft_p95(self) -> float:
        return _percentile(self.ttft_values, 0.95)

    @property
    def doc_ref_hit_rate(self) -> float | None:
        if self.doc_ref_measured_cases <= 0:
            return None
        return self.doc_ref_hit_rate_sum / self.doc_ref_measured_cases

    @property
    def judge_pass_rate(self) -> float | None:
        if self.judge_cases <= 0:
            return None
        return self.judge_pass_cases / max(1, self.judge_cases)

    def summary(self) -> dict[str, object]:
        recall80 = self.recall_at_80
        recallk = self.recall_at_k
        g_score = self.grounding_g_score_beta_2_5
        format_compliance = self.answer_type_format_compliance
        doc_ref_hit = self.doc_ref_hit_rate
        ttft_by_type: dict[str, object] = {}
        for answer_type, values in self.ttft_by_answer_type.items():
            ttft_by_type[answer_type] = {
                "p50_ms": round(_percentile(values, 0.5), 1),
                "p95_ms": round(_percentile(values, 0.95), 1),
                "count": len(values),
            }

        format_by_type: dict[str, float] = {}
        for answer_type, count in self.format_compliance_count_by_answer_type.items():
            if count <= 0:
                continue
            total = self.format_compliance_sum_by_answer_type.get(answer_type, 0.0)
            format_by_type[answer_type] = round(total / count, 4)

        coverage_by_type: dict[str, float] = {}
        for answer_type, count in self.citation_coverage_count_by_answer_type.items():
            if count <= 0:
                continue
            total = self.citation_coverage_sum_by_answer_type.get(answer_type, 0.0)
            coverage_by_type[answer_type] = round(total / count, 4)

        g_score_by_type: dict[str, float] = {}
        for answer_type, count in self.g_score_count_by_answer_type.items():
            if count <= 0:
                continue
            total = self.g_score_sum_by_answer_type.get(answer_type, 0.0)
            g_score_by_type[answer_type] = round(total / count, 4)

        stage_p50 = {name: round(_percentile(values, 0.5), 1) for name, values in self.stage_values_ms.items()}
        stage_p95 = {name: round(_percentile(values, 0.95), 1) for name, values in self.stage_values_ms.items()}
        citation_hallucination_rate = (
            self.citation_hallucination_rate_sum / self.citation_hallucination_cases
            if self.citation_hallucination_cases > 0
            else None
        )

        sorted_cases = sorted(
            self.cases,
            key=lambda row: _coerce_float(row.get("ttft_ms"), default=0.0),
            reverse=True,
        )
        top_10_slowest = sorted_cases[:10]
        top_10_boolean = [
            row
            for row in sorted_cases
            if str(row.get("answer_type", "")).strip().lower() == "boolean"
        ][:10]
        hallucination_suspects = sorted(
            [
                row
                for row in self.cases
                if _coerce_float(row.get("citation_hallucination_rate"), default=0.0) > 0.0
            ],
            key=lambda row: _coerce_float(row.get("citation_hallucination_rate"), default=0.0),
            reverse=True,
        )[:10]
        summary: dict[str, object] = {
            "total_cases": self.total_cases,
            "gold_annotated_cases": self.gold_annotated_cases,
            "answer_type_cases": self.answer_type_cases,
            "recall@80": None if recall80 is None else round(recall80, 4),
            "recall@k": None if recallk is None else round(recallk, 4),
            "grounding_g_score_beta_2_5": None if g_score is None else round(g_score, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "answer_type_format_compliance": None if format_compliance is None else round(format_compliance, 4),
            "doc_ref_cases": self.doc_ref_cases,
            "doc_ref_measured_cases": self.doc_ref_measured_cases,
            "doc_ref_hit_rate": None if doc_ref_hit is None else round(doc_ref_hit, 4),
            "ttft_p50_ms": round(self.ttft_p50, 1),
            "ttft_p95_ms": round(self.ttft_p95, 1),
            "ttft_count": len(self.ttft_values),
            "ttft_by_answer_type": ttft_by_type,
            "format_compliance_by_answer_type": format_by_type,
            "citation_coverage_by_answer_type": coverage_by_type,
            "grounding_g_score_beta_2_5_by_answer_type": g_score_by_type,
            "stage_p50_ms": stage_p50,
            "stage_p95_ms": stage_p95,
            "citation_hallucination_rate": None if citation_hallucination_rate is None else round(citation_hallucination_rate, 4),
            "top_10_slowest_cases": top_10_slowest,
            "top_10_boolean_slowest": top_10_boolean,
            "hallucination_suspects": hallucination_suspects,
            "failures": len(self.failures),
        }

        if self.judge_cases > 0 or self.judge_failures > 0:
            pass_rate = self.judge_pass_rate
            judge_block: dict[str, object] = {
                "cases": int(self.judge_cases),
                "pass_rate": None if pass_rate is None else round(pass_rate, 4),
                "avg_accuracy": round(self.judge_accuracy_sum / max(1, self.judge_cases), 4)
                if self.judge_cases
                else None,
                "avg_grounding": round(self.judge_grounding_sum / max(1, self.judge_cases), 4)
                if self.judge_cases
                else None,
                "avg_clarity": round(self.judge_clarity_sum / max(1, self.judge_cases), 4)
                if self.judge_cases
                else None,
                "avg_uncertainty_handling": round(self.judge_uncertainty_sum / max(1, self.judge_cases), 4)
                if self.judge_cases
                else None,
                "judge_failures": int(self.judge_failures),
                "top_fails": _judge_top_fails(self.cases),
            }
            summary["judge"] = judge_block

        return summary


async def run_evaluation(
    golden_path: str | Path,
    endpoint_url: str = "http://localhost:8000/query",
    concurrency: int = 4,
    emit_cases: bool = False,
    *,
    judge_enabled: bool = False,
    judge_scope: str = "free_text",
    judge_docs_dir: str | Path | None = None,
    judge_out_path: str | Path | None = None,
) -> EvalResult:
    """Run batch eval against live `/query` endpoint and aggregate retrieval/citation metrics."""
    cases = load_golden_dataset(golden_path)
    result = EvalResult(total_cases=len(cases))
    endpoint_semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    lock = asyncio.Lock()
    judge_out_lock = asyncio.Lock()

    settings = get_settings()
    qdrant_client: AsyncQdrantClient | None = None
    citations_cache: dict[str, set[str]] = {}

    judge_scope_key = str(judge_scope or "free_text").strip().lower()
    if judge_scope_key not in {"free_text", "all", "none"}:
        judge_scope_key = "free_text"

    judge_out_file: Path | None = None
    if judge_out_path is not None and str(judge_out_path).strip():
        judge_out_file = Path(judge_out_path)
        judge_out_file.parent.mkdir(parents=True, exist_ok=True)
        judge_out_file.write_text("", encoding="utf-8")

    judge_is_enabled = bool(judge_enabled or settings.judge.enabled)
    judge_client: JudgeClient | None = None
    judge_semaphore = asyncio.Semaphore(1)
    pdf_provider: PdfPageTextProvider | None = None
    qdrant_fallback: QdrantPageTextFallback | None = None

    if judge_is_enabled:
        judge_client = JudgeClient()
        judge_semaphore = asyncio.Semaphore(max(1, int(getattr(settings.judge, "concurrency", 2))))
        docs_dir = Path(judge_docs_dir) if judge_docs_dir is not None else Path(str(settings.judge.docs_dir))
        pdf_provider = PdfPageTextProvider(
            docs_dir,
            max_chars_per_page=int(getattr(settings.judge, "sources_max_chars_per_page", 20000)),
        )

    qdrant_client: AsyncQdrantClient | None = None
    try:
        try:
            qdrant_client = AsyncQdrantClient(
                url=settings.qdrant.url,
                api_key=settings.qdrant.api_key or None,
                timeout=int(settings.qdrant.timeout_s),
                check_compatibility=bool(getattr(settings.qdrant, "check_compatibility", True)),
            )
            await qdrant_client.get_collections()
        except Exception:
            logger.warning("Qdrant unavailable for doc-ref hit-rate metric; skipping", exc_info=True)
            qdrant_client = None
        else:
            if judge_is_enabled:
                qdrant_fallback = QdrantPageTextFallback(
                    qdrant_client=qdrant_client,
                    collection=str(settings.qdrant.collection),
                    max_chars_per_page=int(getattr(settings.judge, "sources_max_chars_per_page", 20000)),
                )

        async def _eval_case(case: GoldenCase, client: httpx.AsyncClient) -> None:
            failure: str | None = None
            r80 = 0.0
            rk = 0.0
            g_score_value: float | None = None
            coverage = 0.0
            format_compliance = 0.0
            ttft_ms: float | None = None
            has_gold = bool(case.gold_chunk_ids)
            has_answer_type = bool(case.answer_type.strip())
            answer_type_key = case.answer_type.strip().lower() or "free_text"
            doc_refs: list[str] = []
            doc_ref_hit_rate: float | None = None
            stage_values: dict[str, float] = {}
            citation_hallucination_rate: float | None = None
            context_ids: list[str] = []
            cited_ids: list[str] = []
            token_text = ""

            payload: dict[str, object] | None = None
            judge_used_pages: list[str] = []
            judge_outcome: JudgeOutcome | None = None
            judge_sources_sha256 = ""
            judge_failure = ""

            try:
                async with endpoint_semaphore:
                    t0 = time.perf_counter()
                    request_payload: dict[str, object] = {
                        "question": case.question,
                        "request_id": case.case_id,
                        # Ticket 25 compatibility: harmless for old API (ignored extra fields).
                        "question_id": case.case_id,
                        "answer_type": case.answer_type,
                    }
                    response = await client.post(
                        endpoint_url,
                        json=request_payload,
                        timeout=30.0,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    response.raise_for_status()

                    sse_events = _parse_sse_body(response.text)
                    answer_final_evt = next(
                        (evt for evt in sse_events if evt.get("type") == "answer_final"), None
                    )
                    if answer_final_evt is not None:
                        token_text = str(answer_final_evt.get("text", "")).strip()
                    else:
                        token_text = "".join(
                            str(event.get("text", "")) for event in sse_events if event.get("type") == "token"
                        )
                    telemetry_event = next((evt for evt in sse_events if evt.get("type") == "telemetry"), None)

                    if telemetry_event is None:
                        failure = f"No telemetry SSE event for: {case.question[:80]}"
                    else:
                        payload_obj = telemetry_event.get("payload", {})
                        payload = cast("dict[str, object]", payload_obj) if isinstance(payload_obj, dict) else {}
                        retrieved_ids = _coerce_str_list(payload.get("retrieved_chunk_ids"))
                        context_ids = _coerce_str_list(payload.get("context_chunk_ids"))
                        cited_ids = _coerce_str_list(payload.get("cited_chunk_ids"))
                        doc_refs = _coerce_str_list(payload.get("doc_refs"))
                        stage_values = {
                            "classify_ms": _coerce_float(payload.get("classify_ms"), default=0.0),
                            "embed_ms": _coerce_float(payload.get("embed_ms"), default=0.0),
                            "qdrant_ms": _coerce_float(payload.get("qdrant_ms"), default=0.0),
                            "rerank_ms": _coerce_float(payload.get("rerank_ms"), default=0.0),
                            "llm_ms": _coerce_float(payload.get("llm_ms"), default=0.0),
                            "verify_ms": _coerce_float(payload.get("verify_ms"), default=0.0),
                        }

                        gold = set(case.gold_chunk_ids)
                        if gold:
                            r80 = len(gold.intersection(set(retrieved_ids))) / len(gold)
                            rk = len(gold.intersection(set(context_ids))) / len(gold)
                            g_score_value = _g_score(set(context_ids), gold, beta=2.5)

                        # Null answers (strict types) and "no information" answers (free_text) have
                        # no citations by design — the pipeline correctly clears context/cited ids.
                        _is_null_answer = token_text.strip().lower() == "null"
                        _is_no_info_answer = token_text.strip().lower().startswith(
                            "there is no information on this question"
                        )
                        if _is_null_answer or _is_no_info_answer:
                            coverage = 1.0
                            parsed_cited_ids: set[str] = set()
                        elif cited_ids and context_ids:
                            coverage = len(set(cited_ids).intersection(set(context_ids))) / max(
                                1, len(set(cited_ids))
                            )
                            parsed_cited_ids = set()
                        else:
                            parsed_cited_ids = CitationCoverage.extract_cited_chunk_ids(token_text)
                            coverage = (
                                len(parsed_cited_ids.intersection(set(context_ids))) / max(1, len(parsed_cited_ids))
                                if parsed_cited_ids and context_ids
                                else 0.0
                            )
                        if coverage < 1.0:
                            print(
                                f"[{case.case_id}] CITATION FAIL: Coverage {coverage}, "
                                f"Cited: {cited_ids or parsed_cited_ids}, Context: {context_ids}"
                            )
                        parsed_for_hallucination = set(cited_ids or list(parsed_cited_ids))
                        if parsed_for_hallucination:
                            invalid = parsed_for_hallucination.difference(set(context_ids))
                            citation_hallucination_rate = len(invalid) / max(1, len(parsed_for_hallucination))
                        else:
                            citation_hallucination_rate = 0.0

                        ttft_value: object = payload.get("ttft_ms", elapsed_ms)
                        ttft_ms = _coerce_float(ttft_value, default=elapsed_ms)

                        if doc_refs and qdrant_client is not None and context_ids:
                            doc_ref_hit_rate = await _doc_ref_hit_rate(
                                qdrant_client=qdrant_client,
                                collection=settings.qdrant.collection,
                                context_chunk_ids=context_ids,
                                doc_refs=doc_refs,
                                citations_cache=citations_cache,
                            )

                if has_answer_type:
                    format_compliance = (
                        1.0
                        if AnswerTypeFormatCompliance.is_answer_format_compliant(token_text, case.answer_type)
                        else 0.0
                    )
                    if format_compliance < 1.0:
                        print(f"[{case.case_id}] FORMAT FAIL: Type {case.answer_type}, Text: {token_text!r}")

            except Exception as exc:
                failure = f"{case.question[:80]}: {exc}"

            if (
                judge_is_enabled
                and payload is not None
                and judge_client is not None
                and pdf_provider is not None
                and judge_scope_key != "none"
            ):
                should_judge = judge_scope_key == "all" or (
                    judge_scope_key == "free_text" and answer_type_key == "free_text"
                )
                if should_judge:
                    judge_used_pages = select_used_pages(
                        payload,
                        max_pages=int(getattr(settings.judge, "sources_max_pages", 6)),
                    )
                    async with judge_semaphore:
                        sources_text = await build_sources_text(
                            judge_used_pages,
                            pdf_provider=pdf_provider,
                            qdrant_fallback=qdrant_fallback,
                            max_chars_total=int(getattr(settings.judge, "sources_max_chars_total", 60000)),
                        )
                        judge_sources_sha256 = hashlib.sha256(sources_text.encode("utf-8")).hexdigest()
                        judge_outcome = await judge_client.evaluate(
                            question=case.question,
                            answer_type=answer_type_key,
                            answer=token_text.strip(),
                            used_pages=judge_used_pages,
                            sources_text=sources_text,
                        )
                        judge_failure = judge_outcome.failure

                        if judge_out_file is not None:
                            row = {
                                "case_id": case.case_id,
                                "question_id": case.case_id,
                                "answer_type": answer_type_key,
                                "question": case.question,
                                "answer": token_text.strip(),
                                "used_pages": judge_used_pages,
                                "sources_text_sha256": judge_sources_sha256,
                                "judge_model": judge_outcome.model,
                                "judge_failure": judge_outcome.failure,
                                "judge_result": (
                                    judge_outcome.result.model_dump(mode="json") if judge_outcome.result else None
                                ),
                            }
                            async with judge_out_lock:
                                with judge_out_file.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            async with lock:
                if has_gold:
                    result.gold_annotated_cases += 1
                result.recall_at_80_sum += r80
                result.recall_at_k_sum += rk
                if g_score_value is not None:
                    result.g_score_sum += g_score_value
                    result.g_score_cases += 1
                    result.g_score_sum_by_answer_type[answer_type_key] = (
                        result.g_score_sum_by_answer_type.get(answer_type_key, 0.0) + g_score_value
                    )
                    result.g_score_count_by_answer_type[answer_type_key] = (
                        result.g_score_count_by_answer_type.get(answer_type_key, 0) + 1
                    )
                result.citation_coverage_sum += coverage
                if has_answer_type:
                    result.answer_type_cases += 1
                    result.answer_type_format_compliance_sum += format_compliance
                if ttft_ms is not None:
                    result.ttft_values.append(ttft_ms)
                    result.ttft_by_answer_type.setdefault(answer_type_key, []).append(ttft_ms)
                if has_answer_type:
                    result.format_compliance_sum_by_answer_type[answer_type_key] = (
                        result.format_compliance_sum_by_answer_type.get(answer_type_key, 0.0) + format_compliance
                    )
                    result.format_compliance_count_by_answer_type[answer_type_key] = (
                        result.format_compliance_count_by_answer_type.get(answer_type_key, 0) + 1
                    )
                result.citation_coverage_sum_by_answer_type[answer_type_key] = (
                    result.citation_coverage_sum_by_answer_type.get(answer_type_key, 0.0) + coverage
                )
                result.citation_coverage_count_by_answer_type[answer_type_key] = (
                    result.citation_coverage_count_by_answer_type.get(answer_type_key, 0) + 1
                )
                for stage_name, stage_value in stage_values.items():
                    result.stage_values_ms.setdefault(stage_name, []).append(stage_value)
                if citation_hallucination_rate is not None:
                    result.citation_hallucination_cases += 1
                    result.citation_hallucination_rate_sum += citation_hallucination_rate
                if judge_outcome is not None:
                    if judge_outcome.result is None:
                        result.judge_failures += 1
                    else:
                        result.judge_cases += 1
                        if judge_outcome.result.verdict == "PASS":
                            result.judge_pass_cases += 1
                        result.judge_accuracy_sum += int(judge_outcome.result.scores.accuracy)
                        result.judge_grounding_sum += int(judge_outcome.result.scores.grounding)
                        result.judge_clarity_sum += int(judge_outcome.result.scores.clarity)
                        result.judge_uncertainty_sum += int(judge_outcome.result.scores.uncertainty_handling)
                if failure is not None:
                    result.failures.append(failure)
                if doc_refs:
                    result.doc_ref_cases += 1
                    if doc_ref_hit_rate is not None:
                        result.doc_ref_measured_cases += 1
                        result.doc_ref_hit_rate_sum += float(doc_ref_hit_rate)
                if emit_cases:
                    case_record: dict[str, object] = {
                        "case_id": case.case_id,
                        "question_id": case.case_id,
                        "answer_type": case.answer_type,
                        "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
                        "doc_ref_hit_rate": None if doc_ref_hit_rate is None else round(doc_ref_hit_rate, 4),
                        "citation_coverage": round(coverage, 4),
                        "grounding_g_score_beta_2_5": (
                            None if g_score_value is None else round(g_score_value, 4)
                        ),
                        "citation_hallucination_rate": (
                            None if citation_hallucination_rate is None else round(citation_hallucination_rate, 4)
                        ),
                        "format_compliance": round(format_compliance, 4) if has_answer_type else None,
                        "context_chunk_count": len(context_ids),
                        "cited_chunk_count": len(cited_ids),
                        "stage_ms": {key: round(value, 1) for key, value in stage_values.items()},
                        "question": case.question,
                        "answer": token_text.strip(),
                        "failure": failure,
                        "telemetry": payload,
                    }
                    if judge_outcome is not None:
                        case_record["judge_used_pages"] = list(judge_used_pages)
                        case_record["judge_sources_sha256"] = judge_sources_sha256
                        case_record["judge_failure"] = judge_failure
                        case_record["judge_model"] = judge_outcome.model
                        if judge_outcome.result is not None:
                            case_record["judge"] = {
                                "verdict": judge_outcome.result.verdict,
                                "scores": judge_outcome.result.scores.model_dump(mode="json"),
                                "format_issues": judge_outcome.result.format_issues[:10],
                                "unsupported_claims": judge_outcome.result.unsupported_claims[:10],
                                "recommended_fix": judge_outcome.result.recommended_fix[:500],
                            }
                    result.cases.append(case_record)

        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[_eval_case(case, client) for case in cases])
        return result

    finally:
        if pdf_provider is not None:
            pdf_provider.close()
        if judge_client is not None:
            await judge_client.close()
        if qdrant_client is not None:
            await qdrant_client.close()


def _parse_sse_body(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(cast("dict[str, object]", parsed))
    return events


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    return [text for item in items if (text := str(item).strip())]


def _coerce_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except ValueError:
            return float(default)
        return max(0.0, number)
    try:
        number = float(str(value))
    except Exception:
        return float(default)
    return max(0.0, number)


def _g_score(predicted: set[str], gold: set[str], beta: float = 2.5) -> float:
    if not gold:
        return 1.0 if not predicted else 0.0

    true_positives = len(predicted.intersection(gold))
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold)
    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta * beta
    denominator = (beta_sq * precision) + recall
    if denominator <= 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denominator


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, int((len(sorted_values) - 1) * q)))
    return float(sorted_values[idx])


def _judge_top_fails(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    if not cases:
        return []
    offenders: list[dict[str, object]] = []
    for row in cases:
        judge_obj = row.get("judge")
        if not isinstance(judge_obj, dict):
            continue
        judge = cast("dict[str, object]", judge_obj)
        verdict = str(judge.get("verdict") or "").strip().upper()
        scores_obj = judge.get("scores")
        scores = cast("dict[str, object]", scores_obj) if isinstance(scores_obj, dict) else {}
        grounding = _coerce_float(scores.get("grounding"), default=0.0)
        accuracy = _coerce_float(scores.get("accuracy"), default=0.0)
        used_pages_obj = row.get("judge_used_pages", [])
        used_pages = (
            [str(v).strip() for v in cast("list[object]", used_pages_obj) if str(v).strip()]
            if isinstance(used_pages_obj, list)
            else []
        )
        unsupported_obj = judge.get("unsupported_claims", [])
        unsupported = (
            [str(v).strip() for v in cast("list[object]", unsupported_obj) if str(v).strip()]
            if isinstance(unsupported_obj, list)
            else []
        )
        offenders.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "answer_type": str(row.get("answer_type") or ""),
                "verdict": verdict,
                "grounding": round(grounding, 3),
                "accuracy": round(accuracy, 3),
                "used_pages": used_pages,
                "unsupported_claims": unsupported[:5],
                "question": str(row.get("question") or "")[:300],
            }
        )
    offenders.sort(
        key=lambda r: (
            0 if str(r.get("verdict") or "").strip().upper() == "FAIL" else 1,
            cast("float", r.get("grounding", 0.0)),
            cast("float", r.get("accuracy", 0.0)),
        )
    )
    return offenders[:10]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legal RAG Evaluation Harness")
    parser.add_argument("--golden", type=Path, required=True, help="Path to golden dataset JSON")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/query")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path for run summary")
    parser.add_argument(
        "--emit-cases",
        action="store_true",
        help="Emit per-case records to output JSON and include top-offender diagnostics in summary.",
    )
    parser.add_argument("--judge", action="store_true", help="Enable optional LLM-as-judge pass (advisory).")
    parser.add_argument(
        "--judge-scope",
        type=str,
        choices=["free_text", "all", "none"],
        default="free_text",
        help="Which answer types to judge (default: free_text only).",
    )
    parser.add_argument("--judge-docs-dir", type=Path, default=None, help="Override docs dir for sources extraction.")
    parser.add_argument("--judge-out", type=Path, default=None, help="Optional JSONL output for per-case judge records.")
    return parser


async def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_format)
    result = await run_evaluation(
        golden_path=args.golden,
        endpoint_url=args.endpoint,
        concurrency=args.concurrency,
        emit_cases=bool(args.emit_cases),
        judge_enabled=bool(getattr(args, "judge", False)),
        judge_scope=str(getattr(args, "judge_scope", "free_text")),
        judge_docs_dir=getattr(args, "judge_docs_dir", None),
        judge_out_path=getattr(args, "judge_out", None),
    )
    summary = result.summary()
    print(json.dumps(summary, indent=2))

    out_path = getattr(args, "out", None)
    if isinstance(out_path, Path):
        out_payload: dict[str, object] = {"summary": summary, "failures": list(result.failures)}
        if bool(args.emit_cases):
            out_payload["cases"] = list(result.cases)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    recall80 = result.recall_at_80
    if recall80 is None:
        logger.warning("No gold_chunk_ids annotations available; skipping recall@80 guardrail")
        return 0
    if recall80 < 0.80:
        logger.warning("recall@80 below guardrail: %.3f", recall80)
        return 1
    return 0


async def _doc_ref_hit_rate(
    *,
    qdrant_client: AsyncQdrantClient,
    collection: str,
    context_chunk_ids: list[str],
    doc_refs: list[str],
    citations_cache: dict[str, set[str]],
) -> float:
    """Proxy grounding metric for public dataset (no gold chunks): do we hit the referenced doc?"""
    if not doc_refs:
        return 0.0
    if not context_chunk_ids:
        return 0.0

    missing_chunk_ids = [chunk_id for chunk_id in context_chunk_ids if chunk_id not in citations_cache]
    if missing_chunk_ids:
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)) for chunk_id in missing_chunk_ids]
        try:
            records = await qdrant_client.retrieve(
                collection_name=collection,
                ids=point_ids,
                with_payload=["chunk_id", "citations"],
                with_vectors=False,
            )
        except Exception:
            logger.debug("Failed retrieving citations payloads from Qdrant", exc_info=True)
            records = []

        for record in records:
            payload_obj: object = getattr(record, "payload", None)
            if not isinstance(payload_obj, dict):
                continue
            payload = cast("dict[str, object]", payload_obj)
            chunk_id_obj = payload.get("chunk_id")
            if not isinstance(chunk_id_obj, str) or not chunk_id_obj.strip():
                continue
            citations_obj = payload.get("citations", [])
            citations_list = (
                [str(item).strip() for item in cast("list[object]", citations_obj) if str(item).strip()]
                if isinstance(citations_obj, list)
                else []
            )
            citations_cache[chunk_id_obj.strip()] = set(citations_list)

        still_missing = [chunk_id for chunk_id in missing_chunk_ids if chunk_id not in citations_cache]
        if still_missing:
            # Snapshot-restored collections might not use our UUID5(point_id) scheme. Fall back to
            # payload lookup by `chunk_id` so the metric remains meaningful across deployments.
            try:
                selector = Filter(must=[FieldCondition(key="chunk_id", match=MatchAny(any=still_missing))])
                records, _next = await qdrant_client.scroll(
                    collection_name=collection,
                    scroll_filter=selector,
                    limit=256,
                    with_payload=["chunk_id", "citations"],
                    with_vectors=False,
                )
            except Exception:
                logger.debug("Failed scrolling citations payloads from Qdrant", exc_info=True)
                records = []

            for record in records:
                payload_obj: object = getattr(record, "payload", None)
                if not isinstance(payload_obj, dict):
                    continue
                payload = cast("dict[str, object]", payload_obj)
                chunk_id_obj = payload.get("chunk_id")
                if not isinstance(chunk_id_obj, str) or not chunk_id_obj.strip():
                    continue
                citations_obj = payload.get("citations", [])
                citations_list = (
                    [str(item).strip() for item in cast("list[object]", citations_obj) if str(item).strip()]
                    if isinstance(citations_obj, list)
                    else []
                )
                citations_cache[chunk_id_obj.strip()] = set(citations_list)

    union: set[str] = set()
    for chunk_id in context_chunk_ids:
        union.update(citations_cache.get(chunk_id, set()))

    hits = sum(1 for ref in doc_refs if ref in union)
    return hits / max(1, len(doc_refs))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
