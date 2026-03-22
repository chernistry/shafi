from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from pydantic import SecretStr

from rag_challenge.core.selective_icr_reranker import (
    SelectiveICRConfig,
    SelectiveICRReranker,
)
from rag_challenge.models import DocType, RetrievedChunk


def _chunk(
    chunk_id: str,
    *,
    text: str,
    doc_title: str = "Example Title",
    section_path: str = "page:1 | title",
    score: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split(":", 1)[0],
        doc_title=doc_title,
        doc_type=DocType.STATUTE,
        section_path=section_path,
        text=text,
        score=score,
        page_family="title",
        doc_family="statute",
    )


def test_selective_icr_prioritizes_structurally_matching_chunk() -> None:
    reranker = SelectiveICRReranker()
    chunks = [
        _chunk("doc-a:0:0:a", text="General introduction and background", section_path="page:1 | intro"),
        _chunk(
            "doc-b:2:0:b",
            text="Article 8. A person may not operate without registration.",
            section_path="page:3 | article 8",
        ),
    ]

    ranked = reranker.rank("Under Article 8, may a person operate without registration?", chunks, top_n=2)

    assert [item.chunk_id for item in ranked] == ["doc-b:2:0:b", "doc-a:0:0:a"]
    assert ranked[0].page_id == "doc-b_3"
    assert reranker.get_last_diagnostics().candidate_count == 2


def test_selective_icr_uses_model_object_when_available() -> None:
    model = SimpleNamespace(predict=lambda pairs: [0.2, 0.9])
    reranker = SelectiveICRReranker(config=SelectiveICRConfig(model_path="local-model"), model_obj=model)
    chunks = [
        _chunk("doc-a:0:0:a", text="first chunk"),
        _chunk("doc-b:1:0:b", text="second chunk"),
    ]

    scores = reranker.score_documents("query", [chunk.text for chunk in chunks])
    ranked = reranker.rank("query", chunks, top_n=2)

    assert scores == [0.0, 1.0]
    assert [item.chunk_id for item in ranked] == ["doc-b:1:0:b", "doc-a:0:0:a"]
    assert reranker.model_name == "local-model"


def test_shadow_rerank_helper_returns_ranked_chunks() -> None:
    from rag_challenge.core.reranker import RerankerClient

    settings = SimpleNamespace(
        reranker=SimpleNamespace(
            primary_model="zerank-2",
            primary_api_url="https://api.zeroentropy.dev/v1/models/rerank",
            primary_api_key=SecretStr("ze-key"),
            primary_batch_size=50,
            primary_latency_mode="fast",
            primary_timeout_s=5.0,
            primary_max_connections=20,
            primary_concurrency_limit=1,
            primary_min_interval_s=0.0,
            primary_connect_timeout_s=10.0,
            fallback_model="rerank-v4.0-fast",
            fallback_api_key=SecretStr("cohere-key"),
            fallback_timeout_s=5.0,
            shadow_selective_icr_enabled=True,
            shadow_selective_icr_provider_exit=False,
            shadow_selective_icr_model_path="",
            shadow_selective_icr_max_chars=1800,
            shadow_selective_icr_candidate_batch_size=32,
            shadow_selective_icr_normalize_scores=True,
            top_n=6,
            rerank_candidates=80,
            retry_attempts=4,
            retry_base_delay_s=0.5,
            retry_max_delay_s=8.0,
            retry_jitter_s=1.0,
            circuit_failure_threshold=3,
            circuit_reset_timeout_s=60.0,
        )
    )
    chunks = [
        _chunk("doc-a:0:0:a", text="Background text"),
        _chunk("doc-b:2:0:b", text="Article 8. A person may not operate without registration."),
    ]
    with patch("rag_challenge.core.reranker.get_settings", return_value=settings), patch(
        "rag_challenge.core.request_limiter._LIMITER_REGISTRY",
        {},
    ):
        async def _run() -> list[str]:
            async with httpx.AsyncClient() as client:
                rc = RerankerClient(client=client, cohere_client=SimpleNamespace(rerank=None))
                ranked = await rc.shadow_rerank("Article 8 query", chunks, top_n=2)
                assert rc.get_last_shadow_used_model() == "selective_icr_heuristic"
                assert rc.get_last_shadow_candidate_count() == 2
                assert rc.get_last_shadow_chunk_ids() == ["doc-b:2:0:b", "doc-a:0:0:a"]
                return [item.chunk_id for item in ranked]

        ranked_ids = asyncio.run(_run())

    assert ranked_ids == ["doc-b:2:0:b", "doc-a:0:0:a"]


def test_shadow_benchmark_loader_and_evaluator(tmp_path) -> None:
    raw_results_path = tmp_path / "raw_results.json"
    reviewed_path = tmp_path / "reviewed.json"
    raw_results_path.write_text(
        json.dumps(
            [
                {
                    "case": {
                        "case_id": "q1",
                        "question": "Who were the claimants in case CFI 010/2024?",
                        "answer_type": "names",
                    },
                    "telemetry": {
                        "rerank_ms": 17,
                        "retrieved_chunk_ids": ["doc-a:0:0:a", "doc-b:2:0:b"],
                        "context_page_ids": ["doc-a_1"],
                        "used_page_ids": ["doc-a_1"],
                        "chunk_snippets": {
                            "doc-a:0:0:a": "Claimants: Fursa Consulting.",
                            "doc-b:2:0:b": "Article 8 text.",
                        },
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    reviewed_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "Who were the claimants in case CFI 010/2024?",
                    "answer_type": "names",
                    "golden_answer": "Fursa Consulting",
                    "golden_page_ids": ["doc-a_1"],
                    "confidence": "high",
                    "label_status": "correct",
                    "audit_note": "",
                    "current_label_problem": "",
                    "trust_tier": "high",
                    "label_weight": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    script_path = Path("/Users/sasha/IdeaProjects/personal_projects/rag_challenge/scripts/evaluate_selective_icr_rerank.py")
    spec = importlib.util.spec_from_file_location("evaluate_selective_icr_rerank", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    evaluate_shadow_cases = module.evaluate_shadow_cases
    load_shadow_cases = module.load_shadow_cases

    cases = load_shadow_cases(raw_results_path, reviewed_path)
    summary, results = evaluate_shadow_cases(cases, reranker=SelectiveICRReranker(), slice_name="reviewed_all_100")

    assert len(cases) == 1
    assert summary.case_count == 1
    assert summary.shadow_hit_rate == 1.0
    assert results[0].shadow_page_ids == ["doc-a_1"]
