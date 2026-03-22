from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shafi.config import get_settings
from shafi.core.classifier import QueryClassifier
from shafi.core.embedding import EmbeddingClient
from shafi.core.pipeline import RAGPipelineBuilder
from shafi.core.qdrant import QdrantStore
from shafi.core.reranker import RerankerClient
from shafi.core.retriever import HybridRetriever
from shafi.core.verifier import AnswerVerifier
from shafi.llm.generator import RAGGenerator
from shafi.llm.provider import LLMProvider
from shafi.prompts.loader import load_prompt

logger = logging.getLogger(__name__)

_WARMUP_SYSTEM_PROMPT = load_prompt("app/warmup_system")
_WARMUP_USER_PROMPT = load_prompt("app/warmup_user")


class AppState:
    """Shared application state created during lifespan."""

    pipeline_builder: RAGPipelineBuilder
    pipeline: Any
    embedder: EmbeddingClient
    store: QdrantStore
    reranker: RerankerClient
    llm: LLMProvider
    verifier: AnswerVerifier | None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    settings = get_settings()
    logger.info("Starting application")

    app_state.embedder = EmbeddingClient()
    app_state.store = QdrantStore()
    app_state.reranker = RerankerClient()
    app_state.llm = LLMProvider()
    app_state.verifier = AnswerVerifier(app_state.llm) if settings.verifier.enabled else None

    retriever = HybridRetriever(store=app_state.store, embedder=app_state.embedder)
    generator = RAGGenerator(llm=app_state.llm)
    classifier = QueryClassifier()

    app_state.pipeline_builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=app_state.reranker,
        generator=generator,
        classifier=classifier,
        verifier=app_state.verifier,
    )
    app_state.pipeline = app_state.pipeline_builder.compile()

    # Build DIFC appeal-chain map (SCT→CFI→CA) for appeal-query doc_ref expansion.
    # Non-blocking: failure is logged and gracefully skipped.
    await app_state.pipeline_builder.build_appeal_chain()

    logger.info("Application ready")
    if bool(getattr(settings.app, "warmup_enabled", False)):
        timeout_s = float(getattr(settings.app, "warmup_timeout_s", 5.0))
        try:
            await asyncio.wait_for(app_state.store.health_check(), timeout=timeout_s)
        except Exception:
            logger.warning("Warmup failed: qdrant", exc_info=True)
        if bool(getattr(settings.app, "warmup_embed", False)):
            try:
                await asyncio.wait_for(app_state.embedder.embed_query("warmup"), timeout=timeout_s)
            except Exception:
                logger.warning("Warmup failed: embedding", exc_info=True)
            try:
                await asyncio.wait_for(
                    retriever.retrieve(
                        "definitions",
                        prefetch_dense=1,
                        prefetch_sparse=1,
                        top_k=1,
                    ),
                    timeout=timeout_s,
                )
            except Exception:
                logger.warning("Warmup failed: retrieval", exc_info=True)
            try:
                await asyncio.wait_for(
                    retriever.retrieve(
                        "Companies Law 2018",
                        query_vector=None,
                        doc_refs=["Companies Law 2018"],
                        sparse_only=True,
                        top_k=1,
                    ),
                    timeout=timeout_s,
                )
            except Exception:
                logger.warning("Warmup failed: sparse doc-ref retrieval", exc_info=True)
        if bool(getattr(settings.app, "warmup_llm", False)):
            try:
                await asyncio.wait_for(
                    app_state.llm.generate(
                        system_prompt=_WARMUP_SYSTEM_PROMPT,
                        user_prompt=_WARMUP_USER_PROMPT,
                        model=settings.llm.simple_model,
                        max_tokens=5,
                        temperature=0.0,
                    ),
                    timeout=timeout_s,
                )
            except Exception:
                logger.warning("Warmup failed: llm", exc_info=True)
    try:
        yield
    finally:
        await app_state.embedder.close()
        await app_state.store.close()
        await app_state.reranker.close()
        await app_state.llm.close()
        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal RAG Challenge 2026",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from shafi.api.routes import router

    app.include_router(router)
    return app
