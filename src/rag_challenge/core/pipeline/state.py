# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false
from __future__ import annotations

from typing import TypedDict

from rag_challenge.models import Citation, QueryComplexity, RankedChunk, RetrievedChunk, TelemetryPayload  # noqa: TC001
from rag_challenge.telemetry import TelemetryCollector  # noqa: TC001


class RAGState(TypedDict, total=False):
    # Input
    query: str
    request_id: str
    question_id: str
    answer_type: str
    doc_refs: list[str]

    # Routing
    complexity: QueryComplexity
    model: str
    max_tokens: int
    sub_queries: list[str]

    # Retrieval/rerank
    retrieved: list[RetrievedChunk]
    must_include_chunk_ids: list[str]
    reranked: list[RankedChunk]
    context_chunks: list[RankedChunk]
    max_rerank_score: float
    conflict_prompt_context: str

    # Generation
    answer: str
    citations: list[Citation]
    cited_chunk_ids: list[str]
    streamed: bool

    # Retry
    retried: bool

    # Telemetry
    collector: TelemetryCollector
    telemetry: TelemetryPayload
