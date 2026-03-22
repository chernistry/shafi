from __future__ import annotations

from typing import TypedDict

from shafi.core.applicability_engine import TemporalResult  # noqa: TC001
from shafi.core.claim_graph import ClaimGraph  # noqa: TC001
from shafi.core.compare_engine import CompareResult  # noqa: TC001
from shafi.core.field_lookup import FieldAnswer  # noqa: TC001
from shafi.core.proof_answerer import ProofAnswer  # noqa: TC001
from shafi.core.query_contract import QueryContract  # noqa: TC001
from shafi.models import Citation, QueryComplexity, RankedChunk, RetrievedChunk, TelemetryPayload  # noqa: TC001
from shafi.telemetry import TelemetryCollector  # noqa: TC001


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
    query_contract: QueryContract
    db_answer: FieldAnswer
    compare_result: CompareResult
    temporal_result: TemporalResult
    sub_queries: list[str]
    claim_graph: ClaimGraph
    proof_answer: ProofAnswer

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

    # Fast-path flags
    noinfo_fastpath: bool

    # Retry
    retried: bool

    # Telemetry
    collector: TelemetryCollector
    telemetry: TelemetryPayload
