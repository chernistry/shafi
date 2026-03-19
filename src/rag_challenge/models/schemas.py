from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# -- Enums --


class DocType(StrEnum):
    STATUTE = "statute"
    CASE_LAW = "case_law"
    CONTRACT = "contract"
    REGULATION = "regulation"
    OTHER = "other"


class QueryComplexity(StrEnum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class SSEEventType(StrEnum):
    TOKEN = "token"
    ANSWER_FINAL = "answer_final"
    TELEMETRY = "telemetry"
    ERROR = "error"
    DONE = "done"


class PageRole(StrEnum):
    """Semantic role of a page within a legal document."""

    TITLE_COVER = "title_cover"
    CAPTION = "caption"
    ISSUED_BY_BLOCK = "issued_by_block"
    ARTICLE_CLAUSE = "article_clause"
    OPERATIVE_ORDER = "operative_order"
    COSTS_BLOCK = "costs_block"
    SCHEDULE_TABLE = "schedule_table"
    COMMENCEMENT = "commencement"
    ADMINISTRATION = "administration"
    REASONS = "reasons"
    OTHER = "other"


class ScopeMode(StrEnum):
    """Query scope classification for grounding evidence selection."""

    SINGLE_FIELD_SINGLE_DOC = "single_field_single_doc"
    EXPLICIT_PAGE = "explicit_page"
    COMPARE_PAIR = "compare_pair"
    FULL_CASE_FILES = "full_case_files"
    NEGATIVE_UNANSWERABLE = "negative_unanswerable"
    BROAD_FREE_TEXT = "broad_free_text"


# -- Domain Objects --


class ChunkMetadata(BaseModel):
    """Payload stored in Qdrant alongside vectors."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: DocType
    jurisdiction: str = ""
    section_path: str = ""
    citations: list[str] = Field(default_factory=list)
    anchors: list[str] = Field(default_factory=list)
    ingest_version: str = ""
    chunk_text: str = ""
    doc_summary: str = ""
    chunk_type: str = ""
    doc_family: str = ""
    page_family: str = ""
    normalized_title: str = ""
    normalized_refs: list[str] = Field(default_factory=list)
    amount_roles: list[str] = Field(default_factory=list)
    shadow_search_text: str = ""
    party_names: list[str] = Field(default_factory=list)
    court_names: list[str] = Field(default_factory=list)
    law_titles: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    case_numbers: list[str] = Field(default_factory=list)
    cross_refs: list[str] = Field(default_factory=list)


class PageMetadata(BaseModel):
    """Payload stored in Qdrant for page-level collection."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    doc_id: str
    page_num: int
    doc_title: str
    doc_type: DocType
    jurisdiction: str = ""
    section_path: str = ""
    ingest_version: str = ""
    page_text: str = ""
    doc_summary: str = ""
    page_family: str = ""
    doc_family: str = ""
    normalized_refs: list[str] = Field(default_factory=list)
    law_titles: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    case_numbers: list[str] = Field(default_factory=list)
    page_role: str = ""
    support_search_text: str = ""
    amount_roles: list[str] = Field(default_factory=list)
    linked_refs: list[str] = Field(default_factory=list)


class RetrievedPage(BaseModel):
    """A page returned from Qdrant page-level hybrid search."""

    page_id: str
    doc_id: str
    page_num: int
    doc_title: str = ""
    doc_type: str = ""
    page_text: str = ""
    score: float = 0.0
    page_family: str = ""
    doc_family: str = ""
    normalized_refs: list[str] = Field(default_factory=list)
    law_titles: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    case_numbers: list[str] = Field(default_factory=list)
    page_role: str = ""
    amount_roles: list[str] = Field(default_factory=list)
    linked_refs: list[str] = Field(default_factory=list)


class SupportFact(BaseModel):
    """A grounding-native fact extracted from a page at ingest time."""

    model_config = ConfigDict(frozen=True)

    fact_id: str
    doc_id: str
    page_id: str
    page_num: int
    doc_title: str
    doc_type: DocType
    doc_family: str = ""
    page_family: str = ""
    page_role: str = ""
    fact_type: str
    normalized_value: str = ""
    quote_text: str = ""
    field_explicitness: float = 1.0
    scope_ref: str = ""
    search_text: str = ""


class SupportFactMetadata(BaseModel):
    """Payload stored in Qdrant alongside support-fact vectors."""

    model_config = ConfigDict(frozen=True)

    fact_id: str
    doc_id: str
    page_id: str
    page_num: int
    doc_title: str
    doc_type: DocType
    doc_family: str = ""
    page_family: str = ""
    page_role: str = ""
    fact_type: str
    normalized_value: str = ""
    quote_text: str = ""
    field_explicitness: float = 1.0
    scope_ref: str = ""
    search_text: str = ""


class RetrievedSupportFact(BaseModel):
    """A support fact returned from Qdrant support-fact search."""

    fact_id: str
    doc_id: str
    page_id: str
    page_num: int
    doc_title: str = ""
    fact_type: str = ""
    normalized_value: str = ""
    quote_text: str = ""
    page_role: str = ""
    page_family: str = ""
    score: float = 0.0


class QueryScopePrediction(BaseModel):
    """Output of query scope classifier for grounding evidence selection."""

    model_config = ConfigDict(frozen=True)

    scope_mode: ScopeMode
    target_page_roles: list[str] = Field(default_factory=list)
    page_budget: int = 1
    requires_all_docs_in_case: bool = False
    hard_anchor_strings: list[str] = Field(default_factory=list)
    should_force_empty_grounding_on_null: bool = False


class RetrievedChunk(BaseModel):
    """A chunk returned from Qdrant hybrid search (pre-rerank)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: DocType
    section_path: str = ""
    text: str
    score: float
    doc_summary: str = ""
    page_family: str = ""
    doc_family: str = ""
    chunk_type: str = ""
    amount_roles: list[str] = Field(default_factory=list)
    normalized_refs: list[str] = Field(default_factory=list)
    shadow_search_text: str = ""
    party_names: list[str] = Field(default_factory=list)
    court_names: list[str] = Field(default_factory=list)
    law_titles: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    case_numbers: list[str] = Field(default_factory=list)
    cross_refs: list[str] = Field(default_factory=list)
    retrieval_sources: list[str] = Field(default_factory=list)


class RankedChunk(BaseModel):
    """A chunk after reranking with Zerank 2 / Cohere."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: DocType
    section_path: str = ""
    text: str
    retrieval_score: float
    rerank_score: float
    doc_summary: str = ""
    page_family: str = ""
    doc_family: str = ""
    chunk_type: str = ""
    amount_roles: list[str] = Field(default_factory=list)
    normalized_refs: list[str] = Field(default_factory=list)


# -- API Request / Response --


class QueryRequest(BaseModel):
    """POST /query request body."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1, max_length=4000)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_id: str = ""
    answer_type: str = "free_text"


class Citation(BaseModel):
    """A grounded citation linking answer text to a source chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_title: str
    section_path: str | None = None
    quote: str | None = None


class TelemetryPayload(BaseModel):
    """Mandatory telemetry included in every response."""

    request_id: str
    question_id: str = ""
    answer_type: str = "free_text"
    ttft_ms: int
    time_per_output_token_ms: int = 0
    total_ms: int
    embed_ms: int
    qdrant_ms: int
    rerank_ms: int
    llm_ms: int
    verify_ms: int = 0
    classify_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    chunk_snippets: dict[str, str] = Field(default_factory=dict)
    retrieved_page_ids: list[str] = Field(default_factory=list)
    context_page_ids: list[str] = Field(default_factory=list)
    cited_page_ids: list[str] = Field(default_factory=list)
    used_page_ids: list[str] = Field(default_factory=list)
    doc_refs: list[str] = Field(default_factory=list)
    model_embed: str = ""
    model_rerank: str = ""
    model_llm: str = ""
    llm_provider: str = ""
    llm_finish_reason: str = ""
    generation_mode: str = ""
    context_chunk_count: int = 0
    context_budget_tokens: int = 0
    malformed_tail_detected: bool = False
    retried: bool = False
    model_upgraded: bool = False
    trained_page_scorer_used: bool = False
    trained_page_scorer_model_path: str = ""
    trained_page_scorer_page_ids: list[str] = Field(default_factory=list)
    trained_page_scorer_fallback_reason: str = ""


class RAGResponse(BaseModel):
    """Full (non-streaming) response model — used for testing/logging."""

    answer: str | None
    citations: list[Citation]
    telemetry: TelemetryPayload


class SSEEvent(BaseModel):
    """Individual SSE event sent during streaming."""

    event: SSEEventType
    data: str  # JSON-encoded payload


# -- Ingestion Models --


class DocumentSection(BaseModel):
    """A structural section within a parsed document."""

    heading: str = ""
    section_path: str = ""
    text: str = ""
    level: int = 0


class ProvidedChunk(BaseModel):
    """A chunk provided by the evaluator/dataset (IDs must be preserved).

    Use this when the competition starter kit provides pre-chunked inputs with stable `chunk_id`s
    that are used for grounding verification (gold-chunk matching).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    section_path: str = ""
    citations: list[str] = Field(default_factory=list)
    anchors: list[str] = Field(default_factory=list)


def _document_sections_factory() -> list[DocumentSection]:
    return []


def _provided_chunks_factory() -> list[ProvidedChunk]:
    return []


class ParsedDocument(BaseModel):
    """Output of document parser — one per source file."""

    doc_id: str
    title: str
    doc_type: DocType
    jurisdiction: str = ""
    source_path: str = ""
    full_text: str = ""
    sections: list[DocumentSection] = Field(default_factory=_document_sections_factory)
    provided_chunks: list[ProvidedChunk] = Field(default_factory=_provided_chunks_factory)
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk ready for embedding and indexing."""

    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: DocType
    jurisdiction: str = ""
    section_path: str = ""
    chunk_text: str
    chunk_text_for_embedding: str  # SAC-augmented version
    doc_summary: str = ""
    citations: list[str] = Field(default_factory=list)
    anchors: list[str] = Field(default_factory=list)
    token_count: int = 0
    chunk_type: str = ""
    doc_family: str = ""
    page_family: str = ""
    normalized_title: str = ""
    normalized_refs: list[str] = Field(default_factory=list)
    amount_roles: list[str] = Field(default_factory=list)
    shadow_search_text: str = ""
    party_names: list[str] = Field(default_factory=list)
    court_names: list[str] = Field(default_factory=list)
    law_titles: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    case_numbers: list[str] = Field(default_factory=list)
    cross_refs: list[str] = Field(default_factory=list)


# -- LangGraph State --


class PipelineTimings(BaseModel):
    """Per-stage timing accumulator."""

    embed_ms: float = 0.0
    qdrant_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    classify_ms: float = 0.0
    verify_ms: float = 0.0
    total_ms: float = 0.0
