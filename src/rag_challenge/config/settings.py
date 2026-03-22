from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Annotated, cast

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _settings_model_config(*, env_prefix: str = "") -> SettingsConfigDict:
    """Build the shared settings config for all settings models.

    Args:
        env_prefix: Prefix applied to environment variables for this settings
            group.

    Returns:
        Pydantic settings config with deterministic env precedence:
        process environment, then `.env.local`, then `.env`, then defaults.
    """
    return SettingsConfigDict(
        env_prefix=env_prefix,
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


class EmbeddingSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="EMBED_")

    provider_mode: str = "api"
    model: str = "kanon-2-embedder"
    api_url: str = "https://api.isaacus.com/v1/embeddings"
    api_key: SecretStr = Field(alias="ISAACUS_API_KEY")
    local_model_path: str = ""
    local_normalize_embeddings: bool = True
    dimensions: int = 1024
    batch_size: int = 128  # Isaacus API hard limit
    concurrency: int = 8
    timeout_s: float = 30.0
    connect_timeout_s: float = 10.0
    retry_attempts: int = 6
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 16.0
    retry_jitter_s: float = 2.0
    circuit_failure_threshold: int = 3
    circuit_reset_timeout_s: float = 60.0


class RerankerSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="RERANK_")

    provider_mode: str = "api"
    primary_model: str = "zerank-2"
    primary_api_url: str = "https://api.zeroentropy.dev/v1/models/rerank"
    primary_api_key: SecretStr = Field(alias="ZEROENTROPY_API_KEY")
    local_model_path: str = ""
    primary_batch_size: int = 50  # ZeroEntropy practical limit
    primary_latency_mode: str = "fast"  # "fast" or "degraded"
    primary_timeout_s: float = 30.0

    # Isaacus kanon-2-reranker (RERANK_PROVIDER_MODE=isaacus)
    isaacus_api_url: str = "https://api.isaacus.com/v1/rerankings"
    isaacus_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="ISAACUS_API_KEY")
    isaacus_model: str = "kanon-2-reranker"

    fallback_model: str = "rerank-v4.0-fast"
    fallback_api_key: SecretStr = Field(alias="COHERE_API_KEY")
    fallback_timeout_s: float = 30.0

    shadow_selective_icr_enabled: bool = False
    shadow_selective_icr_provider_exit: bool = False
    shadow_selective_icr_model_path: str = ""
    shadow_selective_icr_max_chars: int = 1800
    shadow_selective_icr_candidate_batch_size: int = 32
    shadow_selective_icr_normalize_scores: bool = True

    top_n: int = 8  # final context chunks (was 6→10→8; 10 caused 7 G regressions)
    rerank_candidates: int = 120  # from hybrid search, before rerank
    primary_max_connections: int = 20
    primary_concurrency_limit: int = 1
    primary_min_interval_s: float = 0.25
    primary_connect_timeout_s: float = 10.0
    retry_attempts: int = 4
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0
    retry_jitter_s: float = 1.0
    circuit_failure_threshold: int = 3
    circuit_reset_timeout_s: float = 60.0


class QdrantSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="QDRANT_")

    url: str = "http://localhost:6333"
    api_key: str = ""
    collection: str = "legal_chunks"
    shadow_collection: str = "legal_chunks_shadow"
    page_collection: str = "legal_pages"
    segment_collection: str = "legal_segments"
    bridge_fact_collection: str = "legal_bridge_facts"
    support_fact_collection: str = "legal_support_facts"
    pool_size: int = 20
    timeout_s: float = 30.0
    prefetch_dense: int = 120
    prefetch_sparse: int = 120
    use_cloud_inference: bool = False  # False => qdrant-client local fastembed inference
    enable_sparse_bm25: bool = True
    sparse_model: str = "Qdrant/bm25"
    fastembed_cache_dir: str = ""
    sparse_threads: int | None = None
    check_compatibility: bool = False
    fusion_method: str = (
        "RRF"  # "RRF" (robust default) or "DBSF" (preserves score magnitude; better for exact-match citations)
    )
    circuit_failure_threshold: int = 3
    circuit_reset_timeout_s: float = 60.0


class LLMSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="LLM_")

    provider: str = "openai_compatible"
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    )
    openrouter_referer: str = ""
    openrouter_title: str = ""
    anthropic_api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="ANTHROPIC_API_KEY")

    simple_model: str = "gpt-4o-mini"
    complex_model: str = "gpt-4o"
    strict_model: str = "gpt-4o-mini"  # boolean/number/name/names/date (parse-safe, low-latency)
    free_text_simple_model: str = ""  # If set, overrides simple_model for free_text SIMPLE queries; defaults to simple_model
    fallback_model: str = "claude-3-5-sonnet-latest"
    summary_model: str = "gpt-4o-mini"  # for SAC doc summaries
    upgrade_model: str = ""  # Selective upgrade for hardest free_text cases
    synthesis_model: str = "gpt-4.1"  # Stronger model for synthesis/comparison/penalty free_text

    simple_max_tokens: int = 300
    complex_max_tokens: int = 500
    upgrade_max_tokens: int = 1800
    strict_max_tokens: int = 150  # boolean/number/name/names/date
    temperature: float = 0.0
    timeout_s: float = 60.0
    connect_timeout_s: float = 10.0
    max_context_tokens: int = 2500  # cap context window
    stream_include_usage: bool = True
    http2: bool = True
    max_connections: int = 50
    max_keepalive_connections: int = 20
    openrouter_provider_order: Annotated[list[str], NoDecode] = Field(default_factory=list)
    openrouter_allow_fallbacks: bool = True
    circuit_failure_threshold: int = 3
    circuit_reset_timeout_s: float = 60.0

    # Routing thresholds
    complex_min_length: int = 150
    complex_keywords: list[str] = Field(
        default_factory=lambda: [
            "compare",
            "difference",
            "exception",
            "notwithstanding",
            "distinguish",
            "analyze",
            "evaluate",
            "contrast",
            "jurisdiction",
            "precedent",
        ]
    )
    complex_min_entities: int = 2

    def resolved_api_key(self) -> SecretStr:
        if self.api_key.get_secret_value().strip():
            return self.api_key
        return SecretStr("")

    @field_validator("openrouter_provider_order", mode="before")
    @classmethod
    def _parse_openrouter_provider_order(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            items_obj = cast("list[object]", value)
            return [str(item).strip() for item in items_obj if str(item).strip()]
        return []


class JudgeSettings(BaseSettings):
    """Optional LLM-as-judge configuration for eval harness.

    This is intentionally decoupled from the main `LLM_*` settings so we can run:
    - main pipeline on direct OpenAI (speed + stability)
    - judge on OpenRouter cheap/free models
    """

    model_config = _settings_model_config(env_prefix="JUDGE_")

    enabled: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    model: str = "google/gemini-2.5-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 900
    timeout_s: float = 60.0
    connect_timeout_s: float = 10.0
    http2: bool = True
    max_connections: int = 20
    max_keepalive_connections: int = 10
    concurrency: int = 2

    docs_dir: str = "dataset/dataset_documents"
    sources_max_pages: int = 12
    sources_max_chars_per_page: int = 20000
    sources_max_chars_total: int = 60000

    openrouter_provider_order: Annotated[list[str], NoDecode] = Field(default_factory=list)
    openrouter_allow_fallbacks: bool = True

    @field_validator("openrouter_provider_order", mode="before")
    @classmethod
    def _parse_judge_provider_order(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            items_obj = cast("list[object]", value)
            return [str(item).strip() for item in items_obj if str(item).strip()]
        return []


class PipelineSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="PIPELINE_")

    confidence_threshold: float = 0.3  # rerank score below this triggers retry
    max_retries: int = 1
    retry_sparse_bias: int = 90  # sparse prefetch on retry
    retry_dense_bias: int = 40  # dense prefetch on retry
    doc_ref_case_law_filter: bool = True
    case_ref_metadata_first: bool = True  # Skip embedding for case-ref queries; sparse+metadata only
    doc_ref_prefetch_dense: int = 20
    doc_ref_prefetch_sparse: int = 120
    doc_ref_sparse_only: bool = True
    doc_ref_multi_retrieve: bool = True
    doc_ref_multi_top_k_per_ref: int = 30
    max_answer_words: int = 250
    retry_query_max_anchors: int = 3
    premise_guard_enabled: bool = True
    premise_guard_terms: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["jury", "miranda", "parole", "plea bargain", "plea"]
    )

    reranker_enabled: bool = True  # global toggle: False = skip reranker, use raw_ranked for all types
    rerank_enabled_strict_types: bool = True
    rerank_enabled_boolean: bool = True
    rerank_max_candidates_strict_types: int = 30  # was 20; more candidates = higher recall
    boolean_rerank_candidates_cap: int = 30  # was 12; F-beta 2.5 needs more candidates for recall
    free_text_rerank_candidates_cap: int = 0  # 0 = no cap; set >0 to limit free_text reranking pool for TTFT
    strict_types_context_top_n: int = 5
    name_context_top_n: int = 5  # per-type override for name; name G=0.321 worst type
    boolean_context_top_n: int = 5  # was 4; more context for boolean reasoning
    boolean_multi_ref_top_n: int = 3
    boolean_max_tokens: int = 96
    boolean_prefetch_dense: int = 40
    boolean_prefetch_sparse: int = 40
    strict_prefetch_dense: int = 24
    strict_prefetch_sparse: int = 24
    strict_doc_ref_top_k: int = 16
    strict_multi_ref_top_k_per_ref: int = 12
    free_text_targeted_multi_ref_top_k: int = 12
    strict_types_extraction_enabled: bool = True
    strict_types_extraction_max_chunks: int = 4
    strict_types_force_simple_model: bool = True
    strict_types_append_citations: bool = False
    strict_repair_enabled: bool = True
    claim_graph_enabled: bool = False
    proof_compiler_enabled: bool = False

    # BM25 hybrid retrieval
    enable_bm25_hybrid: bool = False  # feature flag: BM25 + RRF fusion
    bm25_weight: float = 0.3  # RRF weight for BM25 (0.7 for dense)
    bm25_rrf_k: int = 60  # RRF constant (standard from literature)
    enable_bm25_alias_expand: bool = False  # expand BM25 query with law alias synonyms
    bm25_alias_map_path: str = "data/legal_aliases.json"
    enable_interleaved_citations: bool = False  # force per-sentence citation in complex/irac prompts
    enable_step_back: bool = False  # step-back query abstraction before retrieval
    proof_min_coverage: float = 0.5
    proof_allow_partial_answers: bool = True
    proof_fluency_pass_enabled: bool = True
    context_token_budget_boolean: int = 900
    context_token_budget_strict: int = 1100
    context_token_budget_named_multi_lookup: int = 1600
    context_token_budget_free_text_simple: int = 1800
    context_token_budget_free_text_complex: int = 2400
    rerank_skip_on_single_doc_ref: bool = False
    rerank_doc_max_chars: int = 2500
    use_fast_reranker_for_simple: bool = False
    enable_conflict_detection: bool = Field(
        default=False,
        validation_alias=AliasChoices("PIPELINE_ENABLE_CONFLICT_DETECTION", "PIPELINE_CONFLICT_DETECTION_ENABLED"),
    )
    conflict_max_chunks: int = 8
    enable_multi_hop: bool = Field(
        default=False,
        validation_alias=AliasChoices("PIPELINE_ENABLE_MULTI_HOP", "PIPELINE_DECOMPOSE_ENABLED"),
    )
    multi_hop_max_subqueries: int = 3
    page_first_enabled: bool = False
    page_first_top_k: int = 15
    enable_shadow_search_text: bool = False
    enable_parallel_anchor_retrieval: bool = False
    enable_entity_boosts: bool = False
    canonical_entity_registry_path: str = ""
    db_answerer_enabled: bool = False
    db_answerer_confidence_threshold: float = 0.85
    known_noinfo_qids_path: str = "data/false_unanswerables.json"
    compare_engine_enabled: bool = False
    temporal_engine_enabled: bool = False
    enable_cross_ref_boosts: bool = True
    enable_segment_retrieval: bool = True
    segment_retrieval_top_k: int = 6
    segment_retrieval_page_budget: int = 4
    enable_bridge_fact_retrieval: bool = False
    bridge_fact_retrieval_top_k: int = 6
    bridge_fact_page_budget: int = 4
    enable_doc_diversity_expansion: bool = True
    doc_diversity_min_unique_docs: int = 8
    doc_diversity_expansion_limit: int = 40
    shadow_retrieval_top_k: int = 24
    anchor_retrieval_top_k: int = 16
    enable_grounding_sidecar: bool = True
    enable_trained_page_scorer: bool = True
    trained_page_scorer_model_path: str = "models/page_scorer/v6_version_full/page_scorer.joblib"
    grounding_support_fact_top_k: int = 32
    grounding_page_top_k: int = 24
    grounding_page_budget_default: int = 2
    grounding_allow_same_doc_hop: bool = True
    grounding_escalation_enabled: bool = True
    grounding_low_rerank_margin_threshold: float = 0.06
    grounding_close_page_margin_threshold: float = 0.2
    grounding_authority_strength_threshold: float = 0.95
    grounding_relevance_verifier_enabled: bool = False
    grounding_relevance_verifier_min_confidence: float = 0.7
    grounding_relevance_verifier_max_candidates: int = 3
    enable_retrieval_escalation: bool = False
    retrieval_escalation_threshold: float = 0.5
    enable_answer_consensus: bool = False
    enable_answer_validation: bool = True
    enable_rag_fusion: bool = False
    rag_fusion_extra_prefetch: int = 40  # dense prefetch limit per variant query
    sparse_fallback_threshold: int = 3  # trigger dense fallback if sparse returns fewer than this

    # HyDE (Hypothetical Document Embeddings) — query expansion for improved recall.
    # Generates hypothetical answer, embeds it, merges retrieval results.
    # Cost: ~150-250ms TTFT overhead (runs parallel with original query embed).
    # Expected: +3-8pp G on multi-hop and complex free_text questions.
    enable_hyde: bool = False
    hyde_extra_prefetch: int = 40  # dense prefetch limit for HyDE hypothetical document

    # Doc-title match boost — boosts chunks from the target document when doc_refs
    # are extracted. Prevents cross-referencing documents from outranking the actual
    # target. Expected: +2-5pp on questions where target doc is retrieved but
    # outranked by docs that merely reference it in their citations.
    enable_doc_title_boost: bool = False
    doc_title_boost_value: float = 0.15

    # Retrieval deduplication — merges chunks from duplicate documents.
    # When multiple doc_ids share the same normalized title, keeps only chunks
    # from the doc_id with the highest total score. Zero LLM cost, O(n) on chunks.
    # Fixes: 13% of collection is exact duplicates (1,336 wasted chunks).
    # Expected: +1-3pp on questions where duplicates crowd out correct context.
    dedup_duplicate_docs: bool = False
    # Minimum title length for dedup (skip very short/generic titles like "." or "1")
    dedup_min_title_length: int = 5

    # Citation graph hop — expands candidates using external_citations enrichment data.
    # Zero LLM cost, zero latency overhead. Requires NOAM enrichment data.
    # Expected: +4-9pp G on multi-hop questions.
    enable_citation_hop: bool = False

    # Post-hoc citation verification — verifies each cited page actually supports the answer.
    # Runs AFTER answer streaming (zero TTFT impact). Uses lightweight LLM for YES/NO per page.
    # Safety guard: never drops all citations. VeriCite (SIGIR-AP 2025): +6-8pp Citation F1.
    enable_citation_verification: bool = False
    citation_verification_model: str = "gpt-4.1-mini"

    # Isaacus Extractive QA — bypasses LLM for name/number/date if extractable.
    # Requires ISAACUS_API_KEY. Cost: ~$1.50/M tokens (very cheap vs LLM).
    # Expected: +5-10 Det by eliminating LLM hallucination on exact-match types.
    enable_extractive_qa: bool = False
    extractive_qa_url: str = "https://api.isaacus.com/v1/extractions/qa"
    extractive_qa_model: str = "kanon-answer-extractor"
    extractive_qa_inextractability_threshold: float = 0.5
    extractive_qa_context_top_n: int = 3  # how many top context chunks to send

    # Predicted Outputs — OpenAI speculative decoding for deterministic types.
    # Predicts output skeleton for boolean/number/date/name to reduce generation time.
    # Supported on GPT-4.1 / 4.1-mini. Minimal overhead on wrong prediction.
    enable_predicted_outputs: bool = False

    @field_validator("premise_guard_terms", mode="before")
    @classmethod
    def _parse_premise_guard_terms(cls, value: object) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            items_obj = cast("list[object]", value)
            items = [str(item).strip() for item in items_obj if str(item).strip()]
            return items
        return ["jury", "miranda", "parole", "plea bargain", "plea"]


class IngestionSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="INGEST_")

    chunk_size_tokens: int = 450
    chunk_overlap_tokens: int = 45  # ~10% overlap
    sac_summary_max_tokens: int = 150
    sac_summary_min_tokens: int = 80
    upsert_batch_size: int = 200
    ingest_version: str = "v2"
    sac_concurrency: int = 4
    sac_doc_excerpt_chars: int = 3000
    parser_pdf_text_min_chars: int = 400
    parser_pdf_text_min_words: int = 80
    build_shadow_collection: bool = True
    build_corpus_registry: bool = False
    corpus_registry_path: str = ""
    build_canonical_entity_registry: bool = False
    canonical_entity_registry_path: str = ""
    build_segment_collection: bool = False
    build_bridge_fact_collection: bool = False
    manifest_hash_chunk_size_bytes: int = 1048576
    manifest_dir: str = ""  # optional absolute/relative override for manifest storage
    manifest_filename: str = ".rag_challenge_ingestion_manifest.json"
    manifest_schema_version: int = 1


class VerifierSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="VERIFY_")

    enabled: bool = True
    max_tokens: int = 500
    temperature: float = 0.0


class AppSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="APP_")

    log_level: str = "INFO"
    log_format: str = "json"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    warmup_enabled: bool = False
    warmup_embed: bool = False
    warmup_llm: bool = False
    warmup_timeout_s: float = 5.0


class PlatformSettings(BaseSettings):
    model_config = _settings_model_config(env_prefix="EVAL_")

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), alias="EVAL_API_KEY")
    base_url: str = Field(
        default="https://platform.agentic-challenge.ai/api/v1",
        alias="EVAL_BASE_URL",
    )
    phase: str = "warmup"
    work_dir: str = "platform_runs"
    documents_dirname: str = "documents"
    questions_filename: str = "questions.json"
    submission_filename: str = "submission.json"
    code_archive_filename: str = "code_archive.zip"
    archive_allowlist_path: str = "src/rag_challenge/submission/archive_allowlist.json"
    collection_prefix: str = "legal_chunks_platform"
    query_concurrency: int = 1
    poll_interval_s: float = 10.0
    poll_timeout_s: float = 1800.0
    architecture_summary: str = (
        "Async legal RAG with OCR-aware ingestion, clause-aware chunking, "
        "hybrid dense+BM25 retrieval in Qdrant, reranking, answer-type routing, "
        "and page-level grounded telemetry over phase-isolated corpora."
    )

    @field_validator("phase")
    @classmethod
    def _validate_phase(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"warmup", "final"}:
            raise ValueError("EVAL_PHASE must be one of: warmup, final")
        return normalized

    @field_validator("architecture_summary")
    @classmethod
    def _validate_architecture_summary(cls, value: str) -> str:
        summary = value.strip()
        if len(summary) > 500:
            raise ValueError("EVAL_ARCHITECTURE_SUMMARY must be <= 500 characters")
        return summary


def _embedding_settings_factory() -> EmbeddingSettings:
    return EmbeddingSettings()  # pyright: ignore[reportCallIssue]


def _reranker_settings_factory() -> RerankerSettings:
    return RerankerSettings()  # pyright: ignore[reportCallIssue]


def _qdrant_settings_factory() -> QdrantSettings:
    return QdrantSettings()


def _llm_settings_factory() -> LLMSettings:
    return LLMSettings()  # pyright: ignore[reportCallIssue]


def _judge_settings_factory() -> JudgeSettings:
    return JudgeSettings()  # pyright: ignore[reportCallIssue]


def _pipeline_settings_factory() -> PipelineSettings:
    return PipelineSettings()


def _ingestion_settings_factory() -> IngestionSettings:
    return IngestionSettings()


def _app_settings_factory() -> AppSettings:
    return AppSettings()


def _platform_settings_factory() -> PlatformSettings:
    return PlatformSettings()  # pyright: ignore[reportCallIssue]


def _verifier_settings_factory() -> VerifierSettings:
    return VerifierSettings()


class Settings(BaseSettings):
    """Root settings aggregating all sub-settings."""

    model_config = _settings_model_config()

    embedding: EmbeddingSettings = Field(default_factory=_embedding_settings_factory)
    reranker: RerankerSettings = Field(default_factory=_reranker_settings_factory)
    qdrant: QdrantSettings = Field(default_factory=_qdrant_settings_factory)
    llm: LLMSettings = Field(default_factory=_llm_settings_factory)
    judge: JudgeSettings = Field(default_factory=_judge_settings_factory)
    pipeline: PipelineSettings = Field(default_factory=_pipeline_settings_factory)
    ingestion: IngestionSettings = Field(default_factory=_ingestion_settings_factory)
    verifier: VerifierSettings = Field(default_factory=_verifier_settings_factory)
    app: AppSettings = Field(default_factory=_app_settings_factory)
    platform: PlatformSettings = Field(default_factory=_platform_settings_factory)


def build_score_settings_snapshot(settings: Settings | None = None) -> dict[str, object]:
    """Build a deterministic snapshot of score-affecting runtime settings.

    Args:
        settings: Optional preloaded settings instance.

    Returns:
        dict[str, object]: JSON-serializable snapshot suitable for lineage and
        artifact fingerprints.
    """

    resolved = settings or get_settings()
    return {
        "platform": {
            "phase": resolved.platform.phase,
            "collection_prefix": resolved.platform.collection_prefix,
        },
        "llm": {
            "provider": resolved.llm.provider,
            "base_url": resolved.llm.base_url,
            "simple_model": resolved.llm.simple_model,
            "complex_model": resolved.llm.complex_model,
            "strict_model": resolved.llm.strict_model,
            "free_text_simple_model": resolved.llm.free_text_simple_model,
            "fallback_model": resolved.llm.fallback_model,
            "summary_model": resolved.llm.summary_model,
            "synthesis_model": resolved.llm.synthesis_model,
            "temperature": resolved.llm.temperature,
            "max_context_tokens": resolved.llm.max_context_tokens,
            "complex_min_length": resolved.llm.complex_min_length,
            "complex_min_entities": resolved.llm.complex_min_entities,
        },
        "reranker": {
            "primary_model": resolved.reranker.primary_model,
            "fallback_model": resolved.reranker.fallback_model,
            "top_n": resolved.reranker.top_n,
            "rerank_candidates": resolved.reranker.rerank_candidates,
            "primary_latency_mode": resolved.reranker.primary_latency_mode,
            "primary_concurrency_limit": resolved.reranker.primary_concurrency_limit,
            "primary_min_interval_s": resolved.reranker.primary_min_interval_s,
            "shadow_selective_icr_enabled": resolved.reranker.shadow_selective_icr_enabled,
            "shadow_selective_icr_provider_exit": resolved.reranker.shadow_selective_icr_provider_exit,
            "shadow_selective_icr_model_path": resolved.reranker.shadow_selective_icr_model_path,
            "shadow_selective_icr_max_chars": resolved.reranker.shadow_selective_icr_max_chars,
            "shadow_selective_icr_candidate_batch_size": resolved.reranker.shadow_selective_icr_candidate_batch_size,
            "shadow_selective_icr_normalize_scores": resolved.reranker.shadow_selective_icr_normalize_scores,
        },
        "qdrant": {
            "url": resolved.qdrant.url,
            "collection": resolved.qdrant.collection,
            "shadow_collection": resolved.qdrant.shadow_collection,
            "page_collection": resolved.qdrant.page_collection,
            "segment_collection": resolved.qdrant.segment_collection,
            "bridge_fact_collection": resolved.qdrant.bridge_fact_collection,
            "support_fact_collection": resolved.qdrant.support_fact_collection,
            "prefetch_dense": resolved.qdrant.prefetch_dense,
            "prefetch_sparse": resolved.qdrant.prefetch_sparse,
            "enable_sparse_bm25": resolved.qdrant.enable_sparse_bm25,
            "fusion_method": resolved.qdrant.fusion_method,
        },
        "pipeline": {
            "confidence_threshold": resolved.pipeline.confidence_threshold,
            "doc_ref_case_law_filter": resolved.pipeline.doc_ref_case_law_filter,
            "doc_ref_prefetch_dense": resolved.pipeline.doc_ref_prefetch_dense,
            "doc_ref_prefetch_sparse": resolved.pipeline.doc_ref_prefetch_sparse,
            "doc_ref_sparse_only": resolved.pipeline.doc_ref_sparse_only,
            "doc_ref_multi_retrieve": resolved.pipeline.doc_ref_multi_retrieve,
            "doc_ref_multi_top_k_per_ref": resolved.pipeline.doc_ref_multi_top_k_per_ref,
            "max_answer_words": resolved.pipeline.max_answer_words,
            "premise_guard_enabled": resolved.pipeline.premise_guard_enabled,
            "premise_guard_terms": resolved.pipeline.premise_guard_terms,
            "rerank_enabled_strict_types": resolved.pipeline.rerank_enabled_strict_types,
            "rerank_enabled_boolean": resolved.pipeline.rerank_enabled_boolean,
            "rerank_max_candidates_strict_types": resolved.pipeline.rerank_max_candidates_strict_types,
            "boolean_rerank_candidates_cap": resolved.pipeline.boolean_rerank_candidates_cap,
            "strict_types_context_top_n": resolved.pipeline.strict_types_context_top_n,
            "boolean_context_top_n": resolved.pipeline.boolean_context_top_n,
            "boolean_multi_ref_top_n": resolved.pipeline.boolean_multi_ref_top_n,
            "strict_types_extraction_enabled": resolved.pipeline.strict_types_extraction_enabled,
            "strict_types_extraction_max_chunks": resolved.pipeline.strict_types_extraction_max_chunks,
            "strict_types_force_simple_model": resolved.pipeline.strict_types_force_simple_model,
            "strict_types_append_citations": resolved.pipeline.strict_types_append_citations,
            "strict_repair_enabled": resolved.pipeline.strict_repair_enabled,
            "claim_graph_enabled": resolved.pipeline.claim_graph_enabled,
            "proof_compiler_enabled": resolved.pipeline.proof_compiler_enabled,
            "proof_min_coverage": resolved.pipeline.proof_min_coverage,
            "proof_allow_partial_answers": resolved.pipeline.proof_allow_partial_answers,
            "proof_fluency_pass_enabled": resolved.pipeline.proof_fluency_pass_enabled,
            "context_token_budget_boolean": resolved.pipeline.context_token_budget_boolean,
            "context_token_budget_strict": resolved.pipeline.context_token_budget_strict,
            "context_token_budget_named_multi_lookup": resolved.pipeline.context_token_budget_named_multi_lookup,
            "context_token_budget_free_text_simple": resolved.pipeline.context_token_budget_free_text_simple,
            "context_token_budget_free_text_complex": resolved.pipeline.context_token_budget_free_text_complex,
            "rerank_skip_on_single_doc_ref": resolved.pipeline.rerank_skip_on_single_doc_ref,
            "rerank_doc_max_chars": resolved.pipeline.rerank_doc_max_chars,
            "use_fast_reranker_for_simple": resolved.pipeline.use_fast_reranker_for_simple,
            "enable_conflict_detection": resolved.pipeline.enable_conflict_detection,
            "conflict_max_chunks": resolved.pipeline.conflict_max_chunks,
            "enable_multi_hop": resolved.pipeline.enable_multi_hop,
            "multi_hop_max_subqueries": resolved.pipeline.multi_hop_max_subqueries,
            "page_first_enabled": resolved.pipeline.page_first_enabled,
            "page_first_top_k": resolved.pipeline.page_first_top_k,
            "enable_shadow_search_text": resolved.pipeline.enable_shadow_search_text,
            "enable_parallel_anchor_retrieval": resolved.pipeline.enable_parallel_anchor_retrieval,
            "enable_entity_boosts": resolved.pipeline.enable_entity_boosts,
            "canonical_entity_registry_path": resolved.pipeline.canonical_entity_registry_path,
            "db_answerer_enabled": resolved.pipeline.db_answerer_enabled,
            "db_answerer_confidence_threshold": resolved.pipeline.db_answerer_confidence_threshold,
            "enable_cross_ref_boosts": resolved.pipeline.enable_cross_ref_boosts,
            "enable_segment_retrieval": resolved.pipeline.enable_segment_retrieval,
            "segment_retrieval_top_k": resolved.pipeline.segment_retrieval_top_k,
            "segment_retrieval_page_budget": resolved.pipeline.segment_retrieval_page_budget,
            "enable_bridge_fact_retrieval": resolved.pipeline.enable_bridge_fact_retrieval,
            "bridge_fact_retrieval_top_k": resolved.pipeline.bridge_fact_retrieval_top_k,
            "bridge_fact_page_budget": resolved.pipeline.bridge_fact_page_budget,
            "enable_doc_diversity_expansion": resolved.pipeline.enable_doc_diversity_expansion,
            "doc_diversity_min_unique_docs": resolved.pipeline.doc_diversity_min_unique_docs,
            "doc_diversity_expansion_limit": resolved.pipeline.doc_diversity_expansion_limit,
            "shadow_retrieval_top_k": resolved.pipeline.shadow_retrieval_top_k,
            "anchor_retrieval_top_k": resolved.pipeline.anchor_retrieval_top_k,
            "enable_grounding_sidecar": resolved.pipeline.enable_grounding_sidecar,
            "enable_trained_page_scorer": resolved.pipeline.enable_trained_page_scorer,
            "grounding_support_fact_top_k": resolved.pipeline.grounding_support_fact_top_k,
            "grounding_page_top_k": resolved.pipeline.grounding_page_top_k,
            "grounding_page_budget_default": resolved.pipeline.grounding_page_budget_default,
            "grounding_allow_same_doc_hop": resolved.pipeline.grounding_allow_same_doc_hop,
            "grounding_escalation_enabled": resolved.pipeline.grounding_escalation_enabled,
            "grounding_low_rerank_margin_threshold": resolved.pipeline.grounding_low_rerank_margin_threshold,
            "grounding_close_page_margin_threshold": resolved.pipeline.grounding_close_page_margin_threshold,
            "grounding_authority_strength_threshold": resolved.pipeline.grounding_authority_strength_threshold,
            "grounding_relevance_verifier_enabled": resolved.pipeline.grounding_relevance_verifier_enabled,
            "grounding_relevance_verifier_min_confidence": resolved.pipeline.grounding_relevance_verifier_min_confidence,
            "grounding_relevance_verifier_max_candidates": resolved.pipeline.grounding_relevance_verifier_max_candidates,
        },
        "ingestion": {
            "ingest_version": resolved.ingestion.ingest_version,
            "chunk_size_tokens": resolved.ingestion.chunk_size_tokens,
            "chunk_overlap_tokens": resolved.ingestion.chunk_overlap_tokens,
            "sac_summary_max_tokens": resolved.ingestion.sac_summary_max_tokens,
            "sac_summary_min_tokens": resolved.ingestion.sac_summary_min_tokens,
            "sac_concurrency": resolved.ingestion.sac_concurrency,
            "sac_doc_excerpt_chars": resolved.ingestion.sac_doc_excerpt_chars,
            "parser_pdf_text_min_chars": resolved.ingestion.parser_pdf_text_min_chars,
            "parser_pdf_text_min_words": resolved.ingestion.parser_pdf_text_min_words,
            "build_shadow_collection": resolved.ingestion.build_shadow_collection,
            "build_corpus_registry": resolved.ingestion.build_corpus_registry,
            "corpus_registry_path": resolved.ingestion.corpus_registry_path,
            "build_canonical_entity_registry": resolved.ingestion.build_canonical_entity_registry,
            "canonical_entity_registry_path": resolved.ingestion.canonical_entity_registry_path,
            "build_segment_collection": resolved.ingestion.build_segment_collection,
            "build_bridge_fact_collection": resolved.ingestion.build_bridge_fact_collection,
            "manifest_schema_version": resolved.ingestion.manifest_schema_version,
        },
        "verifier": {
            "enabled": resolved.verifier.enabled,
            "max_tokens": resolved.verifier.max_tokens,
            "temperature": resolved.verifier.temperature,
        },
    }


def build_score_settings_fingerprint(settings: Settings | None = None) -> dict[str, object]:
    """Build a safe score-settings fingerprint for experiment lineage.

    Args:
        settings: Optional preloaded settings instance.

    Returns:
        dict[str, object]: Snapshot plus SHA-256 digest over the normalized JSON
        representation.
    """

    snapshot = build_score_settings_snapshot(settings)
    payload = json.dumps(snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "settings": snapshot,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
