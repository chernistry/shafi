from __future__ import annotations

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

    model: str = "kanon-2-embedder"
    api_url: str = "https://api.isaacus.com/v1/embeddings"
    api_key: SecretStr = Field(alias="ISAACUS_API_KEY")
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

    primary_model: str = "zerank-2"
    primary_api_url: str = "https://api.zeroentropy.dev/v1/models/rerank"
    primary_api_key: SecretStr = Field(alias="ZEROENTROPY_API_KEY")
    primary_batch_size: int = 50  # ZeroEntropy practical limit
    primary_latency_mode: str = "fast"  # "fast" or "degraded"
    primary_timeout_s: float = 30.0

    fallback_model: str = "rerank-v4.0-fast"
    fallback_api_key: SecretStr = Field(alias="COHERE_API_KEY")
    fallback_timeout_s: float = 30.0

    top_n: int = 6  # final context chunks
    rerank_candidates: int = 80  # from hybrid search, before rerank
    primary_max_connections: int = 20
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
    support_fact_collection: str = "legal_support_facts"
    pool_size: int = 20
    timeout_s: float = 30.0
    prefetch_dense: int = 60
    prefetch_sparse: int = 60
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

    rerank_enabled_strict_types: bool = True
    rerank_enabled_boolean: bool = False
    rerank_max_candidates_strict_types: int = 20
    boolean_rerank_candidates_cap: int = 12
    strict_types_context_top_n: int = 5
    boolean_context_top_n: int = 4
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
    context_token_budget_boolean: int = 900
    context_token_budget_strict: int = 1100
    context_token_budget_named_multi_lookup: int = 1600
    context_token_budget_free_text_simple: int = 1800
    context_token_budget_free_text_complex: int = 2400
    rerank_skip_on_single_doc_ref: bool = True
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
    enable_cross_ref_boosts: bool = False
    shadow_retrieval_top_k: int = 24
    anchor_retrieval_top_k: int = 16
    enable_grounding_sidecar: bool = False
    enable_trained_page_scorer: bool = False
    trained_page_scorer_model_path: str = ""
    grounding_support_fact_top_k: int = 32
    grounding_page_top_k: int = 24
    grounding_page_budget_default: int = 2
    grounding_allow_same_doc_hop: bool = True

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
