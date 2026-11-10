import json
import os
from unittest.mock import patch

import pytest


def test_settings_load_defaults():
    env = {
        "ISAACUS_API_KEY": "test-isa-key",
        "ZEROENTROPY_API_KEY": "test-ze-key",
        "COHERE_API_KEY": "test-co-key",
        "LLM_API_KEY": "test-oai-key",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_SIMPLE_MODEL": "gpt-4o-mini",
        "LLM_COMPLEX_MODEL": "gpt-4o",
        "LLM_STRICT_MODEL": "gpt-4o-mini",
        "QDRANT_CHECK_COMPATIBILITY": "false",
        "QDRANT_COLLECTION": "legal_chunks",
        "RERANK_TOP_N": "6",
        "EMBED_DIMENSIONS": "1024",  # explicit default; .env.local may set 1792
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.embedding.model == "kanon-2-embedder"
        assert s.embedding.dimensions == 1024
        assert s.embedding.batch_size == 128
        assert s.reranker.primary_model == "zerank-2"
        assert s.reranker.primary_batch_size == 50
        assert s.reranker.top_n == 6
        assert s.reranker.shadow_selective_icr_enabled is False
        assert s.reranker.shadow_selective_icr_provider_exit is False
        assert s.reranker.shadow_selective_icr_model_path == ""
        assert s.reranker.shadow_selective_icr_max_chars == 1800
        assert s.reranker.shadow_selective_icr_candidate_batch_size == 32
        assert s.reranker.shadow_selective_icr_normalize_scores is True
        assert s.qdrant.collection == "legal_chunks"
        assert s.qdrant.prefetch_dense == 120
        assert s.qdrant.check_compatibility is False
        assert s.llm.base_url == "https://api.openai.com/v1"
        assert s.llm.simple_model == "gpt-4o-mini"
        assert s.llm.complex_model == "gpt-4o"
        assert s.llm.strict_model == "gpt-4o-mini"
        assert s.llm.stream_include_usage is True
        assert s.pipeline.confidence_threshold == 0.3
        assert s.pipeline.retry_query_max_anchors == 3
        assert s.ingestion.chunk_size_tokens == 450
        assert s.ingestion.parser_pdf_text_min_chars == 400


def test_settings_override_via_env():
    env = {
        "ISAACUS_API_KEY": "test",
        "ZEROENTROPY_API_KEY": "test",
        "COHERE_API_KEY": "test",
        "OPENAI_API_KEY": "test",
        "QDRANT_COLLECTION": "custom_collection",
        "PIPELINE_CONFIDENCE_THRESHOLD": "0.5",
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.qdrant.collection == "custom_collection"
        assert s.pipeline.confidence_threshold == 0.5


def test_pipeline_terms_csv_parses_into_list():
    env = {
        "ISAACUS_API_KEY": "test",
        "ZEROENTROPY_API_KEY": "test",
        "COHERE_API_KEY": "test",
        "OPENAI_API_KEY": "test",
        "PIPELINE_PREMISE_GUARD_TERMS": "jury,miranda,parole",
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.pipeline.premise_guard_terms == ["jury", "miranda", "parole"]


def test_openrouter_settings_override():
    env = {
        "ISAACUS_API_KEY": "test",
        "ZEROENTROPY_API_KEY": "test",
        "COHERE_API_KEY": "test",
        "LLM_BASE_URL": "https://openrouter.ai/api/v1",
        "LLM_API_KEY": "sk-or-test",
        "OPENROUTER_API_KEY": "sk-or-test",
        "OPENAI_API_KEY": "",
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.llm.base_url == "https://openrouter.ai/api/v1"
        assert s.llm.resolved_api_key().get_secret_value() == "sk-or-test"


def test_secret_str_not_leaked():
    env = {
        "ISAACUS_API_KEY": "secret-key-123",
        "ZEROENTROPY_API_KEY": "test",
        "COHERE_API_KEY": "test",
        "OPENAI_API_KEY": "test",
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import Settings

        s = Settings()
        assert "secret-key-123" not in str(s.embedding.api_key)
        assert s.embedding.api_key.get_secret_value() == "secret-key-123"


def test_platform_settings_load_from_eval_env() -> None:
    env = {
        "ISAACUS_API_KEY": "test",
        "ZEROENTROPY_API_KEY": "test",
        "COHERE_API_KEY": "test",
        "OPENAI_API_KEY": "test",
        "EVAL_API_KEY": "mcs_test_platform_key_1234567890",
        "EVAL_BASE_URL": "https://platform.agentic-challenge.ai/api/v1",
        "EVAL_PHASE": "warmup",
        "EVAL_COLLECTION_PREFIX": "phase_chunks",
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.platform.api_key.get_secret_value() == "mcs_test_platform_key_1234567890"
        assert s.platform.base_url == "https://platform.agentic-challenge.ai/api/v1"
        assert s.platform.phase == "warmup"
        assert s.platform.collection_prefix == "phase_chunks"


@pytest.mark.usefixtures("monkeypatch")
def test_settings_env_local_overrides_env(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ISAACUS_API_KEY=from-env",
                "ZEROENTROPY_API_KEY=from-env",
                "COHERE_API_KEY=from-env",
                "OPENAI_API_KEY=from-env",
                "QDRANT_COLLECTION=from-env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text("QDRANT_COLLECTION=from-env-local\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.qdrant.collection == "from-env-local"


@pytest.mark.usefixtures("monkeypatch")
def test_settings_process_env_overrides_env_local(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ISAACUS_API_KEY=from-env",
                "ZEROENTROPY_API_KEY=from-env",
                "COHERE_API_KEY=from-env",
                "OPENAI_API_KEY=from-env",
                "QDRANT_COLLECTION=from-env",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text("QDRANT_COLLECTION=from-env-local\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {"QDRANT_COLLECTION": "from-process-env"}, clear=True):
        from shafi.config.settings import Settings

        s = Settings()
        assert s.qdrant.collection == "from-process-env"


def test_score_settings_fingerprint_is_stable_and_secret_free() -> None:
    env = {
        "ISAACUS_API_KEY": "embed-secret",
        "ZEROENTROPY_API_KEY": "rerank-secret",
        "COHERE_API_KEY": "fallback-secret",
        "OPENAI_API_KEY": "llm-secret",
        "INGEST_INGEST_VERSION": "v3_anchor_retrieval",
        "PIPELINE_ENABLE_GROUNDING_SIDECAR": "true",
        "QDRANT_COLLECTION": "legal_chunks_v2",
    }
    with patch.dict(os.environ, env, clear=False):
        from shafi.config.settings import (
            Settings,
            build_score_settings_fingerprint,
            build_score_settings_snapshot,
        )

        settings = Settings()
        fingerprint = build_score_settings_fingerprint(settings)
        snapshot = build_score_settings_snapshot(settings)
        assert isinstance(fingerprint["sha256"], str) and len(fingerprint["sha256"]) == 64
        serialized = json.dumps(fingerprint, sort_keys=True)
        assert "embed-secret" not in serialized
        assert "rerank-secret" not in serialized
        assert "llm-secret" not in serialized
        assert fingerprint["settings"]["ingestion"]["ingest_version"] == "v3_anchor_retrieval"
        assert fingerprint["settings"]["pipeline"]["enable_grounding_sidecar"] is True
        assert snapshot["reranker"]["shadow_selective_icr_enabled"] is False
        assert snapshot["reranker"]["shadow_selective_icr_candidate_batch_size"] == 32
