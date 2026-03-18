from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_challenge.core.verifier import VerificationResult
from rag_challenge.models import Citation, DocType, QueryComplexity, RankedChunk, RetrievedChunk
from rag_challenge.telemetry import TelemetryCollector


def _make_retrieved(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            doc_title="Test",
            doc_type=DocType.STATUTE,
            section_path=f"Section {i}",
            text=f"text {i}",
            score=0.9 - (i * 0.01),
            doc_summary="",
        )
        for i in range(n)
    ]


def _make_ranked(n: int, *, base_score: float = 0.8) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            doc_title="Test",
            doc_type=DocType.STATUTE,
            section_path=f"Section {i}",
            text=f"text {i}",
            retrieval_score=0.9 - (i * 0.01),
            rerank_score=base_score - (i * 0.01),
            doc_summary="",
        )
        for i in range(n)
    ]


def _make_retrieved_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    doc_title: str,
    section_path: str,
    text: str,
    score: float,
    doc_summary: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_title,
        doc_type=DocType.STATUTE,
        section_path=section_path,
        text=text,
        score=score,
        doc_summary=doc_summary,
    )


def test_extract_question_title_refs_strips_generic_question_leads() -> None:
    from rag_challenge.core.pipeline import _extract_question_title_refs

    refs = _extract_question_title_refs("On what date was the Employment Law Amendment Law enacted?")

    assert refs == ["Employment Law Amendment Law"]


def test_extract_question_title_refs_strips_preposition_leads() -> None:
    from rag_challenge.core.pipeline import _extract_question_title_refs

    refs = _extract_question_title_refs(
        "What are the effective dates for pre-existing and new accounts under the "
        "Common Reporting Standard Law 2018, and what is the date of its enactment?"
    )

    assert refs == ["Common Reporting Standard Law 2018"]


def test_extract_question_title_refs_strips_administers_lead_tokens() -> None:
    from rag_challenge.core.pipeline import _extract_question_title_refs

    refs = _extract_question_title_refs(
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )

    assert refs == ["Intellectual Property Law", "Trust Law"]


def test_extract_title_refs_from_query_strips_administers_lead_tokens(mock_settings) -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    refs = RAGPipelineBuilder._extract_title_refs_from_query(
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )

    assert refs == ["Intellectual Property Law", "Trust Law"]


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        embedding=SimpleNamespace(model="kanon-2-embedder"),
        reranker=SimpleNamespace(primary_model="zerank-2", top_n=6, rerank_candidates=80),
        pipeline=SimpleNamespace(
            confidence_threshold=0.3,
            retry_query_max_anchors=3,
            rerank_max_candidates_strict_types=20,
            boolean_rerank_candidates_cap=12,
            strict_doc_ref_top_k=16,
            strict_multi_ref_top_k_per_ref=12,
            strict_prefetch_dense=24,
            strict_prefetch_sparse=24,
            free_text_targeted_multi_ref_top_k=12,
        ),
        verifier=SimpleNamespace(enabled=True),
    )
    with patch("rag_challenge.core.pipeline.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def pipeline_builder(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(10))
    retriever.retrieve_with_retry = AsyncMock(return_value=_make_retrieved(10))

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=_make_ranked(6))

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "Answer text "
        yield "(cite: c0)."

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("Answer text (cite: c0).", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Test")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    classifier = MagicMock()
    classifier.normalize_query.side_effect = lambda q: q.strip()
    classifier.classify.return_value = QueryComplexity.SIMPLE
    classifier.select_model.return_value = "gpt-4o-mini"
    classifier.select_max_tokens.return_value = 300

    return RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end(pipeline_builder):
    app = pipeline_builder.compile()
    collector = TelemetryCollector(request_id="test-run")

    events: list[dict[str, object]] = []
    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await app.ainvoke(
            {
                "query": "What is the limitation period?",
                "request_id": "test-run",
                "collector": collector,
            }
        )

    assert "answer" in result
    assert "Answer text" in result["answer"]
    assert "telemetry" in result
    telemetry = result["telemetry"]
    assert telemetry.request_id == "test-run"
    assert telemetry.total_ms >= 0
    assert telemetry.model_embed == "kanon-2-embedder"
    assert telemetry.model_rerank == "zerank-2"
    assert telemetry.model_llm == "gpt-4o-mini"
    assert telemetry.cited_chunk_ids == ["c0"]
    assert telemetry.chunk_snippets["c0"] == "Section 0 | text 0"
    assert any(event.get("type") == "token" for event in events)
    assert any(event.get("type") == "answer_final" and "Answer text" in str(event.get("text")) for event in events)
    assert any(event.get("type") == "telemetry" for event in events)


@pytest.mark.asyncio
async def test_retrieve_single_title_strict_query_uses_targeted_title_lookup(pipeline_builder) -> None:
    generic_retrieved: list[RetrievedChunk] = []
    targeted_retrieved = [
        _make_retrieved_chunk(
            chunk_id="employment-amendment:notice",
            doc_id="employment-amendment",
            doc_title="EMPLOYMENT LAW AMENDMENT LAW",
            section_path="page:1",
            text="We hereby enact on this 14 day of September 2021 the Employment Law Amendment Law.",
            score=0.98,
        )
    ]
    pipeline_builder._retriever.retrieve = AsyncMock(side_effect=[generic_retrieved, targeted_retrieved])

    result = await pipeline_builder._retrieve(
        {
            "query": "On what date was the Employment Law Amendment Law enacted?",
            "request_id": "single-title-strict",
            "question_id": "single-title-strict",
            "answer_type": "date",
            "collector": TelemetryCollector(request_id="single-title-strict"),
            "doc_refs": [],
        }
    )

    retrieved = result["retrieved"]
    assert [chunk.chunk_id for chunk in retrieved] == ["employment-amendment:notice"]
    assert result["must_include_chunk_ids"] == ["employment-amendment:notice"]


@pytest.mark.asyncio
async def test_pipeline_retries_on_low_confidence(pipeline_builder):
    low_ranked = _make_ranked(6, base_score=0.1)
    pipeline_builder._reranker.rerank = AsyncMock(side_effect=[low_ranked, _make_ranked(6, base_score=0.8)])

    app = pipeline_builder.compile()
    collector = TelemetryCollector(request_id="retry-test")
    events: list[dict[str, object]] = []

    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await app.ainvoke(
            {
                "query": "Complex question",
                "request_id": "retry-test",
                "collector": collector,
            }
        )

    assert result.get("retried") is True
    assert result["telemetry"].retried is True
    assert pipeline_builder._retriever.retrieve_with_retry.await_count == 1
    assert pipeline_builder._reranker.rerank.await_count == 2
    assert any(event.get("type") == "telemetry" for event in events)


@pytest.mark.asyncio
async def test_retry_retrieve_preserves_broad_enumeration_top_n(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(20))
    retriever.retrieve_with_retry = AsyncMock(return_value=_make_retrieved(20))

    reranker = MagicMock()
    reranker.rerank = AsyncMock(side_effect=[_make_ranked(6, base_score=0.1), _make_ranked(12, base_score=0.8)])
    reranker.get_last_used_model = MagicMock(return_value="zerank-2")

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "1. Test (cite: c0)"

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Test (cite: c0)", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Test")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)

    classifier = MagicMock()
    classifier.normalize_query.side_effect = lambda q: q.strip()
    classifier.classify.return_value = QueryComplexity.SIMPLE
    classifier.select_model.return_value = "gpt-4o-mini"
    classifier.select_max_tokens.return_value = 300

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    app = builder.compile()
    collector = TelemetryCollector(request_id="retry-enum")

    result = await app.ainvoke(
        {
            "query": "Which laws were made by the Ruler of Dubai and their commencement date is specified in an Enactment Notice?",
            "request_id": "retry-enum",
            "collector": collector,
            "answer_type": "free_text",
        }
    )

    assert result.get("retried") is True
    assert len(result["context_chunks"]) == 12
    assert reranker.rerank.await_args_list[1].kwargs["top_n"] == 12


@pytest.mark.asyncio
async def test_pipeline_skips_verifier_for_strict_types(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(5))
    retriever.retrieve_with_retry = AsyncMock(return_value=_make_retrieved(5))

    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=_make_ranked(3))
    reranker.get_last_used_model = MagicMock(return_value="zerank-2")

    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("Draft answer without citations.", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Test")])
    generator.extract_cited_chunk_ids = MagicMock(side_effect=lambda text: ["c0"] if "c0" in str(text) else [])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    verifier = MagicMock()
    verifier.should_verify = MagicMock(return_value=True)
    verifier.verify = AsyncMock(
        return_value=VerificationResult(
            is_grounded=False,
            unsupported_claims=["claim"],
            revised_answer="Yes. Revised answer (cite: c0).",
        )
    )

    classifier = MagicMock()
    classifier.normalize_query.side_effect = lambda q: q.strip()
    classifier.classify.return_value = QueryComplexity.SIMPLE
    classifier.select_model.return_value = "gpt-4o-mini"
    classifier.select_max_tokens.return_value = 300

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
        verifier=verifier,
    )
    app = builder.compile()
    collector = TelemetryCollector(request_id="verify-run")
    result = await app.ainvoke(
        {
            "query": "What is the rule?",
            "request_id": "verify-run",
            "collector": collector,
            "answer_type": "boolean",
        }
    )

    assert str(result["answer"]).lower() in {"no", "null"}
    assert result["telemetry"].verify_ms == 0
    verifier.verify.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_strict_boolean_normalizes_invalid_citations(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(
        return_value=[
            _make_retrieved_chunk(
                chunk_id="c0",
                doc_id="law",
                doc_title="Test Law",
                section_path="page:1",
                text="Article 10 applies in these circumstances.",
                score=0.91,
            ),
            _make_retrieved_chunk(
                chunk_id="c1",
                doc_id="law",
                doc_title="Test Law",
                section_path="page:2",
                text="This page contains background material only.",
                score=0.88,
            ),
        ]
    )
    retriever.retrieve_with_retry = AsyncMock(
        return_value=[
            _make_retrieved_chunk(
                chunk_id="c0",
                doc_id="law",
                doc_title="Test Law",
                section_path="page:1",
                text="Article 10 applies in these circumstances.",
                score=0.91,
            ),
            _make_retrieved_chunk(
                chunk_id="c1",
                doc_id="law",
                doc_title="Test Law",
                section_path="page:2",
                text="This page contains background material only.",
                score=0.88,
            ),
        ]
    )

    reranker = MagicMock()
    reranker.rerank = AsyncMock(
        return_value=[
            RankedChunk(
                chunk_id="c0",
                doc_id="law",
                doc_title="Test Law",
                doc_type=DocType.STATUTE,
                section_path="page:1",
                text="Article 10 applies in these circumstances.",
                retrieval_score=0.91,
                rerank_score=0.91,
                doc_summary="",
            ),
            RankedChunk(
                chunk_id="c1",
                doc_id="law",
                doc_title="Test Law",
                doc_type=DocType.STATUTE,
                section_path="page:2",
                text="This page contains background material only.",
                retrieval_score=0.88,
                rerank_score=0.88,
                doc_summary="",
            ),
        ]
    )
    reranker.get_last_used_model = MagicMock(return_value="zerank-2")

    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("Yes, it applies. (cite: bad-id (cite: c0).", []))
    generator.generate_stream = AsyncMock()
    generator.extract_cited_chunk_ids = MagicMock(
        side_effect=lambda text: ["bad-id (cite: c0"] if "bad-id" in str(text) else (["c0"] if "c0" in str(text) else [])
    )
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Test")])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    classifier = MagicMock()
    classifier.normalize_query.side_effect = lambda q: q.strip()
    classifier.classify.return_value = QueryComplexity.SIMPLE
    classifier.select_model.return_value = "gpt-4o-mini"
    classifier.select_max_tokens.return_value = 300

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    app = builder.compile()
    collector = TelemetryCollector(request_id="strict-cite-fix")
    events: list[dict[str, object]] = []
    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await app.ainvoke(
            {
                "query": "Does article 10 apply?",
                "request_id": "strict-cite-fix",
                "collector": collector,
                "answer_type": "boolean",
            }
        )

    assert result["answer"] == "Yes"
    # Strict outputs are parse-safe; evidence is tracked separately via telemetry.
    assert result["telemetry"].cited_chunk_ids == ["c0"]
    assert any(event.get("type") == "answer_final" and event.get("text") == "Yes" for event in events)


@pytest.mark.asyncio
async def test_pipeline_strict_number_localizes_minimal_support_instead_of_whole_context(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    mock_settings.pipeline.strict_types_extraction_enabled = False

    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("250499.26", []))
    generator.generate_stream = AsyncMock()
    generator.extract_cited_chunk_ids = MagicMock(return_value=[])
    generator.extract_citations = MagicMock(return_value=[])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    classifier = MagicMock()
    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    collector = TelemetryCollector(request_id="strict-number-support")

    context_chunks = [
        RankedChunk(
            chunk_id="claim:value",
            doc_id="claim-doc",
            doc_title="Claim Form",
            doc_type=DocType.CASE_LAW,
            section_path="page:3",
            text="The claim value is 250499.26 AED.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="claim:noise",
            doc_id="claim-doc",
            doc_title="Claim Form",
            doc_type=DocType.CASE_LAW,
            section_path="page:4",
            text="This page contains procedural history only.",
            retrieval_score=0.8,
            rerank_score=0.8,
            doc_summary="",
        ),
    ]

    result = await builder._generate(
        {
            "query": "What is the claim value?",
            "request_id": "strict-number-support",
            "question_id": "strict-number-support",
            "collector": collector,
            "answer_type": "number",
            "context_chunks": context_chunks,
            "model": "gpt-4.1-mini",
            "max_tokens": 64,
            "complexity": QueryComplexity.SIMPLE,
        }
    )

    assert result["answer"] == "250499.26"
    assert collector.finalize().cited_chunk_ids == ["claim:value"]


@pytest.mark.asyncio
async def test_pipeline_strict_number_without_support_does_not_fallback_to_first_context_chunk(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    mock_settings.pipeline.strict_types_extraction_enabled = False

    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("250499.26", []))
    generator.generate_stream = AsyncMock()
    generator.extract_cited_chunk_ids = MagicMock(return_value=[])
    generator.extract_citations = MagicMock(return_value=[])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="strict-number-no-fallback")

    context_chunks = [
        RankedChunk(
            chunk_id="noise:0",
            doc_id="noise-doc",
            doc_title="Procedural Note",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="This page contains procedural history only and no numeric claim amount.",
            retrieval_score=0.8,
            rerank_score=0.8,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="noise:1",
            doc_id="noise-doc",
            doc_title="Procedural Note",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="No relevant amount is stated in this document.",
            retrieval_score=0.79,
            rerank_score=0.79,
            doc_summary="",
        ),
    ]

    result = await builder._generate(
        {
            "query": "What is the claim value?",
            "request_id": "strict-number-no-fallback",
            "question_id": "strict-number-no-fallback",
            "collector": collector,
            "answer_type": "number",
            "context_chunks": context_chunks,
            "model": "gpt-4.1-mini",
            "max_tokens": 64,
            "complexity": QueryComplexity.SIMPLE,
        }
    )

    assert result["answer"] == "250499.26"
    assert collector.finalize().cited_chunk_ids == []


def test_localize_strict_date_support_matches_iso_answer_to_textual_enactment_date() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="enactment:0",
            doc_id="employment-amendment",
            doc_title="ENACTMENT NOTICE",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on this 14 day of September 2021 in the form now attached "
                "the Employment Law Amendment Law DIFC Law No. 4 of 2021."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        )
    ]

    localized = RAGPipelineBuilder._localize_strict_support_chunk_ids(
        answer_type="date",
        answer="2021-09-14",
        query="On what date was the Employment Law Amendment Law enacted?",
        context_chunks=context_chunks,
    )

    assert localized == ["enactment:0"]


def test_localize_strict_name_support_matches_case_caption_tokens() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="case:0",
            doc_id="enf-269-2023",
            doc_title="ENF 269/2023 (1) Ozias (2) Ori (3) Octavio v (1) Obadiah (2) Oaklen",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="ENF 269/2023 (1) OZIAS (2) ORI (3) OCTAVIO Appellants and OBADIAH Defendant.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        )
    ]

    localized = RAGPipelineBuilder._localize_strict_support_chunk_ids(
        answer_type="name",
        answer="Ozias Ori Octavio v Obadiah",
        query="Which case was decided earlier: ENF 269/2023 or SCT 514/2025?",
        context_chunks=context_chunks,
    )

    assert localized == ["case:0"]


def test_localize_strict_name_support_matches_law_title_without_year_in_chunk_text() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="employment:article16",
            doc_id="employment-law",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:8",
            text=(
                "16. Payroll records. An Employer shall keep records of the Employee's Remuneration "
                "(gross and net, where applicable)."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        )
    ]

    localized = RAGPipelineBuilder._localize_strict_support_chunk_ids(
        answer_type="name",
        answer="Employment Law 2019",
        query=(
            "Under Article 16(1)(c) of the Employment Law 2019, what type of remuneration "
            "(gross or net) must an Employer keep records of, where applicable?"
        ),
        context_chunks=context_chunks,
    )

    assert localized == ["employment:article16"]


def test_localize_strict_boolean_support_can_keep_exception_companion_page() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="operating:rule",
            doc_id="operating-law",
            doc_title="Operating Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text=(
                "The Registrar is not liable for any act or omission done in the exercise of the Registrar's "
                "functions under this Law."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="operating:exception",
            doc_id="operating-law",
            doc_title="Operating Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:8",
            text=(
                "The limitation on liability does not apply if the act or omission is shown to have been in bad faith."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    localized = RAGPipelineBuilder._localize_strict_support_chunk_ids(
        answer_type="boolean",
        answer="Yes",
        query=(
            "Under the Operating Law 2018, can the Registrar be held liable for acts or omissions in performing "
            "their functions if the act or omission is shown to have been in bad faith?"
        ),
        context_chunks=context_chunks,
    )

    assert localized == ["operating:exception", "operating:rule"]


def test_localize_free_text_support_keeps_multiple_pages_for_composite_item() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="data:0:0:title",
            doc_id="data-law",
            doc_title="Data Protection Law 2020",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law may be cited as the "Data Protection Law 2020".',
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="data:1:0:update",
            doc_id="data-law",
            doc_title="Data Protection Law 2020",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="Consolidated Version (1 July 2024).",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
    ]

    localized = RAGPipelineBuilder._localize_free_text_support_chunk_ids(
        answer=(
            "1. DIFC Law No. 5 of 2020 - Title: Data Protection Law 2020 (cite: data:title) - "
            "Last updated (consolidated version): 1 July 2024."
        ),
        query="What is the title of DIFC Law No. 5 of 2020 and when was its consolidated version last updated?",
        context_chunks=context_chunks,
    )

    assert set(localized) == {"data:0:0:title", "data:1:0:update"}


def test_apply_support_shape_policy_keeps_both_sides_for_strict_case_compare() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="case-a:title",
            doc_id="cfi-010-2024",
            doc_title="CFI 010/2024 Alpha v Beta",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="CFI 010/2024 Alpha v Beta. Date of Issue: 12 January 2024.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="case-a:page2",
            doc_id="cfi-010-2024",
            doc_title="CFI 010/2024 Alpha v Beta",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="Supporting procedural details for CFI 010/2024.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="case-b:title",
            doc_id="cfi-016-2025",
            doc_title="CFI 016/2025 Gamma v Delta",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="CFI 016/2025 Gamma v Delta. Date of Issue: 4 February 2025.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="case-b:page2",
            doc_id="cfi-016-2025",
            doc_title="CFI 016/2025 Gamma v Delta",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="Supporting procedural details for CFI 016/2025.",
            retrieval_score=0.93,
            rerank_score=0.93,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="name",
        answer="CFI 010/2024",
        query="Which case has an earlier Date of Issue: CFI 010/2024 or CFI 016/2025?",
        context_chunks=context_chunks,
        cited_ids=["case-a:title"],
        support_ids=[],
    )

    assert set(shaped_ids) == {"case-a:title", "case-a:page2", "case-b:title", "case-b:page2"}
    assert flags == []


def test_apply_support_shape_policy_keeps_title_page_for_multi_atom_named_metadata_query() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="found:title",
            doc_id="foundations-law",
            doc_title="Foundations Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law may be cited as the "Foundations Law 2018".',
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="found:update",
            doc_id="foundations-law",
            doc_title="Foundations Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="The consolidated version of this Law was updated on 1 July 2024.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="found:enactment",
            doc_id="foundations-law",
            doc_title="Foundations Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="This Law comes into force on the date specified in the Enactment Notice.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="free_text",
        answer="The title is Foundations Law 2018 and the consolidated version was updated on 1 July 2024.",
        query="What is the title of the Foundations Law 2018 and when was its consolidated version last updated?",
        context_chunks=context_chunks,
        cited_ids=["found:update"],
        support_ids=["found:update"],
    )

    assert set(shaped_ids) == {"found:title", "found:update", "found:enactment"}
    assert flags == []


def test_apply_support_shape_policy_adds_title_page_for_single_atom_named_metadata_query() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="law:title",
            doc_id="personal-property-law",
            doc_title="Personal Property Law 2005",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law may be cited as the "Personal Property Law 2005".',
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:maker",
            doc_id="personal-property-law",
            doc_title="Personal Property Law 2005",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text="This Law is made by the Ruler of Dubai.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="free_text",
        answer="This Law was made by the Ruler of Dubai.",
        query="According to Article 2 of the DIFC Personal Property Law 2005, who made this Law?",
        context_chunks=context_chunks,
        cited_ids=["law:maker"],
        support_ids=["law:maker"],
    )

    assert shaped_ids == ["law:maker", "law:title"]
    assert flags == []


def test_apply_support_shape_policy_keeps_outcome_and_cost_pages() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="appeal:outcome",
            doc_id="ca-009-2024",
            doc_title="CA 009/2024",
            doc_type=DocType.CASE_LAW,
            section_path="page:5",
            text="The Permission to Appeal Application is refused.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="appeal:costs",
            doc_id="ca-009-2024",
            doc_title="CA 009/2024",
            doc_type=DocType.CASE_LAW,
            section_path="page:6",
            text="The Appellant was awarded costs in the sum of USD 40,000.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="free_text",
        answer="The Permission to Appeal Application is refused. The Appellant was awarded costs in the sum of USD 40,000.",
        query="How did the Court of Appeal rule, and what costs were awarded?",
        context_chunks=context_chunks,
        cited_ids=["appeal:outcome"],
        support_ids=["appeal:outcome"],
    )

    assert set(shaped_ids) == {"appeal:outcome", "appeal:costs"}
    assert flags == []


def test_apply_support_shape_policy_forces_explicit_second_page_chunk() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="judgment:page1",
            doc_id="judgment-doc",
            doc_title="CFI 010/2024 Alpha v Beta",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="CFI 010/2024 Alpha v Beta. Cover page with parties.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="judgment:page2",
            doc_id="judgment-doc",
            doc_title="CFI 010/2024 Alpha v Beta",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="The second page contains the judge names and filing details.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="name",
        answer="Justice Example",
        query="Who is named on the second page of CFI 010/2024 Alpha v Beta?",
        context_chunks=context_chunks,
        cited_ids=["judgment:page1"],
        support_ids=["judgment:page1"],
    )

    assert shaped_ids == ["judgment:page2"]
    assert flags == ["explicit_page_reference_forced", "explicit_page_reference_pruned"]


def test_apply_support_shape_policy_forces_title_page_chunk_for_named_metadata_query() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="law:title",
            doc_id="foundations-law",
            doc_title="Foundations Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law may be cited as the "Foundations Law 2018".',
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:update",
            doc_id="foundations-law",
            doc_title="Foundations Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="The consolidated version of this Law was updated on 1 July 2024.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="free_text",
        answer="The consolidated version was updated on 1 July 2024.",
        query="What appears on the title page of the Foundations Law 2018?",
        context_chunks=context_chunks,
        cited_ids=["law:update"],
        support_ids=["law:update"],
    )

    assert shaped_ids == ["law:title"]
    assert flags == ["explicit_page_reference_forced", "explicit_page_reference_pruned"]


@pytest.mark.asyncio
async def test_generate_augments_strict_context_for_explicit_page_claim_origin(mock_settings) -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("Claim No. CA 008/2023", []))
    generator.extract_citations = MagicMock(return_value=[])
    generator.extract_cited_chunk_ids = MagicMock(return_value=[])

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="claim-origin", question_id="claim-origin", answer_type="name")
    collector.set_retrieved_ids(["ca009:10:context", "ca009:1:origin"])
    collector.set_context_ids(["ca009:10:context"])

    result = await builder._generate(
        {
            "query": "According to page 2 of the judgment, from which specific claim number did the appeal in CA 009/2024 originate?",
            "request_id": "claim-origin",
            "question_id": "claim-origin",
            "collector": collector,
            "answer_type": "name",
            "context_chunks": [
                RankedChunk(
                    chunk_id="ca009:10:context",
                    doc_id="ca009",
                    doc_title="CA 009/2024 Oskar v Oron",
                    doc_type=DocType.CASE_LAW,
                    section_path="page:11",
                    text="Argument sections discussing officers and shareholders.",
                    retrieval_score=0.96,
                    rerank_score=0.96,
                    doc_summary="",
                )
            ],
            "retrieved": [
                RetrievedChunk(
                    chunk_id="ca009:10:context",
                    doc_id="ca009",
                    doc_title="CA 009/2024 Oskar v Oron",
                    doc_type=DocType.CASE_LAW,
                    section_path="page:11",
                    text="Argument sections discussing officers and shareholders.",
                    score=0.96,
                    doc_summary="",
                ),
                RetrievedChunk(
                    chunk_id="ca009:1:origin",
                    doc_id="ca009",
                    doc_title="CA 009/2024 Oskar v Oron",
                    doc_type=DocType.CASE_LAW,
                    section_path="page:2",
                    text=(
                        "UPON the hearing of the Appellant's appeal against the Order granting the relief "
                        "sought by the Respondents in their Urgent Application of 1 April 2024 in Claim No. "
                        'ENF-316-2023/2, (the "Application").'
                    ),
                    score=0.94,
                    doc_summary="",
                ),
            ],
            "doc_refs": ["CA 009/2024"],
            "model": "gpt-4.1-mini",
            "max_tokens": 64,
            "complexity": QueryComplexity.SIMPLE,
        }
    )

    telemetry = collector.finalize()

    assert result["answer"] == "ENF-316-2023/2"
    assert result["cited_chunk_ids"] == ["ca009:1:origin"]
    assert "ca009:1:origin" in telemetry.context_chunk_ids
    assert "ca009_2" in telemetry.context_page_ids
    assert generator.generate.await_count == 0


@pytest.mark.asyncio
async def test_strict_null_preserves_context_page_telemetry(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder
    from rag_challenge.core.strict_answerer import StrictAnswerResult

    generator = MagicMock()
    generator.generate = AsyncMock(return_value=("should-not-run", []))
    generator.get_context_debug_stats = MagicMock(return_value=(2, 900))
    generator.extract_citations = MagicMock(return_value=[])
    generator.extract_cited_chunk_ids = MagicMock(return_value=[])

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    builder._strict_answerer.answer = MagicMock(return_value=StrictAnswerResult(answer="null", cited_chunk_ids=[], confident=True))

    collector = TelemetryCollector(
        request_id="strict-null-telemetry",
        question_id="strict-null-telemetry",
        answer_type="boolean",
    )

    result = await builder._generate(
        {
            "query": (
                "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate or conduct business "
                "in or from the DIFC without being incorporated, registered, or continued under a Prescribed Law or "
                "other Legislation administered by the Registrar?"
            ),
            "request_id": "strict-null-telemetry",
            "question_id": "strict-null-telemetry",
            "collector": collector,
            "answer_type": "boolean",
            "context_chunks": [
                RankedChunk(
                    chunk_id="operating:title",
                    doc_id="operating",
                    doc_title="OPERATING LAW",
                    doc_type=DocType.STATUTE,
                    section_path="page:4",
                    text='This Law may be cited as the "Operating Law 2018".',
                    retrieval_score=0.92,
                    rerank_score=0.92,
                    doc_summary="",
                ),
                RankedChunk(
                    chunk_id="operating:article8",
                    doc_id="operating",
                    doc_title="OPERATING LAW",
                    doc_type=DocType.STATUTE,
                    section_path="page:8",
                    text=(
                        "No person shall operate or conduct business in or from the DIFC unless that person is "
                        "incorporated, registered or continued under a Prescribed Law."
                    ),
                    retrieval_score=0.91,
                    rerank_score=0.91,
                    doc_summary="",
                ),
            ],
            "retrieved": [
                RetrievedChunk(
                    chunk_id="operating:title",
                    doc_id="operating",
                    doc_title="OPERATING LAW",
                    doc_type=DocType.STATUTE,
                    section_path="page:4",
                    text='This Law may be cited as the "Operating Law 2018".',
                    score=0.92,
                    doc_summary="",
                ),
                RetrievedChunk(
                    chunk_id="operating:article8",
                    doc_id="operating",
                    doc_title="OPERATING LAW",
                    doc_type=DocType.STATUTE,
                    section_path="page:8",
                    text=(
                        "No person shall operate or conduct business in or from the DIFC unless that person is "
                        "incorporated, registered or continued under a Prescribed Law."
                    ),
                    score=0.91,
                    doc_summary="",
                ),
            ],
            "doc_refs": ["Operating Law 2018"],
            "model": "gpt-4.1-mini",
            "max_tokens": 64,
            "complexity": QueryComplexity.SIMPLE,
        }
    )

    telemetry = collector.finalize()

    assert result["answer"] == "null"
    assert telemetry.retrieved_chunk_ids == ["operating:title", "operating:article8"]
    assert telemetry.context_chunk_ids == ["operating:title", "operating:article8"]
    assert telemetry.retrieved_page_ids == ["operating_4", "operating_8"]
    assert telemetry.context_page_ids == ["operating_4", "operating_8"]
    assert telemetry.cited_page_ids == []
    assert telemetry.used_page_ids == []
    assert telemetry.context_chunk_count == 2
    assert generator.generate.await_count == 0


def test_apply_support_shape_policy_prunes_nonrequested_title_page_noise_when_title_page_already_present() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="law:title",
            doc_id="civil-commercial-law",
            doc_title="DIFC Law on the Application of Civil and Commercial Laws",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This is DIFC Law No. 3 of 2004.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:page3",
            doc_id="civil-commercial-law",
            doc_title="DIFC Law on the Application of Civil and Commercial Laws",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text="Interpretation clause on page 3.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:page7",
            doc_id="civil-commercial-law",
            doc_title="DIFC Law on the Application of Civil and Commercial Laws",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text="Later operative clause on page 7.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    shaped_ids, flags = RAGPipelineBuilder._apply_support_shape_policy(
        answer_type="number",
        answer="Law No. 3 of 2004",
        query="According to the title page of the DIFC Law on the Application of Civil and Commercial Laws, what is its official law number?",
        context_chunks=context_chunks,
        cited_ids=["law:page7", "law:title"],
        support_ids=["law:page3"],
    )

    assert shaped_ids == ["law:title"]
    assert flags == ["explicit_page_reference_pruned"]


@pytest.mark.asyncio
async def test_set_final_used_pages_clamps_citation_override_to_explicit_anchor() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    mock_retriever = MagicMock()
    mock_retriever._settings.pipeline.enable_grounding_sidecar = False
    builder = RAGPipelineBuilder(
        retriever=mock_retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="anchor-clamp", question_id="anchor-clamp", answer_type="number")
    context_chunks = [
        RankedChunk(
            chunk_id="law:page7",
            doc_id="civil-commercial-law",
            doc_title="DIFC Law on the Application of Civil and Commercial Laws",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text="Later operative clause repeating Law No. 3 of 2004 in the body text.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:title",
            doc_id="civil-commercial-law",
            doc_title="DIFC Law on the Application of Civil and Commercial Laws",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This is DIFC Law No. 3 of 2004.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
    ]

    await builder._set_final_used_pages(
        collector=collector,
        query="According to the title page of the DIFC Law on the Application of Civil and Commercial Laws, what is its official law number?",
        answer="Law No. 3 of 2004",
        answer_type="number",
        context_chunks=context_chunks,
        current_used_ids=["law:page7", "law:title"],
    )

    telemetry = collector.finalize()

    assert telemetry.used_page_ids == ["civil-commercial-law_1"]


@pytest.mark.asyncio
async def test_set_final_used_pages_keeps_title_page_compare_to_one_page_per_doc() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    mock_retriever = MagicMock()
    mock_retriever._settings.pipeline.enable_grounding_sidecar = False
    builder = RAGPipelineBuilder(
        retriever=mock_retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="title-compare", question_id="title-compare", answer_type="name")
    context_chunks = [
        RankedChunk(
            chunk_id="ca1:body",
            doc_id="ca1",
            doc_title="CA 001/2024 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="The claimant is Alice Example in the reasons section.",
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="ca1:title",
            doc_id="ca1",
            doc_title="CA 001/2024 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="Between Alice Example and Bob Example.",
            retrieval_score=0.90,
            rerank_score=0.90,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="sct2:body",
            doc_id="sct2",
            doc_title="SCT 002/2024 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:3",
            text="The claimant is Alice Example in the factual background.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="sct2:title",
            doc_id="sct2",
            doc_title="SCT 002/2024 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="Between Alice Example and Carol Example.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
    ]

    await builder._set_final_used_pages(
        collector=collector,
        query="Which claimant appears on the title page of both CA 001/2024 and SCT 002/2024?",
        answer="Alice Example",
        answer_type="name",
        context_chunks=context_chunks,
        current_used_ids=["ca1:body", "ca1:title", "sct2:body", "sct2:title"],
    )

    telemetry = collector.finalize()

    assert telemetry.used_page_ids == ["ca1_1", "sct2_1"]


@pytest.mark.asyncio
async def test_set_final_used_pages_sidecar_empty_override_clears_used_pages() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    mock_retriever = MagicMock()
    mock_retriever._settings.pipeline.enable_grounding_sidecar = True
    builder = RAGPipelineBuilder(
        retriever=mock_retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    builder._grounding_selector_cached = MagicMock(select_page_ids=AsyncMock(return_value=[]))

    collector = TelemetryCollector(request_id="sidecar-empty", question_id="sidecar-empty", answer_type="free_text")
    context_chunks = [
        RankedChunk(
            chunk_id="law:title",
            doc_id="law",
            doc_title="Example Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Title page.",
            retrieval_score=0.91,
            rerank_score=0.91,
            doc_summary="",
        )
    ]

    await builder._set_final_used_pages(
        collector=collector,
        query="What was the jury finding?",
        answer="null",
        answer_type="free_text",
        context_chunks=context_chunks,
        current_used_ids=["law:title"],
    )

    telemetry = collector.finalize()

    assert telemetry.used_page_ids == []


@pytest.mark.asyncio
async def test_set_final_used_pages_null_answer_without_sidecar_clears_used_pages() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    mock_retriever = MagicMock()
    mock_retriever._settings.pipeline.enable_grounding_sidecar = False
    builder = RAGPipelineBuilder(
        retriever=mock_retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )

    collector = TelemetryCollector(request_id="legacy-empty", question_id="legacy-empty", answer_type="free_text")
    context_chunks = [
        RankedChunk(
            chunk_id="law:title",
            doc_id="law",
            doc_title="Example Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Title page.",
            retrieval_score=0.91,
            rerank_score=0.91,
            doc_summary="",
        )
    ]

    await builder._set_final_used_pages(
        collector=collector,
        query="What was the jury finding?",
        answer="There is no information on this question.",
        answer_type="free_text",
        context_chunks=context_chunks,
        current_used_ids=["law:title"],
    )

    telemetry = collector.finalize()

    assert telemetry.used_page_ids == []


def test_expand_page_spanning_support_chunk_ids_includes_adjacent_continuation_page() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="law:0:0:tail",
            doc_id="law",
            doc_title="Example Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "Article 10 provides that the Court may award compensation for losses suffered by the claimant, "
                "together with costs reasonably incurred in bringing"
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:1:0:cont",
            doc_id="law",
            doc_title="Example Law",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="the claim and any further relief that the Court considers appropriate.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    expanded = RAGPipelineBuilder._expand_page_spanning_support_chunk_ids(
        chunk_ids=["law:0:0:tail"],
        context_chunks=context_chunks,
    )

    assert expanded == ["law:0:0:tail", "law:1:0:cont"]


def test_expand_page_spanning_support_chunk_ids_does_not_add_new_section_page() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="law:0:0:complete",
            doc_id="law",
            doc_title="Example Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Article 10 provides that the Court may award compensation for losses suffered by the claimant.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:1:0:new",
            doc_id="law",
            doc_title="Example Law",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="ARTICLE 11 NEW OFFENCE A person commits an offence if they obstruct the Court.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    expanded = RAGPipelineBuilder._expand_page_spanning_support_chunk_ids(
        chunk_ids=["law:0:0:complete"],
        context_chunks=context_chunks,
    )

    assert expanded == ["law:0:0:complete"]


def test_apply_doc_shortlist_gating_filters_wrong_document_candidates() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="wrong:1",
            doc_id="wrong",
            doc_title="Schedule Provisions Digest",
            section_path="page:1",
            text="Schedule 1 contains definitions for several laws including data protection and employment.",
            score=0.99,
        ),
        _make_retrieved_chunk(
            chunk_id="data:1",
            doc_id="data",
            doc_title="Data Protection Law 2020",
            section_path="page:1",
            text='This Law may be cited as the "Data Protection Law 2020".',
            score=0.91,
        ),
        _make_retrieved_chunk(
            chunk_id="employment:1",
            doc_id="employment",
            doc_title="Employment Law 2019",
            section_path="page:1",
            text='This Law may be cited as the "Employment Law 2019".',
            score=0.90,
        ),
    ]

    shortlisted = RAGPipelineBuilder._apply_doc_shortlist_gating(
        query="What are the titles of the Data Protection Law 2020 and the Employment Law 2019?",
        doc_refs=["Data Protection Law 2020", "Employment Law 2019"],
        retrieved=retrieved,
    )

    assert {chunk.doc_id for chunk in shortlisted} == {"data", "employment"}


def test_apply_doc_shortlist_gating_handles_strict_law_number_query() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:1",
            text='This Law may be cited as the "Employment Law 2019". DIFC Law No. 4 of 2019.',
            score=0.98,
            doc_summary="Employment Law 2019.",
        ),
        _make_retrieved_chunk(
            chunk_id="employment-amendment:title",
            doc_id="employment-amendment",
            doc_title="EMPLOYMENT LAW AMENDMENT LAW",
            section_path="page:1",
            text='This Law may be cited as the "Employment Law Amendment Law". DIFC Law No. 1 of 2024.',
            score=0.72,
            doc_summary="Employment Law Amendment Law, DIFC Law No. 1 of 2024.",
        ),
    ]

    shortlisted = RAGPipelineBuilder._apply_doc_shortlist_gating(
        query="What is the law number of the Employment Law Amendment Law?",
        doc_refs=["Employment Law Amendment Law"],
        retrieved=retrieved,
    )

    assert {chunk.doc_id for chunk in shortlisted} == {"employment-amendment"}


@pytest.mark.asyncio
async def test_pipeline_free_text_without_inline_citations_localizes_support_without_first_chunk_patch(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "The title is Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013."

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock()
    generator.extract_cited_chunk_ids = MagicMock(return_value=[])
    generator.extract_citations = MagicMock(return_value=[])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_final_answer = MagicMock(side_effect=lambda answer: answer)

    classifier = MagicMock()
    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    collector = TelemetryCollector(request_id="free-text-support")

    context_chunks = [
        RankedChunk(
            chunk_id="arb:title",
            doc_id="arb",
            doc_title="Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="This Law may be cited as Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013.",
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="This Law may be cited as General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013.",
            retrieval_score=0.91,
            rerank_score=0.91,
            doc_summary="",
        ),
    ]

    result = await builder._generate(
        {
            "query": "What are the titles of DIFC Law No. 6 of 2013 and DIFC Law No. 3 of 2013?",
            "request_id": "free-text-support",
            "question_id": "free-text-support",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": context_chunks,
            "model": "gpt-4.1",
            "max_tokens": 128,
            "complexity": QueryComplexity.SIMPLE,
        }
    )

    assert "(cite:" not in str(result["answer"])
    assert collector.finalize().cited_chunk_ids == ["arb:title", "gp:title"]


@pytest.mark.asyncio
async def test_pipeline_free_text_used_pages_can_extend_beyond_inline_citations(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield (
            "1. DIFC Law No. 5 of 2020 - Title: Data Protection Law 2020 (cite: data:title) - "
            "Last updated (consolidated version): 1 July 2024."
        )

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock()
    generator.extract_cited_chunk_ids = MagicMock(return_value=["data:0:0:title"])
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="data:0:0:title", doc_title="Data Protection Law 2020")])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_final_answer = MagicMock(side_effect=lambda answer: answer)

    classifier = MagicMock()
    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    collector = TelemetryCollector(request_id="free-text-used-pages")

    context_chunks = [
        RankedChunk(
            chunk_id="data:0:0:title",
            doc_id="data",
            doc_title="Data Protection Law 2020",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law may be cited as the "Data Protection Law 2020".',
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="data:1:0:update",
            doc_id="data",
            doc_title="Data Protection Law 2020",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text="Consolidated Version (1 July 2024).",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    await builder._generate(
        {
            "query": "What is the title of DIFC Law No. 5 of 2020 and when was its consolidated version last updated?",
            "request_id": "free-text-used-pages",
            "question_id": "free-text-used-pages",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": context_chunks,
            "model": "gpt-4.1",
            "max_tokens": 128,
            "complexity": QueryComplexity.SIMPLE,
        }
    )

    payload = collector.finalize()
    assert payload.cited_chunk_ids == ["data:0:0:title"]
    assert payload.used_page_ids == ["data_1", "data_2"]


def test_support_question_refs_strips_how_do_lead_tokens() -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    refs = RAGPipelineBuilder._support_question_refs(
        "How do the Limited Liability Partnership Law and the Non Profit Incorporated Organisations Law define their administration?"
    )

    assert refs == [
        "Limited Liability Partnership Law",
        "Non Profit Incorporated Organisations Law",
    ]


@pytest.mark.asyncio
async def test_generate_adds_ruler_enactment_relative_commencement_hint(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    captured: dict[str, object] = {}

    async def _gen_stream(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield "1. Employment Law (cite: c0)"

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Employment Law (cite: c0)", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Employment Law")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="enactment-hint")

    await builder._generate(
        {
            "query": "Which laws were made by the Ruler of Dubai and their commencement date is specified in an Enactment Notice?",
            "request_id": "enactment-hint",
            "question_id": "enactment-hint",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(2),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    prompt_hint = str(captured["prompt_hint"])
    assert "90 days after enactment" in prompt_hint
    assert "relative period" in prompt_hint


@pytest.mark.asyncio
async def test_generate_adds_citation_title_hint_for_registrar_enumeration(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    captured: dict[str, object] = {}

    async def _gen_stream(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield "1. Limited Partnership Law 2006 (cite: c0, c1)"

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Limited Partnership Law 2006 (cite: c0, c1)", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Limited Partnership Law 2006")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="citation-title-hint")

    await builder._generate(
        {
            "query": "Which laws are administered by the Registrar and what are their respective citation titles?",
            "request_id": "citation-title-hint",
            "question_id": "citation-title-hint",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    prompt_hint = str(captured["prompt_hint"])
    assert "exact citation title as written in the source" in prompt_hint
    assert "Do not paraphrase" in prompt_hint
    assert "cite both blocks" in prompt_hint
    assert "minimum number of citations needed" in prompt_hint
    assert "one separate numbered item per matching law" in prompt_hint


@pytest.mark.asyncio
async def test_generate_adds_structural_common_elements_hint_and_strips_trailing_negative(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    captured: dict[str, object] = {}
    context_chunks = [
        RankedChunk(
            chunk_id="c0",
            doc_id="operating",
            doc_title="Operating Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Schedule 1 contains interpretative provisions and a list of defined terms.",
            retrieval_score=0.9,
            rerank_score=0.9,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="c1",
            doc_id="trust",
            doc_title="Trust Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Schedule 1 contains interpretative provisions and a list of defined terms.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="",
        ),
    ]

    async def _gen_stream(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield (
            "Schedule 1 in both laws contains interpretative provisions and a list of defined terms "
            "(cite: c0, c1). There is no information on this question."
        )

    generator.generate_stream = _gen_stream
    async def _generate(*args, **kwargs):
        del args
        captured.update(kwargs)
        return (
            "Schedule 1 in both laws contains interpretative provisions and a list of defined terms "
            "(cite: c0, c1). There is no information on this question.",
            [],
        )

    generator.generate = AsyncMock(side_effect=_generate)
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Operating Law 2018")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)

    def _strip(answer: str) -> str:
        return answer.replace(" There is no information on this question.", "")

    generator.strip_negative_subclaims = MagicMock(side_effect=_strip)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="common-elements-hint")

    result = await builder._generate(
        {
            "query": "What are the common elements found in Schedule 1 of the Operating Law 2018 and the Trust Law 2018?",
            "request_id": "common-elements-hint",
            "question_id": "common-elements-hint",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": context_chunks,
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    prompt_hint = str(captured["prompt_hint"])
    assert "without IRAC" in prompt_hint
    assert "if you cannot cite every referenced document for an element, omit that element" in prompt_hint
    assert "Output ONLY a numbered list of common elements" in prompt_hint
    assert "for example (cite: id_a, id_b, id_c)" in prompt_hint
    assert "merge closely related interpretative rules into one item" in prompt_hint
    assert "same item" in prompt_hint
    assert "structural overlap counts as a valid common element" in prompt_hint
    assert "do not infer those sub-items as common from the other documents" in prompt_hint
    assert result["answer"] == "Schedule 1 in both laws contains interpretative provisions and a list of defined terms (cite: c0, c1)."


@pytest.mark.asyncio
async def test_generate_adds_multi_criteria_year_dual_citation_hint(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    captured: dict[str, object] = {}

    async def _gen_stream(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield "1. Foundations Law 2018 (cite: c0, c1)"

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Foundations Law 2018 (cite: c0, c1)", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Foundations Law 2018")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="year-hint")

    await builder._generate(
        {
            "query": "Which laws, enacted in 2018, include provisions relating to the application of the Arbitration Law in their Schedule 2?",
            "request_id": "year-hint",
            "question_id": "year-hint",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    prompt_hint = str(captured["prompt_hint"])
    assert "Use the exact year shown" in prompt_hint
    assert "cite both blocks in that same item" in prompt_hint
    assert "Each numbered item should mainly give the law title itself" in prompt_hint


@pytest.mark.asyncio
async def test_generate_adds_named_title_refs_hint_for_broad_enumeration(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    captured: dict[str, object] = {}

    async def _gen_stream(*args, **kwargs):
        del args
        captured.update(kwargs)
        yield "1. Incorporated Cell Company (ICC) Regulations (cite: c0)"

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Incorporated Cell Company (ICC) Regulations (cite: c0)", []))
    generator.extract_citations = MagicMock(
        return_value=[Citation(chunk_id="c0", doc_title="Incorporated Cell Company (ICC) Regulations")]
    )
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="named-title-refs-hint")

    await builder._generate(
        {
            "query": "Which laws explicitly mention the Companies Law 2018 and the Insolvency Law 2009 in their regulations concerning company structures?",
            "request_id": "named-title-refs-hint",
            "question_id": "named-title-refs-hint",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    prompt_hint = str(captured["prompt_hint"])
    assert "Companies Law 2018" in prompt_hint
    assert "Insolvency Law 2009" in prompt_hint
    assert "explicitly mentions every named law reference above" in prompt_hint


@pytest.mark.asyncio
async def test_generate_named_title_refs_query_skips_titles_only_cleanup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "1. Incorporated Cell Company (ICC) Regulations mention both the Companies Law 2018 and the Insolvency Law 2009 (cite: c0)."

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Incorporated Cell Company (ICC) Regulations (cite: c0).", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="ICC Regulations")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_broad_enumeration_titles_only = MagicMock(side_effect=lambda answer, **_kwargs: "BROKEN")
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="named-title-cleanup")

    result = await builder._generate(
        {
            "query": "Which laws explicitly mention the Companies Law 2018 and the Insolvency Law 2009 in their regulations concerning company structures?",
            "request_id": "named-title-cleanup",
            "question_id": "named-title-cleanup",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    generator.cleanup_broad_enumeration_titles_only.assert_not_called()
    assert "Incorporated Cell Company (ICC) Regulations" in result["answer"]


@pytest.mark.asyncio
async def test_generate_named_title_refs_query_runs_named_ref_cleanup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "1. Incorporated Cell Company (ICC) Regulations mention both the Companies Law 2018 and the Insolvency Law 2009 (cite: c0)."

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Incorporated Cell Company (ICC) Regulations (cite: c0).", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="ICC Regulations")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_broad_enumeration_titles_only = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_named_ref_enumeration_items = MagicMock(side_effect=lambda answer, **_kwargs: "1. Investment Companies (IC) Regulations (cite: c1)")
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="named-ref-cleanup")

    result = await builder._generate(
        {
            "query": "Which laws explicitly mention the Companies Law 2018 and the Insolvency Law 2009 in their regulations concerning company structures?",
            "request_id": "named-ref-cleanup",
            "question_id": "named-ref-cleanup",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    generator.cleanup_named_ref_enumeration_items.assert_called_once()
    assert result["answer"] == "1. Investment Companies (IC) Regulations (cite: c1)"


@pytest.mark.asyncio
async def test_generate_ruler_enactment_query_runs_cleanup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "1. Wrong Law (cite: c0)."

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Wrong Law (cite: c0).", []))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c1", doc_title="Law of Damages and Remedies 2005")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_broad_enumeration_titles_only = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_ruler_enactment_enumeration_items = MagicMock(
        side_effect=lambda answer, **_kwargs: "1. Law of Damages and Remedies 2005 (cite: c1)"
    )
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="ruler-enactment-cleanup")

    result = await builder._generate(
        {
            "query": "Which laws mention the 'Ruler of Dubai' as the legislative authority and also specify that the law comes into force on the date specified in the Enactment Notice?",
            "request_id": "ruler-enactment-cleanup",
            "question_id": "ruler-enactment-cleanup",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    generator.cleanup_ruler_enactment_enumeration_items.assert_called_once()
    assert result["answer"] == "1. Law of Damages and Remedies 2005 (cite: c1)"


@pytest.mark.asyncio
async def test_generate_citation_title_enumeration_runs_registrar_cleanup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield "1. Citation title (cite: c0, c1)."

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(return_value=("1. Citation title (cite: c0, c1).", []))
    generator.extract_citations = MagicMock(
        return_value=[Citation(chunk_id="c0", doc_title="General Partnership Law 2004")]
    )
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_broad_enumeration_titles_only = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_registrar_enumeration_items = MagicMock(
        side_effect=lambda answer, **_kwargs: "1. General Partnership Law 2004 (cite: c0)"
    )
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="citation-title-cleanup")

    result = await builder._generate(
        {
            "query": "Which laws are administered by the Registrar and what are their respective citation titles?",
            "request_id": "citation-title-cleanup",
            "question_id": "citation-title-cleanup",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(3),
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    generator.cleanup_registrar_enumeration_items.assert_called_once()
    assert result["answer"] == "1. General Partnership Law 2004 (cite: c0)"


@pytest.mark.asyncio
async def test_generate_common_elements_uses_streaming_path_even_with_multiple_title_refs(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    captured: dict[str, object] = {"stream_calls": 0, "generate_calls": 0}

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        captured["stream_calls"] = int(captured["stream_calls"]) + 1
        yield "1. Shared interpretative provisions (cite: c0, c1, c2)."

    async def _gen(*args, **kwargs):
        del args, kwargs
        captured["generate_calls"] = int(captured["generate_calls"]) + 1
        return ("1. Shared interpretative provisions (cite: c0, c1, c2).", [])

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock(side_effect=_gen)
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Operating Law 2018")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1", "c2"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="common-elements-stream")
    events: list[dict[str, object]] = []

    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await builder._generate(
            {
                "query": "What are the common elements found in the interpretation sections of the Operating Law 2018, Trust Law 2018, and Common Reporting Standard Law 2018?",
                "request_id": "common-elements-stream",
                "question_id": "common-elements-stream",
                "collector": collector,
                "answer_type": "free_text",
                "context_chunks": _make_ranked(3),
                "model": "gpt-4o",
                "max_tokens": 300,
                "complexity": QueryComplexity.COMPLEX,
            }
        )

    assert captured["stream_calls"] == 1
    assert captured["generate_calls"] == 0
    assert result["streamed"] is True
    assert any(event.get("type") == "token" for event in events)
    assert any(event.get("type") == "answer_final" for event in events)


@pytest.mark.asyncio
async def test_generate_answer_final_uses_canonical_common_elements_cleanup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield (
            "1. Schedule 1 of the Operating Law 2018 contains interpretative provisions (cite: c0). "
            "2. Schedule 1 of the Trust Law 2018 contains interpretative provisions (cite: c1)."
        )

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock()
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Operating Law 2018")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_common_elements_canonical_answer = MagicMock(
        return_value=(
            "1. Schedule 1 contains interpretative provisions which apply to the Law. (cite: c0, c1)\n"
            "2. Schedule 1 contains a list of defined terms used in the Law. (cite: c0, c1)"
        )
    )
    generator.cleanup_named_commencement_answer = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_account_effective_dates_answer = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_final_answer = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="canonical-common-elements")
    events: list[dict[str, object]] = []

    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await builder._generate(
            {
                "query": "What are the common elements found in Schedule 1 of the Operating Law 2018 and the Trust Law 2018?",
                "request_id": "canonical-common-elements",
                "question_id": "canonical-common-elements",
                "collector": collector,
                "answer_type": "free_text",
                "context_chunks": _make_ranked(2),
                "doc_refs": ["Operating Law 2018", "Trust Law 2018"],
                "model": "gpt-4o",
                "max_tokens": 300,
                "complexity": QueryComplexity.COMPLEX,
            }
        )

    assert result["answer"].startswith("1. Schedule 1 contains interpretative provisions")
    assert any(
        event.get("type") == "answer_final"
        and "Schedule 1 contains a list of defined terms used in the Law." in str(event.get("text"))
        for event in events
    )
    generator.cleanup_common_elements_canonical_answer.assert_called_once()


@pytest.mark.asyncio
async def test_generate_answer_final_uses_account_effective_dates_cleanup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()

    async def _gen_stream(*args, **kwargs):
        del args, kwargs
        yield (
            "1. DIFC References to legislation in the Law: Comes into force on the date specified in the Enactment Notice "
            "(cite: c0). 2. Common Reporting Standard Law DIFC Law No. 2 of 2018: Shall come into force on the 5th "
            "business day after enactment (cite: c1)."
        )

    generator.generate_stream = _gen_stream
    generator.generate = AsyncMock()
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Common Reporting Standard Law 2018")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_common_elements_canonical_answer = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_named_commencement_answer = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_account_effective_dates_answer = MagicMock(
        return_value=(
            "1. Pre-existing Accounts: The effective date is 31 December, 2016 (cite: c0)\n"
            "2. New Accounts: The effective date is 1 January, 2017 (cite: c0)\n"
            "3. Common Reporting Standard Law 2018: The date of enactment is 14th day of March 2018 (cite: c1)"
        )
    )
    generator.cleanup_final_answer = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="canonical-effective-dates")
    events: list[dict[str, object]] = []

    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await builder._generate(
            {
                "query": (
                    "What are the effective dates for pre-existing and new accounts under the Common Reporting "
                    "Standard Law 2018, and what is the date of its enactment?"
                ),
                "request_id": "canonical-effective-dates",
                "question_id": "canonical-effective-dates",
                "collector": collector,
                "answer_type": "free_text",
                "context_chunks": _make_ranked(2),
                "doc_refs": ["Common Reporting Standard Law 2018"],
                "model": "gpt-4o",
                "max_tokens": 300,
                "complexity": QueryComplexity.COMPLEX,
            }
        )

    assert result["answer"].startswith("1. Pre-existing Accounts")
    assert any(
        event.get("type") == "answer_final"
        and "The date of enactment is 14th day of March 2018" in str(event.get("text"))
        for event in events
    )
    generator.cleanup_account_effective_dates_answer.assert_called_once()


@pytest.mark.asyncio
async def test_generate_named_administration_query_uses_structured_bypass_and_sets_debug_telemetry(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    generator.generate_stream = AsyncMock()
    generator.generate = AsyncMock()
    generator.build_structured_free_text_answer = MagicMock(
        return_value=(
            "1. Limited Liability Partnership Law 2004: This Law and any legislation made for the purpose of this Law "
            "is administered by the Registrar (cite: c0)\n"
            "2. Non Profit Incorporated Organisations Law: This Law and any legislation made for the purposes of this Law "
            "are administered by the Registrar (cite: c1)"
        )
    )
    generator.get_context_debug_stats = MagicMock(return_value=(4, 1600))
    generator.extract_citations = MagicMock(return_value=[Citation(chunk_id="c0", doc_title="Limited Liability Partnership Law 2004")])
    generator.extract_cited_chunk_ids = MagicMock(return_value=["c0", "c1"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_final_answer = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="named-admin-stream")
    events: list[dict[str, object]] = []

    with patch("rag_challenge.core.pipeline.get_stream_writer", return_value=lambda event: events.append(event)):
        result = await builder._generate(
            {
                "query": "How do the Limited Liability Partnership Law and the Non Profit Incorporated Organisations Law define their administration?",
                "request_id": "named-admin-stream",
                "question_id": "named-admin-stream",
                "collector": collector,
                "answer_type": "free_text",
                "context_chunks": _make_ranked(3),
                "doc_refs": ["Limited Liability Partnership Law", "Non Profit Incorporated Organisations Law"],
                "model": "gpt-4o",
                "max_tokens": 300,
                "complexity": QueryComplexity.COMPLEX,
            }
        )

    payload = collector.finalize()
    generator.generate_stream.assert_not_called()
    generator.generate.assert_not_called()
    assert result["streamed"] is False
    assert payload.generation_mode == "single_shot"
    assert payload.context_chunk_count == 4
    assert payload.context_budget_tokens == 1600
    assert payload.model_llm == "structured-extractor"
    assert any(event.get("type") == "token" for event in events)
    generator.build_structured_free_text_answer.assert_called_once()


@pytest.mark.asyncio
async def test_generate_named_multi_title_lookup_uses_structured_bypass(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    generator.generate_stream = AsyncMock()
    generator.generate = AsyncMock()
    generator.build_structured_free_text_answer = MagicMock(
        return_value=(
            "1. Law No. 5 of 2018 - Title: COMPANIES LAW - Last updated (consolidated version): March 2022 "
            "(cite: companies:0)\n"
            "2. Law No. 4 of 2019 - Title: Intellectual Property Law - Last updated (consolidated version): "
            "March 2022 (cite: ip-law:0)"
        )
    )
    generator.get_context_debug_stats = MagicMock(return_value=(6, 1600))
    generator.extract_citations = MagicMock(
        return_value=[
            Citation(chunk_id="companies:0", doc_title="COMPANIES LAW"),
            Citation(chunk_id="ip-law:0", doc_title="Intellectual Property Law"),
        ]
    )
    generator.extract_cited_chunk_ids = MagicMock(return_value=["companies:0", "ip-law:0"])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_final_answer = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="named-title-structured")

    result = await builder._generate(
        {
            "query": "What is the title of DIFC Law No. 5 of 2018 and DIFC Law No. 4 of 2019, and when were their consolidated versions last updated?",
            "request_id": "named-title-structured",
            "question_id": "named-title-structured",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": _make_ranked(4),
            "doc_refs": ["Law No. 5 of 2018", "Law No. 4 of 2019"],
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    payload = collector.finalize()
    generator.generate_stream.assert_not_called()
    generator.generate.assert_not_called()
    assert result["streamed"] is False
    assert payload.generation_mode == "single_shot"
    assert payload.model_llm == "structured-extractor"


@pytest.mark.asyncio
async def test_rerank_uses_compact_top_n_for_default_broad_enumeration(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=_make_ranked(8))
    generator = MagicMock()
    classifier = MagicMock()

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    collector = TelemetryCollector(request_id="broad-top-n")

    await builder._rerank(
        {
            "query": "Which laws are administered by the Registrar and were enacted in 2004?",
            "collector": collector,
            "answer_type": "free_text",
            "retrieved": _make_retrieved(20),
        }
    )

    assert reranker.rerank.await_args.kwargs["top_n"] == 8


@pytest.mark.asyncio
async def test_rerank_keeps_expanded_top_n_for_recall_sensitive_broad_enumeration(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=_make_ranked(12))
    generator = MagicMock()
    classifier = MagicMock()

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    collector = TelemetryCollector(request_id="broad-top-n-sensitive")

    await builder._rerank(
        {
            "query": "Which laws were amended by DIFC Law No. 2 of 2022?",
            "collector": collector,
            "answer_type": "free_text",
            "retrieved": _make_retrieved(20),
        }
    )

    assert reranker.rerank.await_args.kwargs["top_n"] == 12


@pytest.mark.asyncio
async def test_retrieve_caps_single_doc_ref_for_strict_queries(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(6))

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="strict-doc-ref")

    await builder._retrieve(
        {
            "query": "According to Article 16(1) of the Operating Law 2018, what document must every Registered Person file?",
            "collector": collector,
            "answer_type": "name",
            "doc_refs": ["Operating Law 2018"],
        }
    )

    doc_ref_call = next(call for call in retriever.retrieve.await_args_list if call.kwargs.get("doc_refs"))
    assert doc_ref_call.kwargs["sparse_only"] is True
    assert doc_ref_call.kwargs["top_k"] == 16


@pytest.mark.asyncio
async def test_retrieve_caps_multi_doc_ref_for_strict_queries(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(6))

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="strict-multi-doc-ref")

    await builder._retrieve(
        {
            "query": "Which case was decided earlier: ENF 269/2023 or SCT 169/2025?",
            "collector": collector,
            "answer_type": "name",
            "doc_refs": ["ENF 269/2023", "SCT 169/2025"],
        }
    )

    assert retriever.retrieve.await_count == 2
    assert all(call.kwargs["top_k"] == 12 for call in retriever.retrieve.await_args_list)


@pytest.mark.asyncio
async def test_retrieve_caps_hybrid_candidates_for_strict_queries(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(10))

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="strict-hybrid")

    await builder._retrieve(
        {
            "query": "Under Article 12 of the Real Property Law 2018, what is the term for the office created as a corporation sole?",
            "collector": collector,
            "answer_type": "name",
            "doc_refs": [],
        }
    )

    assert retriever.embed_query.await_count == 1
    assert retriever.retrieve.await_args.kwargs["top_k"] == 20
    assert retriever.retrieve.await_args.kwargs["prefetch_dense"] == 24
    assert retriever.retrieve.await_args.kwargs["prefetch_sparse"] == 24


@pytest.mark.asyncio
async def test_retrieve_caps_title_multi_retrieve_for_common_elements_queries(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(return_value=_make_retrieved(8))

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="common-elements-multi-title")

    await builder._retrieve(
        {
            "query": "What are the common elements found in Schedule 1 of the Operating Law 2018 and the Trust Law 2018?",
            "collector": collector,
            "answer_type": "free_text",
            "doc_refs": [],
        }
    )

    title_calls = [call for call in retriever.retrieve.await_args_list if call.kwargs.get("sparse_only") is True]
    assert title_calls
    assert all(call.kwargs["top_k"] == 12 for call in title_calls)


@pytest.mark.asyncio
async def test_retrieve_targets_missing_named_ref_for_multi_title_lookup(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.embed_query = AsyncMock(return_value=[0.1] * 8)
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="data:anchor",
                    doc_id="data",
                    doc_title="Data Protection Law 2020",
                    section_path="page:1",
                    text='This Law may be cited as the "Data Protection Law 2020".',
                    score=0.95,
                )
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="data:title",
                    doc_id="data",
                    doc_title="Data Protection Law 2020",
                    section_path="page:1",
                    text='This Law may be cited as the "Data Protection Law 2020".',
                    score=0.91,
                )
            ],
            [],
            [
                _make_retrieved_chunk(
                    chunk_id="employment:title",
                    doc_id="employment",
                    doc_title="Employment Law 2019",
                    section_path="page:1",
                    text='This Law may be cited as the "Employment Law 2019".',
                    score=0.9,
                )
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="missing-named-ref")

    await builder._retrieve(
        {
            "query": "What are the citation titles of the Data Protection Law 2020 and the Employment Law 2019?",
            "collector": collector,
            "answer_type": "free_text",
            "doc_refs": [],
        }
    )

    assert retriever.retrieve.await_count == 4
    targeted_call = retriever.retrieve.await_args_list[-1]
    assert "Employment Law 2019" in targeted_call.args[0]
    assert "may be cited as" in targeted_call.args[0]
    assert targeted_call.kwargs["sparse_only"] is True
    assert targeted_call.kwargs["top_k"] == 12


@pytest.mark.asyncio
async def test_retrieve_targets_restriction_effectiveness_boolean_clause(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="pp:generic",
                    doc_id="pp-law",
                    doc_title="PERSONAL PROPERTY LAW",
                    section_path="page:3",
                    text='This Law may be cited as the "Personal Property Law 2005".',
                    score=0.93,
                    doc_summary="Personal Property Law, DIFC Law No. 9 of 2005.",
                )
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="pp:article23",
                    doc_id="pp-law",
                    doc_title="PERSONAL PROPERTY LAW",
                    section_path="page:10",
                    text=(
                        'This Law may be cited as the "Personal Property Law 2005". '
                        "Article 23. A restriction on transfer of a security is ineffective against any person "
                        "other than a person who had actual knowledge of the restriction. "
                        "If the security is uncertificated and the registered owner has been notified of the restriction, "
                        "the restriction remains effective against a person with actual knowledge."
                    ),
                    score=0.88,
                    doc_summary="Personal Property Law, DIFC Law No. 9 of 2005.",
                )
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="restriction-effectiveness")

    result = await builder._retrieve(
        {
            "query": (
                "Under Article 23 of the Personal Property Law 2005, is a restriction on transfer of a security "
                "imposed by the issuer effective against a person who had actual knowledge of such third party "
                "property interest, if the security is uncertificated and the registered owner has been notified "
                "of the restriction?"
            ),
            "collector": collector,
            "answer_type": "boolean",
            "doc_refs": ["Personal Property Law 2005"],
        }
    )

    assert retriever.retrieve.await_count == 2
    targeted_call = retriever.retrieve.await_args_list[-1]
    assert "article 23" in targeted_call.args[0].lower()
    assert "actual knowledge" in targeted_call.args[0].lower()
    assert targeted_call.kwargs["sparse_only"] is True
    assert "pp:article23" in {chunk.chunk_id for chunk in result["retrieved"]}
    assert "pp:article23" in result["must_include_chunk_ids"]


@pytest.mark.asyncio
async def test_retrieve_targets_article_anchored_strict_doc_ref_clause(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="employment:title",
                    doc_id="employment",
                    doc_title="EMPLOYMENT LAW",
                    section_path="page:4",
                    text='This Employment Law 2019 may be cited as the "Employment Law 2019".',
                    score=0.96,
                    doc_summary="Employment Law 2019, DIFC Law No. 2 of 2019.",
                )
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="employment:article14",
                    doc_id="employment",
                    doc_title="EMPLOYMENT LAW",
                    section_path="page:7",
                    text=(
                        "Article 14. Written Employment Contract. "
                        "An Employer shall provide an Employee with a written Employment Contract "
                        "within seven (7) days of the commencement of the Employee's employment."
                    ),
                    score=0.88,
                    doc_summary="Employment Law 2019, DIFC Law No. 2 of 2019.",
                )
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="article-anchored-strict")

    result = await builder._retrieve(
        {
            "query": (
                "Under Article 14(1) of the Employment Law 2019, how many days does an Employer have to "
                "provide an Employee with a written Employment Contract after the commencement of employment?"
            ),
            "collector": collector,
            "answer_type": "number",
            "doc_refs": ["Employment Law 2019"],
        }
    )

    assert retriever.retrieve.await_count == 2
    targeted_call = retriever.retrieve.await_args_list[-1]
    assert "Employment Law 2019" in targeted_call.args[0]
    assert "Article 14(1)" in targeted_call.args[0]
    assert targeted_call.kwargs["doc_refs"] is None
    assert targeted_call.kwargs["sparse_only"] is True
    assert "employment:article14" in {chunk.chunk_id for chunk in result["retrieved"]}
    assert "employment:article14" in result["must_include_chunk_ids"]


@pytest.mark.asyncio
async def test_retrieve_multi_ref_boolean_judge_compare_keeps_judge_title_pages(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="ca:page4",
                    doc_id="ca-004",
                    doc_title="CA 004/2025 Example v Example",
                    section_path="page:4",
                    text="Reasons about jurisdiction without any judge title markers.",
                    score=0.97,
                ),
                _make_retrieved_chunk(
                    chunk_id="ca:page1",
                    doc_id="ca-004",
                    doc_title="CA 004/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. CHIEF JUSTICE WAYNE MARTIN",
                    score=0.82,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="arb:page2",
                    doc_id="arb-034",
                    doc_title="ARB 034/2025 Example v Example",
                    section_path="page:2",
                    text="Return Date hearing details.",
                    score=0.95,
                ),
                _make_retrieved_chunk(
                    chunk_id="arb:page1",
                    doc_id="arb-034",
                    doc_title="ARB 034/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI",
                    score=0.81,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="ca:page1",
                    doc_id="ca-004",
                    doc_title="CA 004/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. CHIEF JUSTICE WAYNE MARTIN",
                    score=0.83,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="arb:page1",
                    doc_id="arb-034",
                    doc_title="ARB 034/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI",
                    score=0.82,
                ),
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="judges-in-common")

    result = await builder._retrieve(
        {
            "query": "Did cases CA 004/2025 and ARB 034/2025 have any judges in common?",
            "collector": collector,
            "answer_type": "boolean",
            "doc_refs": ["CA 004/2025", "ARB 034/2025"],
        }
    )

    assert retriever.retrieve.await_count == 4
    assert "ca:page1" in result["must_include_chunk_ids"]
    assert "arb:page1" in result["must_include_chunk_ids"]
    assert {"ca:page1", "arb:page1"}.issubset({chunk.chunk_id for chunk in result["retrieved"]})


@pytest.mark.asyncio
async def test_retrieve_multi_ref_boolean_presided_over_both_keeps_judge_title_pages(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="arb:page5",
                    doc_id="arb-034",
                    doc_title="ARB 034/2025 Example v Example",
                    section_path="page:5",
                    text="Later procedural page without judge title markers.",
                    score=0.97,
                ),
                _make_retrieved_chunk(
                    chunk_id="arb:page1",
                    doc_id="arb-034",
                    doc_title="ARB 034/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI",
                    score=0.82,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="cfi:page1",
                    doc_id="cfi-067",
                    doc_title="CFI 067/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. JUSTICE MICHAEL BLACK KC",
                    score=0.81,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="arb:page1",
                    doc_id="arb-034",
                    doc_title="ARB 034/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI",
                    score=0.83,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="cfi:page1",
                    doc_id="cfi-067",
                    doc_title="CFI 067/2025 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. JUSTICE MICHAEL BLACK KC",
                    score=0.81,
                ),
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="judges-presided-both")

    result = await builder._retrieve(
        {
            "query": "Is there a judge who presided over both case ARB 034/2025 and case CFI 067/2025?",
            "collector": collector,
            "answer_type": "boolean",
            "doc_refs": ["ARB 034/2025", "CFI 067/2025"],
        }
    )

    assert retriever.retrieve.await_count == 4
    assert "arb:page1" in result["must_include_chunk_ids"]
    assert "cfi:page1" in result["must_include_chunk_ids"]
    assert {"arb:page1", "cfi:page1"}.issubset({chunk.chunk_id for chunk in result["retrieved"]})


@pytest.mark.asyncio
async def test_retrieve_multi_ref_boolean_participated_in_both_prefers_page_one_judge_pages(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="ca:page8",
                    doc_id="ca-005",
                    doc_title="CA 005/2025 Example v Example",
                    section_path="page:8",
                    text="Later procedural page referring to Justice Wayne Martin in the body text only.",
                    score=0.98,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="tcd:page3",
                    doc_id="tcd-001-appeal",
                    doc_title="TCD 001/2024 Example v Example",
                    section_path="page:3",
                    text="Later page summarising that Chief Justice Wayne Martin heard the matter.",
                    score=0.97,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="ca:page1",
                    doc_id="ca-005",
                    doc_title="CA 005/2025 Example v Example",
                    section_path="page:1",
                    text=(
                        "Claim No: CA 005/2025\n"
                        "hearing held before H.E. Chief Justice Wayne Martin, "
                        "H.E. Justice Rene Le Miere and H.E. Justice Sir Peter Gross."
                    ),
                    score=0.82,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="tcd:page1",
                    doc_id="tcd-001-appeal",
                    doc_title="TCD 001/2024 Example v Example",
                    section_path="page:1",
                    text="ORDER WITH REASONS OF H.E. CHIEF JUSTICE WAYNE MARTIN.",
                    score=0.81,
                ),
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="judges-participated-both")

    result = await builder._retrieve(
        {
            "query": "Considering all documents across case CA 005/2025 and case TCD 001/2024, was there any judge who participated in both cases?",
            "collector": collector,
            "answer_type": "boolean",
            "doc_refs": ["CA 005/2025", "TCD 001/2024"],
        }
    )

    assert retriever.retrieve.await_count == 4
    assert "ca:page1" in result["must_include_chunk_ids"]
    assert "tcd:page1" in result["must_include_chunk_ids"]
    assert {"ca:page1", "tcd:page1"}.issubset({chunk.chunk_id for chunk in result["retrieved"]})


@pytest.mark.asyncio
async def test_retrieve_single_ref_case_outcome_query_keeps_page_one_order_seed(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="cfi067:page2",
                    doc_id="cfi067",
                    doc_title="CFI 067/2025 Example v Example",
                    section_path="page:2",
                    text="This Order concerns the Defendant's Application for immediate judgment and/or strike out, which was dismissed by Order of H.E. Justice Shamlan Al Sawalehi.",
                    score=0.93,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="cfi067:page1",
                    doc_id="cfi067",
                    doc_title="CFI 067/2025 Example v Example",
                    section_path="page:1",
                    text=(
                        "ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI\n"
                        "IT IS HEREBY ORDERED THAT:\n"
                        "1. The Defendant's Application for immediate judgment and/or strike out is dismissed."
                    ),
                    score=0.82,
                ),
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="case-outcome-page-one")

    result = await builder._retrieve(
        {
            "query": "What was the result of the application heard in case CFI 067/2025?",
            "collector": collector,
            "answer_type": "free_text",
            "doc_refs": ["CFI 067/2025"],
        }
    )

    assert retriever.retrieve.await_count == 2
    assert "cfi067:page1" in result["must_include_chunk_ids"]
    assert {"cfi067:page1", "cfi067:page2"}.issubset({chunk.chunk_id for chunk in result["retrieved"]})


@pytest.mark.asyncio
async def test_retrieve_multi_ref_name_issue_date_keeps_issue_pages(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="sct169:page5",
                    doc_id="sct-169",
                    doc_title="SCT 169/2025 Obasi v Oreana",
                    section_path="page:5",
                    text="Later procedural page without issue-date field.",
                    score=0.97,
                ),
                _make_retrieved_chunk(
                    chunk_id="sct169:page2",
                    doc_id="sct-169",
                    doc_title="SCT 169/2025 Obasi v Oreana",
                    section_path="page:2",
                    text="Issued by: Delvin Sumo. Date of Issue: 24 December 2025.",
                    score=0.81,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="sct295:page7",
                    doc_id="sct-295",
                    doc_title="SCT 295/2025 Olexa v Odon",
                    section_path="page:7",
                    text="Later page mentioning the case title only.",
                    score=0.96,
                ),
                _make_retrieved_chunk(
                    chunk_id="sct295:page2",
                    doc_id="sct-295",
                    doc_title="SCT 295/2025 Olexa v Odon",
                    section_path="page:2",
                    text="Issued by: Delvin Sumo. Date of Issue: 10 December 2025.",
                    score=0.8,
                ),
            ],
        ]
    )

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=MagicMock(),
        generator=MagicMock(),
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="issue-date-pair")

    result = await builder._retrieve(
        {
            "query": "Which case has an earlier Date of Issue: SCT 169/2025 or SCT 295/2025?",
            "collector": collector,
            "answer_type": "name",
            "doc_refs": ["SCT 169/2025", "SCT 295/2025"],
        }
    )

    assert retriever.retrieve.await_count == 2
    assert "sct169:page2" in result["must_include_chunk_ids"]
    assert "sct295:page2" in result["must_include_chunk_ids"]
    assert {"sct169:page2", "sct295:page2"}.issubset({chunk.chunk_id for chunk in result["retrieved"]})


def test_ensure_named_commencement_context_keeps_title_and_commencement_chunks(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    reranked = [
        RankedChunk(
            chunk_id="data:title",
            doc_id="data",
            doc_title=".",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Law may be cited as the "Data Protection Law 2020".',
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text='This Employment Law 2019 may be cited as the "Employment Law 2019".',
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="data:title",
            doc_id="data",
            doc_title=".",
            section_path="page:4",
            text='This Law may be cited as the "Data Protection Law 2020".',
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="data:comm",
            doc_id="data",
            doc_title=".",
            section_path="page:4",
            text="4. Commencement This Law comes into force on 1 July 2020.",
            score=0.93,
        ),
        _make_retrieved_chunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:4",
            text='This Employment Law 2019 may be cited as the "Employment Law 2019".',
            score=0.94,
        ),
        _make_retrieved_chunk(
            chunk_id="employment:comm",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:5",
            text="6. Commencement This Law comes into force on the date ninety (90) days following the date specified in the Enactment Notice.",
            score=0.90,
        ),
    ]

    selected = RAGPipelineBuilder._ensure_named_commencement_context(
        query="What is the commencement date for the Data Protection Law 2020 and the Employment Law 2019?",
        doc_refs=["Data Protection Law 2020", "Employment Law 2019"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert [chunk.chunk_id for chunk in selected] == [
        "data:title",
        "data:comm",
        "employment:title",
        "employment:comm",
    ]


async def test_generate_free_text_empty_context_returns_unanswerable_without_llm(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    generator = MagicMock()
    generator.generate = AsyncMock()
    generator.generate_stream = AsyncMock()
    generator.extract_citations = MagicMock(return_value=[])
    generator.extract_cited_chunk_ids = MagicMock(return_value=[])
    generator.sanitize_citations = MagicMock(side_effect=lambda answer, _ctx: answer)
    generator.cleanup_list_answer_preamble = MagicMock(side_effect=lambda answer: answer)
    generator.cleanup_numbered_list_items = MagicMock(side_effect=lambda answer, **_kwargs: answer)
    generator.cleanup_list_answer_postamble = MagicMock(side_effect=lambda answer: answer)
    generator.strip_negative_subclaims = MagicMock(side_effect=lambda answer: answer)

    builder = RAGPipelineBuilder(
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=generator,
        classifier=MagicMock(),
    )
    collector = TelemetryCollector(request_id="free-text-empty-context")

    result = await builder._generate(
        {
            "query": "What is the prescribed penalty for an offense against the Strata Title Law under the Strata Title Regulations, and what is the penalty for using leased premises for an illegal purpose under the Leasing Regulations?",
            "request_id": "free-text-empty-context",
            "question_id": "free-text-empty-context",
            "collector": collector,
            "answer_type": "free_text",
            "context_chunks": [],
            "model": "gpt-4o",
            "max_tokens": 300,
            "complexity": QueryComplexity.COMPLEX,
        }
    )

    assert result["answer"] == "There is no information on this question."
    generator.generate.assert_not_called()
    generator.generate_stream.assert_not_called()


def test_registrar_context_keeps_only_self_administered_laws(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="General Partnership Law 2004",
            section_path="page:1",
            text="General Partnership Law 2004 made by the Ruler of Dubai.",
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="gp:admin",
            doc_id="gp",
            doc_title="General Partnership Law 2004",
            section_path="page:21",
            text="Administration of this Law. The Registrar shall administer this Law and the Regulations made under it.",
            score=0.94,
        ),
        _make_retrieved_chunk(
            chunk_id="contract:title",
            doc_id="contract",
            doc_title="Contract Law 2004",
            section_path="page:1",
            text="Contract Law 2004 made by the Ruler of Dubai.",
            score=0.93,
        ),
        _make_retrieved_chunk(
            chunk_id="operating:def",
            doc_id="operating",
            doc_title="Operating Law",
            section_path="page:41",
            text="Legislation administered by the Registrar means the Prescribed Laws.",
            score=0.92,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[1]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[3]),
    ]

    filtered = RAGPipelineBuilder._ensure_self_registrar_context(
        reranked=reranked,
        retrieved=retrieved,
        top_n=6,
    )

    assert [chunk.doc_id for chunk in filtered] == ["gp", "gp"]


def test_named_penalty_context_promotes_clause_chunk_for_each_ref(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="strata:toc",
            doc_id="strata",
            doc_title="STRATA TITLE REGULATIONS",
            section_path="page:2",
            text="3 PENALTY FOR OFFENCES AGAINST THE LAW",
            score=0.98,
        ),
        _make_retrieved_chunk(
            chunk_id="strata:penalty",
            doc_id="strata",
            doc_title="STRATA TITLE REGULATIONS",
            section_path="page:3",
            text="3. PENALTY FOR OFFENCES AGAINST THE LAW The penalty for an offence against the Law is one thousand dollars (US$ 1,000).",
            score=0.92,
        ),
        _make_retrieved_chunk(
            chunk_id="leasing:penalty",
            doc_id="leasing",
            doc_title="LEASING REGULATIONS",
            section_path="page:5",
            text="Use of the Leased Premises for any purpose that is illegal 10,000",
            score=0.97,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
    ]

    selected = RAGPipelineBuilder._ensure_named_penalty_context(
        query=(
            "What is the prescribed penalty for an offense against the Strata Title Law under the Strata Title "
            "Regulations, and what is the penalty for using leased premises for an illegal purpose under the "
            "Leasing Regulations?"
        ),
        doc_refs=["Strata Title Regulations", "Leasing Regulations"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert {chunk.chunk_id for chunk in selected} >= {"strata:penalty", "leasing:penalty"}


def test_named_administration_context_promotes_clause_chunk_for_each_ref(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="leasing:title",
            doc_id="leasing",
            doc_title="LEASING LAW",
            section_path="page:1",
            text='This Law No. 1 of 2020 may be cited as the "Leasing Law 2020".',
            score=0.98,
        ),
        _make_retrieved_chunk(
            chunk_id="leasing:admin",
            doc_id="leasing",
            doc_title="LEASING LAW",
            section_path="page:4",
            text="This Law and any Regulations made under it shall be administered by the DIFCA.",
            score=0.91,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:title",
            doc_id="trust",
            doc_title="TRUST LAW",
            section_path="page:1",
            text='This Law may be cited as the "Trust Law 2018".',
            score=0.97,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:admin",
            doc_id="trust",
            doc_title="TRUST LAW",
            section_path="page:4",
            text="This Law is administered by the DIFCA.",
            score=0.9,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
    ]

    selected = RAGPipelineBuilder._ensure_named_administration_context(
        query="What entity administers the Leasing Law 2020 and the Trust Law 2018?",
        doc_refs=["Leasing Law 2020", "Trust Law 2018"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert {chunk.chunk_id for chunk in selected} >= {"leasing:admin", "trust:admin"}


def test_boolean_admin_compare_context_recovers_canonical_chunk_outside_capped_pool(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    filler = [
        _make_retrieved_chunk(
            chunk_id=f"ip:filler:{idx}",
            doc_id="ip-law",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path=f"page:{10 + idx}",
            text="Miscellaneous intellectual property provision.",
            score=0.80 - (idx * 0.01),
        )
        for idx in range(10)
    ]
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="ip:admin",
            doc_id="ip-law",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:4",
            text=(
                "INTELLECTUAL PROPERTY LAW DIFC Law No. 4 of 2019. "
                "Administration of this Law. This Law is administered by the DIFC Authority."
            ),
            score=0.99,
            doc_summary="DIFC Law No. 4 of 2019.",
        ),
        _make_retrieved_chunk(
            chunk_id="trust:surrogate",
            doc_id="trust-consolidated",
            doc_title="TRUST LAW",
            section_path="page:4",
            text=(
                "TRUST LAW DIFC Law No. 4 of 2018. "
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            score=0.98,
            doc_summary="Trust Law consolidates DIFC Law No. 4 of 2018 with amendments up to March 2024.",
        ),
        *filler,
        _make_retrieved_chunk(
            chunk_id="trust:canonical",
            doc_id="trust-law",
            doc_title="TRUST LAW",
            section_path="page:5",
            text=(
                "TRUST LAW DIFC Law No. 4 of 2018. "
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            score=0.70,
            doc_summary="Trust Law under DIFC Law No. 4 of 2018.",
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[1]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
    ]

    selected = RAGPipelineBuilder._ensure_boolean_admin_compare_context(
        query=(
            "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
            "that administers the Trust Law No. 4 of 2018?"
        ),
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert [chunk.chunk_id for chunk in selected[:2]] == ["ip:admin", "trust:canonical"]
    assert "trust:surrogate" not in {chunk.chunk_id for chunk in selected[:2]}


def test_missing_named_ref_targets_detects_missing_admin_and_penalty_criteria(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    admin_retrieved = [
        _make_retrieved_chunk(
            chunk_id="leasing:title",
            doc_id="leasing",
            doc_title="LEASING LAW",
            section_path="page:1",
            text='This Law No. 1 of 2020 may be cited as the "Leasing Law 2020".',
            score=0.98,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:admin",
            doc_id="trust",
            doc_title="TRUST LAW",
            section_path="page:4",
            text="This Law is administered by the DIFCA.",
            score=0.9,
        ),
    ]

    penalty_retrieved = [
        _make_retrieved_chunk(
            chunk_id="strata:toc",
            doc_id="strata",
            doc_title="STRATA TITLE REGULATIONS",
            section_path="page:2",
            text="3 PENALTY FOR OFFENCES AGAINST THE LAW",
            score=0.98,
        ),
        _make_retrieved_chunk(
            chunk_id="leasing:penalty",
            doc_id="leasing-regs",
            doc_title="LEASING REGULATIONS",
            section_path="page:5",
            text="Use of the Leased Premises for any purpose that is illegal 10,000",
            score=0.97,
        ),
    ]

    admin_missing = RAGPipelineBuilder._missing_named_ref_targets(
        query="What entity administers the Leasing Law 2020 and the Trust Law 2018?",
        doc_refs=["Leasing Law 2020", "Trust Law 2018"],
        retrieved=admin_retrieved,
    )
    penalty_missing = RAGPipelineBuilder._missing_named_ref_targets(
        query=(
            "What is the prescribed penalty for an offense against the Strata Title Law under the Strata Title "
            "Regulations, and what is the penalty for using leased premises for an illegal purpose under the "
            "Leasing Regulations?"
        ),
        doc_refs=["Strata Title Regulations", "Leasing Regulations"],
        retrieved=penalty_retrieved,
    )

    assert admin_missing == ["Leasing Law 2020"]
    assert penalty_missing == ["Strata Title Regulations"]


def test_common_elements_context_prefers_interpretation_chunk_per_referenced_law(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="operating:intro",
            doc_id="operating-a",
            doc_title="OPERATING LAW",
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions and a list of defined terms.",
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="operating:interp",
            doc_id="operating-a",
            doc_title="OPERATING LAW",
            section_path="page:39",
            text="SCHEDULE 1 INTERPRETATION. Rules of interpretation. A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time.",
            score=0.30,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:intro",
            doc_id="trust-a",
            doc_title="TRUST LAW",
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions and a list of defined terms.",
            score=0.90,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:interp",
            doc_id="trust-a",
            doc_title="TRUST LAW",
            section_path="page:43",
            text="SCHEDULE 1 INTERPRETATION. Rules of interpretation. A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time.",
            score=0.25,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:cont",
            doc_id="crs-a",
            doc_title="COMMON REPORTING",
            section_path="page:15",
            text="Continuation page without the Rules of Interpretation heading.",
            score=0.80,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:interp",
            doc_id="crs-a",
            doc_title="COMMON REPORTING",
            section_path="page:14",
            text="COMMON REPORTING STANDARD LAW. SCHEDULE 1 INTERPRETATION. Rules of interpretation. A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time.",
            score=0.20,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[4]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[5]),
    ]

    filtered = RAGPipelineBuilder._ensure_common_elements_context(
        query="What are the common elements found in the interpretation sections of the Operating Law 2018, Trust Law 2018, and Common Reporting Standard Law 2018?",
        doc_refs=["Operating Law 2018", "Trust Law 2018", "Common Reporting Standard Law 2018"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=6,
    )

    assert [chunk.chunk_id for chunk in filtered[:3]] == ["operating:interp", "trust:interp", "crs:interp"]
    assert "operating:intro" in [chunk.chunk_id for chunk in filtered]


def test_common_elements_context_uses_same_doc_clause_chunk_when_rule_text_lacks_title_tokens(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="operating:intro",
            doc_id="operating-a",
            doc_title="OPERATING LAW",
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions and a list of defined terms.",
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="operating:rule",
            doc_id="operating-a",
            doc_title="",
            section_path="page:40",
            text=(
                "Rules of interpretation. A reference to a statutory provision includes a reference to the "
                "statutory provision as amended or re-enacted from time to time."
            ),
            score=0.30,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:intro",
            doc_id="trust-a",
            doc_title="TRUST LAW",
            section_path="page:4",
            text="Schedule 1 contains interpretative provisions and a list of defined terms.",
            score=0.90,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:rule",
            doc_id="trust-a",
            doc_title="",
            section_path="page:42",
            text=(
                "Rules of interpretation. A reference to a statutory provision includes a reference to the "
                "statutory provision as amended or re-enacted from time to time."
            ),
            score=0.28,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:intro",
            doc_id="crs-a",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:15",
            text="Section 3 defines terms used in the Law.",
            score=0.85,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:rule",
            doc_id="crs-a",
            doc_title="",
            section_path="page:14",
            text=(
                "Rules of interpretation. A reference to a statutory provision includes a reference to the "
                "statutory provision as amended or re-enacted from time to time."
            ),
            score=0.27,
        ),
    ]
    reranked = [RAGPipelineBuilder._raw_to_ranked(chunk) for chunk in retrieved]

    filtered = RAGPipelineBuilder._ensure_common_elements_context(
        query="What are the common elements found in the interpretation sections of the Operating Law 2018, Trust Law 2018, and Common Reporting Standard Law 2018?",
        doc_refs=["Operating Law 2018", "Trust Law 2018", "Common Reporting Standard Law 2018"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=6,
    )

    assert [chunk.chunk_id for chunk in filtered[:3]] == ["operating:rule", "trust:rule", "crs:rule"]


def test_targeted_named_ref_query_boosts_interpretation_common_elements_terms(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    query = "What are the common elements found in the interpretation sections of the Operating Law 2018, Trust Law 2018, and Common Reporting Standard Law 2018?"

    targeted = RAGPipelineBuilder._targeted_named_ref_query(
        query=query,
        ref="Operating Law 2018",
        refs=["Operating Law 2018", "Trust Law 2018", "Common Reporting Standard Law 2018"],
    )

    assert "rules of interpretation" in targeted
    assert "a statutory provision includes a reference" in targeted
    assert "reference to a person includes" in targeted


def test_targeted_named_ref_query_boosts_same_year_title_terms(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    targeted = RAGPipelineBuilder._targeted_named_ref_query(
        query="Was the Employment Law enacted in the same year as the Intellectual Property Law?",
        ref="Employment Law",
        refs=["Employment Law", "Intellectual Property Law"],
    )

    assert "title" in targeted
    assert "law no" in targeted
    assert "enacted" in targeted


def test_targeted_named_ref_query_boosts_account_effective_dates_terms(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    targeted = RAGPipelineBuilder._targeted_named_ref_query(
        query=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard "
            "Law 2018, and what is the date of its enactment?"
        ),
        ref="Common Reporting Standard Law 2018",
        refs=["Common Reporting Standard Law 2018"],
    )

    assert "pre-existing accounts" in targeted
    assert "new accounts" in targeted
    assert "hereby enact" in targeted
    assert "enactment notice" in targeted


def test_targeted_named_ref_query_boosts_admin_title_clause_terms(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    targeted = RAGPipelineBuilder._targeted_named_ref_query(
        query=(
            "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
            "that administers the Trust Law No. 4 of 2018?"
        ),
        ref="Trust Law",
        refs=["Intellectual Property Law", "Trust Law"],
    )

    assert "may be cited as" in targeted
    assert "this law is administered by" in targeted
    assert "shall administer this law" in targeted


def test_is_remuneration_recordkeeping_query_detects_article_16_recordkeeping_phrase() -> None:
    from rag_challenge.core.pipeline import _is_remuneration_recordkeeping_query

    assert _is_remuneration_recordkeeping_query(
        "What does Article 16(1)(c) of the Employment Law 2019 require an Employer to keep records of regarding an Employee's remuneration?"
    )


def test_remuneration_recordkeeping_clause_score_prefers_pay_period_clause(pipeline_builder) -> None:
    best = _make_retrieved_chunk(
        chunk_id="employment:records",
        doc_id="employment",
        doc_title="EMPLOYMENT LAW",
        section_path="page:8",
        text=(
            "16. Payroll records. An Employer shall keep records of the Employee's Remuneration "
            "(gross and net, where applicable), and the applicable Pay Period."
        ),
        score=0.42,
    )
    weak = _make_retrieved_chunk(
        chunk_id="employment:other",
        doc_id="employment",
        doc_title="EMPLOYMENT LAW",
        section_path="page:4",
        text="An Employer must keep records in accordance with this Law.",
        score=0.99,
    )

    assert (
        pipeline_builder._remuneration_recordkeeping_clause_score(best)
        > pipeline_builder._remuneration_recordkeeping_clause_score(weak)
    )


def test_select_targeted_title_seed_chunk_id_prefers_title_year_page_for_same_year_boolean(pipeline_builder) -> None:
    row = [
        _make_retrieved_chunk(
            chunk_id="employment:history",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:26",
            text=(
                "The Employment Law 2019 repeals and replaces the Employment Law 2005 "
                "(DIFC Law No. 4 of 2005)."
            ),
            score=0.97,
        ),
        _make_retrieved_chunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:4",
            text="This Law may be cited as the Employment Law 2019.",
            score=0.42,
        ),
    ]

    seed = pipeline_builder._select_targeted_title_seed_chunk_id(
        query="Was the Employment Law enacted in the same year as the Intellectual Property Law?",
        answer_type="boolean",
        ref="Employment Law",
        chunks=row,
        seed_terms=["title", "law no", "year", "enacted"],
    )

    assert seed == "employment:title"


def test_ensure_boolean_year_compare_context_prefers_title_year_pages(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="employment:body",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:15",
            text="Employment Law body text without enactment year support.",
            score=0.99,
        ),
        _make_retrieved_chunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:4",
            text="This Employment Law 2019 repeals and replaces the Employment Law 2005 (DIFC Law No. 4 of 2005).",
            score=0.25,
        ),
        _make_retrieved_chunk(
            chunk_id="ip:body",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:15",
            text="Intellectual Property Law body text without title support.",
            score=0.98,
        ),
        _make_retrieved_chunk(
            chunk_id="ip:title",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:4",
            text="This Law may be cited as the Intellectual Property Law 2019.",
            score=0.30,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
    ]

    filtered = RAGPipelineBuilder._ensure_boolean_year_compare_context(
        query="Was the Employment Law enacted in the same year as the Intellectual Property Law?",
        reranked=reranked,
        retrieved=retrieved,
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in filtered] == ["employment:title", "ip:title"]


def test_ensure_boolean_admin_compare_context_prefers_one_admin_page_per_ref(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="ip:noise",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW NO. 4 OF 2019",
            section_path="page:31",
            text=(
                "Cross-reference note mentioning the Trust Law No. 4 of 2018. "
                "This is not an administration clause."
            ),
            score=0.99,
        ),
        _make_retrieved_chunk(
            chunk_id="ip:admin",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW NO. 4 OF 2019",
            section_path="page:4",
            text=(
                "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                "The DIFC Authority shall administer this Law."
            ),
            score=0.52,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:admin",
            doc_id="trust-admin",
            doc_title="TRUST LAW NO. 4 OF 2018",
            section_path="page:5",
            text=(
                "Administration of this Law. "
                "The DIFC Authority shall administer this Law."
            ),
            score=0.51,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:title",
            doc_id="trust-title",
            doc_title="TRUST LAW NO. 4 OF 2018",
            section_path="page:4",
            text=(
                "This Law may be cited as the Trust Law No. 4 of 2018. "
                "This Law is enacted on the date specified in the Enactment Notice."
            ),
            score=0.93,
        ),
    ]
    reranked = [RAGPipelineBuilder._raw_to_ranked(retrieved[0])]

    filtered = RAGPipelineBuilder._ensure_boolean_admin_compare_context(
        query=query,
        reranked=reranked,
        retrieved=retrieved,
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in filtered] == ["ip:admin", "trust:admin"]


def test_ensure_page_one_context_matches_exact_page_one_only(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="employment:page15",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:15",
            text="Employment Law body text.",
            score=0.99,
        ),
        _make_retrieved_chunk(
            chunk_id="employment:page4",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            section_path="page:4",
            text="This Law may be cited as the Employment Law 2019.",
            score=0.25,
        ),
        _make_retrieved_chunk(
            chunk_id="ip:page4",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:4",
            text="This Law may be cited as the Intellectual Property Law 2019.",
            score=0.30,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
    ]

    filtered = RAGPipelineBuilder._ensure_page_one_context(
        reranked=reranked,
        retrieved=retrieved,
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in filtered] == ["employment:page15", "ip:page4"]


def test_ensure_account_effective_dates_context_keeps_enactment_notice_page(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="crs:effective",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "The effective date is 31 December, 2016 for Pre-existing Accounts and 1 January, 2017 for New Accounts."
            ),
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="Enactment Notice",
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law DIFC Law No. 2 of 2018."
            ),
            score=0.41,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:body",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:15",
            text="Additional common reporting provisions.",
            score=0.92,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
    ]

    filtered = RAGPipelineBuilder._ensure_account_effective_dates_context(
        query=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard "
            "Law 2018, and what is the date of its enactment?"
        ),
        doc_refs=["Common Reporting Standard Law 2018"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert [chunk.chunk_id for chunk in filtered[:2]] == ["crs:effective", "crs:notice"]


def test_collapse_doc_family_crowding_context_injects_second_doc_for_issue_date_compare(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    reranked = [
        RankedChunk(
            chunk_id="sct169:body1",
            doc_id="sct-169",
            doc_title="SCT 169/2025 Obasi v Oreana",
            doc_type=DocType.CASE_LAW,
            section_path="page:5",
            text="Later procedural page without issue-date field.",
            retrieval_score=0.99,
            rerank_score=0.99,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="sct169:body2",
            doc_id="sct-169",
            doc_title="SCT 169/2025 Obasi v Oreana",
            doc_type=DocType.CASE_LAW,
            section_path="page:6",
            text="Another later procedural page.",
            retrieval_score=0.98,
            rerank_score=0.98,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="sct169:body3",
            doc_id="sct-169",
            doc_title="SCT 169/2025 Obasi v Oreana",
            doc_type=DocType.CASE_LAW,
            section_path="page:8",
            text="Body-only page that crowds out the second case.",
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="",
        ),
    ]
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="sct169:body1",
            doc_id="sct-169",
            doc_title="SCT 169/2025 Obasi v Oreana",
            section_path="page:5",
            text="Later procedural page without issue-date field.",
            score=0.99,
        ),
        _make_retrieved_chunk(
            chunk_id="sct169:body2",
            doc_id="sct-169",
            doc_title="SCT 169/2025 Obasi v Oreana",
            section_path="page:6",
            text="Another later procedural page.",
            score=0.98,
        ),
        _make_retrieved_chunk(
            chunk_id="sct295:page7",
            doc_id="sct-295",
            doc_title="SCT 295/2025 Olexa v Odon",
            section_path="page:7",
            text="Later page mentioning the case title only.",
            score=0.96,
        ),
        _make_retrieved_chunk(
            chunk_id="sct295:page2",
            doc_id="sct-295",
            doc_title="SCT 295/2025 Olexa v Odon",
            section_path="page:2",
            text="Issued by: Delvin Sumo. Date of Issue: 10 December 2025.",
            score=0.8,
        ),
    ]

    collapsed = RAGPipelineBuilder._collapse_doc_family_crowding_context(
        query="Which case has an earlier Date of Issue: SCT 169/2025 or SCT 295/2025?",
        answer_type="name",
        doc_ref_count=2,
        reranked=reranked,
        retrieved=retrieved,
        must_include_chunk_ids=[],
        top_n=3,
    )

    assert [chunk.chunk_id for chunk in collapsed] == ["sct169:body1", "sct169:body2", "sct295:page2"]


def test_rerank_support_pages_within_selected_docs_prefers_page_one_for_named_metadata(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    context_chunks = [
        RankedChunk(
            chunk_id="companies:body",
            doc_id="companies",
            doc_title="Companies Law",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Body text mentioning updated provisions.",
            retrieval_score=0.95,
            rerank_score=0.94,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="companies:title",
            doc_id="companies",
            doc_title="Companies Law",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="COMPANIES LAW. Last updated (consolidated version): March 2022.",
            retrieval_score=0.83,
            rerank_score=0.84,
            doc_summary="",
        ),
    ]

    reranked_ids = RAGPipelineBuilder._rerank_support_pages_within_selected_docs(
        query="What is the title of DIFC Law No. 5 of 2018 and when was its consolidated version last updated?",
        answer_type="free_text",
        context_chunks=context_chunks,
        used_ids=["companies:body"],
    )

    assert reranked_ids == ["companies:title"]


def test_rerank_support_pages_within_selected_docs_keeps_page_family_for_single_atom_metadata(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    context_chunks = [
        RankedChunk(
            chunk_id="ppl:title",
            doc_id="personal-property-law",
            doc_title="Personal Property Law 2005",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text='This Law may be cited as the "Personal Property Law 2005".',
            retrieval_score=0.82,
            rerank_score=0.83,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="ppl:maker",
            doc_id="personal-property-law",
            doc_title="Personal Property Law 2005",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text="This Law is made by the Ruler of Dubai.",
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="ppl:noise",
            doc_id="personal-property-law",
            doc_title="Personal Property Law 2005",
            doc_type=DocType.STATUTE,
            section_path="page:18",
            text="Definitions and schedules unrelated to legislative authority.",
            retrieval_score=0.75,
            rerank_score=0.74,
            doc_summary="",
        ),
    ]

    reranked_ids = RAGPipelineBuilder._rerank_support_pages_within_selected_docs(
        query="According to Article 2 of the DIFC Personal Property Law 2005, who made this Law?",
        answer_type="free_text",
        context_chunks=context_chunks,
        used_ids=["ppl:maker", "ppl:title", "ppl:noise"],
    )

    assert reranked_ids == ["ppl:title", "ppl:maker"]


def test_rerank_support_pages_within_selected_docs_picks_one_page_per_doc_for_compare(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    context_chunks = [
        RankedChunk(
            chunk_id="ca4:body",
            doc_id="ca-004",
            doc_title="CA 004/2025 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:4",
            text="Reasons about jurisdiction without any judge title markers.",
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="ca4:title",
            doc_id="ca-004",
            doc_title="CA 004/2025 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="ORDER WITH REASONS OF H.E. CHIEF JUSTICE WAYNE MARTIN",
            retrieval_score=0.82,
            rerank_score=0.83,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="arb34:body",
            doc_id="arb-034",
            doc_title="ARB 034/2025 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="Return date hearing details without judge names.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="arb34:title",
            doc_id="arb-034",
            doc_title="ARB 034/2025 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="ORDER WITH REASONS OF H.E. JUSTICE SHAMLAN AL SAWALEHI",
            retrieval_score=0.81,
            rerank_score=0.82,
            doc_summary="",
        ),
    ]

    reranked_ids = RAGPipelineBuilder._rerank_support_pages_within_selected_docs(
        query="Was the same judge involved in CA 004/2025 and ARB 034/2025?",
        answer_type="boolean",
        context_chunks=context_chunks,
        used_ids=["ca4:body", "arb34:body"],
    )

    assert reranked_ids == ["ca4:title", "arb34:title"]


def test_rerank_support_pages_within_selected_docs_skips_explicit_page_queries(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    context_chunks = [
        RankedChunk(
            chunk_id="arb34:page2",
            doc_id="arb-034",
            doc_title="ARB 034/2025 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:2",
            text="Return date hearing details.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="arb34:title",
            doc_id="arb-034",
            doc_title="ARB 034/2025 Example v Example",
            doc_type=DocType.CASE_LAW,
            section_path="page:1",
            text="ORDER WITH REASONS",
            retrieval_score=0.81,
            rerank_score=0.82,
            doc_summary="",
        ),
    ]

    reranked_ids = RAGPipelineBuilder._rerank_support_pages_within_selected_docs(
        query="What is stated on page 2 of ARB 034/2025?",
        answer_type="free_text",
        context_chunks=context_chunks,
        used_ids=["arb34:page2"],
    )

    assert reranked_ids == ["arb34:page2"]


def test_rerank_support_pages_within_selected_docs_caps_single_doc_strict_to_one_page(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    context_chunks = [
        RankedChunk(
            chunk_id="law:page3",
            doc_id="law",
            doc_title="LAW",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text="A weaker match mentioning the amount in passing.",
            retrieval_score=0.92,
            rerank_score=0.90,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law:page5",
            doc_id="law",
            doc_title="LAW",
            doc_type=DocType.STATUTE,
            section_path="page:5",
            text="The exact amount awarded is USD 10,000.",
            retrieval_score=0.96,
            rerank_score=0.97,
            doc_summary="",
        ),
    ]

    reranked_ids = RAGPipelineBuilder._rerank_support_pages_within_selected_docs(
        query="What amount was awarded under the law?",
        answer_type="number",
        context_chunks=context_chunks,
        used_ids=["law:page3", "law:page5"],
    )

    assert reranked_ids == ["law:page5"]


def test_collapse_doc_family_crowding_context_keeps_existing_doc_diversity(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    reranked = [
        RankedChunk(
            chunk_id="employment:title",
            doc_id="employment",
            doc_title="EMPLOYMENT LAW",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the Employment Law 2019.",
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:title",
            doc_id="trust",
            doc_title="TRUST LAW NO. 4 OF 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="This Law may be cited as the Trust Law No. 4 of 2018.",
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="",
        ),
    ]

    collapsed = RAGPipelineBuilder._collapse_doc_family_crowding_context(
        query="What are the titles of the Employment Law 2019 and the Trust Law No. 4 of 2018?",
        answer_type="free_text",
        doc_ref_count=2,
        reranked=reranked,
        retrieved=[
            _make_retrieved_chunk(
                chunk_id="employment:title",
                doc_id="employment",
                doc_title="EMPLOYMENT LAW",
                section_path="page:1",
                text="This Law may be cited as the Employment Law 2019.",
                score=0.95,
            ),
            _make_retrieved_chunk(
                chunk_id="trust:title",
                doc_id="trust",
                doc_title="TRUST LAW NO. 4 OF 2018",
                section_path="page:1",
                text="This Law may be cited as the Trust Law No. 4 of 2018.",
                score=0.94,
            ),
        ],
        must_include_chunk_ids=[],
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in collapsed] == ["employment:title", "trust:title"]


def test_collapse_doc_family_crowding_context_skips_explicit_page_queries(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings
    reranked = [
        RankedChunk(
            chunk_id="law-a:body1",
            doc_id="law-a",
            doc_title="LAW A",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text="Requested page is elsewhere.",
            retrieval_score=0.99,
            rerank_score=0.99,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="law-a:body2",
            doc_id="law-a",
            doc_title="LAW A",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text="Another chunk from the same law.",
            retrieval_score=0.98,
            rerank_score=0.98,
            doc_summary="",
        ),
    ]
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="law-a:body1",
            doc_id="law-a",
            doc_title="LAW A",
            section_path="page:3",
            text="Requested page is elsewhere.",
            score=0.99,
        ),
        _make_retrieved_chunk(
            chunk_id="law-b:page2",
            doc_id="law-b",
            doc_title="LAW B",
            section_path="page:2",
            text="This would normally become the alternate doc representative.",
            score=0.97,
        ),
    ]

    collapsed = RAGPipelineBuilder._collapse_doc_family_crowding_context(
        query="What is stated on page 2 of Law A and Law B?",
        answer_type="free_text",
        doc_ref_count=2,
        reranked=reranked,
        retrieved=retrieved,
        must_include_chunk_ids=[],
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in collapsed] == ["law-a:body1", "law-a:body2"]


def test_doc_shortlist_score_accepts_enactment_notice_surrogate_for_account_effective_query(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    notice_doc = [
        _make_retrieved_chunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="_______________________________________________",
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law "
                "DIFC Law No. 2 of 2018."
            ),
            score=0.41,
        )
    ]

    score = RAGPipelineBuilder._doc_shortlist_score(
        query=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard "
            "Law 2018, and what is the date of its enactment?"
        ),
        ref="Common Reporting Standard Law 2018",
        doc_chunks=notice_doc,
    )

    assert score > 0


def test_apply_doc_shortlist_gating_keeps_enactment_notice_doc_for_account_effective_query(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="crs:effective",
            doc_id="crs-a",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "The effective date is 31 December, 2016 for Pre-existing Accounts and 1 January, 2017 for New Accounts."
            ),
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:dup",
            doc_id="crs-b",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "The effective date is 31 December, 2016 for Pre-existing Accounts and 1 January, 2017 for New Accounts."
            ),
            score=0.94,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="_______________________________________________",
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law "
                "DIFC Law No. 2 of 2018."
            ),
            score=0.41,
        ),
    ]

    shortlisted = RAGPipelineBuilder._apply_doc_shortlist_gating(
        query=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard "
            "Law 2018, and what is the date of its enactment?"
        ),
        doc_refs=["Common Reporting Standard Law 2018"],
        retrieved=retrieved,
    )

    assert {chunk.doc_id for chunk in shortlisted} == {"crs-a", "crs-b", "crs-notice"}


def test_apply_doc_shortlist_gating_combines_doc_refs_with_title_refs_for_admin_compare(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="ip:admin",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:4",
            text=(
                "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                "The DIFC Authority shall administer this Law."
            ),
            score=0.91,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:admin",
            doc_id="trust",
            doc_title="TRUST LAW",
            section_path="page:4",
            text=(
                "This Law may be cited as the Trust Law No. 4 of 2018. "
                "The DIFC Authority shall administer this Law."
            ),
            score=0.83,
        ),
    ]

    shortlisted = RAGPipelineBuilder._apply_doc_shortlist_gating(
        query=query,
        doc_refs=["Law No. 4 of 2019", "Law No. 4 of 2018"],
        retrieved=retrieved,
    )

    assert {chunk.doc_id for chunk in shortlisted} == {"ip", "trust"}


def test_apply_doc_shortlist_gating_preserves_must_keep_doc_family(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="ip:admin",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:4",
            text=(
                "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                "The DIFC Authority shall administer this Law."
            ),
            score=0.91,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:title",
            doc_id="trust-title",
            doc_title="TRUST LAW",
            section_path="page:4",
            text=(
                "This Law may be cited as the Trust Law No. 4 of 2018. "
                "This Law is enacted on the date specified in the Enactment Notice."
            ),
            score=0.89,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:admin",
            doc_id="trust-admin",
            doc_title="TRUST LAW",
            section_path="page:5",
            text="Administration of this Law. This Law is administered by the DIFC Authority.",
            score=0.41,
        ),
    ]

    shortlisted = RAGPipelineBuilder._apply_doc_shortlist_gating(
        query=query,
        doc_refs=["Law No. 4 of 2019", "Law No. 4 of 2018"],
        retrieved=retrieved,
        must_keep_chunk_ids=["trust:admin"],
    )

    assert {chunk.doc_id for chunk in shortlisted} == {"ip", "trust-title", "trust-admin"}


def test_ensure_notice_doc_context_prefers_explicit_notice_docs_for_notice_query(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = "Does the enactment notice specify a precise calendar date for the law to come into force?"
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="law:body",
            doc_id="law",
            doc_title="LAW BODY",
            section_path="page:3",
            text=(
                "Date of enactment. This Law is enacted on the date specified in the Enactment Notice. "
                "Commencement. This Law comes into force on the date specified in the Enactment Notice."
            ),
            score=0.97,
        ),
        _make_retrieved_chunk(
            chunk_id="notice:a",
            doc_id="notice-a",
            doc_title="ENACTMENT NOTICE",
            section_path="page:1",
            text=(
                "ENACTMENT NOTICE. We hereby enact on this 01 day of March 2024. "
                "This Law shall come into force on the 5th business day after enactment."
            ),
            score=0.72,
        ),
        _make_retrieved_chunk(
            chunk_id="notice:b",
            doc_id="notice-b",
            doc_title="ENACTMENT NOTICE",
            section_path="page:1",
            text=(
                "ENACTMENT NOTICE. We hereby enact on this 14th day of November 2024. "
                "This Law shall come into force on the 5th business day after enactment."
            ),
            score=0.71,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[1]),
    ]

    selected = RAGPipelineBuilder._ensure_notice_doc_context(
        query=query,
        reranked=reranked,
        retrieved=retrieved,
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in selected] == ["notice:a", "notice:b"]


def test_doc_shortlist_score_prefers_self_admin_family_for_admin_compare(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    trust_admin_chunks = [
        _make_retrieved_chunk(
            chunk_id="trust-admin:p5",
            doc_id="trust-admin",
            doc_title="TRUST LAW",
            section_path="page:5",
            text=(
                "This Law may be cited as the Trust Law No. 4 of 2018. "
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            score=0.82,
        )
    ]
    trust_duties_chunks = [
        _make_retrieved_chunk(
            chunk_id="trust-duties:p34",
            doc_id="trust-duties",
            doc_title="TRUST LAW",
            section_path="page:34",
            text=(
                "Duty to administer a trust. A trustee shall administer the trust solely in the interest "
                "of the beneficiaries."
            ),
            score=0.91,
        )
    ]

    admin_score = RAGPipelineBuilder._doc_shortlist_score(
        query=query,
        ref="Trust Law",
        doc_chunks=trust_admin_chunks,
    )
    duties_score = RAGPipelineBuilder._doc_shortlist_score(
        query=query,
        ref="Trust Law",
        doc_chunks=trust_duties_chunks,
    )

    assert admin_score > duties_score


def test_best_named_administration_chunk_prefers_canonical_family_over_consolidated_family(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    consolidated = _make_retrieved_chunk(
        chunk_id="trust-consolidated:p4",
        doc_id="trust-consolidated",
        doc_title="TRUST LAW",
        section_path="page:4",
        text=(
            "This Law may be cited as the Trust Law No. 4 of 2018. "
            "Administration of this Law. This Law is administered by the DIFCA."
        ),
        score=0.96,
        doc_summary=(
            "Trust Law consolidated with amendments up to March 2024, including Law No. 3 of 2024 "
            "and Law No. 1 of 2024."
        ),
    )
    canonical = _make_retrieved_chunk(
        chunk_id="trust-canonical:p5",
        doc_id="trust-canonical",
        doc_title="TRUST LAW",
        section_path="page:5",
        text=(
            "This Law may be cited as the Trust Law No. 4 of 2018. "
            "Administration of this Law. This Law is administered by the DIFCA."
        ),
        score=0.61,
        doc_summary="Trust Law, DIFC Law No. 4 of 2018.",
    )

    best = RAGPipelineBuilder._best_named_administration_chunk(
        ref="Trust Law No. 4 of 2018",
        chunks=[consolidated, canonical],
    )

    assert best is not None
    assert best.chunk_id == "trust-canonical:p5"


def test_boolean_admin_compare_context_prefers_canonical_family_even_if_surrogate_reranks_higher(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    ip_chunk = _make_retrieved_chunk(
        chunk_id="ip:p4",
        doc_id="ip-law",
        doc_title="INTELLECTUAL PROPERTY LAW",
        section_path="page:4",
        text=(
            'This Law may be cited as the "Intellectual Property Law No. 4 of 2019". '
            "Administration of this Law. This Law shall be administered by the Commissioner of Intellectual Property."
        ),
        score=0.94,
        doc_summary="Intellectual Property Law, DIFC Law No. 4 of 2019.",
    )
    trust_consolidated = _make_retrieved_chunk(
        chunk_id="trust-consolidated:p4",
        doc_id="trust-consolidated",
        doc_title="TRUST LAW",
        section_path="page:4",
        text=(
            'This Law may be cited as the "Trust Law No. 4 of 2018". '
            "Administration of this Law. This Law is administered by the DIFCA."
        ),
        score=0.99,
        doc_summary=(
            "Trust Law consolidated with amendments up to March 2024, including Law No. 3 of 2024 "
            "and Law No. 1 of 2024."
        ),
    )
    trust_canonical = _make_retrieved_chunk(
        chunk_id="trust-canonical:p5",
        doc_id="trust-canonical",
        doc_title="TRUST LAW",
        section_path="page:5",
        text=(
            'This Law may be cited as the "Trust Law No. 4 of 2018". '
            "Administration of this Law. This Law is administered by the DIFCA."
        ),
        score=0.61,
        doc_summary="Trust Law, DIFC Law No. 4 of 2018.",
    )

    reranked = [
        RankedChunk(
            chunk_id=trust_consolidated.chunk_id,
            doc_id=trust_consolidated.doc_id,
            doc_title=trust_consolidated.doc_title,
            doc_type=DocType.STATUTE,
            section_path=trust_consolidated.section_path,
            text=trust_consolidated.text,
            retrieval_score=trust_consolidated.score,
            rerank_score=0.99,
            doc_summary=trust_consolidated.doc_summary,
        ),
        RankedChunk(
            chunk_id=ip_chunk.chunk_id,
            doc_id=ip_chunk.doc_id,
            doc_title=ip_chunk.doc_title,
            doc_type=DocType.STATUTE,
            section_path=ip_chunk.section_path,
            text=ip_chunk.text,
            retrieval_score=ip_chunk.score,
            rerank_score=0.95,
            doc_summary=ip_chunk.doc_summary,
        ),
        RankedChunk(
            chunk_id=trust_canonical.chunk_id,
            doc_id=trust_canonical.doc_id,
            doc_title=trust_canonical.doc_title,
            doc_type=DocType.STATUTE,
            section_path=trust_canonical.section_path,
            text=trust_canonical.text,
            retrieval_score=trust_canonical.score,
            rerank_score=0.62,
            doc_summary=trust_canonical.doc_summary,
        ),
    ]

    selected = RAGPipelineBuilder._ensure_boolean_admin_compare_context(
        query=query,
        reranked=reranked,
        retrieved=[trust_consolidated, ip_chunk, trust_canonical],
        top_n=2,
    )

    assert [chunk.chunk_id for chunk in selected] == ["ip:p4", "trust-canonical:p5"]


def test_missing_named_ref_targets_flags_admin_ref_without_self_admin_clause(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    retrieved = [
        _make_retrieved_chunk(
            chunk_id="ip:admin",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path="page:4",
            text=(
                "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                "This Law shall be administered by the Commissioner of Intellectual Property."
            ),
            score=0.94,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:wrong",
            doc_id="trust-wrong",
            doc_title="TRUST LAW",
            section_path="page:34",
            text=(
                "Duty to administer a trust. A trustee shall administer the trust solely in the interest "
                "of the beneficiaries."
            ),
            score=0.97,
        ),
    ]

    missing = RAGPipelineBuilder._missing_named_ref_targets(
        query=query,
        doc_refs=["Intellectual Property Law", "Trust Law"],
        retrieved=retrieved,
    )

    assert missing == ["Trust Law"]


def test_account_effective_support_family_seed_chunk_ids_selects_effective_and_notice_chunks(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="crs:effective",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "The effective date is 31 December, 2016 for Pre-existing Accounts and 1 January, 2017 for New Accounts."
            ),
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="_______________________________________________",
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law "
                "DIFC Law No. 2 of 2018."
            ),
            score=0.41,
        ),
        _make_retrieved_chunk(
            chunk_id="crs:body",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            section_path="page:15",
            text="Additional common reporting provisions.",
            score=0.92,
        ),
    ]

    seeds = RAGPipelineBuilder._account_effective_support_family_seed_chunk_ids(
        ref="Common Reporting Standard Law 2018",
        retrieved=retrieved,
    )

    assert seeds == ["crs:effective", "crs:notice"]


def test_account_enactment_clause_score_prefers_explicit_notice_over_generic_law_clause(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    generic_law_chunk = _make_retrieved_chunk(
        chunk_id="crs:law",
        doc_id="crs",
        doc_title="COMMON REPORTING STANDARD LAW",
        section_path="page:3",
        text=(
            "This Law may be cited as the Common Reporting Standard Law 2018. "
            "This Law shall come into force on the date specified in the Enactment Notice in respect of this Law."
        ),
        score=0.98,
    )
    notice_chunk = _make_retrieved_chunk(
        chunk_id="crs:notice",
        doc_id="crs-notice",
        doc_title="ENACTMENT NOTICE",
        section_path="page:1",
        text=(
            "ENACTMENT NOTICE. We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law "
            "DIFC Law No. 2 of 2018."
        ),
        score=0.72,
    )

    generic_score = RAGPipelineBuilder._account_enactment_clause_score(
        ref="Common Reporting Standard Law 2018",
        raw=generic_law_chunk,
    )
    notice_score = RAGPipelineBuilder._account_enactment_clause_score(
        ref="Common Reporting Standard Law 2018",
        raw=notice_chunk,
    )

    assert notice_score > generic_score


@pytest.mark.asyncio
async def test_rerank_same_year_boolean_keeps_title_year_chunk_beyond_default_strict_cap(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()
    classifier = MagicMock()

    builder = RAGPipelineBuilder(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        classifier=classifier,
    )
    collector = TelemetryCollector(request_id="same-year-cap")

    retrieved = [
        _make_retrieved_chunk(
            chunk_id=f"ip:body:{idx}",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW",
            section_path=f"page:{20 + idx}",
            text="Intellectual Property Law body text.",
            score=1.0 - (idx * 0.01),
        )
        for idx in range(10)
    ]
    retrieved.extend(
        [
            _make_retrieved_chunk(
                chunk_id="employment:body",
                doc_id="employment",
                doc_title="EMPLOYMENT LAW",
                section_path="page:7",
                text="Employment Law body text without clear enactment year.",
                score=0.6,
            ),
            _make_retrieved_chunk(
                chunk_id="ip:title",
                doc_id="ip",
                doc_title="INTELLECTUAL PROPERTY LAW",
                section_path="page:4",
                text="This Law may be cited as the Intellectual Property Law 2019.",
                score=0.5,
            ),
            _make_retrieved_chunk(
                chunk_id="employment:title",
                doc_id="employment",
                doc_title="EMPLOYMENT LAW",
                section_path="page:4",
                text="This Law may be cited as the Employment Law 2019.",
                score=0.4,
            ),
        ]
    )

    result = await builder._rerank(
        {
            "query": "Was the Employment Law enacted in the same year as the Intellectual Property Law?",
            "collector": collector,
            "answer_type": "boolean",
            "retrieved": retrieved,
        }
    )

    assert {chunk.chunk_id for chunk in result["context_chunks"]} >= {"employment:title", "ip:title"}


@pytest.mark.asyncio
async def test_retrieve_boolean_admin_compare_preserves_title_seed_chunk_across_merge_cap(
    pipeline_builder,
) -> None:
    pipeline_builder._settings.reranker.rerank_candidates = 3
    pipeline_builder._retriever.retrieve = AsyncMock(
        side_effect=[
            [
                _make_retrieved_chunk(
                    chunk_id="ip:body:1",
                    doc_id="ip",
                    doc_title="INTELLECTUAL PROPERTY LAW",
                    section_path="page:31",
                    text="Commissioner of Intellectual Property decision text.",
                    score=0.99,
                ),
                _make_retrieved_chunk(
                    chunk_id="ip:title",
                    doc_id="ip",
                    doc_title="INTELLECTUAL PROPERTY LAW",
                    section_path="page:4",
                    text=(
                        "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                        "The DIFC Authority shall administer this Law."
                    ),
                    score=0.95,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="ip:body:2",
                    doc_id="ip",
                    doc_title="INTELLECTUAL PROPERTY LAW",
                    section_path="page:34",
                    text="Further Commissioner of Intellectual Property text.",
                    score=0.98,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="ip:title",
                    doc_id="ip",
                    doc_title="INTELLECTUAL PROPERTY LAW",
                    section_path="page:4",
                    text=(
                        "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                        "The DIFC Authority shall administer this Law."
                    ),
                    score=0.95,
                ),
            ],
            [
                _make_retrieved_chunk(
                    chunk_id="trust:duty",
                    doc_id="trust",
                    doc_title="TRUST LAW",
                    section_path="page:34",
                    text="Duty to administer a trust.",
                    score=0.94,
                ),
                _make_retrieved_chunk(
                    chunk_id="trust:title",
                    doc_id="trust-title",
                    doc_title="TRUST LAW",
                    section_path="page:4",
                    text=(
                        "This Law may be cited as the Trust Law No. 4 of 2018. "
                        "This Law is enacted on the date specified in the Enactment Notice."
                    ),
                    score=0.21,
                ),
                _make_retrieved_chunk(
                    chunk_id="trust:admin",
                    doc_id="trust-admin",
                    doc_title="TRUST LAW",
                    section_path="page:5",
                    text="Administration of this Law. This Law is administered by the DIFC Authority.",
                    score=0.19,
                ),
            ],
        ]
    )

    result = await pipeline_builder._retrieve(
        {
            "query": (
                "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
                "that administers the Trust Law No. 4 of 2018?"
            ),
            "request_id": "admin-compare-merge",
            "question_id": "admin-compare-merge",
            "answer_type": "boolean",
            "collector": TelemetryCollector(request_id="admin-compare-merge"),
            "doc_refs": ["Law No. 4 of 2019", "Law No. 4 of 2018"],
        }
    )

    assert "trust:admin" in result["must_include_chunk_ids"]
    assert any(chunk.chunk_id == "trust:admin" for chunk in result["retrieved"])


def test_localize_free_text_support_chunk_ids_uses_full_context_for_account_effective_composite(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="crs:effective",
            doc_id="crs",
            doc_title="COMMON REPORTING STANDARD LAW",
            doc_type=DocType.STATUTE,
            section_path="page:3",
            text=(
                "This Law may be cited as the Common Reporting Standard Law 2018. "
                "The effective date is 31 December, 2016 for Pre-existing Accounts and 1 January, 2017 for New Accounts. "
                "This Law is enacted on the date specified in the Enactment Notice in respect of this Law."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="crs:notice",
            doc_id="crs-notice",
            doc_title="_______________________________________________",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on this 14th day of March 2018 the Common Reporting Standard Law "
                "DIFC Law No. 2 of 2018."
            ),
            retrieval_score=0.41,
            rerank_score=0.41,
            doc_summary="",
        ),
    ]

    support_ids = RAGPipelineBuilder._localize_free_text_support_chunk_ids(
        answer=(
            "1. Pre-existing Accounts: The effective date is 31 December, 2016\n"
            "2. New Accounts: The effective date is 1 January, 2017\n"
            "3. Common Reporting Standard Law 2018: The date of enactment is 14th day of March 2018"
        ),
        query=(
            "What are the effective dates for pre-existing and new accounts under the Common Reporting Standard Law 2018, "
            "and what is the date of its enactment?"
        ),
        context_chunks=context_chunks,
    )

    assert set(support_ids) == {"crs:effective", "crs:notice"}


def test_localize_boolean_support_chunk_ids_prefers_one_admin_page_per_ref_for_same_entity_query(
    mock_settings,
) -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    context_chunks = [
        RankedChunk(
            chunk_id="ip:admin",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW NO. 4 OF 2019",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                "The DIFC Authority shall administer this Law."
            ),
            retrieval_score=0.52,
            rerank_score=0.52,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="ip:noise",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW NO. 4 OF 2019",
            doc_type=DocType.STATUTE,
            section_path="page:31",
            text="Historical note mentioning the Trust Law No. 4 of 2018.",
            retrieval_score=0.99,
            rerank_score=0.99,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:admin",
            doc_id="trust",
            doc_title="TRUST LAW NO. 4 OF 2018",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "This Law may be cited as the Trust Law No. 4 of 2018. "
                "The DIFC Authority shall administer this Law."
            ),
            retrieval_score=0.51,
            rerank_score=0.51,
            doc_summary="",
        ),
    ]

    support_ids = RAGPipelineBuilder._localize_boolean_support_chunk_ids(
        answer="Yes",
        query=query,
        context_chunks=context_chunks,
    )

    assert support_ids == ["ip:admin", "trust:admin"]


def test_localize_boolean_support_chunk_ids_prefers_admin_clause_doc_over_title_page(mock_settings) -> None:
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Is the Intellectual Property Law No. 4 of 2019 administered by the same entity "
        "that administers the Trust Law No. 4 of 2018?"
    )
    context_chunks = [
        RankedChunk(
            chunk_id="ip:admin",
            doc_id="ip",
            doc_title="INTELLECTUAL PROPERTY LAW NO. 4 OF 2019",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "This Law may be cited as the Intellectual Property Law No. 4 of 2019. "
                "The DIFC Authority shall administer this Law."
            ),
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:title",
            doc_id="trust-title",
            doc_title="TRUST LAW NO. 4 OF 2018",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                "This Law may be cited as the Trust Law No. 4 of 2018. "
                "This Law is enacted on the date specified in the Enactment Notice."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:admin",
            doc_id="trust-admin",
            doc_title="TRUST LAW NO. 4 OF 2018",
            doc_type=DocType.STATUTE,
            section_path="page:5",
            text="Administration of this Law. This Law is administered by the DIFC Authority.",
            retrieval_score=0.55,
            rerank_score=0.55,
            doc_summary="",
        ),
    ]

    support_ids = RAGPipelineBuilder._localize_boolean_support_chunk_ids(
        answer="Yes",
        query=query,
        context_chunks=context_chunks,
    )

    assert support_ids == ["ip:admin", "trust:admin"]


def test_localize_free_text_support_chunk_ids_preserves_amended_law_slot_pages(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    context_chunks = [
        RankedChunk(
            chunk_id="amendment:enact",
            doc_id="amender",
            doc_title="DIFC Laws Amendment Law, DIFC Law No. 8 of 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "We hereby enact on the 5th day of November 2018 the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018."
            ),
            retrieval_score=0.97,
            rerank_score=0.97,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="trust:amended",
            doc_id="trust",
            doc_title="Trust Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Trust Law 2018 is amended by the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018.",
            retrieval_score=0.88,
            rerank_score=0.88,
            doc_summary="",
        ),
        RankedChunk(
            chunk_id="foundations:amended",
            doc_id="foundations",
            doc_title="Foundations Law 2018",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Foundations Law 2018 is amended by the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018.",
            retrieval_score=0.87,
            rerank_score=0.87,
            doc_summary="",
        ),
    ]

    support_ids = RAGPipelineBuilder._localize_free_text_support_chunk_ids(
        answer=(
            "1. Enactment Date:\n"
            "DIFC Laws Amendment Law, DIFC Law No. 8 of 2018 was enacted on the 5th day of November 2018.\n"
            "2. Laws Amended:\n"
            "- Trust Law 2018\n"
            "- Foundations Law 2018"
        ),
        query="When was the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018 enacted and what laws did it amend?",
        context_chunks=context_chunks,
    )

    assert set(support_ids) == {"amendment:enact", "trust:amended", "foundations:amended"}


def test_suppress_named_administration_family_orphan_support_ids_drops_surrogate_same_family_support(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    context_chunks = [
        RankedChunk(
            chunk_id="leasing:canonical",
            doc_id="leasing-law",
            doc_title="LEASING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Leasing Law 2020". '
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="Leasing Law, DIFC Law No. 1 of 2020.",
        ),
        RankedChunk(
            chunk_id="trust:canonical",
            doc_id="trust-law",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:5",
            text=(
                'This Law may be cited as the "Trust Law No. 4 of 2018". '
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="Trust Law, DIFC Law No. 4 of 2018.",
        ),
        RankedChunk(
            chunk_id="trust:surrogate",
            doc_id="trust-consolidated",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Trust Law No. 4 of 2018". '
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary=(
                "Trust Law consolidated with amendments up to March 2024, including Law No. 3 of 2024 "
                "and Law No. 1 of 2024."
            ),
        ),
    ]

    filtered = RAGPipelineBuilder._suppress_named_administration_family_orphan_support_ids(
        query="What entity administers the Leasing Law 2020 and the Trust Law 2018?",
        cited_ids=["leasing:canonical", "trust:canonical"],
        support_ids=["trust:surrogate"],
        context_chunks=context_chunks,
    )

    assert filtered == []


def test_named_multi_title_context_keeps_one_anchor_and_one_clause_per_named_ref(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="arb:intro",
            doc_id="arb",
            doc_title="Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013",
            section_path="page:1",
            text="We hereby enact the Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013.",
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="arb:title",
            doc_id="arb",
            doc_title="Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013",
            section_path="page:2",
            text="This Law may be cited as Arbitration Law of 2008 Amendment Law, DIFC Law No. 6 of 2013.",
            score=0.94,
        ),
        _make_retrieved_chunk(
            chunk_id="gp:intro",
            doc_id="gp",
            doc_title="General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013",
            section_path="page:1",
            text="We hereby enact the General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013.",
            score=0.93,
        ),
        _make_retrieved_chunk(
            chunk_id="gp:title",
            doc_id="gp",
            doc_title="General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013",
            section_path="page:2",
            text="This Law may be cited as General Partnership Law 2004 Amendment Law, DIFC Law No. 3 of 2013.",
            score=0.92,
        ),
    ]
    reranked = [
        RAGPipelineBuilder._raw_to_ranked(retrieved[0]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[2]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[1]),
        RAGPipelineBuilder._raw_to_ranked(retrieved[3]),
    ]

    filtered = RAGPipelineBuilder._ensure_named_multi_title_context(
        query="What are the titles of DIFC Law No. 6 of 2013 and DIFC Law No. 3 of 2013?",
        doc_refs=["Law No. 6 of 2013", "Law No. 3 of 2013"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert [chunk.chunk_id for chunk in filtered[:4]] == [
        "arb:intro",
        "arb:title",
        "gp:intro",
        "gp:title",
    ]


def test_named_amendment_context_keeps_amender_and_amended_law_chunks(mock_settings):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    retrieved = [
        _make_retrieved_chunk(
            chunk_id="amender:notice",
            doc_id="amender",
            doc_title="ENACTMENT NOTICE",
            section_path="page:1",
            text="We hereby enact on this 5th day of November 2018 the DIFC Laws Amendment Law DIFC Law No. 8 of 2018.",
            score=0.95,
        ),
        _make_retrieved_chunk(
            chunk_id="lp:amended",
            doc_id="lp",
            doc_title="LIMITED PARTNERSHIP LAW",
            section_path="page:1",
            text="LIMITED PARTNERSHIP LAW DIFC LAW NO. 4 OF 2006 As Amended by DIFC Laws Amendment Law DIFC Law No. 8 of 2018",
            score=0.94,
        ),
        _make_retrieved_chunk(
            chunk_id="trust:amended",
            doc_id="trust",
            doc_title="TRUST LAW",
            section_path="page:1",
            text="TRUST LAW DIFC LAW NO. 4 OF 2018 As Amended by DIFC Laws Amendment Law DIFC Law No. 8 of 2018",
            score=0.93,
        ),
        _make_retrieved_chunk(
            chunk_id="noise:0",
            doc_id="noise",
            doc_title="Unrelated Law",
            section_path="page:7",
            text="Miscellaneous provisions unrelated to DIFC Law No. 8 of 2018.",
            score=0.92,
        ),
    ]
    reranked = [RAGPipelineBuilder._raw_to_ranked(chunk) for chunk in retrieved]

    filtered = RAGPipelineBuilder._ensure_named_amendment_context(
        query="When was the DIFC Laws Amendment Law, DIFC Law No. 8 of 2018 enacted and what laws did it amend?",
        doc_refs=["DIFC Law No. 8 of 2018"],
        reranked=reranked,
        retrieved=retrieved,
        top_n=4,
    )

    assert [chunk.chunk_id for chunk in filtered[:3]] == [
        "amender:notice",
        "lp:amended",
        "trust:amended",
    ]


def test_multi_criteria_enumeration_detects_comma_separated_filters() -> None:
    from rag_challenge.core.pipeline import _is_multi_criteria_enumeration_query

    assert (
        _is_multi_criteria_enumeration_query(
            "Which laws, enacted in 2018, include provisions relating to the application of the Arbitration Law in their Schedule 2?"
        )
        is True
    )


def test_best_support_chunk_id_prefers_article_theme_slot_over_same_document_distractors(
    mock_settings,
):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    query = (
        "Which articles of Law No. 12 of 2004 are explicitly superseded by Law No. 16 of 2011, "
        "and what is the overarching theme of the content in Article 4 of Law No. 12 of 2004 that was superseded?"
    )
    context_chunks = [
        RankedChunk(
            chunk_id="article4:noise",
            doc_id="article4-doc",
            doc_title="LAW NO. 12 OF 2004",
            doc_type=DocType.STATUTE,
            section_path="page:7",
            text="Article 7 concerns procedural execution steps and does not describe the Chief Justice.",
            retrieval_score=0.91,
            rerank_score=0.91,
            doc_summary="Law No. 12 of 2004.",
        ),
        RankedChunk(
            chunk_id="superseded:gold",
            doc_id="superseded-doc",
            doc_title="LAW NO. 16 OF 2011",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text=(
                "Law No. 16 of 2011 provides that Articles (2), (4), (5), and (7) of Law No. 12 of 2004 "
                "are superseded by the following new provisions."
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary="Law No. 16 of 2011 amending Law No. 12 of 2004.",
        ),
        RankedChunk(
            chunk_id="article4:gold",
            doc_id="article4-doc",
            doc_title="LAW NO. 12 OF 2004",
            doc_type=DocType.STATUTE,
            section_path="page:2",
            text=(
                "Article 4 sets out the duties and powers of the Chief Justice of the Courts, including "
                "appointment and general supervisory authority."
            ),
            retrieval_score=0.95,
            rerank_score=0.95,
            doc_summary="Law No. 12 of 2004.",
        ),
        RankedChunk(
            chunk_id="article4:summary",
            doc_id="article4-doc",
            doc_title="LAW NO. 12 OF 2004",
            doc_type=DocType.STATUTE,
            section_path="page:1",
            text="Part 1 contains introductory provisions about the DIFC Courts and court structure.",
            retrieval_score=0.89,
            rerank_score=0.89,
            doc_summary="Law No. 12 of 2004.",
        ),
    ]

    chunk_id = RAGPipelineBuilder._best_support_chunk_id(
        answer_type="free_text",
        query=query,
        fragment="Article 4 concerns the duties and powers of the Chief Justice of the Courts.",
        context_chunks=context_chunks,
        allow_first_chunk_fallback=False,
    )

    assert chunk_id == "article4:gold"


def test_suppress_named_administration_family_orphan_support_ids_is_order_independent_for_trusted_sentinel(
    mock_settings,
):
    from rag_challenge.core.pipeline import RAGPipelineBuilder

    del mock_settings

    canonical_context = [
        RankedChunk(
            chunk_id="trust:canonical",
            doc_id="trust-law",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:5",
            text=(
                'This Law may be cited as the "Trust Law No. 4 of 2018". '
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            retrieval_score=0.92,
            rerank_score=0.92,
            doc_summary="Trust Law, DIFC Law No. 4 of 2018.",
        ),
        RankedChunk(
            chunk_id="leasing:canonical",
            doc_id="leasing-law",
            doc_title="LEASING LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Leasing Law 2020". '
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            retrieval_score=0.94,
            rerank_score=0.94,
            doc_summary="Leasing Law, DIFC Law No. 1 of 2020.",
        ),
        RankedChunk(
            chunk_id="trust:surrogate",
            doc_id="trust-consolidated",
            doc_title="TRUST LAW",
            doc_type=DocType.STATUTE,
            section_path="page:4",
            text=(
                'This Law may be cited as the "Trust Law No. 4 of 2018". '
                "Administration of this Law. This Law is administered by the DIFCA."
            ),
            retrieval_score=0.96,
            rerank_score=0.96,
            doc_summary=(
                "Trust Law consolidated with amendments up to March 2024, including Law No. 3 of 2024 "
                "and Law No. 1 of 2024."
            ),
        ),
    ]

    filtered = RAGPipelineBuilder._suppress_named_administration_family_orphan_support_ids(
        query="What entity administers the Leasing Law 2020 and the Trust Law 2018?",
        cited_ids=["leasing:canonical", "trust:canonical"],
        support_ids=["trust:surrogate"],
        context_chunks=list(reversed(canonical_context)),
    )

    assert filtered == []
