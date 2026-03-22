# Architecture Evolution: 13 Days of the RAG Challenge

Trace of architectural decisions from initial commit (2026-03-10) through
final submission (2026-03-22), reconstructed from git history, upload-pack
crisis documents, and current source tree.

---

## Timeline Overview

| Day | Date | Phase | Key Architectural Event |
|-----|------|-------|------------------------|
| 1 | Mar 10 | Warm-up | Initial commit: 49 Python files, 33k LOC, monolith pipeline |
| 2 | Mar 11 | Warm-up | First heuristic answerers (case outcome, judge comparison) |
| 3 | Mar 12 | Warm-up | Truth audit scaffold, isolated runtime per worker |
| 4 | Mar 13 | Warm-up | Grounding probe tooling, support portfolio analysis |
| 5 | Mar 14 | Private | Rescue pack: ranked 13th, doctrine under review |
| 7 | Mar 16 | Private | Page-first crisis: catastrophic 4x latency, reverted |
| 8 | Mar 17 | Private | Retrieval-first overhaul plan, hybrid eval begins |
| 9 | Mar 18 | Private | Grounding sidecar born, ML training scaffold |
| 10 | Mar 19 | Private | Grounding ML pipeline, page scorer, failed challenger reverted |
| 11 | Mar 20 | Private | Temporal routing, HyDE, citation graph, segment retrieval |
| 12 | Mar 21 | Private | DB answerer, query contract, rerank tuning, agent swarm |
| 13 | Mar 22 | Private | Pipeline decomposition, 103 corrections, V2 final submission |

---

## Stage 1: V1 Baseline (Day 1, March 10)

### Architecture

A single-file LangGraph state machine (`core/pipeline.py`, 5,746 lines)
orchestrating the full question-answering flow. All retrieval strategy,
answer generation, support-building, and telemetry logic lived in one class.

```
src/shafi/
  api/              # FastAPI app + SSE streaming route
  config/           # Settings (Pydantic), logging
  core/
    pipeline.py     # 5,746-line monolith: classify, retrieve, rerank,
                    #   generate, verify, emit, finalize
    retriever.py    # HybridRetriever: Qdrant dense + BM25 server-side fusion
    reranker.py     # Zerank 2 primary, Cohere fallback
    classifier.py   # Zero-latency heuristic query classifier
    strict_answerer.py  # Post-generation type coercion (boolean, number, date)
    verifier.py     # LLM-based answer verification
    decomposer.py   # Multi-hop query decomposition
    embedding.py    # Remote embedding client with circuit breaker
    sparse_bm25.py  # FastEmbed BM25 sparse encoder
    qdrant.py       # Qdrant vector store wrapper
    circuit_breaker.py
    conflict_detector.py
    premise_guard.py
  ingestion/
    pipeline.py     # PDF parse -> chunk -> embed -> upsert
    parser.py       # PDF text extraction
    chunker.py      # Section-aware legal chunking
    sac.py          # Summary-Anchor-Citation generation
  llm/
    generator.py    # RAGGenerator: prompt assembly + streaming generation
    provider.py     # LLMProvider: OpenAI API wrapper
  eval/             # Offline evaluation harness, judge, metrics
  models/schemas.py # Pydantic domain models
  prompts/          # Markdown prompt templates
  submission/       # Platform submission driver
  telemetry/        # TelemetryCollector
```

**49 Python source files. ~33,000 lines of code.**

### Diagram

```
                        +---------------------+
                        |    FastAPI + SSE     |
                        +----------+----------+
                                   |
                        +----------v----------+
                        |    pipeline.py       |
                        |    (5,746 lines)     |
                        |                      |
                        |  classify            |
                        |    |                 |
                        |  decompose           |
                        |    |                 |
                        |  retrieve            |
                        |    |                 |
                        |  rerank              |
                        |    |                 |
                        |  generate            |
                        |    |                 |
                        |  verify              |
                        |    |                 |
                        |  emit + finalize     |
                        +----------+-----------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
    +---------v------+   +---------v------+   +---------v------+
    | HybridRetriever|   | RerankerClient |   | RAGGenerator   |
    | (dense + BM25) |   | (Zerank 2 +   |   | (GPT-4.1 /     |
    |                |   |  Cohere)       |   |  GPT-4.1-mini) |
    +--------+-------+   +----------------+   +----------------+
             |
    +--------v-------+
    | Qdrant Store   |
    | (chunks only)  |
    +----------------+
```

### Key V1 Characteristics

- **Retrieval**: Single Qdrant collection of chunk-level embeddings.
  Dense vectors (remote embedding API) fused with BM25 sparse vectors via
  Qdrant server-side Reciprocal Rank Fusion.
- **Reranking**: Zerank 2 API with Cohere fallback. Circuit breaker pattern
  on both paths.
- **Generation**: GPT-4.1 for complex queries, GPT-4.1-mini for simple/strict
  types. Model routing decided by the heuristic classifier.
- **Type handling**: `StrictAnswerer` post-processes generated text into
  boolean/number/date/name/names via regex extraction. No pre-generation
  type-aware routing.
- **Grounding**: Page IDs derived from chunk metadata after answer generation.
  No independent page selection logic.
- **Scoring formula**: `Total = (0.7*Det + 0.3*Asst) * G * T * F`.
  At this stage G=0.801, Total=0.742 (rank 13 on public leaderboard).

### What Drove the Design

The system was built for the warm-up phase where getting a working end-to-end
pipeline mattered more than component quality. LangGraph provided the state
machine skeleton. The monolith `pipeline.py` was fast to iterate on but
accumulated query-type-specific logic (enumeration detection, case outcome
handling, judge comparison, provision references) until it became the primary
risk surface.

---

## Stage 2: V9 Hybrid Retrieval + Reranking (Days 8-11, March 17-20)

### What Changed

The period from the page-first crisis (March 16) through the grounding
closeout (March 19) transformed the retrieval and evidence-selection layers
without fully decomposing the pipeline monolith.

#### Crisis: Page-First Architecture (March 16)

An attempt to replace chunk-level retrieval with page-level retrieval
(full-page embeddings in a second Qdrant collection, cross-encoder page
reranking) failed catastrophically:

- All `names`-type answers returned the same wrong value (context bug)
- 16 empty-page answers vs 5 in V10 baseline
- 4x latency increase (avg 1524ms vs 359ms)
- 52% of answers changed (impossible to evaluate without gold labels)

**Decision**: Revert page-first. Keep chunk-first retrieval as the answer
path. Build page-selection as an independent grounding sidecar instead.

#### Grounding Sidecar (March 18-19)

A standalone evidence-selection subsystem was built to produce `used_page_ids`
independently from the answer generation path:

```
src/shafi/core/grounding/
  evidence_selector.py     # 1,325 lines - main sidecar orchestrator
  scope_policy.py          # Doc-scope candidate selection
  query_scope_classifier.py # Classify single-doc vs multi-doc scope
  authority_priors.py      # Authoritative page heuristics
  evidence_portfolio.py    # Portfolio optimization over page sets
  page_rank_merge.py       # Rank fusion across candidate sources
  page_semantic_lane.py    # Semantic re-ranking of page candidates
  necessity_pruner.py      # Remove redundant pages
  condition_audit.py       # Condition-based page auditing
  law_family_graph.py      # Law family relationship graph
  search_escalation.py     # Escalation for insufficient evidence
  relevance_verifier.py    # Relevance verification
  typed_panel_extractor.py # Panel-type-specific extraction
```

#### ML-Trained Page Scorer (March 19)

A LightGBM model trained on reviewed grounding labels to score candidate
pages. Deployed as a runtime sidecar within the evidence selector.

```
src/shafi/ml/
  training_scaffold.py     # Feature engineering + training loop
  page_scorer_training.py  # LightGBM page scorer training
  page_scorer_runtime.py   # Runtime inference adapter
  grounding_dataset.py     # Dataset export from reviewed labels
  ...
```

#### Enriched Ingestion (March 17-19)

New ingestion modules to extract structured metadata from legal documents:

- `legal_segments.py` -- Article/section/schedule boundary detection
- `page_semantics.py` -- Page-role classification (title, body, schedule)
- `bridge_facts.py` -- Cross-document factual relationships
- `canonical_entities.py` -- Entity normalization and alias resolution
- `corpus_compiler.py` -- Compile structured registry from parsed corpus
- `applicability_graph.py` -- Law applicability relationships
- `page_regionizer.py` -- Sub-page region detection

### Architecture at V9

```
src/shafi/
  core/
    pipeline.py          # Still monolith, but with grounding sidecar callout
    retriever.py         # +BM25Retriever, +CitationGraphExpander, +HyDE
    reranker.py          # Zerank 2 + Cohere (RERANK_TOP_N tuned 12->16)
    classifier.py        # Expanded query-type detection
    strict_answerer.py   # +temporal comparisons, +version extraction
    grounding/           # NEW: 14 files, 4,005 lines
      evidence_selector.py
      scope_policy.py
      query_scope_classifier.py
      ...
    citation_graph.py    # NEW: Cross-document citation expansion
    bm25_retriever.py    # NEW: Standalone BM25 with alias map
    hyde.py              # NEW: Hypothetical document embedding
    span_grounder.py     # NEW: Evidence span extraction
    claim_graph.py       # NEW: Claim decomposition + provenance
    ...
  ingestion/
    pipeline.py          # 2,204 lines — grew with all enrichment
    legal_segments.py    # NEW
    bridge_facts.py      # NEW
    canonical_entities.py # NEW
    corpus_compiler.py   # NEW: 1,351 lines
    ...
  ml/                    # NEW: 15 files — training + runtime
    page_scorer_runtime.py
    training_scaffold.py
    ...
  llm/
    generator.py         # +case outcome, +citation cleanup, +titles
```

### Diagram

```
                        +---------------------+
                        |    FastAPI + SSE     |
                        +----------+----------+
                                   |
                        +----------v-----------+
                        |    pipeline.py        |
                        |    (still monolith,   |
                        |     ~6,000+ lines)    |
                        +---+------+-------+---+
                            |      |       |
              +-------------+   +--+--+    +-------------+
              |                 |     |                   |
    +---------v------+  +------v--+  |  +----------------v---------+
    | HybridRetriever|  |Reranker |  |  |  Grounding Sidecar       |
    | dense + BM25   |  |Zerank 2 |  |  |  (evidence_selector.py)  |
    | + citation hop |  |TOP_N=16 |  |  |                          |
    | + HyDE         |  +---------+  |  |  scope_policy            |
    +--------+-------+               |  |  authority_priors         |
             |              +---------+  |  page_rank_merge         |
    +--------v-------+     |            |  page_semantic_lane       |
    | Qdrant Store   |     |            |  evidence_portfolio       |
    | chunks +       |     |            |  LightGBM page_scorer     |
    | pages +        |     |            +---------------------------+
    | segments +     |     |
    | bridge_facts   |     |
    +----------------+     |
                    +------v---------+
                    | RAGGenerator   |
                    | GPT-4.1 /      |
                    | GPT-4.1-mini   |
                    | + claim graph  |
                    | + proof        |
                    +----------------+
```

### Key V9 Characteristics

- **Retrieval**: Hybrid dense+BM25 on chunks, plus citation graph expansion,
  HyDE for hard queries, segment retrieval (1792-dim collection), bridge
  facts for cross-document joins.
- **Reranking**: Zerank 2 with TOP_N=16 (tuned from 12 via A/B test showing
  +0.04 G).
- **Grounding**: Independent sidecar running after answer generation.
  ML-trained LightGBM page scorer. Evidence portfolio optimization.
  Scope policy classifies single-doc vs multi-doc queries.
- **Ingestion**: Rich structured extraction -- legal segments, page
  semantics, bridge facts, canonical entity registry, applicability graph.
- **Scoring**: G=0.9956, F=1.029, Total projected ~0.88-0.92.

### What Drove the Design

The page-first crisis (March 16) was the pivotal architectural decision.
Rather than replacing chunk retrieval with page retrieval, the team chose to
keep chunks for the answer path and build an independent grounding sidecar
for page selection. This separation of concerns -- answer quality vs
grounding quality -- became the defining pattern of the system.

The grounding sidecar grew from a simple "derive pages from chunks" approach
to a full evidence-selection pipeline with scope classification, authority
priors, portfolio optimization, and an ML-trained page scorer. This was
driven by the scoring formula where G (grounding) multiplies the total score:
even a 0.01 G improvement was worth ~0.009 total.

---

## Stage 3: Final Architecture (Days 12-13, March 21-22)

### What Changed

The final two days brought three major structural changes:

1. **Pipeline decomposition**: The 5,746-line `pipeline.py` monolith was split
   into a `pipeline/` package with 24 files (~11,497 lines total across more
   focused modules).

2. **Database answerer**: A structured-data fast path that resolves ~13% of
   questions (118/900) directly from a compiled corpus registry without
   touching the vector store or LLM -- under 50ms.

3. **Query contract compiler**: A typed intermediate representation of the
   parsed question that routes execution through domain-specific engines
   (database lookup, comparison engine, temporal engine) before falling
   through to full RAG.

#### Pipeline Decomposition

The monolith was split into mixins and focused modules:

```
src/shafi/core/pipeline/
  __init__.py                    # 64 lines — public re-exports
  builder.py                     # 1,551 lines — LangGraph wiring + routing
  state.py                       # 57 lines — RAGState TypedDict
  types.py                       # 66 lines — shared type aliases
  constants.py                   # 210 lines — regex patterns, prompts
  query_rules.py                 # 264 lines — query classification predicates
  orchestration_logic.py         # 576 lines — classify, db_lookup, routing
  retrieval_logic.py             # 216 lines — facade over typed helpers
  retrieval_primitives.py        # 250 lines — core retrieval operations
  retrieval_context.py           # 415 lines — context augmentation
  retrieval_boolean_handlers.py  # 458 lines — boolean-specific retrieval
  retrieval_named_handlers.py    # 644 lines — named-entity retrieval
  retrieval_common_elements.py   # 180 lines — common-element queries
  retrieval_seed_selection.py    # 487 lines — initial seed chunk selection
  generation_logic.py            # 1,600 lines — generate, verify, emit
  support_logic.py               # 243 lines — support page building
  support_page_policy.py         # 923 lines — page selection policy
  support_formatting.py          # 329 lines — output formatting
  support_free_text.py           # 403 lines — free-text-specific support
  support_localization.py        # 411 lines — localization support
  support_scoring.py             # 438 lines — support scoring
  support_query_primitives.py    # 499 lines — query primitive helpers
  answer_validator.py            # 478 lines — answer validation
  answer_quality_gate.py         # 178 lines — quality gate checks
  answer_consensus.py            # 215 lines — multi-answer consensus
  appeal_chain.py                # 198 lines — DIFC appeal chain detection
  free_text_cleanup.py           # 136 lines — free-text post-processing
```

#### Database Answerer Fast Path

```
Question --> QueryContractCompiler --> QueryContract (typed IR)
                |
                v
         DatabaseAnswerer (corpus_registry.json)
                |
          +-----+------+
          |            |
        HIT          MISS
       (<50ms)         |
          |            v
        emit     CompareEngine / TemporalEngine
                       |
                 +-----+------+
                 |            |
               HIT          MISS
                |            |
              emit      full RAG pipeline
```

The `QueryContract` is a Pydantic model capturing:
- `PredicateType`: LOOKUP_FIELD, COMPARE, TEMPORAL, ENUMERATE, etc.
- Resolved entity references (case numbers, law titles)
- Field targets (date_of_issue, parties, commencement_date)
- Comparison parameters

`DatabaseAnswerer` resolves field lookups against a compiled
`corpus_registry.json` built by `ingestion/corpus_compiler.py` from
extracted document metadata. Covers: date_of_issue, parties, judges,
commencement_date, enactment_date, law titles, and other metadata fields.

`CompareEngine` handles "which case has earlier X" questions by looking up
both values and comparing them deterministically.

#### Generator Decomposition

The 4,266-line `llm/generator.py` also sprouted helper modules:

```
src/shafi/llm/
  generator.py               # 4,266 lines — main generation orchestrator
  generator_prompts.py       # 176 lines — prompt assembly
  generator_constants.py     # 346 lines — constants, templates
  generator_question_types.py # 289 lines — type-specific generation
  generator_case_outcome.py  # 239 lines — case outcome extraction
  generator_titles.py        # 485 lines — title extraction + normalization
  generator_citations.py     # 84 lines — citation parsing
  generator_cleanup.py       # 282 lines — answer post-processing
  generator_token_usage.py   # 120 lines — token accounting
  generator_types.py         # 44 lines — type definitions
```

### Final Architecture Diagram

```
                            +---------------------+
                            |    FastAPI + SSE     |
                            |    (api/routes.py)   |
                            +----------+----------+
                                       |
                            +----------v----------+
                            | pipeline/builder.py  |
                            | (LangGraph DAG)      |
                            +---+--+--+--+--+-----+
                                |  |  |  |  |
         +----------------------+  |  |  |  +---------------------------+
         |                         |  |  |                              |
+--------v---------+    +----------v--+  |                   +----------v----------+
| Classify         |    | Query        |  |                   | Finalize            |
| (orchestration   |    | Contract     |  |                   | (telemetry,         |
|  _logic.py)      |    | Compiler     |  |                   |  SSE emit)          |
+--------+---------+    +------+-------+  |                   +---------------------+
         |                     |          |
         v                     v          |
+------------------+  +--------+-------+  |
| QueryClassifier  |  | DB Answerer    |  |
| (heuristic,      |  | (<50ms,        |  |
|  zero-latency)   |  |  13% coverage) |  |
+------------------+  +---+----+-------+  |
                           |    |         |
                         HIT  MISS        |
                           |    |         |
                         emit   v         |
                        +-------+------+  |
                        | Compare /    |  |
                        | Temporal     |  |
                        | Engine       |  |
                        +---+----+-----+  |
                            |    |        |
                          HIT  MISS       |
                            |    |        |
                          emit   v        |
              +--------------+---+--------v---------+
              |              |                       |
    +---------v------+ +-----v-------+   +-----------v----------+
    | HybridRetriever| | Reranker    |   | Grounding Sidecar    |
    |                | | (Zerank 2)  |   |                      |
    | Dense (Kanon 2)| | TOP_N=12   |   | evidence_selector    |
    | + BM25 sparse  | +-----+-------+   | scope_policy         |
    | + citation hop |       |           | LightGBM scorer      |
    | + HyDE         |       |           | evidence_portfolio   |
    | + segments     |       |           | authority_priors     |
    | + bridge facts |       v           | necessity_pruner     |
    +--------+-------+ +-----+--------+  +----------------------+
             |         | RAGGenerator  |
    +--------v-------+ | GPT-4.1 /    |
    | Qdrant Store   | | GPT-4.1-mini |
    |                | |              |
    | Collections:   | | + claim graph|
    |  chunks        | | + proof      |
    |  pages         | | + strict     |
    |  segments      | |   answerer   |
    |  bridge_facts  | +--------------+
    +----------------+
```

### LangGraph Node Sequence (Final)

```
classify
    |
compile_query_contract
    |
database_lookup ---[HIT]---> emit ---> finalize
    |
    |---[COMPARE]---> compare_lookup ---[HIT]---> emit
    |---[TEMPORAL]--> temporal_lookup --[HIT]---> emit
    |
decompose
    |
retrieve
    |
rerank
    |
detect_conflicts
    |
confidence_check ---[LOW]---> retry_retrieve ---> rerank (loop)
    |
generate
    |
build_claim_graph
    |
proof_compile
    |
verify
    |
emit ---> finalize
```

### Final System Statistics

| Metric | V1 (Day 1) | Final (Day 13) | Change |
|--------|-----------|----------------|--------|
| Python source files | 49 | 172 | +251% |
| Total Python LOC | ~33,000 | ~65,500 | +98% |
| `pipeline.py` | 5,746 lines (1 file) | 11,497 lines (24 files) | split |
| `retriever.py` | ~600 lines | 2,129 lines | +255% |
| `strict_answerer.py` | ~800 lines | 2,534 lines | +217% |
| `generator.py` | ~1,500 lines | 4,266 + 1,800 helpers | +305% |
| Grounding subsystem | 0 files | 14 files, 4,005 lines | new |
| ML subsystem | 0 files | 15 files | new |
| Ingestion modules | 4 files | 18 files, 8,908 lines | +350% |
| Qdrant collections | 1 (chunks) | 4 (chunks, pages, segments, bridge_facts) | +300% |
| Domain engines | 0 | 3 (DB, Compare, Temporal) | new |

### Key Final Characteristics

- **Retrieval**: 4 Qdrant collections. Dense (Kanon 2) + BM25 sparse with
  server-side fusion. Citation graph expansion for cross-document hops. HyDE
  for hard queries. Segment-level retrieval for article-specific questions.
  Bridge facts for cross-document joins.
- **Structured fast paths**: DatabaseAnswerer resolves 13% of questions from
  corpus registry in <50ms. CompareEngine handles "which case has
  earlier/later X" deterministically. TemporalEngine handles date comparisons.
- **Reranking**: Zerank 2 primary, Cohere fallback, with query-type-aware
  instruction generation (`rerank_instructions.py`).
- **Generation**: GPT-4.1 for complex + free_text, GPT-4.1-mini for strict
  types. Predicted output for boolean type (speculative decoding). Evidence-
  first prompts. Claim graph with proof-carrying answers.
- **Grounding**: Independent sidecar with scope classification, authority
  priors, evidence portfolio optimization, LightGBM page scorer, necessity
  pruning, semantic re-ranking. 14 modules, 4,005 lines.
- **Post-processing**: StrictAnswerer (2,534 lines) handles boolean, number,
  date, name, names extraction with type-specific regex, temporal comparison,
  version number extraction, penalty extraction, party name normalization.
- **Answer quality**: Answer validator, quality gate, free-text cleanup,
  consensus across multiple answer attempts.
- **Scoring**: G=0.9967, F=1.032, null=3, nopg=3, projected Total ~0.93.

---

## Key Architectural Decisions and What Drove Them

### 1. Keep Chunks for Answers, Build Sidecar for Grounding

**Decision (March 16)**: After the page-first crisis, separate the answer
path (chunk retrieval -> LLM generation) from the grounding path (independent
page selection for `used_page_ids`).

**Driver**: The page-first architecture caused catastrophic regressions in
answer quality, latency, and reliability. Chunk-level retrieval produced
better LLM context. But the competition scores grounding (G) as a multiplier
on the total score, so page selection had to be excellent independently.

**Impact**: This separation of concerns became the defining architectural
pattern. The grounding sidecar grew into 14 modules with its own ML model
but never touched the answer text.

### 2. Monolith Pipeline Split into Mixin Package

**Decision (March 21-22)**: Decompose the 5,746-line `pipeline.py` into a
`pipeline/` package with 24 focused modules using mixin composition on the
builder class.

**Driver**: The monolith accumulated query-type-specific handlers (boolean
comparisons, named entity retrieval, common elements, enumeration, amendment
queries, etc.) that made it impossible to change one behavior without risking
others. Agent-driven development (8+ named agents making concurrent changes)
made merge conflicts frequent.

**Impact**: Each query-type family got its own handler module. Retrieval,
generation, support-building, and orchestration separated into distinct
mixins. The builder class remained large (1,551 lines) as the wiring hub.

### 3. Database Answerer as a Pre-RAG Short Circuit

**Decision (March 21)**: Build a structured-data lookup that answers
metadata questions directly from a compiled corpus registry, bypassing
vector retrieval and LLM generation entirely.

**Driver**: ~40% of private-set questions (362/900) are answerable from
document metadata (date of issue, parties, judges, commencement dates).
These questions hit the full RAG pipeline unnecessarily, adding latency and
introducing LLM hallucination risk. A deterministic lookup is both faster
(<50ms vs ~1500ms) and more accurate.

**Impact**: 118/900 questions (13%) resolved via DB answerer at submission
time. Improved TTFT coefficient (F) and reduced null-answer rate. Spawned
CompareEngine and TemporalEngine for relational questions.

### 4. Query Contract as Typed Intermediate Representation

**Decision (March 21)**: Introduce `QueryContract` -- a Pydantic model that
captures the parsed semantic structure of a question (predicate type, entity
references, field targets, comparison parameters) before execution routing.

**Driver**: The classifier produced a simple complexity enum, but the pipeline
needed to route through multiple domain engines (DB, Compare, Temporal, full
RAG) based on fine-grained query semantics. Without a typed IR, routing
logic was scattered across the pipeline as ad-hoc regex checks.

**Impact**: Clean routing: `classify -> compile_contract -> route`. Entity
resolution happens once. Downstream engines receive strongly-typed input.
The 954-line `query_contract.py` became the single source of truth for
query understanding.

### 5. Multiplicative Scoring Drives All Priority

**Decision (entire timeline)**: Every architectural investment was prioritized
by expected impact on `Total = S * G * T * F` where each factor is a
multiplier.

**Driver**: The scoring formula means a 0.01 improvement in G (grounding)
with G~0.99 is worth ~0.009 total, while a 0.01 improvement in Det
(deterministic accuracy) is worth only 0.007 total. TTFT coefficient F
ranges from 0.85 to 1.05 -- a 20% swing. Telemetry T must be near-perfect
or it destroys the total.

**Impact**: Grounding sidecar got 14 modules and an ML model. TTFT was
optimized ruthlessly (DB answerer, predicted output, rerank candidates cap).
Telemetry was audited to 100% compliance. Free-text quality improvements
(Asst) were pursued last because they have the smallest multiplier (0.3).

### 6. Evidence-First Prompts Over RAG Dogma

**Decision (March 20-21)**: Replace "answer the question using the
following context" prompts with "cite specific evidence from the passages
before answering."

**Driver**: The LLM-judged Asst score penalizes answers that do not
ground claims in the provided evidence. Standard RAG prompts produced
answers that started with "The..." and made assertions without citing
specific articles or sections. Evidence-first prompts forced the LLM to
name its sources.

**Impact**: +3-5% Asst improvement on audited samples. Required splitting
prompt templates by question type (5 variants) and retuning generation
parameters.

### 7. Multi-Agent Development as Architecture Pressure

**Decision (March 20-22)**: Deploy 8+ named AI agents (KEREN, DAGAN, EYAL,
SHAI, NOGA, KESHET, LIRON, OREV, TZUF, TAMAR, NOAM, AMIR, RASO, GILAD)
working concurrently on the codebase with a task-server coordination layer.

**Driver**: The 13-day deadline required parallel work streams. Agent
specialization: KEREN (director), DAGAN (coordination), EYAL (TTFT/scoring),
SHAI (prompts), NOGA (testing), KESHET (smoke testing), LIRON (auditing),
OREV (heuristics), TZUF (reranking), TAMAR (answer cleanup).

**Impact**: Forced modularization -- agents working on the same file created
merge conflicts. Drove the pipeline split, the extraction of handler modules,
and the separation of grounding from answer logic. Also created risk:
confounded submissions, stale agent state, and trust issues (OREV inflated
scores, TAMAR/GILAD generated noise -- per trust audit).

---

## Evolution Summary

```
Day 1 (V1)              Day 8-11 (V9)              Day 12-13 (Final)
-----------             -----------------          -------------------

 [monolith]             [monolith+sidecar]         [decomposed pipeline]
     |                       |    |                    |   |   |   |
 single Qdrant          4 collections            DB  Cmp Tmp  RAG
 collection             + grounding ML           |    |   |    |
     |                       |                   fast paths   full path
 chunk -> rerank         chunk -> rerank              |
 -> generate             -> generate              pipeline/
     |                   + sidecar pages          24 modules
 derive pages            + evidence portfolio     + grounding/
                         + page scorer            14 modules
                                                  + ml/
                                                  15 modules

 49 files                ~120 files               172 files
 33k LOC                 ~50k LOC                 65k LOC
 G=0.801                 G=0.9956                 G=0.9967
 Total=0.742             Total~0.88               Total~0.93
```

The system evolved from a straightforward LangGraph RAG pipeline into a
multi-layered architecture with structured fast paths, independent grounding
optimization, ML-trained page scoring, and type-aware execution routing --
all driven by the multiplicative scoring formula and the relentless pressure
of a 13-day competition timeline.
