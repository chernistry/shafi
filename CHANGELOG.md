# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-03-22

**Final competition submission.** Private phase complete -- V2 verified, all gates PASS.

### Final State
- 900 questions, 300 documents, 10,109 pages (+209% registry enrichment)
- null=3, nopg=3, T=50, F=143
- 103+ corrections: 56 DOI dates, 11 comparisons, 32 booleans, 16 numbers
- Evals V9.1 through V17 completed; V18 rejected (nopg=41)

### Rejected Approaches (87.5% ablation rejection rate)
- BM25-only retrieval
- RAG Fusion
- HyDE (Hypothetical Document Embeddings)
- FlashRank reranking
- Isaacus EQA (Extractive QA)
- Step-back prompting
- Citation-first retrieval

## [0.9.0] - 2026-03-20

**Multi-agent sprint.** Six AI agents (OREV, DAGAN, TAMAR, LIRON, KESHET, TZUF) working in parallel on the final push. Agent communication protocol, real-time coordination, and task dispatch system deployed.

### Added
- Multi-agent coordination framework with polling dashboard
- Agent watcher and universal protocol instructions
- TTFT optimization (+23.5% identified as biggest remaining lever)
- Page scorer retraining with TAMAR ground-truth data
- Citation attribution fixes across the pipeline
- Private data execution runbook

### Changed
- Bumped page_budget to 2 universally for F-beta 2.5 recall optimization
- Removed warmup-specific language from all prompts
- Unanswerable answer calibration

## [0.8.0] - 2026-03-18

**Grounding sidecar.** Hardened grounding pipeline for multi-document and cross-reference scenarios.

### Added
- Offline grounding ablation harness
- Multi-doc grounding scope policy
- Anchor-based page selection hardening
- Article-aware page trimming for grounding precision
- Retrieval-first shadow/anchor challenger

### Changed
- Split pipeline into orchestration and generation modules
- Extracted retriever filter helpers into standalone module

## [0.7.0] - 2026-03-13

**Reranking and page-level retrieval.** Introduced ColBERT reranking and bounded doc-family strategies.

### Added
- Local ColBERT page reranker tooling
- Bounded doc-family rerank collapse
- Bounded doc-page rerank (phase 1)
- Explicit anchor support gap auditing

### Rejected
- Page localizer approach (ticket recorded in matrix)
- Bounded page-branch experiments

## [0.5.0] - 2026-03-11

**Initial pipeline.** Core RAG system operational with LangGraph orchestration, Qdrant hybrid search, and the first eval harness.

### Added
- FastAPI server with SSE streaming
- LangGraph pipeline with classify, retrieve, rerank, generate, verify stages
- Qdrant hybrid search (dense + BM25) with Kanon-2 embeddings
- DB answerer short-circuit for metadata questions
- Platform submission tooling with preflight checks and code archive
- Eval harness with judge support
- Legal query handling: case outcomes, provision references, comparisons
- Docker Compose stack (API + Qdrant + tool services)

## [0.1.0] - 2026-03-10

**Initial commit.** Project scaffolding, dashboard, and base configuration.
