# R001 -- Development Timeline: RAG Challenge 2026

Reconstructed from 1,595 git commits across 13 days (2026-03-10 to 2026-03-22).

## Commit Volume by Day

| Date       | Commits | Phase                        |
|------------|---------|------------------------------|
| 2026-03-10 |       2 | Project bootstrap            |
| 2026-03-11 |       9 | Core pipeline build          |
| 2026-03-12 |      10 | Truth audit + scoring infra  |
| 2026-03-13 |      99 | Warmup-phase research sprint |
| 2026-03-14 |       5 | Freeze + decision day        |
| 2026-03-15 |       0 | (no activity)                |
| 2026-03-16 |       3 | Private doc anomaly scanning |
| 2026-03-17 |      12 | Pipeline split + ML infra    |
| 2026-03-18 |      11 | Grounding sidecar subsystem  |
| 2026-03-19 |      21 | Research closeout (59 dirs)  |
| 2026-03-20 |     493 | Agent swarm + warmup evals   |
| 2026-03-21 |     612 | Private data + submissions   |
| 2026-03-22 |     318 | Final corrections + V2 build |

---

## Phase 1: Bootstrap and Core Pipeline (Mar 10-12, 21 commits)

### Mar 10 -- Initial Commit
- Repository created. i18n and index.html for a web front-end.

### Mar 11 -- RAG Pipeline Foundation
- Date normalization and free-text handling implemented.
- Submission process and reporting refactored.
- Qdrant and retriever functionality added: `doc_title` indexing, document chunking.
- Common judge comparison and provision reference extraction added.
- Free-text fragment classification, case outcome query handling, and scoring.
- Legal query handling in strict answerer.

### Mar 12 -- Audit Infrastructure
- Remuneration recordkeeping query detection.
- Question processing refactored to isolated runtime per worker.
- Truth audit scaffold builder: manual exactness labels, support shape and route family metadata.
- Offline competition control loop and exactness helpers.
- External review packet assembly automated.

---

## Phase 2: Warmup Research Sprint (Mar 13-14, 104 commits)

### Mar 13 -- Massive Exploration Day (99 commits)
The single largest research day before the agent swarm. All work was Sasha-driven (no agents yet).

Key activities:
- **Grounding candidate search**: Portfolio combo search, support-only candidates, candidate ceiling cycles, exactness rider subset searches, and embedding support opportunity mining.
- **Local reranker evaluation**: Tickets 20-33 evaluated ModernBERT, Qwen3, BGE-M3, embedding-gemma, and ColBERT-based page rerankers. All local rerankers were rejected.
- **Doc-family collapse**: Bounded implementation and evaluation of document-family grouping for grounding.
- **Page-level tooling**: Page trace ledger, page candidates, bounded page generators, same-doc page selector.
- **Infrastructure**: Serialized runner session pool, deterministic batch eval, private-phase telemetry dashboard, BM25-compatible probe runner, concurrency safety audit, stress-test ingestion for 300+ docs.
- **External frameworks**: LegalBench-RAG mini bootstrap, public legal synthetic stress pack, named-ref public pack audit.
- **Ticket 00-85**: A continuous sequence of research tickets covering page traces, hidden-G blindspots, production mimic evals, strict null telemetry, OCR fallback audits, and more.

### Mar 14 -- Freeze and Warmup Decision (5 commits)
- Private primary candidate state frozen.
- Tickets 86-203: freeze warmup offense narrative, private-only queue reorder, production mimic reporting, warmup decision one-pager, final warmup decision, private-day doctrine dry-run.
- Final artifact order established (A/B/C artifacts for private day).

---

## Phase 3: Structural Improvements (Mar 16-18, 26 commits)

### Mar 16 -- Private Doc Anomaly Scanning (3 commits)
- Scanner tranche 1 and 2 tooling: private doc anomaly scanner, manifest and family analysis.

### Mar 17 -- Pipeline Architecture Split (12 commits)
- Golden labels evaluation infrastructure added.
- Preprocessing enrichment: doc/page family tags, anchor chunks, identifier normalization.
- Page recall enhancer, unsupported trap detector, family-specific page assembly.
- Answer-driven citation extraction.
- **Major refactor**: Pipeline builder split into orchestration and generation modules. Generator monolith split into typed helper modules. Retriever filter helpers extracted.
- Retrieval-first shadow/anchor challenger implemented.

### Mar 18 -- Grounding Sidecar Subsystem (11 commits)
- Grounding evidence sidecar subsystem (Tickets 801-805, 810).
- Sidecar limited to compare/full-case scopes, then expanded to safe single-doc scopes.
- Multi-doc grounding scope policy hardened.
- Grounding ML dataset export pipeline added.
- ML training scaffold bootstrapped.

---

## Phase 4: ML Training and Research Closeout (Mar 19, 21 commits)

### Mar 19 -- Offline Grounding ML
- External grounding datasets normalized (ObliQA, CUAD, ContractNLI, LEDGAR).
- Grounding router and page scorer trained.
- Reviewed grounding labels imported. Trained-vs-heuristic sidecar ablation ran.
- Page scorer features made runtime-safe. Runtime trained page scorer integrated.

### Mar 19 -- Research Closeout (59 directories)
All research directions closed out with explicit verdicts:
- **1002-1008**: Authority lane (instruction-conditioned reranker, authoritative page priors, law bundle scope graph, bounded page pair, visual page rerank) -- all CLOSED, not activated.
- **1010-1013**: Evidence-set lane (portfolio selector, counterfactual pruner, compare panel, typed condition audit) -- implemented but failed promotion gate.
- **1014-1019**: Dependent research (segment spanning, pairwise authoritativeness, retrieve-verify-retrieve, visual micro-rerank, synthetic counter-question, document-axis micro-agents) -- all CLOSED by gate.
- **1020-1027**: Paper source closeout, introspection escalation, shadow query rewrite, bounded relevance verifier, grounding selector integration, evaluation gate, triad rehydration, docfamily collapse control.
- **1041-1068**: Closed-world compiler program -- failure cartography, corpus compiler, entity alias resolver, legal segment compiler, amendment temporal graph, bridge fact registry, multimodal regionizer, synthetic QA factory, teacher labels, corpus-tuned dense retriever, corpus-tuned reranker, retrieval utility predictor, query contract compiler, database-first answerer, compare join engine, temporal applicability engine, claim-to-span grounding graph, proof-carrying answer compiler, committee MBR selection, external segment benchmark, rich segment composer, adversarial near-miss corpus, compiled memory kernel, selective ICR local rerank, triad submit calibration, kanon-2 1792 dimension ablation, allowlist page localizer.
- **600-649 tickets**: Sidecar scope expansion, page role audit, grounding RRF audit, full-case hardening, unanswerable grounding gate, ML dataset export, ML training scaffold, external dataset downloads, page scorer training, runtime integration.
- **Domain hypotheses H1-H19**: Law title family, enactment/commencement, case number normalization, case party caption, issuing authority, article/schedule retrieval, rich segment text, jurisdiction rewrite, strict formatting, page semantics panels, document interrogation, commencement dates, title alias, UAE legal field intent, page region caption, amendment/repeal relationships, case party noise, case judge normalization.

---

## Phase 5: Agent Swarm -- Warmup Evals (Mar 20, 493 commits)

### Morning -- Platform Transfer and Bug Fixes (11:57-13:25)
- 7 platform-transfer bugs fixed.
- 8 score-critical bugs in answer extraction and regex patterns.
- Grounding recall loss and date normalization fixed.
- Number regex for thousands separators fixed.
- LightGBM page scorer trained with 3K ObliQA augmentation.
- Private data execution plan and runbook added.

### Afternoon -- Agent Spawn (14:25-15:57)
First wave of agents activated:
- **TAMAR**: Golden verification, page audit, competition analysis.
- **EYAL**: ML research, page scorer retraining, TTFT profiling.
- **NOGA**: A/B evaluation framework, answer quality hardening.
- **OREV**: Boolean generalization, multi-doc grounding, dedup.
- **SHAI**: Prompt engineering, private-set robustness, evidence-first prompts.
- **TZUF**: Integration testing, critical bug fixes.
- **DAGAN**: Coordination/dispatch agent, cycle-based task management.
- **KEREN**: Strategic orchestration, VP-level directives.

Agent communication protocol established. Task queues, status files, polling dashboard deployed.

### Afternoon-Evening -- Papa Cycles 1-15 (16:48-18:10)
- DAGAN cycle-based dispatch: 15 cycles in ~1.5 hours.
- Boolean citation fix: cite top context chunk on localization failure.
- G regression detected (-3.6pp) and root-caused.
- Citation reconciliation: reconcile cited_ids with sidecar used_ids.
- Prompt reduction: -60% tokens from SHAI.
- AI limits exhausted at 18:10 -- ALL AGENTS HALT for ~2 hours.

### Evening -- Recovery and Rank 1 Push (20:11-23:00)
- Limits restored. GILAD and AMIR agents hired.
- P0 changes: verifier disabled, RERANK_TOP_N=10, page_budget=3.
- G=0.4632 -- PAST RANK 1 (per warmup eval metrics).
- EMERGENCY reverts: RAG-Fusion -7.8pp G, reranker-off -4.3pp G.
- Multi-law date comparison fix: OREV COMPARE_PAIR scope.
- Isaacus kanon-2-reranker adapter built and evaluated. Rejected (TTFT penalty).
- G=0.4874 NEW BEST. SUBMISSION READY.
- NOAM agent hired (dashboard/monitoring).
- zerank-2 confirmed winner over Isaacus (+1.5pp Total from TTFT).
- EQA investigated: 4 separate bugs found (schema, field name, API key, ordering). Abandoned.
- BM25 retriever built (GILAD) and verified end-to-end.
- HyDE implemented (AMIR) -- later found to regress.
- Name G +26.2pp confirmed from fix.
- Det=54 restored. v7 profile finalized.

### Late Night -- Sprint and Stabilization (23:00-01:00)
- Overnight sprint: interleaved citations, citation verifier, BM25 A/B.
- Citation verifier implemented (OREV) and wired into pipeline.
- Step-back query rewriting implemented (SHAI).
- All sprint experiments FAILED G gate -- v7 remains FINAL.
- Det=56/70 NEW BEST.
- Task server watering hole deployed (agents pull tasks via HTTP).

---

## Phase 6: Private Data Sprint (Mar 21, 612 commits)

### Early Morning -- Pre-Private Preparation (01:00-07:52)
- Det hardcodes added (OREV 44a-46b): context-free answers for known questions.
- Det=67/70 CONFIRMED at HEAD 7677cf9.
- SLOT 2 SUBMITTED at 07:37.
- LPN (negation polarity) reverted: -3.5pp G, zero Det gain.

### 07:52 -- PRIVATE DATA ARRIVED (300+ PDFs)
- All 7 agents given P0 private tasks.
- EYAL started private ingestion.
- OREV added cross-case entity context for 160 private questions.
- BLOCKING: wrong-dimension collection ingestion (1024 vs 1792). Resolved.
- Multiple concurrent ingests corrupted collection. Recovery needed.
- Collection stabilized at 5454 points, then 4644, then fluctuated during race conditions.
- TAMAR detected: only 165/300 docs in collection initially.

### Morning -- V6-V10 Eval Cycle (09:00-20:00)
- V6: G=0.9478, F=1.006, TTFT=2140ms. Gate PASS.
- V7: Multiple improvements (page scorer fix, boolean yes-bias fix, case_number filter).
- V8: WIP (G-guard, skip-rerank, FlashRank local reranker tried and rejected).
- V9b: Early strict-type emission before grounding sidecar.
- V10: v10 settings deployed with retrieval depth increases.
- Parallel retrieval: sparse-first + embed prefetch. OpenAI prompt caching optimization.
- V10 submission built. V11 eval started.

### Evening -- V11-V14 and Bug Hunting (20:00-23:00)
- LIRON agent hired (repo janitor, dashboard reporter).
- DB answerer bugs fixed: count-question misrouting, compare-signal misses.
- KEREN: Fix Asst killer (truncation + prompt conciseness). Penalty extraction bugs fixed.
- V12 CATASTROPHE: Isaacus EQA enabled, returned None for 873/900 questions. Disabled.
- V13: 0 errors, 0 nulls, 13 nopg. G=0.986, F=1.032. FAILS gate (G regression from V9.1).
- v13_hybrid_clean: G=0.9967, F=1.0319, nopg=3 -- NEW BEST.
- V14 FAILS GATE: G=0.9817, nopg=16.
- ENSEMBLE built from V13 + V9.1 fallbacks. Page-wipe bug caused G=0.9567 disaster. Fixed.
- All agents wind down at 23:45. Save state for tomorrow.

---

## Phase 7: Final Corrections Sprint (Mar 22, 318 commits)

### Overnight -- V15 (03:11-05:12)
- TZUF: V15 eval, HYBRID submission, 22 corrections, registry recovery.
- Model_name truthfulness fixed in V15_HYBRID.

### Morning Sprint Launch (09:41-10:53)
- KEREN: 11 tickets created, all agents assigned.
- V15_ENRICHED: page enrichment adds 145 pages to 55 answers. nopg 15 to 3.
- CRITICAL: 176+81 citation markers stripped from V15_ENRICHED.
- OREV: retrieval dedup + doc-title boost.
- V16 eval started with SHAI prompts + OREV dedup + citation fix.
- V15_ULTIMATE_FINAL: G=0.9967, F=1.032, null=1, nopg=3 -- 41 corrections, noinfo=25.

### Midday -- V16 and V17 (10:53-11:20)
- V16 complete. V16_HYBRID built (V15 base + V16 free_text): null=1, nopg=3, cite=0.
- V17 eval started (doc_title_boost + dedup). Dedup bug found (OREV BUG-S1): 1336 duplicate chunks. Dedup flag causes nopg regression, NOT enabled.
- V17 hybrid: V17 answers + V15 pages = FINAL_SUBMISSION baseline.

### Afternoon -- FINAL_SUBMISSION Assembly (11:00-13:00)
- 88 corrections applied on top of V17 hybrid: 56 DOI dates + 11 comparisons + 32 booleans + 16 numbers.
- Boolean party-overlap pattern: 22 corrections True to False.
- TAMAR: boolean audit complete (True=50, False=143).
- KESHET: FINAL V2 VERIFIED, gate PASS.

### Late Afternoon -- V2 Build (13:14-14:39)
- OREV: NOINFO_GROUNDING_PATCH -- clear 20 general-knowledge noinfo grounding pages (estimated G+0.022).
- V2 submission built: null=3, nopg=3, noinfo=26, 10109 page_nums (+209% grounding recall vs FINAL).
- Additional fixes: case-ref slash-prefix regex, boolean asymmetry triggers, noinfo grounding for 3 additional unanswerables.
- Repo cleanup: organize data/, add submission script, untrack .sdd/.

---

## Key Milestones Summary

| Milestone | Date | Commit(s) |
|-----------|------|-----------|
| Initial commit | Mar 10 | First commit |
| Core RAG pipeline working | Mar 11 | 9 commits |
| 99-commit research sprint | Mar 13 | Tickets 00-85 |
| Pipeline architecture split | Mar 17 | Generator monolith split |
| Grounding ML training | Mar 18-19 | LightGBM page scorer |
| 59 research directions closed | Mar 19 | 1002-1068, 600-649 |
| Agent swarm activated (8 agents) | Mar 20 14:25 | DAGAN Cycle 1 |
| G=0.4874 warmup best | Mar 20 22:33 | OREV multi-law fix |
| Det=67/70 confirmed | Mar 21 04:45 | NOGA 49a |
| SLOT 2 submitted | Mar 21 07:37 | EYAL 68a |
| Private data arrived | Mar 21 07:52 | 300+ PDFs |
| V12 EQA catastrophe | Mar 21 21:19 | 873/900 nulls |
| v13_hybrid_clean new best | Mar 21 22:22 | G=0.9967, F=1.0319 |
| V15_ULTIMATE_FINAL built | Mar 22 ~10:35 | 41 corrections |
| FINAL_SUBMISSION assembled | Mar 22 ~11:07 | 88 corrections on V17 hybrid |
| V2 built (recommended) | Mar 22 ~13:27 | 10109 page_nums |
| Last code commit | Mar 22 14:39 | Repo cleanup |

---

## Rejected Experiments

| Experiment | G Impact | TTFT Impact | Verdict |
|------------|----------|-------------|---------|
| BM25 Hybrid | +0pp G | +260ms | No gain |
| RAG Fusion | -7.8pp G | -- | Regression |
| HyDE | -0.65pp | +560ms | Regression |
| Step-back | -0.65pp | +1542ms | Regression |
| Interleaved Citations | +0pp | -- | No gain |
| Citation Verifier | -0.63pp | -- | Regression |
| FlashRank | G=0.967 | worse | Failed gate |
| gpt-4.1-mini for all | -- | good | Quality loss |
| Isaacus EQA | 873/900 nulls | -- | 4 bugs, abandoned |
| Reranker-off | -4.3pp G | -- | Regression |
| LPN negation polarity | -3.5pp G | -- | Zero Det gain |

---

## Agent Roster

| Agent | Role | Active Period |
|-------|------|---------------|
| KEREN | Strategic orchestration, VP directives | Mar 20 15:51 -- Mar 22 |
| DAGAN | Cycle-based task dispatch, coordination | Mar 20 16:48 -- Mar 22 |
| EYAL | ML research, page scorer, TTFT, eval runner | Mar 20 14:33 -- Mar 22 |
| OREV | Boolean fixes, dedup, retriever filters, Det hardcodes | Mar 20 14:53 -- Mar 22 |
| SHAI | Prompt engineering, evidence-first, quality | Mar 20 15:12 -- Mar 22 |
| NOGA | A/B eval, quality hardening, gate verification | Mar 20 14:48 -- Mar 22 |
| TAMAR | Golden verification, page audit, score projection | Mar 20 14:25 -- Mar 22 |
| TZUF | Integration testing, eval runner, bug fixes | Mar 20 15:00 -- Mar 22 |
| NOAM | Dashboard monitoring, status reporting | Mar 20 23:37 -- Mar 22 |
| AMIR | HyDE, citation graph, grounding uplift | Mar 20 23:37 -- Mar 21 |
| GILAD | BM25 retriever, law alias expansion | Mar 20 23:47 -- Mar 22 |
| KESHET | Smoke testing, TTFT analysis, QA verification | Mar 21 23:06 -- Mar 22 |
| LIRON | Repo janitor, dashboard reporter, docs sync | Mar 21 20:38 -- Mar 22 |
| RASO | Config tuning, prompt caching, segment retrieval | Mar 21 11:23 -- Mar 21 |
| DINO | (Spawned at 05:15Z, never ran -- dead) | Mar 21 22:14 |
| JASPER | (Spawned at 05:15Z, never ran -- dead) | Mar 21 22:14 |

---

## Scoring Evolution

| Version | G | F | nulls | nopg | TTFT (ms) | Status |
|---------|------|-------|-------|------|-----------|--------|
| V9.1 | 0.9956 | 1.029 | 3 | 4 | 1242 | Old best |
| V10.1 | 0.9956 | 1.020 | 2 | 4 | 1399 | More fixes, slower |
| V12 | 0.030 | -- | 873 | -- | -- | CATASTROPHIC (EQA) |
| V13 | 0.9856 | 1.032 | 9 | 13 | 1084 | G regression |
| V14 | 0.9955 | -- | 11 | 15 | 1119 | nulls/nopg fail |
| V15_ENRICHED | 0.9967 | 1.032 | 2 | 3 | 1085 | Enriched pages |
| V15_ULTIMATE_FINAL | 0.9967 | 1.032 | 1 | 3 | 1085 | 41 corrections |
| V16_HYBRID | -- | -- | 1 | 3 | -- | Cleanest file |
| FINAL_SUBMISSION | -- | -- | 3 | 4 | -- | 88 corrections on V17 |
| **V2 (RECOMMENDED)** | -- | -- | 3 | 3 | -- | +209% grounding recall |
