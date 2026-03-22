# R006: Failure Taxonomy

A systematic catalogue of every failed approach in the DIFC RAG Legal Challenge 2026, with root cause analysis and domain-specific lessons.

---

## 1. Master Failure Table

| # | Experiment | Delta Metric | Category | Root Cause | Lesson |
|---|-----------|-------------|----------|------------|--------|
| F1 | RAG Fusion | G: -7.8pp | Retrieval | Multi-query RRF injects noise; legal queries are precise, not ambiguous | Query expansion harms precision in closed legal corpora |
| F2 | HyDE | G: -0.65pp, TTFT: +560ms | Retrieval | Hypothetical documents hallucinate non-existent legal provisions | Synthetic generation poisons retrieval in factual domains |
| F3 | Step-back rewriting | G: -0.65pp, TTFT: +1542ms | Retrieval | Over-abstraction loses article numbers and case references | Legal retrieval requires exact terms, not abstractions |
| F4 | BM25 Hybrid | G: +0pp, TTFT: +260ms | Retrieval | Dense retrieval already captures lexical signal in legal text | Redundant retrieval paths waste latency for zero gain |
| F5 | FlashRank reranker | G: 0.967 (fail gate), TTFT: 3483ms | Reranking | ONNX inference scales with chunk length; 1500-char legal chunks too long | Off-the-shelf rerankers fail on long legal passages |
| F6 | Citation verifier | G: -0.63pp net | Scoring | Verification step adds TTFT cost that exceeds marginal G gain | Post-hoc verification is net negative when F is multiplicative |
| F7 | Interleaved citations | G: +0pp | Generation | G is page-level, not passage-level; citation format is irrelevant | Grounding metric does not reward inline citations |
| F8 | gpt-4.1-mini for all types | TTFT: good, Asst: bad | Generation | Mini model lacks precision for complex free-text legal analysis | Cost of quality loss exceeds TTFT gain in multiplicative formula |
| F9 | Isaacus EQA (V12) | 873/900 nulls | Infrastructure | EQA returns None for most questions; no fallback path | New LLM components need slice tests before full deployment |
| F10 | V13 rerank cap | G: -1.0pp (9 new nopg) | Reranking | Candidate cap restricted page diversity for outlier questions | Average TTFT optimization can destroy tail retrieval |
| F11 | V14 null regression | null: 10+, nopg: 16+ | Infrastructure | Multiple changes confounded; routing bugs + dedup interacted | Never ship confounded changes |
| F12 | PAGE_ENRICHED / ULTIMATE | nopg: 3 to 32 | Infrastructure | Page-wipe commit removed retrieved_chunk_pages from no-info answers | Post-processing bugs can destroy submission silently |
| F13 | V15 raw (dedup enabled) | nopg: 43 | Infrastructure | PIPELINE_DEDUP_DUPLICATE_DOCS=true removes valid chunks along with dupes | Dedup at query time is destructive when index has collisions |
| F14 | gpt-4.1 for ALL free_text (V5) | TTFT: 5086ms, F: 0.973 | Generation | Without prompt caching, full model TTFT destroys F coefficient | Model upgrade without latency mitigation is net negative |
| F15 | Claude Sonnet for Asst | Never tested | Infrastructure | ANTHROPIC_API_KEY blocked for entire sprint duration | External dependencies must be validated on day 1 |
| F16 | SHAI-35a chain-of-logic | Det: -4 | Generation | Prompt change for boolean reasoning introduced regression on 4 answers | Prompt changes propagate unpredictably across question types |

---

## 2. Detailed Analysis by Category

### 2.1 Retrieval Failures

#### F1: RAG Fusion -- The Noise Amplifier

**Hypothesis:** Multiple reformulated queries merged via Reciprocal Rank Fusion (RRF) improve recall on ambiguous questions.

**What happened:** G dropped 7.8 percentage points. The worst regression in the project.

**Root cause:** Legal queries in this competition are precise, not ambiguous. A question like "What is the Date of Issue of CFI 022/2025?" has exactly one correct interpretation. RAG Fusion generated 3-5 query variants, each introducing slightly different lexical signals. RRF merged them, promoting pages that appeared across variants but were not relevant to the precise original query. In general-domain QA, query ambiguity makes multi-query retrieval valuable. In a closed legal corpus with specific case numbers and article references, the original query is already optimal.

**Academic context:** Raudaschl et al. (2023) show RAG Fusion improves on ambiguous open-domain queries. Our result is consistent with the known limitation: fusion degrades when the original query has high lexical specificity (Formal Information Retrieval, Manning et al., Ch. 9 -- query expansion hurts precision-oriented tasks).

---

#### F2: HyDE -- Hallucinated Legal Provisions

**Hypothesis:** Generate a hypothetical answer document, embed it, and retrieve similar real documents (Gao et al., 2022).

**What happened:** G dropped 0.65pp, TTFT increased 560ms.

**Root cause:** The LLM generated hypothetical answers that referenced plausible but non-existent legal provisions. For example, given "What fine was imposed in ENF 003/2024?", HyDE might generate a passage referencing "Article 47(2) of the Regulatory Law" -- which sounds correct but retrieves pages about Article 47 generally rather than the specific enforcement decision. In the legal domain, invented specifics are worse than the original query because they confidently point to the wrong pages.

**Academic context:** HyDE was designed for knowledge-intensive open-domain QA where the model's parametric knowledge provides useful signal (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"). In specialized legal corpora, the model's parametric knowledge about DIFC law is thin, making the hypothetical document a noise source rather than a signal amplifier.

---

#### F3: Step-Back Rewriting -- Losing Legal Precision

**Hypothesis:** Abstract the question to a broader form, retrieve on the abstraction, then answer the specific question (Zheng et al., 2023).

**What happened:** G dropped 0.65pp, TTFT increased 1,542ms (the largest latency penalty of any experiment).

**Root cause:** Step-back abstraction strips the very specifics that make legal retrieval work. "What is the Date of Issue of CFI 022/2025?" becomes something like "What are the key dates associated with CFI cases?" -- which retrieves dozens of irrelevant CFI case pages. Legal retrieval depends on exact case numbers, article references, and party names. Abstracting these away is actively harmful.

**The TTFT penalty:** An additional LLM call to generate the step-back question added ~1.5s. Under multiplicative scoring where F penalizes TTFT > 3s, this alone would reduce F from 1.02 to ~0.99, costing ~3pp Total before even accounting for G regression.

---

#### F4: BM25 Hybrid -- Redundant Signal

**Hypothesis:** Adding sparse BM25 retrieval alongside dense embeddings captures lexical matches that dense models miss.

**What happened:** G unchanged (+0pp), TTFT increased 260ms.

**Root cause:** Our dense retriever (Kanon 2, 1792-dim) already captured lexical signal in the legal domain. DIFC legal text has distinctive vocabulary (specific law names, case numbers, article references) that dense embeddings encode well. BM25 retrieved nearly identical page sets -- Jaccard similarity was high between dense-only and hybrid results. The 260ms overhead for sparse index lookup was pure waste.

**Why this contradicts general RAG wisdom:** Hybrid search is widely recommended (e.g., Pinecone, Weaviate docs) because general-domain embeddings often miss exact lexical matches. In a specialized legal corpus with domain-tuned embeddings, the dense model already handles lexical specificity. The BM25 contribution was already captured.

---

### 2.2 Reranking Failures

#### F5: FlashRank -- ONNX Does Not Scale

**Hypothesis:** Replace Zerank-2 cloud reranker with local FlashRank (ONNX-based) to eliminate network latency and improve TTFT.

**What happened:** G=0.967 (below 0.99 threshold), TTFT=3,483ms (above 1,500ms threshold). Both gates failed.

**Root cause:** Two independent issues:
1. **Latency:** ONNX inference time scales with input text length. Legal chunks average 1,500 characters (vs ~300 in general web text). Reranking 80 chunks at 1,500 chars each took 750ms -- far slower than the Zerank-2 API call.
2. **Quality:** Jaccard overlap between FlashRank and Zerank-2 page selections was only 0.676. FlashRank was trained on general-domain data and lacked legal domain understanding. It systematically ranked procedural boilerplate (headers, footers, standard clauses) higher than substantive legal content.

**Lesson:** Local rerankers are not free. Their latency depends on input size, and their quality depends on training domain. For specialized legal text, a cloud API with domain-aware training (Zerank-2) was both faster and better.

---

#### F10: V13 Rerank Candidate Cap -- Tail Risk

**Hypothesis:** Capping rerank candidates reduces average TTFT by processing fewer chunks.

**What happened:** Average TTFT improved 14% (1,242ms to 1,084ms). But G regressed 1.0pp due to 9 new nopg answers.

**Root cause:** The cap improved the average case but destroyed the tail. Questions referencing obscure provisions or multiple documents needed the full candidate set to surface relevant pages. The 6 identified regressions were all multi-document or cross-reference questions where the relevant pages ranked 9th-15th in the uncapped set. Capping at 8 excluded them.

**Lesson:** In a multiplicative formula, tail failures are catastrophic. Saving 158ms average TTFT (staying in the same F band) while losing 9 grounding answers costs far more than it saves. Optimize average only when the tail is protected.

---

### 2.3 Generation Failures

#### F7: Interleaved Citations -- Wrong Abstraction Level

**Hypothesis:** Embedding "[page X]" citations inline in the answer text will improve the grounding score.

**What happened:** G unchanged. Zero effect.

**Root cause:** The grounding metric operates at the page level, comparing the set of `retrieved_chunk_pages` in telemetry against the golden page set. It does not parse the answer text for citation markers. Interleaved citations are cosmetic -- they may help human readers but are invisible to the scoring function.

**Lesson:** Always verify which signal the metric actually reads before optimizing. We spent engineering time on a feature the scorer literally cannot see.

---

#### F8: gpt-4.1-mini for All Answer Types

**Hypothesis:** Using the faster mini model for all question types improves F coefficient enough to offset any Asst quality loss.

**What happened:** TTFT improved but Asst quality degraded on complex free-text questions.

**Root cause:** The mini model produced shorter, less grounded answers for complex legal analysis questions. It could handle simple boolean/number/date extraction (which have ground truth and are scored by Det, not Asst) but struggled with nuanced legal reasoning expected in free-text responses. The 5-criteria LLM judge penalized shallow analysis, vague phrasing, and missing legal citations.

**Resolution:** We adopted typed model routing: gpt-4.1-mini for strict types (boolean, number, date, name -- scored by Det only, no Asst impact) and gpt-4.1 for free_text and complex types (where Asst quality matters). This gave us fast TTFT on 70% of questions while preserving quality on the 30% where it counted.

---

#### F14: gpt-4.1 Without Prompt Caching (V5)

**Hypothesis:** Switching from gpt-4.1-mini to gpt-4.1 for free_text improves Asst quality.

**What happened:** Asst improved, but free_text TTFT=5,086ms. 15.4% of answers exceeded 5s (F=0.85 penalty). Net F=0.973, wiping out the Asst gain.

**Root cause:** gpt-4.1 with 1,500+ token system prompts has high time-to-first-token without prompt caching. The model processes the full prompt from scratch on every request.

**Resolution:** RASO prompt caching (commit cda03f0) restructured prompts so the static prefix exceeded 1,024 tokens, triggering OpenAI's automatic prompt cache. This cut free_text TTFT from 5,086ms to 1,931ms -- eliminating the F penalty while keeping the quality gain. The lesson: model upgrades and latency mitigation must be shipped together.

---

#### F16: SHAI-35a Chain-of-Logic Prompt

**Hypothesis:** Adding explicit chain-of-logic reasoning steps to the boolean prompt improves accuracy.

**What happened:** Det dropped by 4 points (4 boolean answers flipped from correct to incorrect).

**Root cause:** The chain-of-logic prompt encouraged the model to "reason through" boolean questions rather than extract the answer directly from evidence. For questions like "Was the Claimant successful?", the model would construct a plausible-sounding reasoning chain that reached the wrong conclusion, overriding the correct answer visible in the retrieved evidence. In legal QA, evidence-first extraction beats chain-of-thought reasoning when the answer is explicitly stated in the source material.

**Academic context:** This is consistent with findings from Shi et al. (2023, "Large Language Models Can Be Easily Distracted by Irrelevant Context") -- chain-of-thought can lead models astray when evidence is present but the reasoning chain introduces confounding logic.

---

### 2.4 Scoring Traps

#### F6: Citation Verifier -- TTFT Penalty Exceeds G Gain

**Hypothesis:** A post-generation step that verifies each cited page actually supports the answer will improve grounding precision.

**What happened:** Net Total: -0.29pp. Small G gain was overwhelmed by TTFT penalty.

**Root cause:** The verifier added an additional LLM call per answer (~300-500ms). Under the multiplicative formula, this pushed several answers from the 1-2s F band (F=1.02) into the 2-3s band (F=1.00), losing 2pp F on those answers. The G improvement from removing ~3 incorrect page citations was only +0.2pp G. Net: -0.29pp Total.

**Lesson:** In a multiplicative formula, any post-processing step that adds latency must clear a high bar: its G/quality improvement must exceed the F penalty with margin. For a 400ms step, the G improvement must be >0.5pp to break even. Most verification steps cannot clear this bar.

---

#### F12: Page-Wipe Bug -- Silent Submission Destruction

**Hypothesis:** (Not an experiment -- an infrastructure bug.)

**What happened:** PAGE_ENRICHED and ULTIMATE submissions had nopg=32, G=0.9644. Three separate submission files were silently corrupted.

**Root cause:** A commit that cleaned up no-info answer formatting also removed `retrieved_chunk_pages` from those answers. The build script ran without error. The submission passed basic validation (900 answers, correct schema). Only TAMAR's file-size monitoring (493KB to 517KB was suspicious) caught the regression.

**Lesson:** Submission sanity checks must include nopg count, not just schema validation. We added a pre-submit check (`scripts/pre_submit_sanity_check.py`) with 7 quality gates after this incident. Under multiplicative scoring, a silent grounding regression is existential.

---

### 2.5 Infrastructure Failures

#### F9: Isaacus EQA -- The V12 Catastrophe

**Hypothesis:** Isaacus EQA (span extraction model) can replace LLM generation for name/number/date questions.

**What happened:** 873/900 null answers. G=0.030. Total near zero. Compounded by a concurrent server crash (797 connection refused errors).

**Root cause:** Two independent failures:
1. EQA returned None for most questions it could not handle -- but the pipeline treated None as a valid "null" answer rather than falling back to LLM generation.
2. The server crashed during the first run attempt, adding connection errors that were promoted over successful retries in the checkpoint loader (see F11-related dedup bug).

**Lesson:** Never enable a new LLM component without a 50-question slice test first. The pipeline must have safe fallbacks for None/error returns -- null should only come from deliberate no-info decisions, not from component failures.

---

#### F13: Dedup at Query Time -- V15 nopg=43

**Hypothesis:** `PIPELINE_DEDUP_DUPLICATE_DOCS=true` removes 1,336 duplicate chunks (13% of index) at query time to improve retrieval quality.

**What happened:** nopg jumped from 3 to 43. The dedup logic removed valid chunks that shared content with duplicates.

**Root cause:** The Qdrant collection contained 1,336 duplicate chunks from a flawed ingestion run. The dedup flag was supposed to filter these out at query time, but the dedup logic used content hashing -- and some legitimate chunks from different pages had identical content (e.g., standard regulatory headers). Removing them eliminated grounding evidence for 40 additional questions.

**Resolution:** Never enabled in production. Used V15 pages (pre-dedup) as the grounding source, with V17 answers (post-prompt-improvements) for content. The lesson: fix data quality at ingestion, not at query time.

---

#### F15: Missing Anthropic API Key

**Hypothesis:** Claude Sonnet could improve Asst score (estimated +3pp Total from Asst 0.693 to 0.780).

**What happened:** Never tested. The ANTHROPIC_API_KEY was blocked for the entire sprint.

**Root cause:** External API dependency was not validated on day 1. By the time the blocker was discovered, there was no time to obtain and integrate the key.

**Lesson:** Validate ALL external API keys, model access, and rate limits on day 1 of any competition. Our biggest remaining gap (Asst=0.693 vs leader 0.833) might have been partially closed with a different model, but we will never know.

---

## 3. Counter-Intuitive Findings

Things that work in general-purpose RAG but fail in legal domain QA:

### 3.1 Query Expansion is Harmful

Standard RAG advice: expand queries to improve recall. In legal RAG, the original query contains precise identifiers (case numbers, article references, party names) that are already optimal search terms. Any expansion dilutes these. RAG Fusion (-7.8pp), HyDE (-0.65pp), and Step-back (-0.65pp) all demonstrated this.

**Why:** General-domain queries are often vague ("how to fix a leaky faucet"). Legal queries are specific ("What is the penalty under Article 47(2) of DIFC Law No. 1 of 2020?"). The specificity IS the retrieval signal.

### 3.2 More Context Can Be Faster

Increasing RERANK_TOP_N from 8 to 12 was expected to increase TTFT (more pages in the LLM context). Instead, TTFT decreased from 1.93s to 1.85s. More relevant context led to more concise, confident answers -- the model spent fewer tokens hedging and qualifying.

**Why:** LLM generation time depends on output tokens, not just input tokens. Better context reduces output uncertainty, leading to shorter answers. In legal QA, the model generates fewer "based on the available information" hedges when it has stronger evidence.

### 3.3 Hardcodes Beat Generic RAG

14 domain-specific hardcodes in `strict_answerer` (e.g., "if question asks about Date of Issue of CFI X, look up doi_lookup.json") contributed +10 Det points -- more than any generic retrieval improvement. Each took minutes to implement with zero regression risk (LLM fallback when pattern does not match).

**Why:** The competition uses a fixed legal corpus. Some question patterns repeat with different case numbers. A hardcoded lookup that maps "Date of Issue of CFI X" to the correct date from a verified registry is more reliable than any retrieval + extraction pipeline. This is domain-specific engineering, not generic RAG -- and it won more points than all retrieval experiments combined.

### 3.4 Prompt Caching Outperforms Model Selection

RASO prompt caching (+3.0pp Total via F improvement) delivered more Total score than switching from gpt-4.1-mini to gpt-4.1 for quality (+1.9pp estimated Asst improvement, partially offset by F penalty). Infrastructure optimization beat model optimization.

**Why:** The multiplicative formula amplifies latency improvements through the F coefficient. A 50% TTFT reduction that moves answers into a better F band is worth more than a quality improvement that is only weighted 0.3 in Q = 0.7*Det + 0.3*Asst.

### 3.5 Recall-Heavy Grounding Does Not Mean "Cite Everything"

Despite F-beta(2.5) weighting recall 6.25x over precision, bulk page enrichment was risky. Targeted enrichment (registry-based, document-specific) worked; blanket "add all pages from referenced documents" created precision collapse on questions with small golden sets. See the grounding paradox analysis in R005.

### 3.6 Chain-of-Thought Hurts Evidence Extraction

Chain-of-logic prompting (-4 Det) caused the model to override clear evidence with plausible-sounding but incorrect reasoning chains. For boolean questions where the answer is stated in the text ("The appeal was dismissed"), direct extraction beats reasoning. This contradicts the general advice to use CoT for complex questions.

**Why:** In legal QA with high-quality retrieved evidence, the answer is often explicitly stated. CoT adds a reasoning layer that can contradict the evidence. Evidence-first extraction ("read the text, extract the answer") is more reliable than "reason about the text, conclude the answer" when the text is authoritative.

---

## 4. Meta-Lessons

### 4.1 The Rejection Ratio

Of 16 significant experiments attempted, 14 were rejected (87.5%). Only 2 delivered measurable gains: RASO prompt caching and boolean prompt simplification. The rejection rate for retrieval-specific experiments was 100% (6/6 rejected).

This suggests that in a well-tuned legal RAG system with good embeddings and reranking, the retrieval stage is already near-optimal. Further gains come from post-retrieval optimization (generation quality, TTFT, telemetry) and domain-specific engineering (hardcodes, registry corrections).

### 4.2 The Confounding Problem

V14 shipped multiple changes simultaneously and failed. V12 had two independent failures (EQA None + server crash) that were initially diagnosed as one. The multiplicative formula makes confounded experiments especially dangerous because any G regression destroys the value of all co-shipped improvements.

**Rule that emerged:** One hypothesis per commit. One change per submission. Isolate variables or lose the ability to learn.

### 4.3 The Measurement Gap

Local evaluation systematically overestimated Det by ~4pp vs platform (TZUF-20a: local Det=59, platform Det=55). This meant experiments that appeared to improve Det locally sometimes showed no gain on platform. The only reliable signal was platform evaluation, which was slow and budget-limited.

**Rule that emerged:** Never trust local-only metrics. Budget platform evaluations for gating decisions. Use local metrics for smoke tests only.
