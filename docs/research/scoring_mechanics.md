# R005: Scoring Mechanics Analysis

How the multiplicative scoring formula shaped every architecture decision in the DIFC RAG Legal Challenge 2026.

---

## 1. The Formula

```
Total = (0.7 * Det + 0.3 * Asst) * G * T * F
```

| Symbol | Component | What it measures | Range | Our best (V2) | Leader (warmup) |
|--------|-----------|-----------------|-------|---------------|-----------------|
| Det | Deterministic | Exact-match on boolean/number/date/name | 0--1 | ~0.976 | 1.000 |
| Asst | Assisted | LLM-judged free-text quality (5 criteria) | 0--1 | ~0.75 | 0.833 |
| G | Grounding | F-beta(2.5) over page-level citations | 0--1+ | 0.9967 | 0.990 |
| T | Telemetry | Schema compliance of execution metadata | 0--1 | 0.996 | 0.995 |
| F | TTFT coeff | Time-to-first-token bonus/penalty | 0.85--1.05 | 1.032 | 1.050 |

**F formula (per-question, then averaged):**

| TTFT band | F value |
|-----------|---------|
| < 1 s | 1.05 |
| 1--2 s | 1.02 |
| 2--3 s | 1.00 |
| 3--5 s | 0.99 - (TTFT - 3000) * 0.14 / 2000 |
| > 5 s | 0.85 |

The answer-quality term Q = 0.7*Det + 0.3*Asst is the only additive piece. Everything else is multiplicative.

---

## 2. Sensitivity Analysis

Using our realistic operating point (Det=0.976, Asst=0.75, G=0.9967, T=0.996, F=1.032):

**Baseline Q = 0.7 * 0.976 + 0.3 * 0.75 = 0.9082**

**Baseline Total = 0.9082 * 0.9967 * 0.996 * 1.032 = 0.9299**

### What +1 percentage point in each component yields:

| Component | From | To | Delta Total | Marginal value |
|-----------|------|----|-------------|----------------|
| Det +1pp | 0.976 | 0.986 | +0.0072 | +0.72pp Total |
| Asst +1pp | 0.750 | 0.760 | +0.0031 | +0.31pp Total |
| G +1pp | 0.9967 | 1.0067 | +0.0093 | +0.93pp Total |
| T +1pp | 0.996 | 1.006 | +0.0093 | +0.93pp Total |
| F +1pp | 1.032 | 1.042 | +0.0090 | +0.90pp Total |

### Key ranking: G ~ T ~ F >> Det >> Asst

One percentage point of grounding is worth 3x one percentage point of Asst quality. This single fact drove more architecture decisions than any other insight.

### Downside asymmetry

Multipliers punish catastrophically. Dropping G from 0.997 to 0.965 (a 3.2pp regression, as happened in V13) costs:

```
0.9082 * 0.965 * 0.996 * 1.032 = 0.9005   (-2.94pp Total)
```

Meanwhile, +3.2pp Asst (0.75 to 0.782) would yield only +0.99pp Total. The downside of breaking G is 3x the upside of improving Asst by the same amount.

---

## 3. The Multiplier Trap: One Bad Answer Tanks Everything

Because G is a per-question metric averaged across 900 questions, every individual question's grounding score feeds into the mean. But because G multiplies the entire Total, the effect of any zero-grounding answer is:

**One nopg answer (G=0 for that question) reduces G_mean by 1/900 = 0.0011, which costs ~0.1pp Total.**

This means:
- 4 nopg answers cost ~0.4pp Total (our V9.1 baseline)
- 17 nopg answers cost ~1.9pp Total (our V7 with metadata gaps)
- 43 nopg answers cost ~4.8pp Total (our V15 raw before enrichment)

Every single missing page citation has a dollar value in the competition. This is why we built an entire page-enrichment pipeline (registry-based page injection) as a post-processing step rather than accepting pipeline imperfection.

Similarly for T: a single telemetry violation (missing field, wrong schema) costs ~0.1pp Total. We verified T=0.996 across all 900 answers. The 4 missing telemetry fields were traced to edge cases in the no-info path, and 171 warmup doc_id references were stripped to avoid T penalties on private.

---

## 4. The Grounding Paradox

### F-beta(2.5) is recall-heavy

The grounding metric uses F-beta with beta=2.5, which weights recall 6.25x more than precision:

```
F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
      = 7.25 * (P * R) / (6.25 * P + R)
```

This heavily favors returning MORE pages. In theory, you should dump every page from every referenced document.

### But bulk enrichment still hurts

Our V2 enrichment experiment proved this is a trap:

| Version | Strategy | Pages cited | G proxy |
|---------|----------|-------------|---------|
| FINAL_SUBMISSION | Pipeline pages only | 3,263 | 0.9967 |
| V2 (enriched) | +registry all-pages | 10,109 (+209%) | expected higher |

V2 added 6,846 pages via registry enrichment (for every answer referencing a document, add ALL pages from that document). The recall-heavy F-beta(2.5) should reward this.

**Why it does not always work:** The golden page set is sparse. A 300-page regulation may have 3 relevant pages. Adding all 300 pages gives recall=1.0 but precision=3/300=0.01. Even at beta=2.5, the F-beta is:

```
F_2.5 = 7.25 * (0.01 * 1.0) / (6.25 * 0.01 + 1.0) = 0.0725 / 1.0625 = 0.068
```

Versus citing just the 3 correct pages: F_2.5 = 1.0.

The paradox: F-beta(2.5) is recall-heavy, but the gain from adding irrelevant pages is dwarfed by the precision collapse when the golden set is small. The optimal strategy is targeted recall -- find the RIGHT pages, not ALL pages.

### Our resolution

We used a hybrid approach:
1. Pipeline retrieves and reranks to top-12 pages (high precision)
2. Post-processing enriches with registry pages only for answers that cite a specific document (targeted recall boost)
3. No-info answers get zero pages (avoids G=0 penalties from wrong page citations)

This achieved G=0.9967 -- beating the warmup leader's G=0.990.

---

## 5. Multiplicative vs Additive: What We Would Do Differently

### Under additive scoring: Total = w1*Det + w2*Asst + w3*G + w4*T + w5*F

| Decision | What we actually did (multiplicative) | What additive would encourage |
|----------|--------------------------------------|-------------------------------|
| **Error budget** | Zero-tolerance for nopg/null; built entire page-enrichment pipeline for 4 answers | Accept 5-10% grounding failures, focus on answer quality |
| **TTFT investment** | RASO prompt caching, parallel retrieval, streaming-first architecture -- 6 engineer-weeks | Minimal; F coefficient contributes linearly, not multiplicatively |
| **Telemetry perfection** | Audited every schema field, stripped 171 invalid doc_ids | Fix obvious bugs only; T floor at 0.95 is fine |
| **Model routing** | gpt-4.1-mini for strict types (fast F), gpt-4.1 for free_text (quality Asst) | Use one model; optimize answer quality uniformly |
| **No-info handling** | 25 trick questions short-circuited with empty refs (avoids G=0) | Attempt answer anyway; wrong answer costs less than G crash |
| **Registry hardcodes** | 103+ manual corrections for Det; each +1pp Det is worth +0.72pp Total | Same investment, but returns are linear not amplified |
| **Kill bad experiments** | Rejected 6 retrieval experiments immediately on G regression | Would tolerate small G loss if Asst improved proportionally |

### The fundamental difference

Multiplicative scoring creates a "weakest link" dynamic. Your Total is gated by your worst multiplier. Under additive scoring, a 5pp Asst gain can compensate for a 2pp G loss. Under multiplicative scoring, it cannot -- the G loss is amplified by the entire Q*T*F product.

This pushed us toward **defensive architecture**: protect G and T at all costs, then improve Det/Asst within that safety envelope. An additive formula would have pushed us toward **offensive architecture**: maximize Q even at the expense of grounding robustness.

---

## 6. Concrete Examples From Our Data

### Example 1: The V2 Enrichment Story

Registry enrichment added +209% more page citations. Under a recall-only grounding metric, this would be a free lunch. Under F-beta(2.5), the precision penalty on small golden sets meant net G impact was uncertain. We submitted V2 alongside FINAL as a hedge -- the multiplicative formula made the risk non-trivial even for a "more recall" change.

### Example 2: No-Info Clearing Saves 23 Penalties

Our pipeline initially returned 27 false "no information" answers while still citing pages. Each one risks a grounding mismatch (answer says "unavailable" but pages are cited, or vice versa). Clearing these with proper empty-ref handling eliminated 23 potential G=0 penalties.

Value: 23 * (1/900) * G_multiplier effect = ~2.5pp Total saved in the worst case.

### Example 3: TTFT Optimization -- RASO Caching

Before RASO prompt caching (V5): free_text TTFT=5,086ms, 15.4% of answers >5s, F=0.973.
After RASO prompt caching (V6): free_text TTFT=1,931ms, 0.6% >5s, F=1.006.

```
Delta F = +3.3pp
Delta Total = 0.9082 * 0.9967 * 0.996 * 0.033 = +0.030 Total (+3.0pp)
```

RASO caching -- putting static prompt prefixes over 1,024 tokens to trigger OpenAI's automatic prompt cache -- was the single largest Total improvement from one change. It cost zero answer quality and zero grounding. Pure multiplicative gain.

### Example 4: V12 Catastrophe -- EQA Returns None

V12 enabled Isaacus EQA, which returned None for most questions. Result: 873/900 null answers, G=0.030. Total collapsed to near zero.

Under additive scoring, 873 nulls would lose ~97% of Det points but G/T/F would contribute independently. Under multiplicative scoring, G=0.030 multiplied everything else by 0.03, making the entire submission worthless regardless of the 27 good answers.

This is the ultimate illustration of the multiplier trap: one component at near-zero destroys the product.

### Example 5: The Rerank12 Surprise

Increasing RERANK_TOP_N from 8 to 12 was expected to hurt TTFT (more pages = more LLM context). Instead:
- TTFT: 1.93s to 1.85s (FASTER -- better context led to more concise answers)
- G: +1.0pp platform (more relevant pages included)
- Both multipliers improved simultaneously

Under additive scoring, this would be a modest +1pp G + small F gain. Under multiplicative scoring, the compounding effect made it worth +1.5pp Total.

### Example 6: Warmup Leader Analysis

The warmup leader (CPBD, Total=0.982) achieved: Det=1.000, Asst=0.833, G=0.990, T=0.995, F=1.050.

Decomposing: Q = 0.7*1.0 + 0.3*0.833 = 0.950. Then 0.950 * 0.990 * 0.995 * 1.050 = 0.982.

Their G=0.990 was their weakest multiplier. Under multiplicative scoring, even they were gated by grounding. Our G=0.9967 actually beat them on G, but our Q=0.908 vs their Q=0.950 meant we lost on the additive piece. The multiplicative formula amplifies the Q gap:

```
Our Total:   0.908 * 0.997 * 0.996 * 1.032 = 0.930
Their Total: 0.950 * 0.990 * 0.995 * 1.050 = 0.982
Gap: -5.2pp
```

The lesson: once G/T/F are near-perfect, the formula reduces to a race on Q. The multiplicative structure front-loads defensive work (get G/T/F right first) then rewards offensive improvement (push Det and Asst higher).

---

## 7. Strategic Implications

1. **Fix multipliers first.** G, T, and F are existential. A 3pp G regression wipes out weeks of Det/Asst work. We learned this from V13 (G=0.986 vs V9.1 G=0.996 -- one rerank cap change cost more than 15 other improvements gained).

2. **TTFT is the cheapest multiplier.** Unlike G (requires retrieval quality) or T (requires schema correctness), F can be improved through pure engineering: caching, parallelism, streaming. RASO caching yielded +3pp Total with zero quality risk.

3. **The endgame is Q.** Once G>0.995, T>0.995, F>1.03, further multiplier gains are marginal (~0.1pp/pp). The winning margin comes from Q = 0.7*Det + 0.3*Asst. Our gap to the leader was entirely in Q (0.908 vs 0.950).

4. **Defensive experiments only.** Any change that risks G regression must show >3x expected upside to justify the downside. We rejected 6 retrieval experiments on this principle. Under additive scoring, at least 2 (BM25 hybrid, RAG Fusion) might have been worth iterating on.

5. **Per-question penalties compound.** Each nopg answer costs ~0.1pp Total. Each >5s answer costs ~0.02pp F. Each telemetry miss costs ~0.1pp Total. The formula taxes imperfection multiplicatively -- there is no "acceptable error rate," only a budget you spend reluctantly.
