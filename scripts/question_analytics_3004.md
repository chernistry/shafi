# Ticket 3004: Question Analytics — Statistical Profiling
# Completed by TAMAR, 2026-03-22

## Dataset: 900 private questions, V15_HYBRID submission

---

## 1. ANSWER TYPE DISTRIBUTION

| Type       | Count | % | Avg TTFT | Null | NoPg |
|------------|------:|--:|--------:|-----:|-----:|
| free_text  |   270 | 30.0% | 1668ms |  0   |   3  |
| boolean    |   193 | 21.4% |  829ms |  0   |   0  |
| number     |   159 | 17.7% | 1223ms |  1   |   0  |
| name       |    95 | 10.6% |  807ms |  1   |   0  |
| date       |    93 | 10.3% |  634ms |  0   |   0  |
| names      |    90 | 10.0% |  403ms |  0   |   0  |

**Key insight**: free_text is 3.7× slower than names. names/date/name/boolean all <1s avg.

---

## 2. CASE-LAW vs REGULATORY

| Category         | Count | % | Avg TTFT | Null | NoPg |
|------------------|------:|--:|--------:|-----:|-----:|
| Case-law (case ref) |  505 | 56.1% | — | — | — |
| Regulatory (no case) | 395 | 43.9% | 1663ms | — | — |

- **Regulatory free_text = main bottleneck**: 244 questions, avg ~2000ms, 23 questions >3s
- Case-law questions: handled fast by DB answerer for cross-case, moderate for single-case

---

## 3. CROSS-CASE vs SINGLE vs REGULATORY BREAKDOWN

| Pattern            | Count | % | Avg TTFT | Null | NoPg |
|--------------------|------:|--:|--------:|-----:|-----:|
| Cross-case (2+ refs)|  178 | 19.8% |  553ms |  1   |   0  |
| Single-case (1 ref) |  327 | 36.3% |  678ms |  1   |   2  |
| No case refs       |  395 | 43.9% | 1663ms |  0   |   1  |

**Cross-case type breakdown**: boolean=102, name=73, names=2, date=1
- DB answerer handles 102+73=175 of 178 cross-case questions efficiently

---

## 4. QUESTION LENGTH

- Min: 5 words, Max: 80 words, Mean: 17.6, Median: 14.0
- 1-10w: 117 | 11-15w: 385 | 16-20w: 183 | 21-30w: 125 | >30w: 90

**TTFT by length**:
| Length  | n  | Avg TTFT |
|---------|----|---------|
| 1-10w   | 117 | 1315ms |
| 11-15w  | 385 |  795ms |
| 16-20w  | 183 | 1053ms |
| 21-30w  | 125 | 1352ms |
| >30w    |  90 | 1721ms |

Longer questions = slower TTFT. 11-15w is the sweet spot (most common, fastest).

---

## 5. CASE TYPE BREAKDOWN

| Court | Count | % |
|-------|------:|--:|
| SCT   |  201  | 22.3% |
| CFI   |  183  | 20.3% |
| ARB   |   41  |  4.6% |
| CA    |   40  |  4.4% |
| TCD   |   29  |  3.2% |
| ENF   |   11  |  1.2% |
| None  |  395  | 43.9% |

---

## 6. TOP LEGAL KEYWORDS (top 10)

SCT(338), CFI(264), law(247), difc(244), date(105), under(104), regulations(102),
court(85), document(85), issue(77)

---

## 7. NUMBER QUESTION SUBTYPES

| Sub-type           | Count | Avg TTFT |
|--------------------|------:|--------:|
| "How many/total"   |    77 | 1033ms |
| Amount/fine/penalty|    58 | 1682ms |

Amount questions are 60% slower (require full doc retrieval to find monetary values).

---

## 8. ⚠️ CRITICAL FINDING: 28 "No Information" Answers — Avg 2598ms TTFT

28 questions return "There is no information on this question." — but go through
full pipeline BEFORE deciding this. Avg TTFT = 2598ms.

**Subcategory A: 15+ OBVIOUS TRICK QUESTIONS (non-legal content)**
These are entirely outside the legal domain. They waste 3-7s on useless retrieval.

| QID prefix  | TTFT  | Question |
|-------------|------:|---------|
| a32ee724    | 6755ms | What is the largest mammal in the world? |
| 4a4a1bd4    | 4256ms | What is the speed of light in a vacuum? |
| 9d353a5b    | 3477ms | What gas do plants absorb during photosynthesis? |
| bef758a8    | 3356ms | What does 'CPU' stand for in computing? |
| 96a9f58b    | 3341ms | What is the largest planet in the Solar System? |
| 0d5fed78    | 3321ms | What is the smallest country in the world by area? |
| 5101dcb7    | 3205ms | What is the name of the largest ocean on Earth? |
| d01eb8ea    | 3160ms | Who developed the theory of relativity? |
| 36d17890    | 3079ms | In which year did the Berlin Wall fall? |
| eeb35993    | 2941ms | Who wrote the play 'Romeo and Juliet'? |
| fd48ed84    | 2895ms | In which country was the first modern Olympic Games held? |
| 8cbd7b17    | 2861ms | What is the square root of 144? |
| e498ae14    | 2755ms | Who was the first person to walk on the Moon? |
| 13a42ce6    | 2514ms | What is the currency of the United Kingdom? |
| 5cfb2abc    | 2231ms | What is the average distance from the Earth to the Moon? |
| 33e56b2b    | 2172ms | What is the freezing point of water in Celsius? |
| 01de4aad    | 2048ms | How many players are on a football team on the field at one time? |
| 0ef0cdb6    | 2032ms | Who painted the Mona Lisa? |
| cfdc61c3    | 1745ms | What is the chemical symbol for gold? |
| 3392b510    | 1626ms | What is the currency of Japan? |

**Subcategory B: 8 legal unanswerable (doc absent/out-of-scope)**
712913eb, 8780a9d4, 83e54d9d (confirmed nopg=0), plus others.

---

## 9. F FACTOR IMPACT ANALYSIS

**Current TTFT distribution (V15_HYBRID)**:
- <1s  (F=1.05): 505 (56.1%)
- 1-2s (F=1.02): 246 (27.3%)
- 2-3s (F=1.00): 119 (13.2%)
- 3-5s (F<0.99):  29  (3.2%)
- >5s  (F=0.85):   1  (0.1%)

**Avg F = 1.0319**

**Trick question fast-path opportunity**:
- 28 no-info questions currently avg 2598ms → avg F ≈ 1.002
- If fast-pathed to <500ms: all get F=1.05 → gain = +0.047 per question
- Averaged over 900: **delta avg F = +0.0019, delta Total = +0.0017**

**All 29 questions >3s moved to <1s**:
- Delta avg F = +0.0032, delta Total = +0.0029

---

## 10. ACTIONABLE RECOMMENDATIONS

### HIGH ROI — IMPLEMENT IF TIME:
**TRICK QUESTION FAST-PATH** (+0.0017-0.0029 Total)
- Owner: OREV (query classifier) or DAGAN (pipeline)
- Action: At classify step, detect non-legal questions (no legal keywords, trivial general knowledge)
- Use embeddings or keyword: if question has no DIFC/court/regulation/article/contract terms → null immediately
- Impact: 20+ questions go from 2-7s to <500ms
- Risk: LOW — answers are already correct ("no information"), we just return faster
- Cost: ~2h (code + server restart + 28-question re-run + rebuild)

### DATA INSIGHT — ALREADY KNOWN:
- Free_text is the Asst bottleneck: 30% of questions, slowest TTFT, most complex
- Cross-case questions already well-handled by DB answerer (553ms avg)
- Amount/penalty numbers are slow (1682ms) — full retrieval required

### LOW ROI:
- Optimizing "how many" number questions: already 1033ms (fine)
- Regulatory non-free_text: boolean/name/date are all fast even without case refs

---

## FINDING SUMMARY FOR KEREN

```
TAMAR TICKET 3004: Statistical profiling complete.

DATASET: 30% free_text, 21.4% boolean, 17.7% number. 56.1% case-law, 43.9% regulatory.
PERFORMANCE: names fastest (403ms), free_text slowest (1668ms). Cross-case 553ms (DB answerer).

KEY DISCOVERY: 28 questions return "no information" with avg TTFT=2598ms.
  - 20 are CLEARLY NON-LEGAL trick Qs (mammal, CPU, Berlin Wall, Mona Lisa, etc.)
  - These waste 2-7s on useless retrieval
  - Fast-path to null in <500ms: +0.0017 Total score
  - Risk: LOW (answers are already correct — just makes them faster)

F DISTRIBUTION: 56.1% <1s (F=1.05), 3.2% 3-5s (F<0.99), 1 >5s (F=0.85).
FULL REPORT: scripts/question_analytics_3004.md
```
