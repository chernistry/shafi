# Prompt Regression Safety Review — SHAI

Review of all prompt changes for warmup regression risk.

## Methodology
Every change classified as SAFE (no warmup regression risk), MINOR RISK (possible but unlikely), or CONCERN (needs monitoring).

## File-by-file Review

### generator_system_strict.md
| Change | Assessment |
|--------|-----------|
| Added generic disambiguation examples ("Companies Act" etc.) | SAFE — additive to existing "X Law" ≠ "X Law Amendment Law" rule which remains |
| "article number" → "provision (article, section, clause, or rule)" | SAFE — broader, still catches articles |
| Added CRITICAL Grounding section | SAFE — reinforces existing "If you are not 100% certain... output null" rule |
| Added jurisdiction neutrality | SAFE — purely additive |

### generator_system_simple.md
| Change | Assessment |
|--------|-----------|
| Added 280-char truncation awareness | SAFE — additive |
| Added citation quality rules | SAFE — additive |
| Added jurisdiction neutrality | SAFE — additive |

### generator_system_complex.md / complex_irac.md
All changes are purely additive new bullet points. SAFE.

### generator_user_strict.md
| Change | Assessment |
|--------|-----------|
| "PRINCIPAL law" → "PRINCIPAL instrument" | SAFE — generalization, same semantics |
| Removed DIFC-specific year example, replaced with generic | MINOR RISK — loses the specific example but the abstract instruction conveys identical meaning |
| "six (6) years" → general count/quantity rule | SAFE — same concept, more general |
| Added preposition preservation for names | SAFE — strengthens existing behavior |
| Added "do NOT compute or infer dates" | SAFE — strengthens existing anti-arithmetic rule |
| "article reference" → "provision reference" + prohibition/permission/exception keywords | SAFE — generalization that enhances boolean handling |
| Verification steps: broadened provision types | SAFE — generalization |
| "enactment notice page" → "exact passage" | SAFE — generalization |

### generator_user.md
All changes are purely additive new sections. SAFE.

### generator_prompts.py
| Change | Assessment |
|--------|-----------|
| `_build_type_supplementary_hint()` for boolean | SAFE — injected between answer_type_instruction and prompt_hint, does not modify any existing prompt text |
| `_build_type_supplementary_hint()` for name/names | SAFE — same injection point |

## Potential Risks (all MINOR)

1. **Grounding rule may increase null rate**: "If the answer requires inference... output null" could cause more null answers on borderline strict-type questions. For warmup with good retrieval, answers should be directly extractable, so risk is low. For private set, higher precision is desirable.

2. **Removed DIFC-specific year example**: The specific "Employment Law Amendment Law (enacted 2021) ≠ Employment Law (enacted 2019)" example was replaced with "An amendment instrument and its parent instrument are different and may have different years". Same concept, less specific. Very low risk of regression.

3. **280-char warning may shorten complex answers**: Could cause the LLM to generate shorter complex answers. This is actually beneficial since answers are truncated to 280 chars anyway — front-loading key information improves truncated answer quality.

## Conclusion
All changes are ADDITIVE — no existing instructions were removed. Every existing rule retains its semantic content. The only modifications are generalizations of DIFC-specific patterns to generic legal patterns. Overall warmup regression risk: **LOW**.
