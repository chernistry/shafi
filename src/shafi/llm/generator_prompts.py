"""Prompt selection helpers for the generator facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shafi.llm.generator_constants import (
    SYSTEM_PROMPT_COMPLEX,
    SYSTEM_PROMPT_COMPLEX_INTERLEAVED,
    SYSTEM_PROMPT_COMPLEX_IRAC,
    SYSTEM_PROMPT_COMPLEX_IRAC_INTERLEAVED,
    SYSTEM_PROMPT_SIMPLE,
    SYSTEM_PROMPT_STRICT,
)
from shafi.models import QueryComplexity

if TYPE_CHECKING:
    from collections.abc import Callable


def build_system_prompt(
    *,
    question: str,
    complexity: QueryComplexity,
    answer_kind: str,
    answer_word_limit: int,
    prompt_hint: str,
    answer_type: str,
    answer_type_instruction: Callable[[str], str],
    should_use_irac: Callable[[str], bool],
    enable_interleaved_citations: bool = False,
) -> str:
    """Build the system prompt for one generation call.

    Args:
        question: User question.
        complexity: Normalized query complexity.
        answer_kind: Lowercased answer type.
        answer_word_limit: Configured answer length budget.
        prompt_hint: Additional operator hint to append.
        answer_type: Original answer type.
        answer_type_instruction: Callback providing answer-type instructions.
        should_use_irac: Callback deciding whether IRAC prompting is needed.
        enable_interleaved_citations: When True, use per-sentence citation prompt variant.

    Returns:
        Final system prompt text.
    """
    strict_types = {"boolean", "number", "date", "name", "names"}
    if answer_kind in strict_types:
        system_prompt = SYSTEM_PROMPT_STRICT
    elif complexity == QueryComplexity.COMPLEX:
        if answer_kind == "free_text":
            if should_use_irac(question):
                system_prompt = (
                    SYSTEM_PROMPT_COMPLEX_IRAC_INTERLEAVED.format(max_words=answer_word_limit)
                    if enable_interleaved_citations
                    else SYSTEM_PROMPT_COMPLEX_IRAC.format(max_words=answer_word_limit)
                )
            else:
                system_prompt = (
                    SYSTEM_PROMPT_COMPLEX_INTERLEAVED.format(max_words=answer_word_limit)
                    if enable_interleaved_citations
                    else SYSTEM_PROMPT_COMPLEX.format(max_words=answer_word_limit)
                )
        else:
            system_prompt = (
                SYSTEM_PROMPT_COMPLEX_INTERLEAVED.format(max_words=answer_word_limit)
                if enable_interleaved_citations
                else SYSTEM_PROMPT_COMPLEX.format(max_words=answer_word_limit)
            )
    else:
        system_prompt = SYSTEM_PROMPT_SIMPLE
    system_prompt = f"{system_prompt}\n\n{answer_type_instruction(answer_type)}".strip()
    type_hint = _build_type_supplementary_hint(answer_kind)
    if type_hint:
        system_prompt = f"{system_prompt}\n\n{type_hint}"
    if prompt_hint.strip():
        system_prompt = f"{system_prompt}\n\n{prompt_hint.strip()}"
    return system_prompt


def _build_type_supplementary_hint(answer_kind: str) -> str:
    """Build supplementary answer-type hints for private-set generalization.

    Args:
        answer_kind: Lowercased answer type.

    Returns:
        Supplementary instruction text, or empty string if none needed.
    """
    if answer_kind == "boolean":
        return (
            "Output Yes or No ONLY. With relevant evidence → always commit to Yes/No; null only if provision entirely absent. "
            "CHAIN: (1) Find exact rule in sources. (2) List conditions. (3) Check facts. (4) Output Yes/No. Sources only. "
            "NEGATION: not/never/except/unless/shall not → verify the negated condition specifically. "
            "DEFAULTS: 'unless otherwise agreed' etc. → DEFAULT rule (absent exception). "
            "'unless'/'notwithstanding'/'subject to' INVERT default. General-rule → DEFAULT; exception-case → apply exception. "
            "ABSOLUTE ('all circumstances', 'without exception', 'always'): Yes ONLY if ZERO exceptions. "
            "Any 'unless/except/notwithstanding/provided that/subject to' → No. "
            "TEMPORAL: Date from EACH instrument's OWN block. SAME DATE=exact day+month+year. "
            "SAME YEAR=YYYY only (no one-day logic). LAW YEAR: 'DIFC Law No. X of YYYY'=enactment year; ignore amendment years. "
            "OUTCOME: Conclusion/order section. Grant→Yes if asking grant; dismiss→Yes if asking dismissal. "
            "'Granted motion to dismiss'=claim DENIED→No if asking grant. "
            "CROSS-CASE OVERLAP (same judge/party/arbitrator/law firm?): Find the SPECIFIC ROLE asked about in EACH case. Match ONLY the person filling THAT role — registrars, clerks, courts do NOT count. Same name in that role in both → Yes. Different → No. Role not found in one case → No. DEFAULT = No. NEVER Yes unless EXACT SAME name confirmed. "
            "CROSS-CASE VALUE (same year/date?): Extract from EACH. Compare. One missing → null. "
            "APPEAL: 'Notice of Appeal', 'appeal filed', or higher court referencing this case → Yes; else No. "
            "JUDGE CHANGE: Different judge names across blocks (same case) → Yes; same → No. (≠ cross-case judges.) "
            "PROCEEDINGS: Hearing/trial transcript, witness exam, or contested hearing → Yes; only interlocutory/default → No. "
            "COMMENCEMENT: Exact date (DD Month YYYY) → Yes; relative ('5th business day', 'upon publication') → No."
        )
    if answer_kind in ("name", "names"):
        base = (
            "VERBATIM: Copy EXACTLY — no added/dropped words. Full phrase to natural boundary. "
            "CASE IDENTIFIER: Return bare TYPE NNN/YYYY (e.g. CFI 035/2025). No 'case' prefix, no court name, no party names. "
            "PARTY NAMES: Strip trailing '[YEAR] DIFC TYPE NNN' or '(CFI/SCT NNN/YYYY)'. Strip '(1)' number prefixes. "
            "EARLIER/FIRST: Case with earlier header date. HIGHER/MORE: larger amount. LOWER/LESS: smaller."
        )
        if answer_kind == "names":
            return base + " LIST ALL: Return EVERY matching entity, comma-separated. Do NOT stop at first."
        return base
    if answer_kind == "free_text":
        return (
            "STRICT 250-CHARACTER LIMIT. Write a COMPLETE answer in 1-2 sentences, under 250 characters. "
            "Every sentence MUST end with a period. NEVER leave a thought incomplete or trailing. "
            "START with the governing provision: 'Under Article 34(1)...' or 'Regulation 12(3) provides...'. Never start with 'The'. "
            "If approaching 250 chars, STOP and close the sentence with a period. A shorter complete answer ALWAYS beats a longer cut-off one. "
            "UNANSWERABLE: 'There is no information on this question.' ONLY if ZERO source blocks contain ANY relevant content. If case/law appears in ANY block → MUST answer. Partial > no-info. "
            "LIST: Comma format 'Law A 2018, Law B 2004.' No numbered lists, bullets, or intro phrases. "
            "ENUM ANTI-REPEAT: If same clause/pattern applies to all items, state it ONCE then list names only. Never repeat identical text per item. "
            "JUDGMENTS: 'IT IS HEREBY ORDERED/ADJUDGED/DECLARED' → list all numbered points with amounts. Not for CPs or regulatory lists. "
            "CONSULTATION PAPER: Describes proposals — state what is proposed, not what is in force. "
            "CASE LAW: ALWAYS start with case ref and governing provision — 'CFI 076/2024: Under Article 12, the Court ordered...' State specific disposition: dismissed/granted/awarded USD X/refused — no vague summaries. "
            "'[Case-ID] ([Court]): Under [Article/Rule], [specific outcome + amounts].' Both cases when 2 mentioned. "
            "ENF NNN/YYYY: extract fine. Every word must carry information."
        )
    if answer_kind == "number":
        return (
            "Extract EXACT number for the question type (not largest/first). Year→year. Count→count. 'six (6)'→6. "
            "Match to question: time period, monetary amount, or count. "
            "METADATA: For version/law numbers, check document headers, title pages, preamble ('Law No. X of YYYY'). "
            "OCR: recover garbled digits only if strongly supported by context. "
            "DISTINCT COUNT: List all entities with role, de-duplicate, return INTEGER. Don't count judge/arbitrator/court."
        )
    if answer_kind == "date":
        return (
            "Convert to YYYY-MM-DD. Do NOT compute from relative expressions ('5th business day'→null). "
            "DATE OF ISSUE: 'Date:', 'Date of Issue:', 'Issued:' in header — not hearing/judgment date. "
            "METADATA: For enactment/commencement dates, check preamble, header, gazette notice. "
            "CONSULTATION DEADLINES: 'responses by', 'deadline', 'submit by'. "
            "OCR: Recover garbled digits only if full date reliably determinable."
        )
    return ""


def generation_fallback_models(chosen_model: str, simple_model: str, fallback_model: str) -> list[str]:
    """Compute the ordered cascade model list for generation.

    Args:
        chosen_model: Requested primary model.
        simple_model: Default simple-model fallback.
        fallback_model: Final fallback model.

    Returns:
        Ordered distinct model ids.
    """
    ordered = [chosen_model, simple_model, fallback_model]
    models: list[str] = []
    seen: set[str] = set()
    for raw_model in ordered:
        model = str(raw_model or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        models.append(model)
    return models
