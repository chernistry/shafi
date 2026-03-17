"""Prompt selection helpers for the generator facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_challenge.llm.generator_constants import (
    SYSTEM_PROMPT_COMPLEX,
    SYSTEM_PROMPT_COMPLEX_IRAC,
    SYSTEM_PROMPT_SIMPLE,
    SYSTEM_PROMPT_STRICT,
)
from rag_challenge.models import QueryComplexity

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

    Returns:
        Final system prompt text.
    """
    strict_types = {"boolean", "number", "date", "name", "names"}
    if answer_kind in strict_types:
        system_prompt = SYSTEM_PROMPT_STRICT
    elif complexity == QueryComplexity.COMPLEX:
        if answer_kind == "free_text":
            if should_use_irac(question):
                system_prompt = SYSTEM_PROMPT_COMPLEX_IRAC.format(max_words=answer_word_limit)
            else:
                system_prompt = SYSTEM_PROMPT_COMPLEX.format(max_words=answer_word_limit)
        else:
            system_prompt = SYSTEM_PROMPT_COMPLEX.format(max_words=answer_word_limit)
    else:
        system_prompt = SYSTEM_PROMPT_SIMPLE
    system_prompt = f"{system_prompt}\n\n{answer_type_instruction(answer_type)}".strip()
    if prompt_hint.strip():
        system_prompt = f"{system_prompt}\n\n{prompt_hint.strip()}"
    return system_prompt


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
