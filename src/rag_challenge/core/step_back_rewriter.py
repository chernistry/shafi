"""Step-back query rewriting for improved legal retrieval.

Abstracts specific clause questions to general-principle form before retrieval.
Validated on legal QA (Springer 2025): beats multi-query and decomposition.
Expected: +2-5pp recall on specific-clause questions.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_STEP_BACK_SYSTEM = (
    "You are a legal information retrieval expert. "
    "Rewrite the query to a more general, abstract form that will better retrieve "
    "relevant legal provisions. Remove specific article numbers, party names, and "
    "case-specific details. Focus on the underlying legal principle or rule being asked about. "
    "Output ONLY the rewritten query, nothing else."
)

_STEP_BACK_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Does Article 14(3)(b) of the Employment Law apply to contractors?",
        "rules governing contractor status and employment law applicability",
    ),
    (
        "What penalty does CFI 010/2024 impose on the defendant?",
        "penalties and remedies in commercial court judgments",
    ),
    (
        "What does Article 5(2)(a) say about overtime pay?",
        "overtime pay obligations under employment regulations",
    ),
)


async def rewrite_step_back(
    query: str,
    api_key: str,
    base_url: str | None = None,
    timeout_s: float = 10.0,
) -> str:
    """Rewrite query to abstract form for improved retrieval.

    On any failure, returns the original query unchanged.

    Args:
        query: Original user query.
        api_key: OpenAI-compatible API key.
        base_url: Optional custom base URL (e.g., Azure or proxy).
        timeout_s: Request timeout in seconds.

    Returns:
        Abstract rewritten query string, or original query on any failure.
    """
    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        examples_text = "\n".join(
            f"Original: {orig}\nRewritten: {rewritten}"
            for orig, rewritten in _STEP_BACK_EXAMPLES
        )
        user_msg = f"{examples_text}\n\nOriginal: {query}\nRewritten:"
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": _STEP_BACK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        if rewritten and len(rewritten) > 5:
            logger.debug("step_back: %r -> %r", query, rewritten)
            return rewritten
        return query
    except Exception:
        logger.debug("step_back rewrite failed for %r, using original", query, exc_info=True)
        return query
