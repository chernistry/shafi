"""Post-hoc citation verification: checks each cited page actually supports the answer.

Called AFTER answer generation (no TTFT impact). Uses a lightweight LLM with a
YES/NO prompt per cited page to filter unsupported citations.

Literature basis:
- VeriCite (SIGIR-AP 2025): +6-8pp Citation F1 on Llama3-8B and Qwen2.5.
- VeriFact-CoT (2024): +15pp Citation F1 with NLI-based verification.

Feature flag: PIPELINE_ENABLE_CITATION_VERIFICATION (default False).
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from rag_challenge.models import RankedChunk

logger = logging.getLogger(__name__)

_VERIFY_SYSTEM = (
    "You are a citation verifier. Given an answer and a source passage, "
    "determine if the passage DIRECTLY supports any claim in the answer. "
    "Answer with exactly one word: YES or NO."
)

_VERIFY_USER_TEMPLATE = """\
Answer: {answer}

Source passage: {passage}

Does this passage directly support any claim in the answer? YES or NO:"""

_MAX_PASSAGE_CHARS = 600  # keep prompt short for speed
_MAX_ANSWER_CHARS = 1000  # avg GPT-4.1 legal answer is ~536 chars; 300 caused 30-50% false drops (NOGA-42c)
_MAX_CONCURRENT = 5  # limit parallel API calls


async def _verify_single(
    answer: str,
    chunk_text: str,
    client: AsyncOpenAI,
    model: str,
) -> bool:
    """Return True if chunk_text supports the answer.

    Args:
        answer: The generated answer text.
        chunk_text: The cited passage text to verify.
        client: Async OpenAI-compatible client.
        model: Model name to use for verification.

    Returns:
        True if the passage supports the answer (or on API error — safe default).
    """
    passage = chunk_text[:_MAX_PASSAGE_CHARS]
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": _VERIFY_USER_TEMPLATE.format(
                        answer=answer[:_MAX_ANSWER_CHARS],
                        passage=passage,
                    ),
                },
            ],
            max_tokens=3,
            temperature=0.0,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("Y")
    except Exception:
        logger.debug("Citation verification API call failed — keeping citation", exc_info=True)
        return True  # safe default: keep citation on error


async def verify_citations(
    answer: str,
    cited_chunks: list[RankedChunk],
    api_key: str,
    base_url: str | None = None,
    model: str = "gpt-4.1-mini",
    timeout_s: float = 15.0,
) -> list[RankedChunk]:
    """Filter cited_chunks to only those that directly support the answer.

    Runs all verification calls in parallel (bounded by semaphore). Falls back
    gracefully on API errors (keeps citation). Never returns an empty list —
    if all citations would be dropped, returns the original list unchanged.

    Args:
        answer: The generated answer text.
        cited_chunks: Chunks currently cited in the answer.
        api_key: OpenAI-compatible API key.
        base_url: Optional custom API base URL (e.g. OpenRouter).
        model: Model to use for YES/NO verification.
        timeout_s: Per-request timeout in seconds.

    Returns:
        Subset of cited_chunks verified to support the answer.
        Returns original list unchanged if all citations would be dropped.
    """
    if not cited_chunks or not answer.strip():
        return cited_chunks

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _check(chunk: RankedChunk) -> tuple[RankedChunk, bool]:
        async with semaphore:
            chunk_text = getattr(chunk, "text", "") or getattr(chunk, "content", "") or ""
            supported = await _verify_single(answer, chunk_text, client, model)
            return chunk, supported

    results = await asyncio.gather(*(_check(c) for c in cited_chunks))
    verified = [chunk for chunk, ok in results if ok]

    if not verified:
        # Safety guard: never drop ALL citations — return original to avoid score cliff
        logger.debug(
            "Citation verifier dropped all %d citations — keeping original set",
            len(cited_chunks),
        )
        return cited_chunks

    dropped = len(cited_chunks) - len(verified)
    if dropped:
        logger.debug(
            "Citation verifier dropped %d/%d unsupported citations",
            dropped,
            len(cited_chunks),
        )

    return verified
