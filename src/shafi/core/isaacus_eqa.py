"""Isaacus Extractive QA adapter for strict-type legal questions.

For name/number/date questions, bypasses LLM generation by calling the
Isaacus kanon-answer-extractor API. Returns the extracted answer span
and the 0-based passage index for chunk-level citation.

API endpoint: POST /v1/extractions/qa
Request body: {"model": str, "query": str, "texts": list[str]}
Response (actual Isaacus format):
    {
        "extractions": [
            {
                "index": int,                     # 0-based index into `texts` (passage)
                "inextractability_score": float,  # 0.0 = extractable, 1.0 = not
                "answers": [
                    {
                        "text": str,              # extracted answer span
                        "start": int,             # char offset in passage
                        "end": int,               # char offset in passage
                        "score": float            # confidence of this answer span
                    }
                ]
            }
        ],
        "usage": {"input_tokens": int}
    }

When inextractability_score >= threshold (default 0.5): answer is not
present in the provided passages → fall through to LLM generation.
When score < threshold: use extraction.text as the answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import httpx

logger = logging.getLogger(__name__)

_EQA_API_URL = "https://api.isaacus.com/v1/extractions/qa"
_EQA_MODEL = "kanon-answer-extractor"


@dataclass(frozen=True)
class EQAResult:
    """Result from an Isaacus extractive QA call.

    Attributes:
        answer: Extracted answer text (empty string when not extractable).
        passage_index: 0-based index of the source passage in the input texts.
        inextractability_score: 0.0 = easily extractable, 1.0 = not found.
        used: True when a valid extraction was returned below the threshold.
    """

    answer: str
    passage_index: int
    inextractability_score: float
    used: bool


async def call_isaacus_eqa(
    *,
    question: str,
    texts: list[str],
    api_key: str,
    api_url: str = _EQA_API_URL,
    model: str = _EQA_MODEL,
    inextractability_threshold: float = 0.5,
    timeout: float = 5.0,
) -> EQAResult | None:
    """Call the Isaacus kanon-answer-extractor API.

    Args:
        question: Natural-language question to extract an answer for.
        texts: Ordered list of passage texts to search (top-N context chunks).
        api_key: Isaacus API key (ISAACUS_API_KEY env var).
        api_url: Endpoint URL override.
        model: Model name override.
        inextractability_threshold: Score above this → not extractable.
        timeout: HTTP timeout in seconds.

    Returns:
        EQAResult with ``used=True`` when extraction succeeds, ``used=False``
        when the answer is inextractable, or ``None`` on API failure
        (caller should fall through to LLM generation).
    """
    if not api_key or not texts:
        return None

    payload: dict[str, object] = {
        "model": model,
        "query": question,  # Isaacus API uses "query" not "question"
        "texts": texts,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(api_url, json=payload, headers=headers)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        logger.warning("Isaacus EQA request failed: %s", exc)
        return None

    if resp.status_code in (429, 500, 502, 503, 504):
        logger.warning("Isaacus EQA HTTP %d (retryable)", resp.status_code)
        return None
    if resp.status_code >= 400:
        logger.warning("Isaacus EQA HTTP %d (permanent)", resp.status_code)
        return None

    try:
        data: object = resp.json()
    except Exception:
        logger.warning("Isaacus EQA returned non-JSON response")
        return None

    if not isinstance(data, dict):
        return None

    data_d = cast("dict[str, object]", data)

    # Actual Isaacus response: {"extractions": [{"index": int, "inextractability_score": float,
    # "answers": [{"text": str, "start": int, "end": int, "score": float}]}], "usage": {...}}
    extractions_raw = data_d.get("extractions")
    if not isinstance(extractions_raw, list) or not extractions_raw:
        logger.warning("Isaacus EQA: missing or empty 'extractions' in response")
        return None

    first_extraction: object = cast("list[object]", extractions_raw)[0]
    if not isinstance(first_extraction, dict):
        return None
    ext_d = cast("dict[str, object]", first_extraction)

    raw_score = ext_d.get("inextractability_score", 1.0)
    inext_score = float(raw_score) if isinstance(raw_score, (int, float)) else 1.0

    if inext_score >= inextractability_threshold:
        return EQAResult(
            answer="",
            passage_index=0,
            inextractability_score=inext_score,
            used=False,
        )

    answers_raw = ext_d.get("answers")
    if not isinstance(answers_raw, list) or not answers_raw:
        return None

    best_answer: object = cast("list[object]", answers_raw)[0]
    if not isinstance(best_answer, dict):
        return None
    ans_d = cast("dict[str, object]", best_answer)

    raw_text = ans_d.get("text", "")
    text = str(raw_text).strip() if raw_text else ""
    if not text:
        return None

    raw_idx = ext_d.get("index", 0)
    passage_index = int(raw_idx) if isinstance(raw_idx, int) else 0

    return EQAResult(
        answer=text,
        passage_index=passage_index,
        inextractability_score=inext_score,
        used=True,
    )
