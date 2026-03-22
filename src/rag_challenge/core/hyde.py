"""HyDE (Hypothetical Document Embeddings) for query expansion.

Generates a hypothetical legal document excerpt that would answer the query,
then embeds it alongside the original query to improve retrieval recall.
"""

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from rag_challenge.config.settings import Settings

logger = logging.getLogger(__name__)

HYDE_SYSTEM = (
    "You are a legal document generator. Generate a short excerpt (50-100 words) "
    "from a legal regulation that would directly answer the following question. "
    "Write as if it's from the actual regulation text, not as an answer to a question. "
    "Be specific, use legal language, cite article numbers if relevant."
)


async def generate_hypothetical_document(
    query: str,
    settings: "Settings",
) -> str:
    """Generate a hypothetical answer document for HyDE query expansion.

    Args:
        query: The user's question.
        settings: Application settings with LLM configuration.

    Returns:
        str: Generated hypothetical legal document excerpt.
    """
    llm_settings = settings.llm
    api_key = llm_settings.resolved_api_key().get_secret_value()

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=llm_settings.base_url,
        timeout=max(30.0, float(llm_settings.timeout_s)),
    )

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1-mini",  # Fast, cheap model for HyDE
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        content = resp.choices[0].message.content or ""
        hypo_doc = content.strip()
        if not hypo_doc:
            logger.warning("HyDE returned empty content")
            return ""
        logger.debug("HyDE hypothetical document generated: %s", hypo_doc[:200])
        return hypo_doc
    except Exception as exc:
        logger.warning("HyDE generation failed, returning empty: %s", exc)
        return ""
