"""Token counting and usage accounting helpers for the generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging

    from rag_challenge.llm.generator_types import TokenEncoder


def count_tokens(text: str, encoding: TokenEncoder) -> int:
    """Count tokens using the configured encoder.

    Args:
        text: Source text.
        encoding: Token encoder.

    Returns:
        Number of encoded tokens.
    """
    return len(encoding.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, encoding: TokenEncoder) -> str:
    """Trim text to a token budget.

    Args:
        text: Source text.
        max_tokens: Maximum allowed tokens.
        encoding: Token encoder.

    Returns:
        Token-truncated text.
    """
    if max_tokens <= 0:
        return ""
    token_ids = list(encoding.encode(text))
    if len(token_ids) <= max_tokens:
        return text
    if max_tokens <= 3:
        return encoding.decode(token_ids[:max_tokens])
    return f"{encoding.decode(token_ids[: max_tokens - 1]).rstrip()}..."


def estimate_usage(
    *,
    system_prompt: str,
    user_prompt: str,
    completion_text: str,
    encoding: TokenEncoder,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    """Estimate token usage when the provider does not report it.

    Args:
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        completion_text: Completion text.
        encoding: Token encoder.
        logger: Logger used for audit-safe fallback reporting.

    Returns:
        Estimated prompt, completion, and total tokens.
    """
    prompt_tokens = count_tokens(system_prompt, encoding) + count_tokens(user_prompt, encoding)
    completion_tokens = count_tokens(completion_text, encoding) if completion_text.strip() else 0
    total_tokens = prompt_tokens + completion_tokens
    logger.info(
        "llm_usage_provider_missing_fallback",
        extra={
            "prompt_tokens_est": prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "total_tokens_est": total_tokens,
        },
    )
    return (prompt_tokens, completion_tokens, total_tokens)


def resolve_usage(
    *,
    system_prompt: str,
    user_prompt: str,
    completion_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    encoding: TokenEncoder,
    logger: logging.Logger,
) -> tuple[int, int, int]:
    """Resolve provider-reported usage with a safe estimate fallback.

    Args:
        system_prompt: System prompt text.
        user_prompt: User prompt text.
        completion_text: Completion text.
        prompt_tokens: Provider-reported prompt tokens.
        completion_tokens: Provider-reported completion tokens.
        total_tokens: Provider-reported total tokens.
        encoding: Token encoder.
        logger: Logger used for fallback reporting.

    Returns:
        Final prompt, completion, and total token counts.
    """
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    total = max(0, int(total_tokens))
    if total > 0 and (prompt > 0 or completion > 0):
        if total < (prompt + completion):
            total = prompt + completion
        return (prompt, completion, total)
    return estimate_usage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        completion_text=completion_text,
        encoding=encoding,
        logger=logger,
    )
