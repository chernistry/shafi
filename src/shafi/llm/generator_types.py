"""Shared typing helpers for the generator package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

type CaseOutcomeCandidate = tuple[int, str, str]
type RecoveredTitleCandidate = tuple[str, str]


class TokenEncoder(Protocol):
    """Protocol for token encoders used by generator token helpers.

    Args:
        text: Text to encode or decode.

    Returns:
        Token ids for ``encode`` and decoded text for ``decode``.
    """

    def encode(self, text: str) -> Sequence[int]:
        """Encode text into token ids.

        Args:
            text: Source text.

        Returns:
            Encoded token ids.
        """
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token ids back to text.

        Args:
            tokens: Token ids.

        Returns:
            Decoded text.
        """
        ...
