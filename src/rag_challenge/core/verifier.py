from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from rag_challenge.config import get_settings
from rag_challenge.prompts import load_prompt

if TYPE_CHECKING:
    from rag_challenge.llm.provider import LLMProvider
    from rag_challenge.models import RankedChunk

logger = logging.getLogger(__name__)

_STRONG_ASSERTION_RE = re.compile(
    r"\b(must|shall|required|prohibited|illegal|mandatory|obligated|entitled)\b",
    re.IGNORECASE,
)
_CITE_RE = re.compile(r"\(cite:\s*[^)]+\)", re.IGNORECASE)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_VERIFY_SYSTEM_PROMPT = load_prompt("verifier/system")
_VERIFY_USER_PROMPT = load_prompt("verifier/user")


@dataclass(frozen=True)
class VerificationResult:
    is_grounded: bool
    unsupported_claims: list[str]
    revised_answer: str
    verified: bool = True


class AnswerVerifier:
    """Conditional verifier that checks answer grounding against retrieved chunks."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._settings = get_settings()

    def should_verify(
        self,
        answer: str,
        cited_chunk_ids: list[str] | tuple[str, ...],
        *,
        force: bool = False,
    ) -> bool:
        if force:
            return True
        if not answer.strip():
            return True
        if not cited_chunk_ids:
            return True

        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            if not sentence.strip():
                continue
            has_assertion = _STRONG_ASSERTION_RE.search(sentence) is not None
            has_cite = _CITE_RE.search(sentence) is not None
            if has_assertion and not has_cite:
                return True
        return False

    async def verify(
        self,
        question: str,
        answer: str,
        chunks: list[RankedChunk] | tuple[RankedChunk, ...],
    ) -> VerificationResult:
        sources = self._format_sources(chunks)
        user_prompt = _VERIFY_USER_PROMPT.format(
            question=question,
            answer=answer,
            sources=sources,
        )

        result = await self._llm.generate(
            system_prompt=_VERIFY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=self._settings.llm.simple_model,
            max_tokens=int(self._settings.verifier.max_tokens),
            temperature=float(self._settings.verifier.temperature),
        )

        parsed = self._parse_verification_json(result.text)
        if parsed is None:
            logger.warning("Verifier returned unparsable JSON; passing answer through")
            return VerificationResult(
                is_grounded=True,
                unsupported_claims=[],
                revised_answer="",
            )
        return parsed

    @staticmethod
    def _format_sources(chunks: list[RankedChunk] | tuple[RankedChunk, ...]) -> str:
        if not chunks:
            return "[NO_SOURCES]"

        parts: list[str] = []
        for chunk in chunks:
            header = f"[{chunk.chunk_id}] {chunk.doc_title}"
            if chunk.section_path:
                header += f" | {chunk.section_path}"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_verification_json(text: str) -> VerificationResult | None:
        raw = text.strip()
        if not raw:
            return None

        candidates: list[str] = [raw]
        if raw.startswith("```"):
            stripped = re.sub(r"^```[\w-]*\s*", "", raw)
            stripped = re.sub(r"\s*```$", "", stripped)
            candidates.append(stripped.strip())
        match = _JSON_BLOCK_RE.search(raw)
        if match is not None:
            candidates.append(match.group(0).strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            parsed = AnswerVerifier._coerce_payload(payload)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _coerce_payload(payload: Any) -> VerificationResult | None:
        if not isinstance(payload, dict):
            return None
        obj = cast("dict[str, object]", payload)

        is_grounded: object = obj.get("is_grounded", True)
        unsupported_claims: object = obj.get("unsupported_claims", [])
        revised_answer: object = obj.get("revised_answer", "")
        verified: object = obj.get("verified", True)

        if not isinstance(is_grounded, bool):
            is_grounded = bool(is_grounded)
        if not isinstance(unsupported_claims, list):
            unsupported_claims = []
        claims_list = cast("list[object]", unsupported_claims)
        clean_claims = [text for item in claims_list if (text := str(item).strip())]
        if not isinstance(revised_answer, str):
            revised_answer = str(revised_answer)
        if not isinstance(verified, bool):
            verified = bool(verified)

        return VerificationResult(
            is_grounded=is_grounded,
            unsupported_claims=clean_claims,
            revised_answer=revised_answer.strip(),
            verified=verified,
        )
