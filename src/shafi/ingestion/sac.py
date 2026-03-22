from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from shafi.config import get_settings
from shafi.prompts import load_prompt

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.llm import LLMProvider
    from shafi.models import Chunk, ParsedDocument

_SAC_SYSTEM_PROMPT = load_prompt("ingestion/sac_system")
_SAC_USER_PROMPT_TEMPLATE = load_prompt("ingestion/sac_user")
_STRUCTURAL_CITATION_RE = re.compile(r"^(?:article|section|schedule|chapter|part|rule)\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*0*\d{1,4}\s*/\s*\d{4}\b", re.IGNORECASE)
_LAW_ALIAS_RE = re.compile(r"\b(?:law|regulations?)\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


class SACGenerator:
    """Summary-Augmented Chunking helper."""

    def __init__(self, llm: LLMProvider) -> None:
        settings = get_settings()
        self._llm = llm
        self._ingestion_settings = settings.ingestion
        self._summary_model = settings.llm.summary_model

    async def generate_doc_summary(self, doc: ParsedDocument) -> str:
        text_excerpt = self._doc_excerpt(doc)
        if not text_excerpt.strip():
            return ""

        result = await self._llm.generate(
            system_prompt=_SAC_SYSTEM_PROMPT,
            user_prompt=_SAC_USER_PROMPT_TEMPLATE.format(
                title=doc.title,
                doc_type=doc.doc_type.value,
                text_excerpt=text_excerpt,
            ),
            model=self._summary_model,
            max_tokens=self._ingestion_settings.sac_summary_max_tokens,
            temperature=0.0,
        )
        summary = result.text.strip()
        logger.info("Generated SAC summary for %s (%d chars)", doc.doc_id, len(summary))
        return summary

    def augment_chunks(self, chunks: Sequence[Chunk], doc_summary: str) -> list[Chunk]:
        summary = doc_summary.strip()
        if not chunks:
            return []

        augmented: list[Chunk] = []
        for chunk in chunks:
            embedding_text = self._build_retrieval_text(chunk=chunk, doc_summary=summary)
            augmented.append(
                chunk.model_copy(
                    update={
                        "chunk_text_for_embedding": embedding_text,
                        "doc_summary": summary,
                    }
                )
            )
        return augmented

    @staticmethod
    def _truncate_excerpt(text: str, limit: int) -> str:
        return text[:limit]

    def _doc_excerpt(self, doc: ParsedDocument) -> str:
        limit = int(self._ingestion_settings.sac_doc_excerpt_chars)
        if doc.full_text.strip():
            return self._truncate_excerpt(doc.full_text, limit)
        if doc.sections:
            joined = "\n\n".join(section.text for section in doc.sections[:5] if section.text.strip())
            return self._truncate_excerpt(joined, limit)
        if doc.provided_chunks:
            joined = "\n\n".join(chunk.text for chunk in doc.provided_chunks[:10] if chunk.text.strip())
            return self._truncate_excerpt(joined, limit)
        return ""

    @classmethod
    def _build_retrieval_text(cls, *, chunk: Chunk, doc_summary: str) -> str:
        parts: list[str] = []
        title = re.sub(r"\s+", " ", (chunk.doc_title or "").strip())
        if title:
            parts.append(f"[DOC_TITLE]\n{title}")

        aliases = cls._retrieval_aliases(chunk=chunk, canonical_title=title)
        if aliases:
            parts.append("[DOC_ALIASES]\n" + "\n".join(aliases))

        section_path = re.sub(r"\s+", " ", (chunk.section_path or "").strip())
        if section_path:
            parts.append(f"[SECTION_PATH]\n{section_path}")

        if doc_summary:
            parts.append(f"[DOC_SUMMARY]\n{doc_summary}")

        parts.append(f"[CHUNK]\n{chunk.chunk_text}")
        return "\n\n".join(parts)

    @classmethod
    def _retrieval_aliases(cls, *, chunk: Chunk, canonical_title: str) -> list[str]:
        seen: set[str] = set()
        aliases: list[str] = []
        canonical_key = canonical_title.casefold()

        def add_alias(raw: str) -> None:
            normalized = re.sub(r"\s+", " ", raw).strip(" ,.;:")
            if not normalized:
                return
            key = normalized.casefold()
            if key == canonical_key or key in seen:
                return
            if _STRUCTURAL_CITATION_RE.match(normalized):
                return
            is_case_ref = _CASE_REF_RE.search(normalized) is not None
            is_lawish = _LAW_ALIAS_RE.search(normalized) is not None
            has_year = _YEAR_RE.search(normalized) is not None
            if not (is_case_ref or is_lawish or has_year):
                return
            seen.add(key)
            aliases.append(normalized)

        for citation in chunk.citations:
            add_alias(str(citation))
            if len(aliases) >= 6:
                break

        return aliases
