from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_challenge.models import DocType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.models import RankedChunk

_ARTICLE_RE = re.compile(r"\bArticle\s+\d+(?:\([^)]*\))*", re.IGNORECASE)
_POSITIVE_RE = re.compile(r"\b(shall|must|required|entitled)\b", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\b(shall\s+not|must\s+not|prohibited|not\s+permitted)\b", re.IGNORECASE)

_AUTHORITY_SCORE: dict[DocType, int] = {
    DocType.STATUTE: 8,
    DocType.REGULATION: 6,
    DocType.CASE_LAW: 5,
    DocType.CONTRACT: 3,
    DocType.OTHER: 2,
}


@dataclass(frozen=True)
class ConflictItem:
    chunk_id_a: str
    chunk_id_b: str
    description: str
    conflict_type: str
    authority_winner: str


@dataclass(frozen=True)
class ConflictReport:
    has_conflict: bool
    conflicts: list[ConflictItem]
    resolution_hint: str

    def to_prompt_context(self) -> str:
        if not self.has_conflict or not self.conflicts:
            return ""
        lines = [
            "Conflict advisory:",
            self.resolution_hint,
        ]
        for item in self.conflicts:
            lines.append(
                f"- {item.conflict_type}: {item.chunk_id_a} vs {item.chunk_id_b}; "
                f"prefer {item.authority_winner}. {item.description}"
            )
        return "\n".join(lines)


class ConflictDetector:
    """Heuristic conflict detector with legal authority tie-break."""

    def detect(self, chunks: Sequence[RankedChunk]) -> ConflictReport:
        if len(chunks) < 2:
            return ConflictReport(False, [], "")

        conflicts: list[ConflictItem] = []
        top_chunks = list(chunks[:8])
        for i, chunk_a in enumerate(top_chunks):
            for chunk_b in top_chunks[i + 1 :]:
                conflict = self._detect_pair(chunk_a, chunk_b)
                if conflict is not None:
                    conflicts.append(conflict)
                    if len(conflicts) >= 3:
                        break
            if len(conflicts) >= 3:
                break

        if not conflicts:
            return ConflictReport(False, [], "")

        hint = (
            "When chunks disagree, rely on higher-authority sources first "
            "(statute > regulation > case law > contract > guidance)."
        )
        return ConflictReport(True, conflicts, hint)

    def _detect_pair(self, chunk_a: RankedChunk, chunk_b: RankedChunk) -> ConflictItem | None:
        text_a = chunk_a.text[:2000]
        text_b = chunk_b.text[:2000]
        article_a = self._article_key(text_a)
        article_b = self._article_key(text_b)
        if article_a and article_b and article_a.lower() != article_b.lower():
            return None

        pos_a = _POSITIVE_RE.search(text_a) is not None
        neg_a = _NEGATIVE_RE.search(text_a) is not None
        pos_b = _POSITIVE_RE.search(text_b) is not None
        neg_b = _NEGATIVE_RE.search(text_b) is not None

        contradictory = (pos_a and neg_b) or (neg_a and pos_b)
        if not contradictory:
            return None

        winner = self._authority_winner(chunk_a.doc_type, chunk_b.doc_type)
        article = article_a or article_b or "same legal point"
        description = f"Potentially contradictory obligations around {article}."
        return ConflictItem(
            chunk_id_a=chunk_a.chunk_id,
            chunk_id_b=chunk_b.chunk_id,
            description=description,
            conflict_type="obligation_conflict",
            authority_winner=winner,
        )

    @staticmethod
    def _article_key(text: str) -> str:
        match = _ARTICLE_RE.search(text)
        return match.group(0).strip() if match is not None else ""

    @staticmethod
    def _authority_winner(doc_type_a: DocType, doc_type_b: DocType) -> str:
        score_a = _AUTHORITY_SCORE.get(doc_type_a, 0)
        score_b = _AUTHORITY_SCORE.get(doc_type_b, 0)
        if score_a >= score_b:
            return doc_type_a.value
        return doc_type_b.value
