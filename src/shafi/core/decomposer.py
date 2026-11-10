from __future__ import annotations

import re
from typing import TYPE_CHECKING

from shafi.models import QueryComplexity

if TYPE_CHECKING:
    from shafi.core.query_contract import QueryContract

_SPLIT_RE = re.compile(r"\b(?:and|vs\.?|versus|compare|contrast|difference between|as well as)\b", re.IGNORECASE)
_MULTI_CLAUSE_RE = re.compile(r"[;,]\s*")


class QueryDecomposer:
    """Lightweight decomposition for multi-hop legal questions."""

    def should_decompose(
        self,
        query: str,
        complexity: QueryComplexity | str,
        *,
        query_contract: QueryContract | None = None,
    ) -> bool:
        """Decide whether a query should be decomposed.

        Args:
            query: Normalized user query.
            complexity: Complexity routing label.
            query_contract: Optional structured contract compiled earlier in the pipeline.

        Returns:
            bool: True when decomposition should run.
        """

        if query_contract is not None and any(
            engine.value != "standard_rag" for engine in query_contract.execution_plan
        ):
            return False
        raw = str(complexity).strip().lower()
        is_complex = raw == QueryComplexity.COMPLEX.value
        if not is_complex:
            return False
        lowered = query.lower()
        return any(marker in lowered for marker in ("compare", "contrast", "difference", "relationship")) or bool(
            _MULTI_CLAUSE_RE.search(query)
        )

    def decompose(self, query: str, *, max_subqueries: int = 3) -> list[str]:
        max_items = max(1, int(max_subqueries))
        parts: list[str] = []
        for clause in _MULTI_CLAUSE_RE.split(query):
            clause = clause.strip()
            if not clause:
                continue
            split = [segment.strip() for segment in _SPLIT_RE.split(clause) if segment.strip()]
            if len(split) > 1:
                parts.extend(split)
            else:
                parts.append(clause)
            if len(parts) >= max_items:
                break

        # Dedupe while preserving order and keep meaningful fragments.
        seen: set[str] = set()
        out: list[str] = []
        for item in parts:
            text = item.strip().rstrip("?")
            if len(text) < 8:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= max_items:
                break
        return out
