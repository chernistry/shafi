"""Citation graph expansion for retrieval candidate enrichment.

Loads Isaacus external_citations enrichment data and expands retrieved chunks
to include cross-referenced articles, improving grounding recall on multi-hop
questions at zero LLM cost.

Reverse index (data/enrichments/reverse_index.json) maps:
  cited_text -> [chunk_id, ...]

Built by NOAM (noam-1a). Falls back gracefully if not present.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Normalize cited text for fuzzy lookup: lowercase, collapse whitespace,
# strip punctuation that varies across sources (e.g. "NO." vs "NO")
_NORMALIZE_RE = re.compile(r"[\s\-\.]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.strip().lower())


class CitationGraphExpander:
    """Expands retrieval candidates using Isaacus citation graph data."""

    def __init__(self, enrichments_dir: str = "data/enrichments"):
        self.enrichments_dir = Path(enrichments_dir)
        # normalized_cited_text -> list[chunk_id]
        self._index: dict[str, list[str]] | None = None

    def _load_index(self) -> dict[str, list[str]]:
        """Load reverse_index.json (built by NOAM) or fall back to scanning."""
        if self._index is not None:
            return self._index

        reverse_path = self.enrichments_dir / "reverse_index.json"
        if reverse_path.exists():
            try:
                raw: dict[str, list[str]] = json.loads(reverse_path.read_text())
                self._index = {_normalize(k): v for k, v in raw.items()}
                logger.info(
                    "Citation graph: loaded reverse index (%d entries) from %s",
                    len(self._index),
                    reverse_path,
                )
                return self._index
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Citation graph: failed to load reverse_index.json: %s", exc)

        # Fallback: build from enrichment files (no doc_title → empty index)
        self._index = {}
        logger.debug("Citation graph: reverse_index.json not found, index empty")
        return self._index

    def expand_candidates(
        self,
        chunk_ids: list[str],
        max_hops: int = 1,
    ) -> list[str]:
        """Given retrieved chunk IDs, find all cross-referenced chunk IDs.

        Args:
            chunk_ids: Initial retrieved chunk IDs.
            max_hops: Citation hops (1 = direct references only).

        Returns:
            list[str]: Additional chunk IDs to fetch (deduped, excluding inputs).
        """
        if not self.enrichments_dir.exists():
            return []

        index = self._load_index()
        if not index:
            return []

        input_set = set(chunk_ids)
        expanded: set[str] = set()
        frontier = list(chunk_ids)

        for _ in range(max_hops):
            next_frontier: list[str] = []
            for chunk_id in frontier:
                enrichment_file = self._find_enrichment(chunk_id)
                if not enrichment_file:
                    continue
                try:
                    data = json.loads(enrichment_file.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                for citation in data.get("external_citations", []):
                    for cid in self._resolve(citation, index):
                        if cid not in input_set and cid not in expanded:
                            expanded.add(cid)
                            next_frontier.append(cid)
            frontier = next_frontier
            if not frontier:
                break

        logger.debug(
            "Citation graph: %d input chunks → %d expanded candidates",
            len(chunk_ids),
            len(expanded),
        )
        return list(expanded)

    def _find_enrichment(self, chunk_id: str) -> Optional[Path]:
        # 1. Exact match
        exact = self.enrichments_dir / f"{chunk_id}.json"
        if exact.exists():
            return exact
        # 2. Zero-padded page: chunk_id may be "hash_1", file is "hash_0001.json"
        parts = chunk_id.rsplit("_", 1)
        if len(parts) == 2:
            doc_hash, page_part = parts
            try:
                padded = self.enrichments_dir / f"{doc_hash}_{int(page_part):04d}.json"
                if padded.exists():
                    return padded
            except ValueError:
                pass
        return None

    def _resolve(self, citation: dict, index: dict[str, list[str]]) -> list[str]:
        """Resolve a citation dict to chunk_ids via the reverse index."""
        cited_text = citation.get("cited", "")
        if not cited_text:
            return []
        key = _normalize(cited_text)
        # Exact normalized match
        if key in index:
            return index[key]
        # Substring match: find any index key that contains or is contained by key
        matches: list[str] = []
        for idx_key, ids in index.items():
            if key in idx_key or idx_key in key:
                matches.extend(ids)
        return matches
