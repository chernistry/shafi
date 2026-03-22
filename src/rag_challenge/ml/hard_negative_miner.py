"""Deterministic hard-negative mining for offline retrieval training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class NegativeStrategy(StrEnum):
    """Supported hard-negative mining strategies."""

    SAME_DOC = "same_doc"
    ALIAS_CONFUSABLE = "alias_confusable"
    CROSS_DOC = "cross_doc"
    LEXICAL_NEAR_MISS = "lexical_near_miss"
    EMBEDDING_NEAR_MISS = "embedding_near_miss"


@dataclass(frozen=True, slots=True)
class EvalResultRecord:
    """One evaluation result for mining false-positive negatives."""

    question_id: str
    query: str
    gold_page_ids: Sequence[str]
    gold_doc_ids: Sequence[str]
    predicted_page_ids: Sequence[str]


@dataclass(frozen=True, slots=True)
class NegativeCandidate:
    """One hard-negative page candidate."""

    page_id: str
    text: str
    strategy: NegativeStrategy
    similarity_score: float
    doc_id: str


class HardNegativeMiner:
    """Mine hard negatives from a deterministic offline page pool."""

    def __init__(
        self,
        *,
        page_texts: Mapping[str, str],
        page_doc_ids: Mapping[str, str],
        aliases_by_doc_id: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Initialize the miner.

        Args:
            page_texts: Page text keyed by page ID.
            page_doc_ids: Parent doc IDs keyed by page ID.
            aliases_by_doc_id: Optional alias strings keyed by doc ID.
        """
        self._page_texts = dict(page_texts)
        self._page_doc_ids = dict(page_doc_ids)
        self._aliases_by_doc_id = {doc_id: list(aliases) for doc_id, aliases in (aliases_by_doc_id or {}).items()}

    def mine(
        self,
        *,
        query: str,
        gold_page_ids: Sequence[str],
        gold_doc_ids: Sequence[str],
        top_k: int = 5,
        embedding_similarities: Mapping[str, float] | None = None,
    ) -> list[NegativeCandidate]:
        """Mine negatives across all supported strategies.

        Args:
            query: The user query.
            gold_page_ids: Ground-truth page IDs.
            gold_doc_ids: Ground-truth document IDs.
            top_k: Maximum number of candidates to return.
            embedding_similarities: Optional page_id -> cosine similarity
                mapping.  When provided, the EMBEDDING_NEAR_MISS strategy
                is included in mining.
        """
        candidates = [
            *self.mine_same_doc(query=query, gold_doc_ids=gold_doc_ids, gold_page_ids=gold_page_ids),
            *self.mine_alias_confusable(query=query, gold_page_ids=gold_page_ids),
            *self.mine_cross_doc(query=query, gold_doc_ids=gold_doc_ids, gold_page_ids=gold_page_ids),
            *self.mine_lexical_near_miss(query=query, gold_page_ids=gold_page_ids),
        ]
        if embedding_similarities is not None:
            candidates.extend(
                self.mine_embedding_near_miss(
                    query=query,
                    gold_page_ids=gold_page_ids,
                    embedding_similarities=embedding_similarities,
                )
            )
        return self.deduplicate(candidates)[: max(1, top_k)]

    def mine_same_doc(
        self,
        *,
        query: str,
        gold_doc_ids: Sequence[str],
        gold_page_ids: Sequence[str],
    ) -> list[NegativeCandidate]:
        """Mine negatives from the same gold documents but wrong pages."""
        gold_doc_set = set(gold_doc_ids)
        gold_page_set = set(gold_page_ids)
        return self._rank_candidates(
            query=query,
            page_ids=[
                page_id
                for page_id, doc_id in self._page_doc_ids.items()
                if doc_id in gold_doc_set and page_id not in gold_page_set
            ],
            strategy=NegativeStrategy.SAME_DOC,
            bonus=0.35,
        )

    def mine_alias_confusable(
        self,
        *,
        query: str,
        gold_page_ids: Sequence[str],
    ) -> list[NegativeCandidate]:
        """Mine alias-confusable negatives from docs whose aliases hit the query."""
        gold_page_set = set(gold_page_ids)
        query_blob = _normalize_text(query)
        page_ids = [
            page_id
            for page_id, doc_id in self._page_doc_ids.items()
            if page_id not in gold_page_set
            and any(alias.casefold() in query_blob for alias in self._aliases_by_doc_id.get(doc_id, []))
        ]
        return self._rank_candidates(query=query, page_ids=page_ids, strategy=NegativeStrategy.ALIAS_CONFUSABLE, bonus=0.25)

    def mine_cross_doc(
        self,
        *,
        query: str,
        gold_doc_ids: Sequence[str],
        gold_page_ids: Sequence[str],
    ) -> list[NegativeCandidate]:
        """Mine negatives from non-gold docs with lexical overlap."""
        gold_doc_set = set(gold_doc_ids)
        gold_page_set = set(gold_page_ids)
        page_ids = [
            page_id
            for page_id, doc_id in self._page_doc_ids.items()
            if doc_id not in gold_doc_set and page_id not in gold_page_set
        ]
        return self._rank_candidates(query=query, page_ids=page_ids, strategy=NegativeStrategy.CROSS_DOC, bonus=0.1)

    def mine_lexical_near_miss(
        self,
        *,
        query: str,
        gold_page_ids: Sequence[str],
    ) -> list[NegativeCandidate]:
        """Mine lexically similar non-gold pages."""
        gold_page_set = set(gold_page_ids)
        page_ids = [page_id for page_id in self._page_texts if page_id not in gold_page_set]
        return self._rank_candidates(query=query, page_ids=page_ids, strategy=NegativeStrategy.LEXICAL_NEAR_MISS, bonus=0.0)

    def mine_embedding_near_miss(
        self,
        *,
        query: str,
        gold_page_ids: Sequence[str],
        embedding_similarities: Mapping[str, float],
    ) -> list[NegativeCandidate]:
        """Mine negatives that have high embedding similarity but are not gold.

        These are the most valuable hard negatives: pages the retriever
        ranks highly but that are actually irrelevant.

        Args:
            query: The user query (unused for ranking but kept for API
                consistency and potential future use).
            gold_page_ids: Ground-truth page IDs to exclude.
            embedding_similarities: Mapping of page_id to cosine
                similarity with the query embedding.

        Returns:
            Candidates ranked by embedding similarity descending.
        """
        gold_page_set = set(gold_page_ids)
        bonus = 0.45
        ranked: list[NegativeCandidate] = []
        for page_id, sim in embedding_similarities.items():
            if page_id in gold_page_set:
                continue
            text = self._page_texts.get(page_id, "")
            if not text:
                continue
            ranked.append(
                NegativeCandidate(
                    page_id=page_id,
                    text=text,
                    strategy=NegativeStrategy.EMBEDDING_NEAR_MISS,
                    similarity_score=float(sim) + bonus,
                    doc_id=self._page_doc_ids.get(page_id, page_id.rpartition("_")[0]),
                )
            )
        return sorted(ranked, key=lambda item: (-item.similarity_score, item.page_id))

    def deduplicate(self, candidates: Sequence[NegativeCandidate]) -> list[NegativeCandidate]:
        """Deduplicate candidates by page ID, keeping the strongest score."""
        deduped: dict[str, NegativeCandidate] = {}
        for candidate in sorted(
            candidates,
            key=lambda item: (item.similarity_score, item.strategy.value, item.page_id),
            reverse=True,
        ):
            deduped.setdefault(candidate.page_id, candidate)
        return list(deduped.values())

    def _rank_candidates(
        self,
        *,
        query: str,
        page_ids: Sequence[str],
        strategy: NegativeStrategy,
        bonus: float,
    ) -> list[NegativeCandidate]:
        query_tokens = set(_tokenize(query))
        ranked: list[NegativeCandidate] = []
        for page_id in page_ids:
            text = self._page_texts.get(page_id, "")
            if not text:
                continue
            overlap = len(query_tokens & set(_tokenize(text)))
            if overlap <= 0:
                continue
            ranked.append(
                NegativeCandidate(
                    page_id=page_id,
                    text=text,
                    strategy=strategy,
                    similarity_score=float(overlap) + bonus,
                    doc_id=self._page_doc_ids.get(page_id, page_id.rpartition("_")[0]),
                )
            )
        return sorted(ranked, key=lambda item: (-item.similarity_score, item.page_id))


def mine_from_eval_results(
    *,
    eval_results: Sequence[EvalResultRecord],
    page_texts: Mapping[str, str],
    page_doc_ids: Mapping[str, str],
    top_k: int = 5,
) -> dict[str, list[NegativeCandidate]]:
    """Mine hard negatives from evaluation false positives.

    For each eval result, pages that were predicted but are not gold are
    treated as high-confidence SAME_DOC negatives (the system already
    confused them).  Standard mining strategies are then layered on top.

    Args:
        eval_results: Evaluation records with predictions and gold labels.
        page_texts: Page text keyed by page ID.
        page_doc_ids: Parent doc IDs keyed by page ID.
        top_k: Maximum candidates per question.

    Returns:
        Dict mapping question_id to its mined NegativeCandidate list.
    """
    miner = HardNegativeMiner(page_texts=page_texts, page_doc_ids=page_doc_ids)
    out: dict[str, list[NegativeCandidate]] = {}
    for rec in eval_results:
        gold_page_set = set(rec.gold_page_ids)
        # False positives: predicted but not gold.
        fp_page_ids = [pid for pid in rec.predicted_page_ids if pid not in gold_page_set]
        fp_candidates: list[NegativeCandidate] = []
        for page_id in fp_page_ids:
            text = page_texts.get(page_id, "")
            if not text:
                continue
            fp_candidates.append(
                NegativeCandidate(
                    page_id=page_id,
                    text=text,
                    strategy=NegativeStrategy.SAME_DOC,
                    similarity_score=100.0,  # high confidence
                    doc_id=page_doc_ids.get(page_id, page_id.rpartition("_")[0]),
                )
            )
        standard = miner.mine(
            query=rec.query,
            gold_page_ids=list(rec.gold_page_ids),
            gold_doc_ids=list(rec.gold_doc_ids),
            top_k=top_k,
        )
        out[rec.question_id] = miner.deduplicate([*fp_candidates, *standard])[: max(1, top_k)]
    return out


def export_training_triples(
    mined: dict[str, list[NegativeCandidate]],
    *,
    queries: Mapping[str, str],
    gold_page_ids: Mapping[str, Sequence[str]],
    page_texts: Mapping[str, str],
) -> list[dict[str, str]]:
    """Export query/positive/negative triples for page-scorer retraining.

    Args:
        mined: Mining output keyed by question_id.
        queries: Query text keyed by question_id.
        gold_page_ids: Gold page IDs keyed by question_id.
        page_texts: Page text keyed by page ID.

    Returns:
        List of dicts each containing ``query``, ``positive_page_id``,
        ``positive_text``, ``negative_page_id``, ``negative_text``, and
        ``negative_strategy``.
    """
    triples: list[dict[str, str]] = []
    for qid, candidates in mined.items():
        query = queries.get(qid, "")
        if not query:
            continue
        golds = gold_page_ids.get(qid, [])
        for gold_pid in golds:
            pos_text = page_texts.get(gold_pid, "")
            if not pos_text:
                continue
            for neg in candidates:
                triples.append(
                    {
                        "query": query,
                        "positive_page_id": gold_pid,
                        "positive_text": pos_text,
                        "negative_page_id": neg.page_id,
                        "negative_text": neg.text,
                        "negative_strategy": neg.strategy.value,
                    }
                )
    return triples


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]
