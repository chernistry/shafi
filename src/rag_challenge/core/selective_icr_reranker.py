"""Selective ICR reranking utilities for a bounded local shadow lane.

The goal of this module is to provide a deterministic, audit-friendly local
reranker that can be benchmarked against the current provider path without
changing retrieval breadth or answer generation behavior.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from rag_challenge.core.local_cross_encoder_reranker import LocalCrossEncoderReranker

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_LEGAL_REF_RE = re.compile(
    r"\b(?:article|section|schedule|clause|provision|law|law no\.?|cfi|arb|tcd|case|appeal|judgment|decision|judgement)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:/\d+)?(?:\(\d+\))?\b")
_PAGE_RE = re.compile(r":(\d+):")
_PAGE_ONE_HINTS = (
    "title",
    "caption",
    "claimant",
    "respondent",
    "party",
    "parties",
    "case number",
    "case no",
    "judge",
    "authority",
    "date",
)


@dataclass(frozen=True, slots=True)
class SelectiveICRConfig:
    """Configuration for the selective ICR shadow reranker.

    Args:
        model_path: Optional local model path for a learned reranker.
        max_chars: Maximum characters considered per candidate.
        normalize_scores: Whether to normalize heuristic/model scores to `[0, 1]`.
        provider_exit: Whether the caller intends to promote this path later.
    """

    model_path: str = ""
    max_chars: int = 1800
    normalize_scores: bool = True
    provider_exit: bool = False


@dataclass(frozen=True, slots=True)
class SelectiveICRDiagnostics:
    """Diagnostics captured for a single selective rerank run."""

    model_name: str
    candidate_count: int
    latency_ms: float
    fallback_reason: str = ""
    provider_exit: bool = False


@dataclass(frozen=True, slots=True)
class SelectiveICRRankedCandidate:
    """A scored rerank candidate emitted by the selective ICR lane."""

    chunk_id: str
    doc_id: str
    page_id: str
    rerank_score: float
    retrieval_score: float
    text: str
    doc_title: str = ""
    section_path: str = ""
    page_family: str = ""
    doc_family: str = ""
    chunk_type: str = ""
    amount_roles: tuple[str, ...] = ()


class SelectiveICRReranker:
    """Score candidate chunks with a bounded local shadow rerank lane."""

    def __init__(
        self,
        *,
        config: SelectiveICRConfig | None = None,
        model_obj: Any | None = None,
    ) -> None:
        """Initialize the shadow reranker.

        Args:
            config: Optional reranker configuration.
            model_obj: Optional injected model for tests.
        """
        self._config = config or SelectiveICRConfig()
        self._fallback_reason = ""
        self._last_diagnostics = SelectiveICRDiagnostics(
            model_name=self._config.model_path or "selective_icr_heuristic",
            candidate_count=0,
            latency_ms=0.0,
            fallback_reason="",
            provider_exit=bool(self._config.provider_exit),
        )
        self._model: LocalCrossEncoderReranker | None = None
        if model_obj is not None:
            self._model = LocalCrossEncoderReranker(
                model_path=self._config.model_path or "selective_icr_model",
                model_obj=model_obj,
            )
        elif self._config.model_path.strip():
            try:
                self._model = LocalCrossEncoderReranker(model_path=self._config.model_path.strip())
            except Exception as exc:  # pragma: no cover - exercised in integration only
                self._fallback_reason = str(exc)
        self._model_name = (
            self._model.model_path if self._model is not None else (self._config.model_path or "selective_icr_heuristic")
        )

    @property
    def model_name(self) -> str:
        """Return the active model identifier or heuristic label."""

        return self._model_name

    def get_last_diagnostics(self) -> SelectiveICRDiagnostics:
        """Return diagnostics from the most recent ranking call."""

        return self._last_diagnostics

    def score_documents(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score plain documents for a query.

        Args:
            query: Query text.
            documents: Candidate documents.

        Returns:
            One score per input document.
        """
        if not documents:
            return []
        if self._model is not None:
            raw_scores = self._model.score_documents(query=query, documents=list(documents))
            return self._normalize_scores(raw_scores)
        return self._normalize_scores([self._heuristic_score(query, document) for document in documents])

    def rank(self, query: str, candidates: Sequence[object], *, top_n: int | None = None) -> list[SelectiveICRRankedCandidate]:
        """Rank chunk-like candidates for a query.

        Args:
            query: Query text.
            candidates: Chunk-like objects with text and chunk identifiers.
            top_n: Optional cap on returned candidates.

        Returns:
            Ranked candidates sorted by descending score.
        """
        t0 = time.perf_counter()
        scored = self._rank_candidates(query, candidates)
        scored.sort(key=lambda item: (-item.rerank_score, item.page_id, item.chunk_id))
        if top_n is not None:
            scored = scored[: max(0, int(top_n))]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._last_diagnostics = SelectiveICRDiagnostics(
            model_name=self.model_name,
            candidate_count=len(candidates),
            latency_ms=elapsed_ms,
            fallback_reason=self._fallback_reason,
            provider_exit=bool(self._config.provider_exit),
        )
        return scored

    def _rank_candidates(
        self,
        query: str,
        candidates: Sequence[object],
    ) -> list[SelectiveICRRankedCandidate]:
        documents = [self._candidate_text(candidate) for candidate in candidates]
        scores = self.score_documents(query, documents)
        ranked: list[SelectiveICRRankedCandidate] = []
        for candidate, score, text in zip(candidates, scores, documents, strict=True):
            chunk_id = self._candidate_attr(candidate, "chunk_id")
            doc_id = self._candidate_attr(candidate, "doc_id") or chunk_id.split(":", 1)[0]
            page_id = _chunk_id_to_page_id(chunk_id)
            ranked.append(
                SelectiveICRRankedCandidate(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_id=page_id,
                    rerank_score=score,
                    retrieval_score=self._candidate_float(candidate, "retrieval_score", "score"),
                    text=text,
                    doc_title=self._candidate_attr(candidate, "doc_title"),
                    section_path=self._candidate_attr(candidate, "section_path"),
                    page_family=self._candidate_attr(candidate, "page_family"),
                    doc_family=self._candidate_attr(candidate, "doc_family"),
                    chunk_type=self._candidate_attr(candidate, "chunk_type"),
                    amount_roles=tuple(self._candidate_list(candidate, "amount_roles")),
                )
            )
        return ranked

    def _candidate_text(self, candidate: object) -> str:
        pieces = [
            self._candidate_attr(candidate, "doc_title"),
            self._candidate_attr(candidate, "section_path"),
            self._candidate_attr(candidate, "page_family"),
            self._candidate_attr(candidate, "doc_family"),
            self._candidate_attr(candidate, "chunk_type"),
            self._candidate_attr(candidate, "text"),
            " ".join(self._candidate_list(candidate, "amount_roles")),
            " ".join(self._candidate_list(candidate, "normalized_refs")),
            " ".join(self._candidate_list(candidate, "law_titles")),
            " ".join(self._candidate_list(candidate, "article_refs")),
            " ".join(self._candidate_list(candidate, "case_numbers")),
            " ".join(self._candidate_list(candidate, "cross_refs")),
            " ".join(self._candidate_list(candidate, "canonical_entity_ids")),
        ]
        text = " ".join(piece for piece in pieces if piece).strip()
        if not text:
            text = self._candidate_attr(candidate, "chunk_id")
        max_chars = max(0, int(self._config.max_chars))
        return text[:max_chars] if max_chars > 0 else text

    def _heuristic_score(self, query: str, document: str) -> float:
        query_norm = _normalize(query)
        doc_norm = _normalize(document)
        query_tokens = _tokenize(query_norm)
        doc_tokens = _tokenize(doc_norm)
        if not query_tokens or not doc_tokens:
            return 0.0

        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        overlap = len(query_set & doc_set) / max(1, len(query_set))

        query_bigrams = _bigrams(query_tokens)
        doc_bigrams = _bigrams(doc_tokens)
        bigram_overlap = len(query_bigrams & doc_bigrams) / max(1, len(query_bigrams))

        exact_phrase = 1.0 if query_norm and query_norm in doc_norm else 0.0
        ordered_overlap = _ordered_token_overlap(query_tokens, doc_tokens)
        ref_overlap = _reference_overlap(query_norm, doc_norm)
        family_boost = _family_boost(query_norm, doc_norm)
        page_boost = _page_boost(query_norm, doc_norm)

        raw_score = (
            0.28 * overlap
            + 0.18 * bigram_overlap
            + 0.16 * ordered_overlap
            + 0.18 * exact_phrase
            + 0.12 * ref_overlap
            + 0.08 * family_boost
            + 0.05 * page_boost
        )
        return raw_score

    def _normalize_scores(self, scores: Sequence[float]) -> list[float]:
        values = [float(score) for score in scores]
        if not values:
            return []
        if not self._config.normalize_scores:
            return values
        low = min(values)
        high = max(values)
        if high <= low:
            return [0.5 for _ in values]
        span = high - low
        return [(value - low) / span for value in values]

    @staticmethod
    def _candidate_attr(candidate: object, attr: str) -> str:
        value = getattr(candidate, attr, "")
        return str(value).strip()

    @staticmethod
    def _candidate_float(candidate: object, *attrs: str) -> float:
        for attr in attrs:
            value = getattr(candidate, attr, None)
            if isinstance(value, (int, float, str)):
                try:
                    return float(value)
                except ValueError:
                    continue
        return 0.0

    @staticmethod
    def _candidate_list(candidate: object, attr: str) -> list[str]:
        value = getattr(candidate, attr, None)
        if isinstance(value, list):
            raw_items = cast("list[Any]", value)
            items: list[str] = []
            for item in raw_items:
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        if isinstance(value, tuple):
            raw_items = cast("tuple[Any, ...]", value)
            items: list[str] = []
            for item in raw_items:
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        return []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_RE.finditer(text)]


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(max(0, len(tokens) - 1))}


def _ordered_token_overlap(query_tokens: Sequence[str], doc_tokens: Sequence[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_positions: dict[str, list[int]] = {}
    for index, token in enumerate(doc_tokens):
        doc_positions.setdefault(token, []).append(index)
    matches = 0
    last_position = -1
    for token in query_tokens:
        positions = doc_positions.get(token)
        if not positions:
            continue
        next_position = next((position for position in positions if position > last_position), None)
        if next_position is None:
            continue
        matches += 1
        last_position = next_position
    return matches / max(1, len(query_tokens))


def _reference_overlap(query: str, document: str) -> float:
    query_refs = set(_LEGAL_REF_RE.findall(query))
    doc_refs = set(_LEGAL_REF_RE.findall(document))
    query_numbers = set(_NUMBER_RE.findall(query))
    doc_numbers = set(_NUMBER_RE.findall(document))
    if not query_refs and not query_numbers:
        return 0.0
    ref_hits = len(query_refs & doc_refs)
    number_hits = len(query_numbers & doc_numbers)
    return min(1.0, (ref_hits + number_hits) / max(1, len(query_refs) + len(query_numbers)))


def _family_boost(query: str, document: str) -> float:
    score = 0.0
    if any(term in query for term in ("claimant", "respondent", "party", "parties", "caption")) and any(
        term in document for term in ("claimant", "respondent", "party", "caption", "parties")
    ):
        score += 0.5
    if any(term in query for term in ("judge", "court", "tribunal")) and any(
        term in document for term in ("judge", "court", "tribunal")
    ):
        score += 0.3
    if any(term in query for term in ("article", "section", "schedule", "provision", "clause")) and any(
        term in document for term in ("article", "section", "schedule", "provision", "clause")
    ):
        score += 0.5
    if any(term in query for term in ("authority", "date", "law no", "law number", "commencement", "enactment")) and any(
        term in document for term in ("authority", "date", "law", "commencement", "enactment")
    ):
        score += 0.4
    return min(1.0, score)


def _page_boost(query: str, document: str) -> float:
    if any(hint in query for hint in _PAGE_ONE_HINTS) and ":1:" in document:
        return 1.0
    if any(hint in query for hint in ("article", "section", "provision", "schedule")) and ":anchor:" in document:
        return 0.4
    if any(hint in query for hint in ("authority", "date", "law number")) and ":anchor:" in document:
        return 0.3
    return 0.0


def _chunk_id_to_page_id(chunk_id: str) -> str:
    if not chunk_id:
        return ""
    if ":" not in chunk_id and "_" in chunk_id:
        return chunk_id
    parts = chunk_id.split(":")
    if len(parts) < 2:
        return ""
    doc_id = parts[0].strip()
    page_raw = parts[1].strip()
    if not doc_id or not page_raw.isdigit():
        return ""
    return f"{doc_id}_{int(page_raw) + 1}"
