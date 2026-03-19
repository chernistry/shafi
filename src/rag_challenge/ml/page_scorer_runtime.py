# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Runtime adapter for the reviewed page-scorer artifact."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import joblib

from rag_challenge.ml.training_scaffold import PAGE_SCORER_FEATURE_POLICY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rag_challenge.models.schemas import QueryScopePrediction, RetrievedPage

logger = logging.getLogger(__name__)


class _FeatureVectorizer(Protocol):
    """Minimal vectorizer protocol for runtime page scoring."""

    def transform(self, features: Sequence[Mapping[str, object]]) -> object:
        """Transform feature mappings into model input."""
        ...


class _ProbabilityModel(Protocol):
    """Probability model protocol for runtime page scoring."""

    def predict_proba(self, features: object) -> object:
        """Predict positive-class probabilities."""
        ...


class _DecisionModel(Protocol):
    """Decision-function model protocol for runtime page scoring."""

    def decision_function(self, features: object) -> object:
        """Predict ranking scores without probabilities."""
        ...


class _SupportsToList(Protocol):
    """Protocol for array-like objects exposing ``tolist``."""

    def tolist(self) -> object:
        """Return a Python-native representation."""
        ...


@dataclass(frozen=True)
class RuntimePageScorerResult:
    """Runtime page-scorer decision payload.

    Args:
        used: Whether the trained scorer actively reordered pages.
        model_path: Model artifact path associated with the attempt.
        ranked_page_ids: Candidate page IDs ranked by the trained scorer.
        fallback_reason: Fail-closed reason when the scorer was skipped.
    """

    used: bool
    model_path: str
    ranked_page_ids: list[str]
    fallback_reason: str = ""


@dataclass(frozen=True)
class RuntimePageScoringRequest:
    """Runtime-safe feature inputs for page-scoring.

    Args:
        query: Raw user question.
        normalized_answer: Normalized answer string.
        answer_type: Final answer type.
        scope: Query-scope prediction.
        doc_ids: Document IDs already in scope.
        page_candidates: Retrieved page candidates available to the sidecar.
        context_page_ids: Ordered answer-path context page IDs.
        legacy_used_page_ids: Ordered answer-path used page IDs before sidecar override.
        heuristic_scores: Current heuristic page-score mapping for tie-breaking.
    """

    query: str
    normalized_answer: str
    answer_type: str
    scope: QueryScopePrediction
    doc_ids: Sequence[str]
    page_candidates: Sequence[RetrievedPage]
    context_page_ids: Sequence[str]
    legacy_used_page_ids: Sequence[str]
    heuristic_scores: Mapping[str, float]


class RuntimePageScorer:
    """Fail-closed runtime wrapper around the trained page-scorer bundle.

    Args:
        model_path: Absolute or repo-relative path to the trained scorer bundle.
    """

    def __init__(self, *, model_path: str) -> None:
        self._model_path = str(model_path).strip()

    @property
    def model_path(self) -> str:
        """Return the configured model path.

        Returns:
            Configured model path string.
        """

        return self._model_path

    def rank_pages(self, request: RuntimePageScoringRequest) -> RuntimePageScorerResult:
        """Rank candidate pages using the trained scorer.

        Args:
            request: Runtime-safe scoring inputs.

        Returns:
            Ranking result or a fail-closed fallback payload.
        """
        bundle = _load_page_scorer_bundle(self._model_path)
        if bundle is None:
            return RuntimePageScorerResult(
                used=False,
                model_path=self._model_path,
                ranked_page_ids=[],
                fallback_reason="model_bundle_missing_or_invalid",
            )

        feature_policy = str(bundle.get("feature_policy") or "").strip()
        if feature_policy != PAGE_SCORER_FEATURE_POLICY:
            return RuntimePageScorerResult(
                used=False,
                model_path=self._model_path,
                ranked_page_ids=[],
                fallback_reason="unsupported_feature_policy",
            )

        vectorizer = bundle.get("vectorizer")
        model = bundle.get("model")
        if vectorizer is None or model is None:
            return RuntimePageScorerResult(
                used=False,
                model_path=self._model_path,
                ranked_page_ids=[],
                fallback_reason="incomplete_model_bundle",
            )

        doc_rank_map = _build_doc_rank_map(request.page_candidates)
        context_rank_map = _build_rank_map(request.context_page_ids)
        legacy_used_docs = {page_id.rpartition("_")[0] for page_id in request.legacy_used_page_ids}
        feature_rows = [
            _build_runtime_feature_row(
                request=request,
                page=page,
                doc_rank_map=doc_rank_map,
                context_rank_map=context_rank_map,
                legacy_used_docs=legacy_used_docs,
                retrieved_rank=index + 1,
            )
            for index, page in enumerate(request.page_candidates)
        ]
        if not feature_rows:
            return RuntimePageScorerResult(
                used=False,
                model_path=self._model_path,
                ranked_page_ids=[],
                fallback_reason="no_feature_rows",
            )

        typed_vectorizer = cast("_FeatureVectorizer", vectorizer)
        feature_matrix = typed_vectorizer.transform(feature_rows)
        raw_scores = _predict_scores(model=model, feature_matrix=feature_matrix)
        if len(raw_scores) != len(request.page_candidates):
            return RuntimePageScorerResult(
                used=False,
                model_path=self._model_path,
                ranked_page_ids=[],
                fallback_reason="score_count_mismatch",
            )

        ranked = sorted(
            zip(request.page_candidates, raw_scores, strict=False),
            key=lambda item: (
                -float(item[1]),
                -float(request.heuristic_scores.get(item[0].page_id, 0.0)),
                item[0].page_id,
            ),
        )
        ranked_page_ids = [page.page_id for page, _score in ranked if page.page_id]
        if not ranked_page_ids:
            return RuntimePageScorerResult(
                used=False,
                model_path=self._model_path,
                ranked_page_ids=[],
                fallback_reason="empty_ranked_pages",
            )
        return RuntimePageScorerResult(
            used=True,
            model_path=self._model_path,
            ranked_page_ids=ranked_page_ids,
        )


@lru_cache(maxsize=8)
def _load_page_scorer_bundle(model_path: str) -> dict[str, object] | None:
    """Load and cache a page-scorer bundle.

    Args:
        model_path: Artifact path.

        Returns:
            Loaded bundle dictionary, or `None` when unavailable.
        """

    path = Path(model_path).expanduser()
    if not path.is_file():
        return None
    try:
        loaded = joblib.load(path)
    except Exception:
        logger.warning("Failed to load trained page scorer bundle", exc_info=True)
        return None
    if not isinstance(loaded, dict):
        return None
    return cast("dict[str, object]", loaded)


def _build_doc_rank_map(page_candidates: Sequence[RetrievedPage]) -> dict[str, int]:
    """Build a 1-based doc rank mapping from candidate order.

    Args:
        page_candidates: Candidate pages in retrieval order.

    Returns:
        Mapping from document ID to 1-based rank.
    """

    rank_map: dict[str, int] = {}
    for page in page_candidates:
        if page.doc_id and page.doc_id not in rank_map:
            rank_map[page.doc_id] = len(rank_map) + 1
    return rank_map


def _build_rank_map(page_ids: Sequence[str]) -> dict[str, int]:
    """Build a 1-based page rank map from ordered page IDs.

    Args:
        page_ids: Ordered page IDs.

    Returns:
        Mapping from page ID to 1-based rank.
    """

    return {page_id: index for index, page_id in enumerate(page_ids, start=1) if page_id}


def _build_runtime_feature_row(
    *,
    request: RuntimePageScoringRequest,
    page: RetrievedPage,
    doc_rank_map: Mapping[str, int],
    context_rank_map: Mapping[str, int],
    legacy_used_docs: set[str],
    retrieved_rank: int,
) -> dict[str, str | int | bool]:
    """Build one runtime-safe feature row for a candidate page.

    Args:
        request: Runtime scoring request.
        page: Candidate page.
        doc_rank_map: 1-based doc rank mapping.
        context_rank_map: 1-based context page rank mapping.
        legacy_used_docs: Docs already used by the answer path.
        retrieved_rank: 1-based rank in page-candidate retrieval order.

    Returns:
        Runtime-safe feature dictionary.
    """
    anchor_hits = _anchor_hits(request=request, page=page)
    page_text = (page.page_text or "").casefold()
    return {
        "scope_mode": request.scope.scope_mode.value,
        "answer_type": request.answer_type,
        "doc_rank": int(doc_rank_map.get(page.doc_id, 0)),
        "page_num": int(page.page_num),
        "is_first_page": int(page.page_num == 1),
        "doc_selected_by_legacy": int(page.doc_id in legacy_used_docs),
        "doc_candidate_count": len(doc_rank_map),
        "page_candidate_count": len(request.page_candidates),
        "anchor_hit_count": len(anchor_hits),
        "has_anchor_hit": int(bool(anchor_hits)),
        "answer_in_snippet": int(bool(request.normalized_answer) and request.normalized_answer.casefold() in page_text),
        "requires_all_docs_in_case": int(request.scope.requires_all_docs_in_case),
        "should_force_empty_grounding_on_null": int(request.scope.should_force_empty_grounding_on_null),
        "explicit_anchor_count": len(request.scope.hard_anchor_strings),
        "target_page_roles_count": len(request.scope.target_page_roles),
        "doc_ref_count": len(request.doc_ids),
        "legacy_context_page_count": len(request.context_page_ids),
        "sidecar_retrieved_page_count": len(request.page_candidates),
        "targets_title_cover": int("title_cover" in request.scope.target_page_roles),
        "targets_caption": int("caption" in request.scope.target_page_roles),
        "targets_issued_by_block": int("issued_by_block" in request.scope.target_page_roles),
        "targets_operative_order": int("operative_order" in request.scope.target_page_roles),
        "targets_costs_block": int("costs_block" in request.scope.target_page_roles),
        "targets_article_clause": int("article_clause" in request.scope.target_page_roles),
        "targets_schedule_table": int("schedule_table" in request.scope.target_page_roles),
        "legacy_context_rank": int(context_rank_map.get(page.page_id, 0)),
        "sidecar_retrieved_rank": int(retrieved_rank),
        "from_legacy_context": int(page.page_id in context_rank_map),
        "from_legacy_used": int(page.page_id in request.legacy_used_page_ids),
        "from_sidecar_retrieved": 1,
    }


def _anchor_hits(*, request: RuntimePageScoringRequest, page: RetrievedPage) -> list[str]:
    """Collect hard-anchor hits that are available from the page payload.

    Args:
        request: Runtime scoring request.
        page: Candidate page.

    Returns:
        Matched hard-anchor strings.
    """
    if not request.scope.hard_anchor_strings:
        return []
    haystacks = [
        *(page.article_refs or []),
        *(page.normalized_refs or []),
        *(page.linked_refs or []),
        page.page_text or "",
    ]
    matches: list[str] = []
    for anchor in request.scope.hard_anchor_strings:
        anchor_text = str(anchor).strip()
        if not anchor_text:
            continue
        normalized_anchor = anchor_text.casefold()
        if any(normalized_anchor in str(haystack).casefold() for haystack in haystacks if haystack):
            matches.append(anchor_text)
    return matches


def _predict_scores(*, model: object, feature_matrix: object) -> list[float]:
    """Predict page scores from a runtime scorer bundle.

    Args:
        model: Loaded model object.
        feature_matrix: Vectorized feature matrix.

    Returns:
        Float scores aligned with feature rows.

    Raises:
        TypeError: If the model does not expose a supported scoring method.
    """
    if hasattr(model, "predict_proba"):
        probability_model = cast("_ProbabilityModel", model)
        probabilities = probability_model.predict_proba(feature_matrix)
        rows_obj = cast("_SupportsToList", probabilities).tolist() if hasattr(probabilities, "tolist") else probabilities
        rows = cast("Sequence[Sequence[float | int]]", rows_obj)
        return [float(row[1]) for row in rows]
    if hasattr(model, "decision_function"):
        decision_model = cast("_DecisionModel", model)
        scores = decision_model.decision_function(feature_matrix)
        values_obj = cast("_SupportsToList", scores).tolist() if hasattr(scores, "tolist") else scores
        values = cast("Sequence[float | int]", values_obj)
        return [float(value) for value in values]
    raise TypeError(f"Unsupported trained page scorer model: {type(model)!r}")
