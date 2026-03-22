"""Offline corpus-tuned reranker and compact set-selector training helpers."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from rag_challenge.ml.teacher_labels import TrainingTriple

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from rag_challenge.ml.grounding_dataset import GroundingMlRow


@dataclass(frozen=True, slots=True)
class PairwiseLabel:
    """One pairwise reranker label."""

    query: str
    page_id: str
    text: str
    label: int
    source: str
    difficulty: str


@dataclass(frozen=True, slots=True)
class SetUtilityLabel:
    """One compact evidence-set supervision label."""

    query: str
    candidate_pages: list[str]
    candidate_page_texts: dict[str, str]
    selected_pages: list[str]
    is_correct: bool
    answer_coverage: float


@dataclass(frozen=True, slots=True)
class SetSelectorConfig:
    """Configuration for pairwise and set-selector training."""

    min_set_size: int = 3
    max_set_size: int = 8
    negative_weight: float = 0.4


@dataclass(frozen=True, slots=True)
class SetSelectorMetrics:
    """Aggregate set-selector evaluation metrics."""

    page_precision: float
    page_recall: float
    set_utility_rate: float
    avg_set_size: float
    downstream_correctness: float


@dataclass(frozen=True, slots=True)
class _SelectorArtifact:
    base_model: str
    token_weights: dict[str, float]
    min_set_size: int
    max_set_size: int


class SetSelectorTrainer:
    """Train and evaluate a deterministic pairwise scorer and set selector."""

    def prepare_pairwise_data(self, triples: Sequence[TrainingTriple]) -> list[PairwiseLabel]:
        """Convert training triples into pairwise labels."""
        labels: list[PairwiseLabel] = []
        for triple in triples:
            labels.append(
                PairwiseLabel(
                    query=triple.query,
                    page_id=triple.positive_page_id,
                    text=triple.positive_text,
                    label=1,
                    source=triple.source,
                    difficulty=triple.difficulty,
                )
            )
            labels.append(
                PairwiseLabel(
                    query=triple.query,
                    page_id=triple.negative_page_id,
                    text=triple.negative_text,
                    label=0,
                    source=triple.source,
                    difficulty=triple.difficulty,
                )
            )
        return labels

    def prepare_set_utility_data(
        self,
        rows: Sequence[GroundingMlRow],
        *,
        historical_selected_pages: Mapping[str, Sequence[str]] | None = None,
    ) -> list[SetUtilityLabel]:
        """Build set-utility labels from grounding rows and optional historical selections."""
        labels: list[SetUtilityLabel] = []
        historical_map = {question_id: list(page_ids) for question_id, page_ids in (historical_selected_pages or {}).items()}
        for row in rows:
            gold_pages = list(row.label_page_ids or row.sidecar_selected_pages or row.legacy_selected_pages)
            if not gold_pages:
                continue
            selected = historical_map.get(row.question_id, gold_pages)
            candidate_page_texts = {
                candidate.page_id: candidate.snippet_excerpt or candidate.page_id
                for candidate in row.page_candidates
            }
            candidate_pages = list(candidate_page_texts)
            if not candidate_pages:
                continue
            coverage = len(set(selected) & set(gold_pages)) / len(gold_pages)
            labels.append(
                SetUtilityLabel(
                    query=row.question,
                    candidate_pages=candidate_pages,
                    candidate_page_texts=candidate_page_texts,
                    selected_pages=list(selected),
                    is_correct=coverage > 0.0,
                    answer_coverage=coverage,
                )
            )
        return labels

    def train_cross_encoder(
        self,
        *,
        base_model: str,
        pairwise_data: Sequence[PairwiseLabel],
        config: SetSelectorConfig,
        output_dir: Path,
    ) -> Path:
        """Train the deterministic pairwise scorer and write the artifact."""
        if config.min_set_size <= 0 or config.max_set_size < config.min_set_size:
            raise ValueError("invalid set size configuration")
        token_weights: dict[str, float] = {}
        for label in pairwise_data:
            sign = 1.0 if label.label == 1 else -config.negative_weight
            for token in _tokenize(f"{label.query} {label.text}"):
                token_weights[token] = token_weights.get(token, 0.0) + sign
        artifact = _SelectorArtifact(
            base_model=base_model,
            token_weights=dict(sorted(token_weights.items())),
            min_set_size=config.min_set_size,
            max_set_size=config.max_set_size,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "set_selector_artifact.json"
        path.write_text(json.dumps(asdict(artifact), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return path

    def train_set_selector(
        self,
        *,
        cross_encoder_path: Path,
        set_utility_data: Sequence[SetUtilityLabel],
        config: SetSelectorConfig,
        output_dir: Path,
    ) -> Path:
        """Copy the deterministic artifact into the selector output lane."""
        del set_utility_data
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "trained_set_selector.json"
        output_path.write_text(cross_encoder_path.read_text(encoding="utf-8"), encoding="utf-8")
        return output_path

    def score_candidate_pages(
        self,
        *,
        query: str,
        candidate_page_texts: Mapping[str, str],
        artifact_path: Path,
    ) -> dict[str, float]:
        """Score candidate pages with the deterministic pairwise artifact."""
        artifact = _load_selector_artifact(artifact_path)
        query_tokens = set(_tokenize(query))
        scored: dict[str, float] = {}
        for page_id, text in candidate_page_texts.items():
            text_tokens = set(_tokenize(text))
            overlap = query_tokens & text_tokens
            weight = sum(artifact.token_weights.get(token, 0.0) for token in overlap)
            scored[page_id] = float(weight + (0.01 * len(overlap)))
        return dict(sorted(scored.items(), key=lambda item: (-item[1], item[0])))

    def select_compact_evidence_set(
        self,
        *,
        query: str,
        candidate_page_texts: Mapping[str, str],
        artifact_path: Path,
    ) -> list[str]:
        """Greedily select a compact evidence set."""
        artifact = _load_selector_artifact(artifact_path)
        scores = self.score_candidate_pages(query=query, candidate_page_texts=candidate_page_texts, artifact_path=artifact_path)
        query_tokens = set(_tokenize(query))
        covered_tokens: set[str] = set()
        selected: list[str] = []
        for page_id, _score in scores.items():
            page_tokens = set(_tokenize(candidate_page_texts.get(page_id, "")))
            new_tokens = query_tokens & page_tokens - covered_tokens
            if len(selected) < artifact.min_set_size or new_tokens:
                selected.append(page_id)
                covered_tokens |= query_tokens & page_tokens
            if len(selected) >= artifact.max_set_size:
                break
        return selected

    def evaluate_selector(
        self,
        *,
        artifact_path: Path,
        eval_labels: Sequence[SetUtilityLabel],
    ) -> SetSelectorMetrics:
        """Evaluate compact evidence selection metrics."""
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        utility_scores: list[float] = []
        set_sizes: list[float] = []
        downstream_scores: list[float] = []
        for label in eval_labels:
            selected = self.select_compact_evidence_set(
                query=label.query,
                candidate_page_texts=label.candidate_page_texts,
                artifact_path=artifact_path,
            )
            gold_set = set(label.selected_pages)
            selected_set = set(selected)
            precision_scores.append(len(selected_set & gold_set) / len(selected_set) if selected_set else 0.0)
            recall_scores.append(len(selected_set & gold_set) / len(gold_set) if gold_set else 0.0)
            coverage = len(selected_set & gold_set) / len(gold_set) if gold_set else 0.0
            utility_scores.append(1.0 if coverage >= label.answer_coverage else 0.0)
            set_sizes.append(float(len(selected)))
            downstream_scores.append(1.0 if label.is_correct and coverage > 0.0 else 0.0)
        return SetSelectorMetrics(
            page_precision=statistics.fmean(precision_scores) if precision_scores else 0.0,
            page_recall=statistics.fmean(recall_scores) if recall_scores else 0.0,
            set_utility_rate=statistics.fmean(utility_scores) if utility_scores else 0.0,
            avg_set_size=statistics.fmean(set_sizes) if set_sizes else 0.0,
            downstream_correctness=statistics.fmean(downstream_scores) if downstream_scores else 0.0,
        )


def load_pairwise_triples(path: Path) -> list[TrainingTriple]:
    """Load training triples from JSONL."""
    triples: list[TrainingTriple] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        triples.append(TrainingTriple(**json.loads(line)))
    return triples


def write_set_utility_jsonl(path: Path, labels: Sequence[SetUtilityLabel]) -> None:
    """Write set-utility labels as JSONL."""
    lines = [json.dumps(asdict(label), ensure_ascii=True, sort_keys=True) for label in labels]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_set_utility_labels(path: Path) -> list[SetUtilityLabel]:
    """Load set-utility labels from JSONL."""
    labels: list[SetUtilityLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        labels.append(SetUtilityLabel(**json.loads(line)))
    return labels


def _load_selector_artifact(path: Path) -> _SelectorArtifact:
    return _SelectorArtifact(**json.loads(path.read_text(encoding="utf-8")))


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
