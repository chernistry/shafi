"""Offline corpus-tuned dense retriever training and evaluation helpers."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from shafi.ml.teacher_labels import TrainingTriple

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class DenseRetrieverConfig:
    """Configuration for the deterministic dense retriever trainer."""

    dimensions: int = 128
    negative_weight: float = 0.5
    min_token_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalEvalCase:
    """One retrieval evaluation case."""

    question_id: str
    question: str
    family: str
    gold_page_ids: list[str]
    candidate_page_texts: dict[str, str]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics for one model or baseline."""

    mrr_at_10: float
    recall_at_10: float
    recall_at_20: float
    purity: float
    per_family_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class DenseRetrieverArtifact:
    """Serialized artifact for the trained deterministic dense retriever."""

    base_model: str
    dimensions: int
    negative_weight: float
    token_weights: dict[str, float]


class DenseRetrieverTrainer:
    """Train and evaluate a deterministic hashed dense retriever."""

    def prepare_dataset(self, triples_path: Path) -> list[TrainingTriple]:
        """Load training triples from JSONL."""
        triples: list[TrainingTriple] = []
        for line in triples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            triples.append(TrainingTriple(**row))
        return triples

    def train(
        self,
        *,
        base_model: str,
        dataset: Sequence[TrainingTriple],
        config: DenseRetrieverConfig,
        output_dir: Path,
    ) -> Path:
        """Train the deterministic dense retriever and write an artifact."""
        if config.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not dataset:
            raise ValueError("training dataset must not be empty")
        token_weights: dict[str, float] = {}
        for triple in dataset:
            for token in _tokenize(f"{triple.query} {triple.positive_text}"):
                token_weights[token] = token_weights.get(token, 0.0) + 1.0
            for token in _tokenize(f"{triple.query} {triple.negative_text}"):
                token_weights[token] = token_weights.get(token, 0.0) - config.negative_weight
        filtered = {token: weight for token, weight in token_weights.items() if abs(weight) > config.min_token_weight}
        artifact = DenseRetrieverArtifact(
            base_model=base_model,
            dimensions=config.dimensions,
            negative_weight=config.negative_weight,
            token_weights=dict(sorted(filtered.items())),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "dense_retriever_artifact.json"
        artifact_path.write_text(json.dumps(asdict(artifact), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        return artifact_path

    def embed_queries(self, *, questions: Sequence[str], artifact: DenseRetrieverArtifact) -> list[list[float]]:
        """Embed queries using the trained artifact."""
        return [_embed_text(question, artifact=artifact) for question in questions]

    def embed_pages(self, *, pages: Sequence[str], artifact: DenseRetrieverArtifact) -> list[list[float]]:
        """Embed page texts using the trained artifact."""
        return [_embed_text(page, artifact=artifact) for page in pages]

    def evaluate_against_gold(
        self,
        *,
        artifact: DenseRetrieverArtifact,
        eval_cases: Sequence[RetrievalEvalCase],
    ) -> RetrievalMetrics:
        """Evaluate the artifact against retrieval gold labels."""
        return _evaluate_cases(artifact=artifact, eval_cases=eval_cases)

    def compare_to_baseline(
        self,
        *,
        artifact: DenseRetrieverArtifact,
        eval_cases: Sequence[RetrievalEvalCase],
    ) -> dict[str, float]:
        """Compare the tuned artifact to a uniform-weight lexical baseline."""
        baseline_weights = {token: 1.0 for token in artifact.token_weights}
        baseline_artifact = DenseRetrieverArtifact(
            base_model="baseline_uniform",
            dimensions=artifact.dimensions,
            negative_weight=artifact.negative_weight,
            token_weights=baseline_weights,
        )
        tuned = _evaluate_cases(artifact=artifact, eval_cases=eval_cases)
        baseline = _evaluate_cases(artifact=baseline_artifact, eval_cases=eval_cases)
        return {
            "mrr_at_10_delta": tuned.mrr_at_10 - baseline.mrr_at_10,
            "recall_at_10_delta": tuned.recall_at_10 - baseline.recall_at_10,
            "recall_at_20_delta": tuned.recall_at_20 - baseline.recall_at_20,
            "purity_delta": tuned.purity - baseline.purity,
        }


def load_artifact(path: Path) -> DenseRetrieverArtifact:
    """Load a dense retriever artifact from JSON."""
    return DenseRetrieverArtifact(**json.loads(path.read_text(encoding="utf-8")))


def load_eval_cases(path: Path) -> list[RetrievalEvalCase]:
    """Load evaluation cases from JSONL."""
    cases: list[RetrievalEvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(RetrievalEvalCase(**json.loads(line)))
    return cases


def _evaluate_cases(*, artifact: DenseRetrieverArtifact, eval_cases: Sequence[RetrievalEvalCase]) -> RetrievalMetrics:
    reciprocal_ranks: list[float] = []
    recall_at_10_scores: list[float] = []
    recall_at_20_scores: list[float] = []
    purity_scores: list[float] = []
    per_family_rows: dict[str, list[tuple[float, float, float, float]]] = {}
    for case in eval_cases:
        ranked_page_ids = _rank_pages(
            question=case.question, candidate_page_texts=case.candidate_page_texts, artifact=artifact
        )
        top_10 = ranked_page_ids[:10]
        top_20 = ranked_page_ids[:20]
        gold_set = set(case.gold_page_ids)
        rr = 0.0
        for index, page_id in enumerate(top_10, start=1):
            if page_id in gold_set:
                rr = 1.0 / index
                break
        recall_10 = len(gold_set & set(top_10)) / len(gold_set) if gold_set else 0.0
        recall_20 = len(gold_set & set(top_20)) / len(gold_set) if gold_set else 0.0
        purity = len(gold_set & set(top_10)) / len(top_10) if top_10 else 0.0
        reciprocal_ranks.append(rr)
        recall_at_10_scores.append(recall_10)
        recall_at_20_scores.append(recall_20)
        purity_scores.append(purity)
        per_family_rows.setdefault(case.family, []).append((rr, recall_10, recall_20, purity))
    per_family_metrics = {
        family: {
            "mrr_at_10": statistics.fmean(metric[0] for metric in rows),
            "recall_at_10": statistics.fmean(metric[1] for metric in rows),
            "recall_at_20": statistics.fmean(metric[2] for metric in rows),
            "purity": statistics.fmean(metric[3] for metric in rows),
        }
        for family, rows in sorted(per_family_rows.items())
    }
    return RetrievalMetrics(
        mrr_at_10=statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        recall_at_10=statistics.fmean(recall_at_10_scores) if recall_at_10_scores else 0.0,
        recall_at_20=statistics.fmean(recall_at_20_scores) if recall_at_20_scores else 0.0,
        purity=statistics.fmean(purity_scores) if purity_scores else 0.0,
        per_family_metrics=per_family_metrics,
    )


def _rank_pages(
    *,
    question: str,
    candidate_page_texts: dict[str, str],
    artifact: DenseRetrieverArtifact,
) -> list[str]:
    query_vector = _embed_text(question, artifact=artifact)
    scores = [
        (_cosine_similarity(query_vector, _embed_text(text, artifact=artifact)), page_id)
        for page_id, text in candidate_page_texts.items()
    ]
    return [page_id for _score, page_id in sorted(scores, key=lambda item: (item[0], item[1]), reverse=True)]


def _embed_text(text: str, *, artifact: DenseRetrieverArtifact) -> list[float]:
    vector = [0.0] * artifact.dimensions
    for token in _tokenize(text):
        hashed = hashlib.sha1(token.encode()).hexdigest()
        index = int(hashed[:8], 16) % artifact.dimensions
        sign = -1.0 if int(hashed[8:10], 16) % 2 else 1.0
        weight = artifact.token_weights.get(token, 0.1)
        vector[index] += weight * sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True)))


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
