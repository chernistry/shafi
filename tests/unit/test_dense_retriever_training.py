from __future__ import annotations

import json

from rag_challenge.ml.dense_retriever_training import (
    DenseRetrieverConfig,
    DenseRetrieverTrainer,
    RetrievalEvalCase,
    load_artifact,
)


def _triples_path(tmp_path):
    path = tmp_path / "triples.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "What does Employment Law require?",
                        "positive_text": "Employment Law requires notice before termination.",
                        "positive_page_id": "law-a_2",
                        "negative_text": "Employment Amendment Law sets transitional obligations.",
                        "negative_page_id": "law-b_1",
                        "source": "synthetic",
                        "difficulty": "medium",
                        "strategy": "lexical_near_miss",
                    }
                ),
                json.dumps(
                    {
                        "query": "Who issued Employment Law?",
                        "positive_text": "Employment Law 2020 was issued by DIFC Authority.",
                        "positive_page_id": "law-a_1",
                        "negative_text": "Alice Smith v Registrar. Justice Patel.",
                        "negative_page_id": "case-a_1",
                        "source": "compiled:law",
                        "difficulty": "easy",
                        "strategy": "cross_doc",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_prepare_dataset_and_train_round_trip(tmp_path) -> None:
    trainer = DenseRetrieverTrainer()
    dataset = trainer.prepare_dataset(_triples_path(tmp_path))
    artifact_path = trainer.train(
        base_model="deterministic-hash-r1",
        dataset=dataset,
        config=DenseRetrieverConfig(dimensions=32, negative_weight=0.5),
        output_dir=tmp_path / "artifact",
    )
    artifact = load_artifact(artifact_path)

    assert len(dataset) == 2
    assert artifact.dimensions == 32
    assert "employment" in artifact.token_weights


def test_train_rejects_invalid_config(tmp_path) -> None:
    trainer = DenseRetrieverTrainer()
    dataset = trainer.prepare_dataset(_triples_path(tmp_path))

    try:
        trainer.train(
            base_model="deterministic-hash-r1",
            dataset=dataset,
            config=DenseRetrieverConfig(dimensions=0),
            output_dir=tmp_path / "artifact",
        )
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_evaluate_against_gold_and_compare_baseline(tmp_path) -> None:
    trainer = DenseRetrieverTrainer()
    dataset = trainer.prepare_dataset(_triples_path(tmp_path))
    artifact_path = trainer.train(
        base_model="deterministic-hash-r1",
        dataset=dataset,
        config=DenseRetrieverConfig(dimensions=64),
        output_dir=tmp_path / "artifact",
    )
    artifact = load_artifact(artifact_path)
    eval_cases = [
        RetrievalEvalCase(
            question_id="eval-1",
            question="Who issued Employment Law?",
            family="factoid_authority",
            gold_page_ids=["law-a_1"],
            candidate_page_texts={
                "law-a_1": "Employment Law 2020 was issued by DIFC Authority.",
                "law-b_1": "Employment Amendment Law sets transitional obligations.",
                "case-a_1": "Alice Smith v Registrar. Justice Patel.",
            },
        )
    ]

    metrics = trainer.evaluate_against_gold(artifact=artifact, eval_cases=eval_cases)
    delta = trainer.compare_to_baseline(artifact=artifact, eval_cases=eval_cases)
    query_vectors = trainer.embed_queries(questions=["Who issued Employment Law?"], artifact=artifact)

    assert metrics.mrr_at_10 > 0.0
    assert metrics.recall_at_10 == 1.0
    assert "mrr_at_10_delta" in delta
    assert len(query_vectors[0]) == 64
