from __future__ import annotations

from pathlib import Path

from rag_challenge.core.retrieval_utility import BundleSnapshot, EscalationTarget, RetrievalUtilityPredictor
from rag_challenge.ml.utility_predictor import UtilityPredictorTrainer, load_raw_results, load_reviewed_gold


def test_bundle_snapshot_from_raw_result_extracts_pre_generation_features() -> None:
    """BundleSnapshot should expose retrieval-side signals only."""
    snapshot = BundleSnapshot.from_raw_result(
        {
            "case": {
                "case_id": "q1",
                "question": "Compare the effective date and the amendment date.",
                "answer_type": "date",
            },
            "telemetry": {
                "retrieved_page_ids": ["a", "b", "c"],
                "context_page_ids": ["a", "b"],
                "cited_page_ids": ["a"],
                "used_page_ids": ["a"],
                "retrieved_chunk_ids": ["c1", "c2", "c3"],
                "context_chunk_ids": ["c1", "c2"],
                "cited_chunk_ids": ["c1"],
                "doc_refs": ["Doc A"],
                "context_budget_tokens": 900,
            },
        }
    )

    assert snapshot.compare_signal is True
    assert snapshot.temporal_signal is True
    assert snapshot.field_signal is True
    assert snapshot.retrieved_page_count == 3
    assert snapshot.cited_page_count == 1
    assert snapshot.doc_ref_count == 1


def test_predictor_without_model_uses_heuristic_and_route_rules() -> None:
    """A model-free predictor should still make deterministic decisions."""
    predictor = RetrievalUtilityPredictor(threshold=0.5)
    snapshot = BundleSnapshot(
        question_id="q2",
        question="Who were the claimants in the case?",
        answer_type="names",
        retrieved_page_count=4,
        context_page_count=1,
        cited_page_count=0,
        used_page_count=0,
        retrieved_chunk_count=4,
        context_chunk_count=1,
        cited_chunk_count=0,
        doc_ref_count=1,
        bridge_hit_count=0,
        entity_confidence=0.0,
        context_budget_tokens=800,
        compare_signal=False,
        temporal_signal=False,
        field_signal=True,
    )

    prediction = predictor.predict(snapshot)

    assert prediction.bundle_sufficiency < 0.5
    assert prediction.escalation_target is EscalationTarget.STRUCTURED_DB
    assert predictor.should_escalate(prediction) is True


def test_trainer_cross_validation_beats_heuristic_on_public_slice() -> None:
    """The offline utility predictor should beat a trivial heuristic baseline."""
    repo_root = Path(__file__).resolve().parents[2]
    trainer = UtilityPredictorTrainer()
    raw_results = load_raw_results(
        repo_root
        / ".sdd"
        / "researches"
        / "639_grounding_resume_after_devops_baseline_r1_2026-03-19"
        / "raw_results_reviewed_public100_sidecar_current.json"
    )
    reviewed_gold = load_reviewed_gold(repo_root / ".sdd" / "golden" / "reviewed" / "reviewed_all_100.json")
    examples = trainer.build_training_examples(raw_results, reviewed_gold)

    metrics = trainer.cross_validate(examples, folds=5)

    assert metrics.sample_count == 100
    assert metrics.accuracy > metrics.heuristic_accuracy
    assert metrics.f1 > metrics.heuristic_f1


def test_trainer_save_and_reload_round_trip(tmp_path: Path) -> None:
    """Saved artifacts should reload into a predictor with the same interface."""
    trainer = UtilityPredictorTrainer()
    examples = [
        trainer.build_training_examples(
            [
                {
                    "case": {
                        "case_id": "q1",
                        "question": "Who were the claimants?",
                        "answer_type": "names",
                    },
                    "telemetry": {
                        "cited_page_ids": ["p1"],
                        "context_page_ids": ["p1"],
                        "used_page_ids": ["p1"],
                        "retrieved_page_ids": ["p1", "p2"],
                        "retrieved_chunk_ids": ["c1", "c2"],
                        "context_chunk_ids": ["c1"],
                        "doc_refs": ["Doc A"],
                    },
                    "answer_text": "Fursa Consulting",
                },
                {
                    "case": {
                        "case_id": "q2",
                        "question": "Who were the claimants?",
                        "answer_type": "names",
                    },
                    "telemetry": {
                        "cited_page_ids": [],
                        "context_page_ids": ["p2"],
                        "used_page_ids": [],
                        "retrieved_page_ids": ["p1", "p2"],
                        "retrieved_chunk_ids": ["c1", "c2"],
                        "context_chunk_ids": ["c1"],
                        "doc_refs": ["Doc A"],
                    },
                    "answer_text": "Wrong Answer",
                },
            ],
            {
                "q1": {"question_id": "q1", "golden_answer": "Fursa Consulting", "answer_type": "names"},
                "q2": {"question_id": "q2", "golden_answer": "Fursa Consulting", "answer_type": "names"},
            },
        )
    ][0]
    fitted = trainer.fit(examples)
    artifact_path = trainer.save(fitted, tmp_path)
    reloaded = trainer.load(artifact_path)

    prediction = reloaded.predict(
        BundleSnapshot(
            question_id="q1",
            question="Who were the claimants?",
            answer_type="names",
            retrieved_page_count=2,
            context_page_count=1,
            cited_page_count=1,
            used_page_count=1,
            retrieved_chunk_count=2,
            context_chunk_count=1,
            cited_chunk_count=0,
            doc_ref_count=1,
            bridge_hit_count=0,
            entity_confidence=0.0,
            context_budget_tokens=900,
            compare_signal=False,
            temporal_signal=False,
            field_signal=True,
        )
    )

    assert prediction.bundle_sufficiency > 0.5
