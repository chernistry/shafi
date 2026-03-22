from __future__ import annotations

from rag_challenge.eval.failure_cartography import (
    FailureTaxonomy,
    ReviewedGoldenCase,
    RunObservation,
    answers_match,
    build_failure_ledger,
    classify_miss,
    compute_drift,
)


def _golden(*, qid: str = "q1", question: str, answer_type: str = "name", answer: str = "Alice", pages: list[str] | None = None) -> ReviewedGoldenCase:
    return ReviewedGoldenCase(
        question_id=qid,
        question=question,
        answer_type=answer_type,
        golden_answer=answer,
        golden_page_ids=pages or ["doc_1"],
        trust_tier="reviewed",
        label_weight=1.0,
    )


def _obs(*, qid: str = "q1", answer: str = "Bob", used: list[str] | None = None, retrieved: list[str] | None = None, run_label: str = "run_a") -> RunObservation:
    return RunObservation(
        run_label=run_label,
        source_path=f"/tmp/{run_label}.json",
        question_id=qid,
        question="placeholder",
        answer_type="name",
        predicted_answer=answer,
        used_page_ids=used or [],
        retrieved_page_ids=retrieved or [],
    )


def test_answers_match_handles_boolean_and_names() -> None:
    assert answers_match(answer_type="boolean", predicted="Yes", golden="true")
    assert answers_match(answer_type="names", predicted="Alice, Bob", golden="Bob; Alice")


def test_classify_miss_marks_retrieval_and_field_failures() -> None:
    golden = _golden(question="Who is the claimant in the case?", pages=["doc_1"])
    failure_types = classify_miss(golden, _obs(answer="Charlie", used=[], retrieved=[]))
    assert FailureTaxonomy.FIELD_MISS in failure_types
    assert FailureTaxonomy.RETRIEVAL_MISS in failure_types


def test_classify_miss_marks_ranking_and_grounding_when_answer_is_correct() -> None:
    golden = _golden(question="What is the title of the law?", answer="DIFC Law", pages=["law_2"])
    observation = _obs(answer="DIFC Law", used=["law_1"], retrieved=["law_2"])
    failure_types = classify_miss(golden, observation)
    assert FailureTaxonomy.RANKING_MISS in failure_types
    assert FailureTaxonomy.GROUNDING_MISS in failure_types


def test_compute_drift_counts_answer_and_page_variants() -> None:
    drift = compute_drift(
        [
            _obs(answer="Alice", used=["doc_1"], run_label="run_a"),
            _obs(answer="Bob", used=["doc_1"], run_label="run_b"),
            _obs(answer="Bob", used=["doc_2"], run_label="run_c"),
        ]
    )
    assert drift.answer_drift_count == 1
    assert drift.page_drift_count == 1


def test_build_failure_ledger_aggregates_counts() -> None:
    reviewed = {
        "q1": _golden(question="What article applies?", answer_type="number", answer="5", pages=["law_5"]),
    }
    observations = [
        RunObservation(
            run_label="run_a",
            source_path="/tmp/run_a.json",
            question_id="q1",
            question="What article applies?",
            answer_type="number",
            predicted_answer="7",
            used_page_ids=[],
            retrieved_page_ids=["law_4"],
        ),
        RunObservation(
            run_label="run_b",
            source_path="/tmp/run_b.json",
            question_id="q1",
            question="What article applies?",
            answer_type="number",
            predicted_answer="5",
            used_page_ids=["law_4"],
            retrieved_page_ids=["law_5"],
        ),
    ]
    ledger = build_failure_ledger(reviewed=reviewed, observations=observations)
    assert ledger.summary["reviewed_questions"] == 1
    assert ledger.summary["run_observations"] == 2
    assert ledger.records[0].doc_family == "law_provision"
