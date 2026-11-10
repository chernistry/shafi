from __future__ import annotations

from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "golden_sample.json"


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="Golden dataset not found")
def test_retrieval_recall_regression() -> None:
    deepeval = pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase

    from shafi.eval.golden import load_golden_dataset
    from shafi.eval.metrics import AnswerTypeFormatCompliance, GoldChunkRecallAtK

    assert hasattr(deepeval, "assert_test")
    cases = load_golden_dataset(GOLDEN_PATH)
    recall_metric = GoldChunkRecallAtK(k=80, threshold=0.8)
    format_metric = AnswerTypeFormatCompliance()

    for case in cases[:10]:
        if case.answer_type == "boolean":
            placeholder_answer = "Yes (cite: placeholder)"
        elif case.answer_type == "number":
            placeholder_answer = "42 (cite: placeholder)"
        elif case.answer_type == "date":
            placeholder_answer = "2024-01-01 (cite: placeholder)"
        elif case.answer_type == "names":
            placeholder_answer = "Alice, Bob (cite: placeholder)"
        else:
            placeholder_answer = "Placeholder answer (cite: placeholder)"

        tc = LLMTestCase(
            input=case.question,
            actual_output=placeholder_answer,
            expected_output=case.to_expected_output(),
            retrieval_context=[f"chunk_id={cid}\nplaceholder text" for cid in case.gold_chunk_ids],
            additional_metadata={"answer_type": case.answer_type, "id": case.case_id},
        )
        if case.gold_chunk_ids:
            score = recall_metric.measure(tc)
            assert score >= 0.8
            assert recall_metric.is_successful() is True
        format_score = format_metric.measure(tc)
        assert format_score == 1.0
        assert format_metric.is_successful() is True
