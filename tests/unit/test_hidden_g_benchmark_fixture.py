from __future__ import annotations

from pathlib import Path

from scripts.score_page_benchmark import _load_benchmark


def test_hidden_g_benchmark_seed_fixture_has_expected_shape() -> None:
    fixture_path = Path("tests/fixtures/internal_hidden_g_benchmark_seed.json")

    cases = _load_benchmark(fixture_path)

    assert len(cases) == 100
    assert len({case.question_id for case in cases}) == 100
    assert sum(1 for case in cases if case.wrong_document_risk) >= 16
    assert sum(1 for case in cases if case.trust_tier == "trusted") >= 20
    assert sum(1 for case in cases if case.trust_tier == "suspect") >= 50
    assert sum(1 for case in cases if case.items) >= 21
    assert any(
        len({page for item in case.items for slot in item.slots for page in slot.gold_page_ids}) >= 2
        for case in cases
    )


def test_hidden_g_benchmark_trusted_cases_have_reviewable_metadata() -> None:
    fixture_path = Path("tests/fixtures/internal_hidden_g_benchmark_seed.json")

    cases = _load_benchmark(fixture_path)
    trusted_cases = [case for case in cases if case.trust_tier == "trusted"]

    assert trusted_cases
    assert all(case.audit_note for case in trusted_cases)
    assert all(case.gold_origin in {"manual_override", "reviewed_correction"} for case in trusted_cases)


def test_hidden_g_benchmark_keeps_frozen_trusted_regression_sentinels() -> None:
    fixture_path = Path("tests/fixtures/internal_hidden_g_benchmark_seed.json")

    cases = _load_benchmark(fixture_path)
    by_question_id = {case.question_id: case for case in cases}

    sentinels = {
        "117267649104e2ac88d57b64c615721dc2b3f0631b7d4914f6f85323651e8cb4": "manual_override",
        "4ce050c0d6261bf3ee2eafa9c7d5fc7273e390a4a1c09ab6e26f691c68199d1b": "manual_override",
        "c595f1180b440f4e6ea5e130563fb4c2e9705557d3abf10e401948c0eb73b268": "manual_override",
        "7700103c51940db23ba51a0efefbef679201af5b0a60935853d10bf81a260466": "reviewed_correction",
    }

    for question_id, expected_origin in sentinels.items():
        case = by_question_id[question_id]
        assert case.trust_tier == "trusted"
        assert case.gold_origin == expected_origin
        assert case.audit_note
