from __future__ import annotations

from pathlib import Path

from scripts.score_page_benchmark import _load_benchmark


def test_hidden_g_benchmark_seed_fixture_has_expected_shape() -> None:
    fixture_path = Path("tests/fixtures/internal_hidden_g_benchmark_seed.json")

    cases = _load_benchmark(fixture_path)

    assert len(cases) >= 109
    assert len({case.question_id for case in cases}) == len(cases)
    assert sum(1 for case in cases if case.wrong_document_risk) >= 16
    assert sum(1 for case in cases if case.trust_tier == "trusted") >= 30
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


def test_hidden_g_benchmark_keeps_scaffold_backed_anchor_cases() -> None:
    fixture_path = Path("tests/fixtures/internal_hidden_g_benchmark_seed.json")

    cases = _load_benchmark(fixture_path)
    by_question_id = {case.question_id: case for case in cases}

    expected = {
        "802d12a65530cd728da4c2f4430275488285ced247123431a905f8d525e80ede": ["58eae81bf668e7f6c58619f419a49b5e35e2e5c9c7475475ace28ec562580545_2"],
        "1a80e5dc69fb76a97c0ada049df2425c9f06de373bbb9ee5b39dbd1f12f85387": [
            "437568a8709ce09a56f69e804f93f13ae099d84539b6ccca519e7f3102ec8eae_1",
            "897ab23ed5a70034d3d708d871ad1da8bc7b6608d94b1ca46b5d578d985d3c13_1",
        ],
        "2e211d0cdb29134de759b277c8b8ed5b8fe43f033dcec1e4bd15feafb1dbb8ab": [
            "3f8a5ea07f3cbfb9f993f214bdb0907080fcf4f0c4f0061d53d28374abf7ccfa_1",
            "6306079a16b1dec85690f75c715cdbd78b0685a3e19ee30250d481bc32f2e29a_1",
        ],
        "2e8b251fd560f4446bfc9c5fa4b83ecbce69d650a8fd00fc9815c1fb34a764ad": [
            "6248961be609f0274db7ce51637e31ff5d498a06dc9225310f709573e10305ca_1",
            "62930da3ce81071f64e0ff652c0605d0475e7e0ac2a5b3ce5f52a5a71bbf0d39_1",
        ],
        "39890efe7db1258f568e838c284020a9fc79b8bf0deddab76fdf853c9b0171ed": [
            "09660f784dfd6cf819cb0be11bf67f61c190b7f8f390ee29507f9f628d7ff3f0_1",
            "1b446e192bfdfa28fd0f5eb531548fe0e6be5f8c70b13f245cfc6b3214efc24c_1",
        ],
        "d374bee20e02f7f384e766dd792856c6457f99fc1491e5fee2e6cd50a4d856b7": [
            "5d3df6d63c5d54e93e8df5ee3fe7235c49faec3ef43f31c2586a172937ea61c1_1",
            "6306079a16b1dec85690f75c715cdbd78b0685a3e19ee30250d481bc32f2e29a_1",
        ],
        "5d271fced60d88e008a69adc2da21de427906206bfb49f5554a3bf1dd6f72772": ["fbdd7f9dd299d83b1f398778da2e6765dfaaed62005667264734a1f76ec09071_1"],
        "ff1b357588e9b5881ec9b1558ede2892e7cbf42a6787a1cbb922b45318f66778": ["ff746f7b583490a80ba104361c0a82a1ebbf7ed9097cd03dc49d744cb5057761_1"],
        "922dcd5ae0d2420ec5577ed8094f593124f7b69f7e61b12c24ad46410114e8ae": ["4e387152960c1029b3711cacb05b287b13c977bc61f2558059a62b7b427a62eb_1"],
        "e153746c20cc385a520728ac381151f424c9eee10e4c582904fee70afe9af243": ["33bc02044716acdfedb164b065bdaec098aaadcae863c591f9931c88e7307d16_6"],
    }

    for question_id, gold_page_ids in expected.items():
        case = by_question_id[question_id]
        assert case.trust_tier == "trusted"
        assert case.gold_origin == "manual_override"
        assert case.gold_page_ids == gold_page_ids
        assert "scaffold" in case.audit_note.casefold()
