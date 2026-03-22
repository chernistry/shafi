import pytest

from shafi.eval.judge import parse_judge_result


def _json(verdict: str = "PASS") -> str:
    return (
        "{"
        f'"verdict": "{verdict}",'
        '"scores": {"accuracy": 5, "grounding": 4, "clarity": 3, "uncertainty_handling": 2},'
        '"format_issues": [],'
        '"unsupported_claims": [],'
        '"grounding_evidence": [],'
        '"recommended_fix": ""'
        "}"
    )


def test_parse_judge_result_accepts_raw_json() -> None:
    parsed = parse_judge_result(_json("PASS"))
    assert parsed.verdict == "PASS"
    assert parsed.scores.grounding == 4


def test_parse_judge_result_accepts_fenced_json() -> None:
    raw = "```json\n" + _json("FAIL") + "\n```"
    parsed = parse_judge_result(raw)
    assert parsed.verdict == "FAIL"


def test_parse_judge_result_accepts_leading_trailing_prose() -> None:
    raw = "Here you go:\n" + _json("PASS") + "\nThanks."
    parsed = parse_judge_result(raw)
    assert parsed.verdict == "PASS"


def test_parse_judge_result_raises_on_missing_json() -> None:
    with pytest.raises(ValueError):
        parse_judge_result("not json at all")
