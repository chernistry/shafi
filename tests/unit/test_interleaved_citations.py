"""Test that enable_interleaved_citations routes to the correct system prompt variant."""

from __future__ import annotations

from shafi.llm.generator_prompts import build_system_prompt
from shafi.models import QueryComplexity


def _make_build_args(**kwargs: object) -> dict[str, object]:
    """Build default args for build_system_prompt."""
    return {
        "question": "What does Article 5(1) require?",
        "complexity": QueryComplexity.COMPLEX,
        "answer_kind": "free_text",
        "answer_word_limit": 150,
        "prompt_hint": "",
        "answer_type": "free_text",
        "answer_type_instruction": lambda _: "",
        "should_use_irac": lambda _: False,
        **kwargs,
    }


_INTERLEAVED_MARKER = "citation rule"  # Unique to interleaved prompts — standard prompts use "EVIDENCE ANCHOR"


def test_interleaved_flag_uses_per_sentence_citation_prompt() -> None:
    """Interleaved variant must include the CITATION RULE header."""
    prompt_on = build_system_prompt(**_make_build_args(enable_interleaved_citations=True))
    assert _INTERLEAVED_MARKER in prompt_on.lower(), "Interleaved prompt must contain 'CITATION RULE' header"


def test_default_flag_uses_standard_prompt() -> None:
    """Standard variant must NOT contain the interleaved CITATION RULE header."""
    prompt_off = build_system_prompt(**_make_build_args(enable_interleaved_citations=False))
    assert _INTERLEAVED_MARKER not in prompt_off.lower()


def test_irac_interleaved_flag_routes_to_irac_interleaved() -> None:
    """IRAC path with flag=True uses IRAC interleaved variant."""
    prompt_on = build_system_prompt(
        **_make_build_args(
            enable_interleaved_citations=True,
            should_use_irac=lambda _: True,
        )
    )
    assert _INTERLEAVED_MARKER in prompt_on.lower()


def test_irac_default_flag_uses_standard_irac_prompt() -> None:
    """IRAC path with flag=False uses standard IRAC variant."""
    prompt_off = build_system_prompt(
        **_make_build_args(
            enable_interleaved_citations=False,
            should_use_irac=lambda _: True,
        )
    )
    assert _INTERLEAVED_MARKER not in prompt_off.lower()


def test_strict_type_unaffected_by_interleaved_flag() -> None:
    """Strict answer types (name/number/date) are NOT affected by interleaved flag."""
    for strict_type in ("name", "names", "number", "date", "boolean"):
        prompt = build_system_prompt(
            **_make_build_args(
                answer_kind=strict_type,
                answer_type=strict_type,
                enable_interleaved_citations=True,
            )
        )
        assert _INTERLEAVED_MARKER not in prompt.lower(), (
            f"Strict type '{strict_type}' should not be affected by interleaved flag"
        )


def test_simple_complexity_unaffected_by_interleaved_flag() -> None:
    """SIMPLE complexity questions always get the simple prompt, flag has no effect."""
    prompt = build_system_prompt(
        **_make_build_args(
            complexity=QueryComplexity.SIMPLE,
            enable_interleaved_citations=True,
        )
    )
    assert _INTERLEAVED_MARKER not in prompt.lower()
