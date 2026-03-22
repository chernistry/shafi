from __future__ import annotations

import pytest

from rag_challenge.eval.harness import _g_score


def test_g_score_perfect_overlap() -> None:
    assert _g_score({"a", "b"}, {"a", "b"}) == 1.0


def test_g_score_no_overlap() -> None:
    assert _g_score({"a"}, {"b"}) == 0.0


def test_g_score_empty_gold_edge_cases() -> None:
    assert _g_score(set(), set()) == 1.0
    assert _g_score({"a"}, set()) == 0.0


def test_g_score_recall_weighted_beta_2_5() -> None:
    value = _g_score({"a", "b", "c", "d", "e"}, {"a", "b"}, beta=2.5)
    assert value == pytest.approx(0.8285714286, rel=1e-6)
