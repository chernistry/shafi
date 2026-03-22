from __future__ import annotations

from scripts.rank_support_frontier import (
    FrontierCandidate,
    _dominates,
    _pareto_frontier,
    _select_aggressive,
    _select_conservative,
)


def _candidate(
    *,
    qid: str,
    page_drift: int,
    trusted_delta: float,
    all_delta: float,
    judge_pass_delta: float,
    judge_grounding_delta: float,
) -> FrontierCandidate:
    return FrontierCandidate(
        qids=[qid],
        labels=[qid],
        recommendation="PROMISING",
        answer_changed_count=0,
        retrieval_page_projection_changed_count=page_drift,
        benchmark_all_delta=all_delta,
        benchmark_trusted_delta=trusted_delta,
        judge_pass_delta=judge_pass_delta,
        judge_grounding_delta=judge_grounding_delta,
        candidate_page_p95=4,
    )


def test_select_conservative_prefers_lower_page_drift_at_same_hidden_g() -> None:
    low_drift = _candidate(
        qid="core",
        page_drift=2,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.5,
        judge_grounding_delta=3.0,
    )
    high_drift = _candidate(
        qid="aggr",
        page_drift=3,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.6667,
        judge_grounding_delta=3.6,
    )

    assert _select_conservative([low_drift, high_drift]) == low_drift


def test_select_aggressive_prefers_higher_all_case_signal() -> None:
    base = _candidate(
        qid="core",
        page_drift=2,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.5,
        judge_grounding_delta=3.0,
    )
    wider = _candidate(
        qid="aggr",
        page_drift=3,
        trusted_delta=0.0455,
        all_delta=0.0205,
        judge_pass_delta=0.6667,
        judge_grounding_delta=3.6,
    )

    assert _select_aggressive([base, wider]) == wider


def test_dominates_requires_nonworse_metrics_and_one_strict_gain() -> None:
    better = _candidate(
        qid="better",
        page_drift=2,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.5,
        judge_grounding_delta=3.0,
    )
    worse = _candidate(
        qid="worse",
        page_drift=3,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.5,
        judge_grounding_delta=3.0,
    )

    assert _dominates(better, worse) is True
    assert _dominates(worse, better) is False


def test_pareto_frontier_keeps_nondominated_variants() -> None:
    conservative = _candidate(
        qid="core",
        page_drift=2,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.5,
        judge_grounding_delta=3.0,
    )
    aggressive = _candidate(
        qid="aggr",
        page_drift=3,
        trusted_delta=0.0455,
        all_delta=0.0205,
        judge_pass_delta=0.6667,
        judge_grounding_delta=3.6,
    )
    dominated = _candidate(
        qid="dom",
        page_drift=3,
        trusted_delta=0.0455,
        all_delta=0.02,
        judge_pass_delta=0.4,
        judge_grounding_delta=2.0,
    )

    frontier = _pareto_frontier([conservative, aggressive, dominated])

    assert conservative in frontier
    assert aggressive in frontier
    assert dominated not in frontier
