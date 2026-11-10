from __future__ import annotations

from shafi.eval.resurrection_matrix import (
    HOT_SET_PREFIXES,
    ResurrectionEvidence,
    assess_resurrection_candidate,
    hot_set_overlap_prefixes,
)


def test_hot_set_overlap_prefixes_detects_consensus_qids() -> None:
    overlap = hot_set_overlap_prefixes(
        [
            "9f9fb4b911d75c22f2c9a42bb852848ac45594179a7d0d126c8ef0ac8941b18d",
            "d374bee20e02f7f384e766dd792856c6457f99fc1491e5fee2e6cd50a4d856b7",
        ]
    )

    assert overlap == [HOT_SET_PREFIXES[0], HOT_SET_PREFIXES[-1]]


def test_assess_resurrection_candidate_marks_high_signal_clean_candidate_as_over_penalized() -> None:
    evidence = ResurrectionEvidence(
        label="triad_f331_e0798",
        candidate_class="support_only_offense",
        baseline_label="v6_context_seed",
        lineage_confidence="high",
        answer_changed_count=0,
        page_changed_count=4,
        hidden_g_trusted_delta=0.0425,
        strict_total_estimate=0.7606,
        platform_like_total_estimate=0.7529,
        paranoid_total_estimate=0.74156,
        no_submit_reason="citation coverage below strict local bar; weak same-doc page choices where stronger anchors were already available (28/100)",
        tracked_artifacts_ok=True,
        hot_set_touched_prefixes=["e0798b", "f33177"],
    )

    assessment = assess_resurrection_candidate(evidence, public_anchor_total=0.74156)

    assert assessment.status == "probably_over_penalized"
    assert assessment.resurrection_score > 0.0
    assert assessment.toxicity_score < 0.8


def test_assess_resurrection_candidate_marks_missing_low_lineage_branch_as_confounded() -> None:
    evidence = ResurrectionEvidence(
        label="t54_single_doc_rerank_gate_r1",
        candidate_class="single_doc_explicit_provision_page_rerank_gate",
        baseline_label="v6_context_seed",
        lineage_confidence="low",
        answer_changed_count=19,
        page_changed_count=57,
        hidden_g_trusted_delta=0.0584,
        strict_total_estimate=0.7621,
        platform_like_total_estimate=0.7527,
        paranoid_total_estimate=0.5881,
        no_submit_reason="lineage_confidence=low; page-id precision below strict local floor; page-id recall below strict local floor",
        tracked_artifacts_ok=False,
        hot_set_touched_prefixes=["e0798b"],
    )

    assessment = assess_resurrection_candidate(evidence, public_anchor_total=0.74156)

    assert assessment.status == "confounded_but_interesting"
    assert "missing_tracked_candidate_artifacts" in assessment.confounded_reasons
    assert "low_lineage_confidence" in assessment.confounded_reasons


def test_assess_resurrection_candidate_marks_zero_hidden_gain_with_toxic_blockers_as_toxic() -> None:
    evidence = ResurrectionEvidence(
        label="v_t21_doc_page_rerank_phase1_surrogate_r1",
        candidate_class="page_localization_rerank",
        baseline_label="v6_context_seed",
        lineage_confidence="high",
        answer_changed_count=0,
        page_changed_count=6,
        hidden_g_trusted_delta=0.0,
        strict_total_estimate=0.7433,
        platform_like_total_estimate=0.7312,
        paranoid_total_estimate=0.7191,
        no_submit_reason="page drift without trusted hidden-G gain; page-id precision below strict local floor; citation overbreadth exceeded strict local budget",
        tracked_artifacts_ok=True,
        hot_set_touched_prefixes=[],
    )

    assessment = assess_resurrection_candidate(evidence, public_anchor_total=0.74156)

    assert assessment.status == "toxic_even_if_locally_shiny"
    assert assessment.toxicity_score >= 0.8
