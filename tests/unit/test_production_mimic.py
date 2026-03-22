from __future__ import annotations

from rag_challenge.eval.production_mimic import (
    aggregate_hybrid_strict_eval,
    build_page_trace_summary,
    build_public_history_calibration,
    build_support_shape_report,
    estimate_production_mimic,
)


def test_aggregate_hybrid_strict_eval_prefers_worse_metrics() -> None:
    cheap = {
        "summary": {
            "citation_coverage": 0.95,
            "citation_coverage_by_answer_type": {"boolean": 0.95, "free_text": 0.7},
            "answer_type_format_compliance": 1.0,
            "grounding_g_score_beta_2_5": 0.84,
            "judge": {
                "cases": 4,
                "pass_rate": 1.0,
                "avg_accuracy": 4.8,
                "avg_grounding": 4.4,
                "avg_clarity": 4.9,
                "avg_uncertainty_handling": 4.7,
                "judge_failures": 0,
            },
        }
    }
    strict = {
        "summary": {
            "citation_coverage": 1.0,
            "citation_coverage_by_answer_type": {"boolean": 0.85, "free_text": 0.5},
            "answer_type_format_compliance": 0.9,
            "grounding_g_score_beta_2_5": 0.72,
            "judge": {
                "cases": 2,
                "pass_rate": 0.5,
                "avg_accuracy": 4.2,
                "avg_grounding": 3.9,
                "avg_clarity": 4.5,
                "avg_uncertainty_handling": 4.0,
                "judge_failures": 1,
            },
        }
    }

    aggregated = aggregate_hybrid_strict_eval(cheap_payload=cheap, strict_payload=strict)
    judge = aggregated["judge"]

    assert aggregated["citation_coverage"] == 0.95
    assert aggregated["citation_coverage_by_answer_type"] == {"boolean": 0.85, "free_text": 0.5}
    assert aggregated["answer_type_format_compliance"] == 0.9
    assert aggregated["grounding_g_score_beta_2_5"] == 0.72
    assert judge["pass_rate"] == 0.5
    assert judge["avg_grounding"] == 3.9
    assert judge["avg_accuracy"] == 4.2
    assert judge["judge_failures"] == 1
    assert judge["disagreement"] is True


def test_aggregate_hybrid_strict_eval_accepts_production_mimic_payload() -> None:
    cheap = {
        "production_mimic": {
            "eval": {
                "citation_coverage": 0.8,
                "citation_coverage_by_answer_type": {"boolean": 0.82, "name": 0.9},
                "answer_type_format_compliance": 0.95,
                "grounding_g_score_beta_2_5": 0.7,
            },
            "judge": {
                "cases": 3,
                "pass_rate": 1.0,
                "avg_accuracy": 5.0,
                "avg_grounding": 4.5,
                "judge_failures": 0,
            },
        }
    }

    aggregated = aggregate_hybrid_strict_eval(cheap_payload=cheap, strict_payload=None)

    assert aggregated["citation_coverage"] == 0.8
    assert aggregated["citation_coverage_by_answer_type"] == {"boolean": 0.82, "name": 0.9}
    assert aggregated["answer_type_format_compliance"] == 0.95
    assert aggregated["grounding_g_score_beta_2_5"] == 0.7
    assert aggregated["judge"]["avg_grounding"] == 4.5
    assert aggregated["judge"]["judge_timeout_or_failure"] is False


def test_estimate_production_mimic_penalizes_lineage_and_unresolved_cases() -> None:
    calibration = build_public_history_calibration(
        [
            {
                "external_total": 0.71,
                "strict_total_estimate": 0.73,
                "paranoid_total_estimate": 0.72,
                "platform_like_total_estimate": 0.725,
            }
        ]
    )
    result = estimate_production_mimic(
        subject_summary={
            "total": 0.74156,
        },
        candidate_row={
            "label": "triad_f331_e0798_plus_dotted",
            "strict_total_estimate": 0.78,
            "upper_total_estimate": 0.80,
            "paranoid_total_estimate": 0.77,
            "hidden_g_trusted_delta": 0.0425,
            "page_drift": 4,
            "lineage_ok": True,
        },
        exactness_report={
            "resolved_incorrect_qids": ["43f77", "f950"],
            "still_mismatched_incorrect_qids": ["5046"],
        },
        equivalence_report=None,
        cheap_eval_payload={
            "summary": {
                "citation_coverage": 0.9,
                "answer_type_format_compliance": 0.95,
                "grounding_g_score_beta_2_5": 0.75,
                "judge": {
                    "cases": 2,
                    "pass_rate": 0.5,
                    "avg_accuracy": 4.2,
                    "avg_grounding": 4.1,
                    "avg_clarity": 4.5,
                    "avg_uncertainty_handling": 4.0,
                    "judge_failures": 0,
                },
            }
        },
        strict_eval_payload=None,
        calibration=calibration,
    )

    assert result["candidate_class"] == "triad_f331_e0798_plus_dotted"
    assert result["lineage_confidence"] == "medium"
    assert result["submit_eligibility"] is False
    assert result["paranoid_total_estimate"] < result["strict_total_estimate"] < result["upper_total_estimate"]
    assert "lineage_confidence=medium" in str(result["no_submit_reason"])
    assert "known incorrect scaffold cases remain unresolved" in str(result["no_submit_reason"])
    assert "judge pass rate below perfect" in str(result["no_submit_reason"])


def test_build_support_shape_report_detects_anchor_miss_and_overbreadth() -> None:
    raw_results = [
        {
            "case": {"question": "Compare the parties in these cases."},
            "telemetry": {
                "question_id": "q1",
                "used_page_ids": ["docA_3", "docA_4", "docA_5", "docB_2"],
                "retrieved_page_ids": ["docA_1", "docA_3", "docA_4", "docA_5", "docB_1", "docB_2"],
            },
        }
    ]
    scaffold = {
        "records": [
            {
                "question_id": "q1",
                "question": "Compare the parties in these cases.",
                "support_shape_class": "comparison",
                "support_shape_requirements": {"requires_title_anchor": True},
                "minimal_required_support_pages": ["docA_1", "docB_1"],
            }
        ]
    }

    report = build_support_shape_report(raw_results_payload=raw_results, scaffold_payload=scaffold)

    assert report["cases_scored"] == 1
    assert report["weak_same_doc_anchor_case_count"] == 1
    assert report["page_budget_case_count"] == 1
    assert report["citation_overbreadth_case_count"] == 1
    assert report["weak_same_doc_anchor_qids"] == ["q1"]


def test_build_page_trace_summary_preserves_stage_counts() -> None:
    summary = build_page_trace_summary(
        {
            "summary": {
                "cases_scored": 4,
                "trusted_case_count": 2,
                "gold_in_retrieved_count": 3,
                "gold_in_reranked_count": 2,
                "gold_in_used_count": 1,
                "false_positive_case_count": 2,
                "failure_stage_counts": {"lost_after_context": 1, "wrong_page_used_same_doc": 2},
                "stage_examples": {"lost_after_context": ["5046"], "wrong_page_used_same_doc": ["9f9f"]},
                "explained_ratio": 1.0,
            },
            "records": [
                {
                    "qid": "q1",
                    "gold_pages": ["docA_1", "docB_1"],
                    "used_pages": ["docA_1", "docA_4"],
                    "trust_tier": "trusted",
                },
                {
                    "qid": "q2",
                    "gold_pages": ["docC_2"],
                    "used_pages": ["docC_2", "docD_9"],
                    "trust_tier": "suspect",
                },
            ],
        }
    )

    assert summary["cases_scored"] == 4
    assert summary["gold_in_reranked_count"] == 2
    assert summary["failure_stage_counts"] == {"lost_after_context": 1, "wrong_page_used_same_doc": 2}
    assert summary["stage_examples"] == {"lost_after_context": ["5046"], "wrong_page_used_same_doc": ["9f9f"]}
    assert summary["page_true_positive_count"] == 2
    assert summary["page_used_count"] == 4
    assert summary["page_gold_count"] == 3
    assert summary["page_precision"] == 0.5
    assert summary["page_recall"] == 2 / 3
    assert summary["trusted_page_precision"] == 0.5
    assert summary["trusted_page_recall"] == 0.5
    trusted_bootstrap = summary["trusted_bootstrap"]
    assert trusted_bootstrap["record_count"] == 1
    assert trusted_bootstrap["sample_count"] == 500
    assert trusted_bootstrap["unstable_small_slice"] is True
    assert trusted_bootstrap["precision_p50"] == 0.5
    assert trusted_bootstrap["recall_p50"] == 0.5


def test_estimate_production_mimic_penalizes_support_shape_issues() -> None:
    result = estimate_production_mimic(
        subject_summary={"total": 0.74156},
        candidate_row={
            "label": "v10_local_support_shape_v2_r1",
            "strict_total_estimate": 0.78,
            "upper_total_estimate": 0.79,
            "paranoid_total_estimate": 0.775,
            "hidden_g_trusted_delta": 0.0,
            "page_drift": 1,
            "lineage_ok": True,
        },
        exactness_report={"resolved_incorrect_qids": [], "still_mismatched_incorrect_qids": []},
        equivalence_report={"safe_baselines": ["v6"]},
        cheap_eval_payload=None,
        strict_eval_payload=None,
        calibration={},
        raw_results_payload=[
            {
                "case": {"question": "Compare the parties in these cases."},
                "telemetry": {
                    "question_id": "q1",
                    "used_page_ids": ["docA_3", "docA_4", "docA_5", "docB_2"],
                    "retrieved_page_ids": ["docA_1", "docA_3", "docA_4", "docA_5", "docB_1", "docB_2"],
                },
            }
        ],
        scaffold_payload={
            "records": [
                {
                    "question_id": "q1",
                    "question": "Compare the parties in these cases.",
                    "support_shape_class": "comparison",
                    "support_shape_requirements": {"requires_title_anchor": True},
                    "minimal_required_support_pages": ["docA_1", "docB_1"],
                }
            ]
        },
    )

    assert result["support_shape"]["weak_same_doc_anchor_case_count"] == 1
    assert result["support_shape"]["page_budget_case_count"] == 1
    assert result["paranoid_total_estimate"] < result["platform_like_total_estimate"] < result["strict_total_estimate"]
    assert "weak same-doc page choices" in str(result["no_submit_reason"])


def test_estimate_production_mimic_penalizes_page_trace_and_citation_floor_failures() -> None:
    result = estimate_production_mimic(
        subject_summary={"total": 0.74156},
        candidate_row={
            "label": "strict_eval_candidate",
            "strict_total_estimate": 0.79,
            "upper_total_estimate": 0.80,
            "paranoid_total_estimate": 0.785,
            "hidden_g_trusted_delta": 0.01,
            "page_drift": 0,
            "lineage_ok": True,
        },
        exactness_report={"resolved_incorrect_qids": [], "still_mismatched_incorrect_qids": []},
        equivalence_report={"safe_baselines": ["v6"]},
        cheap_eval_payload={
            "summary": {
                "citation_coverage": 0.9,
                "citation_coverage_by_answer_type": {
                    "boolean": 0.75,
                    "free_text_structured": 0.65,
                    "free_text_model": 0.62,
                },
                "answer_type_format_compliance": 1.0,
                "grounding_g_score_beta_2_5": 0.83,
                "judge": {
                    "cases": 3,
                    "pass_rate": 1.0,
                    "avg_accuracy": 5.0,
                    "avg_grounding": 5.0,
                    "avg_clarity": 5.0,
                    "avg_uncertainty_handling": 5.0,
                    "judge_failures": 0,
                },
            }
        },
        strict_eval_payload=None,
        calibration={},
        raw_results_payload=None,
        scaffold_payload=None,
        page_trace_payload={
            "summary": {
                "cases_scored": 5,
                "trusted_case_count": 0,
                "gold_in_retrieved_count": 4,
                "gold_in_reranked_count": 3,
                "gold_in_used_count": 2,
                "false_positive_case_count": 4,
                "failure_stage_counts": {"lost_after_context": 2},
                "stage_examples": {"lost_after_context": ["q1", "q2"]},
                "explained_ratio": 0.8,
            },
            "records": [
                {"qid": "q1", "gold_pages": ["docA_1"], "used_pages": ["docA_3"]},
                {"qid": "q2", "gold_pages": ["docB_1"], "used_pages": ["docB_1", "docB_5"]},
                {"qid": "q3", "gold_pages": ["docC_1"], "used_pages": ["docC_5", "docD_2"]},
                {"qid": "q4", "gold_pages": ["docD_2"], "used_pages": ["docD_2", "docE_1"]},
                {"qid": "q5", "gold_pages": ["docF_9"], "used_pages": ["docF_4", "docG_1"]},
            ],
        },
    )

    assert result["eval"]["citation_floor_failures"] == [
        {"answer_type": "boolean", "observed": 0.75, "floor": 0.8, "gap": 0.05},
        {"answer_type": "free_text_structured", "observed": 0.65, "floor": 0.7, "gap": 0.05},
    ]
    assert result["eval"]["citation_floor_failure_count"] == 2
    assert result["eval"]["citation_floor_failure_answer_types"] == ["boolean", "free_text_structured"]
    assert result["eval"]["citation_hard_floor_blocked"] is True
    assert result["eval"]["citation_page_trace_disagreement"] is True
    assert result["eval"]["citation_page_trace_disagreement_penalty"] > 0.0
    assert result["page_trace"]["page_precision"] == 2 / 9
    assert result["page_trace"]["page_recall"] == 0.4
    assert result["submit_eligibility"] is False
    assert "citation coverage below strict local bar" not in str(result["no_submit_reason"])
    assert "citation floor miss" in str(result["no_submit_reason"])
    assert "citation/page-trace disagreement requires explanation" in str(result["no_submit_reason"])
    assert "page-id precision below strict local floor" in str(result["no_submit_reason"])
    assert "changed-set page trace has no trusted page-id cases" in str(result["no_submit_reason"])
    assert result["strict_raw_total_estimate"] == result["strict_total_estimate"]
    assert result["strict_policy_blocked_total_estimate"] < result["strict_raw_total_estimate"]
    assert result["policy_debt"]["citation_aggregate_penalty"] > 0.0
    assert result["policy_debt"]["citation_floor_penalty"] > 0.0
    assert result["policy_debt"]["total"] > 0.0


def test_estimate_production_mimic_does_not_block_clean_strict_run_on_citation_floor() -> None:
    result = estimate_production_mimic(
        subject_summary={"total": 0.74156},
        candidate_row={
            "label": "clean_strict_candidate",
            "strict_total_estimate": 0.79,
            "upper_total_estimate": 0.80,
            "paranoid_total_estimate": 0.785,
            "hidden_g_trusted_delta": 0.02,
            "page_drift": 0,
            "lineage_ok": True,
        },
        exactness_report={"resolved_incorrect_qids": [], "still_mismatched_incorrect_qids": []},
        equivalence_report={"safe_baselines": ["v6"]},
        cheap_eval_payload={
            "summary": {
                "citation_coverage": 0.88,
                "citation_coverage_by_answer_type": {"boolean": 0.82, "name": 0.9},
                "answer_type_format_compliance": 1.0,
                "grounding_g_score_beta_2_5": 0.9,
                "judge": {
                    "cases": 2,
                    "pass_rate": 1.0,
                    "avg_accuracy": 5.0,
                    "avg_grounding": 5.0,
                    "avg_clarity": 5.0,
                    "avg_uncertainty_handling": 5.0,
                    "judge_failures": 0,
                },
            }
        },
        strict_eval_payload=None,
        calibration={},
        page_trace_payload={
            "summary": {
                "cases_scored": 5,
                "trusted_case_count": 5,
                "gold_in_retrieved_count": 5,
                "gold_in_reranked_count": 5,
                "gold_in_used_count": 5,
                "false_positive_case_count": 0,
                "failure_stage_counts": {"retained_to_used": 5},
                "stage_examples": {"retained_to_used": ["q1", "q2"]},
                "explained_ratio": 1.0,
            },
            "records": [
                {"qid": "q1", "gold_pages": ["docA_1"], "used_pages": ["docA_1"], "trust_tier": "trusted"},
                {"qid": "q2", "gold_pages": ["docB_1"], "used_pages": ["docB_1"], "trust_tier": "trusted"},
                {"qid": "q3", "gold_pages": ["docC_1"], "used_pages": ["docC_1"], "trust_tier": "trusted"},
                {"qid": "q4", "gold_pages": ["docD_1"], "used_pages": ["docD_1"], "trust_tier": "trusted"},
                {"qid": "q5", "gold_pages": ["docE_1"], "used_pages": ["docE_1"], "trust_tier": "trusted"},
            ],
        },
    )

    assert result["eval"]["citation_floor_failures"] == []
    assert result["eval"]["citation_hard_floor_blocked"] is False
    assert result["eval"]["citation_page_trace_disagreement"] is False
    trusted_bootstrap = result["page_trace"]["trusted_bootstrap"]
    assert trusted_bootstrap["record_count"] == 5
    assert trusted_bootstrap["sample_count"] == 500
    assert trusted_bootstrap["unstable_small_slice"] is False
    assert trusted_bootstrap["precision_p50"] == 1.0
    assert trusted_bootstrap["recall_p50"] == 1.0
