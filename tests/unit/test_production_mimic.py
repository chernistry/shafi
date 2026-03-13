from __future__ import annotations

from rag_challenge.eval.production_mimic import (
    aggregate_hybrid_strict_eval,
    build_public_history_calibration,
    estimate_production_mimic,
)


def test_aggregate_hybrid_strict_eval_prefers_worse_metrics() -> None:
    cheap = {
        "summary": {
            "citation_coverage": 0.95,
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
    assert aggregated["answer_type_format_compliance"] == 0.9
    assert aggregated["grounding_g_score_beta_2_5"] == 0.72
    assert judge["pass_rate"] == 0.5
    assert judge["avg_grounding"] == 3.9
    assert judge["avg_accuracy"] == 4.2
    assert judge["judge_failures"] == 1
    assert judge["disagreement"] is True


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
