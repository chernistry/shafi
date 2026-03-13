from __future__ import annotations

from typing import cast

JsonDict = dict[str, object]


def _as_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _as_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in cast("list[object]", value) if (text := str(item).strip())]


def _eval_summary(payload: JsonDict | None) -> JsonDict:
    if payload is None:
        return {}
    summary_obj = payload.get("summary")
    if isinstance(summary_obj, dict):
        return cast("JsonDict", summary_obj)
    return payload


def extract_judge_summary(payload: JsonDict | None) -> JsonDict:
    summary = _eval_summary(payload)
    judge_obj = summary.get("judge")
    if isinstance(judge_obj, dict):
        return cast("JsonDict", judge_obj)
    return {}


def extract_eval_metrics(payload: JsonDict | None) -> JsonDict:
    summary = _eval_summary(payload)
    return {
        "citation_coverage": summary.get("citation_coverage"),
        "answer_type_format_compliance": summary.get("answer_type_format_compliance"),
        "grounding_g_score_beta_2_5": summary.get("grounding_g_score_beta_2_5"),
        "judge": extract_judge_summary(payload),
    }


def aggregate_hybrid_strict_judge(
    *,
    cheap_payload: JsonDict | None,
    strict_payload: JsonDict | None,
) -> JsonDict:
    cheap = extract_judge_summary(cheap_payload)
    strict = extract_judge_summary(strict_payload)

    def _pick_min(key: str) -> float | None:
        values = [
            _as_float(block.get(key))
            for block in (cheap, strict)
            if key in block and block.get(key) is not None
        ]
        if not values:
            return None
        return min(values)

    pass_rate = _pick_min("pass_rate")
    avg_accuracy = _pick_min("avg_accuracy")
    avg_grounding = _pick_min("avg_grounding")
    avg_clarity = _pick_min("avg_clarity")
    avg_uncertainty = _pick_min("avg_uncertainty_handling")
    cases = max(_as_int(cheap.get("cases")), _as_int(strict.get("cases")))
    failures = max(_as_int(cheap.get("judge_failures")), _as_int(strict.get("judge_failures")))
    strict_present = bool(strict)
    cheap_present = bool(cheap)
    disagreement = False
    if cheap_present and strict_present:
        for key in ("pass_rate", "avg_accuracy", "avg_grounding", "avg_clarity", "avg_uncertainty_handling"):
            cheap_value = cheap.get(key)
            strict_value = strict.get(key)
            if cheap_value is None or strict_value is None:
                continue
            if abs(_as_float(cheap_value) - _as_float(strict_value)) > 1e-9:
                disagreement = True
                break

    return {
        "cases": cases,
        "pass_rate": pass_rate,
        "avg_accuracy": avg_accuracy,
        "avg_grounding": avg_grounding,
        "avg_clarity": avg_clarity,
        "avg_uncertainty_handling": avg_uncertainty,
        "judge_failures": failures,
        "strict_present": strict_present,
        "cheap_present": cheap_present,
        "disagreement": disagreement,
        "judge_timeout_or_failure": failures > 0 or not cheap_present,
        "top_fails": strict.get("top_fails") or cheap.get("top_fails") or [],
    }


def aggregate_hybrid_strict_eval(
    *,
    cheap_payload: JsonDict | None,
    strict_payload: JsonDict | None,
) -> JsonDict:
    cheap = extract_eval_metrics(cheap_payload)
    strict = extract_eval_metrics(strict_payload)

    def _pick_min(metric: str) -> float | None:
        values = [
            _as_float(block.get(metric))
            for block in (cheap, strict)
            if block.get(metric) is not None
        ]
        if not values:
            return None
        return min(values)

    return {
        "citation_coverage": _pick_min("citation_coverage"),
        "answer_type_format_compliance": _pick_min("answer_type_format_compliance"),
        "grounding_g_score_beta_2_5": _pick_min("grounding_g_score_beta_2_5"),
        "judge": aggregate_hybrid_strict_judge(
            cheap_payload=cheap_payload,
            strict_payload=strict_payload,
        ),
    }


def infer_lineage_confidence(
    *,
    candidate_row: JsonDict,
    equivalence_report: JsonDict | None,
) -> str:
    if not _as_bool(candidate_row.get("lineage_ok")):
        return "low"
    if equivalence_report is None:
        return "medium"
    safe_baselines = _as_str_list(equivalence_report.get("safe_baselines"))
    if safe_baselines:
        return "high"
    unexpected_answer = _as_str_list(equivalence_report.get("unexpected_answer_qids"))
    unexpected_page = _as_str_list(equivalence_report.get("unexpected_page_qids"))
    return "high" if not unexpected_answer and not unexpected_page else "medium"


def build_public_history_calibration(rows: list[JsonDict]) -> JsonDict:
    strict_gaps: list[float] = []
    paranoid_gaps: list[float] = []
    platform_like_gaps: list[float] = []
    for row in rows:
        external_total = row.get("external_total")
        if external_total is None:
            continue
        external = _as_float(external_total, default=-1.0)
        if external < 0.0:
            continue
        if row.get("strict_total_estimate") is not None:
            strict_gaps.append(max(0.0, _as_float(row.get("strict_total_estimate")) - external))
        if row.get("paranoid_total_estimate") is not None:
            paranoid_gaps.append(max(0.0, _as_float(row.get("paranoid_total_estimate")) - external))
        if row.get("platform_like_total_estimate") is not None:
            platform_like_gaps.append(max(0.0, _as_float(row.get("platform_like_total_estimate")) - external))

    strict_offset = max(strict_gaps) if strict_gaps else 0.0
    paranoid_offset = max(paranoid_gaps) if paranoid_gaps else strict_offset
    platform_like_offset = max(platform_like_gaps) if platform_like_gaps else max(strict_offset, paranoid_offset)
    return {
        "history_rows_used": len(strict_gaps) or len(paranoid_gaps) or len(platform_like_gaps),
        "strict_offset": strict_offset,
        "paranoid_offset": paranoid_offset,
        "platform_like_offset": platform_like_offset,
    }


def estimate_production_mimic(
    *,
    subject_summary: JsonDict,
    candidate_row: JsonDict,
    exactness_report: JsonDict | None,
    equivalence_report: JsonDict | None,
    cheap_eval_payload: JsonDict | None,
    strict_eval_payload: JsonDict | None,
    calibration: JsonDict | None,
) -> JsonDict:
    calibration_block = calibration or {}
    hybrid_eval = aggregate_hybrid_strict_eval(
        cheap_payload=cheap_eval_payload,
        strict_payload=strict_eval_payload,
    )
    hybrid_judge = cast("JsonDict", hybrid_eval.get("judge") or {})

    strict_total = _as_float(
        candidate_row.get("strict_total_estimate"),
        default=_as_float(subject_summary.get("total")),
    )
    upper_total = _as_float(
        candidate_row.get("upper_total_estimate"),
        default=strict_total,
    )
    paranoid_total = _as_float(
        candidate_row.get("paranoid_total_estimate"),
        default=min(strict_total, _as_float(subject_summary.get("total"))),
    )

    unresolved_qids = _as_str_list((exactness_report or {}).get("still_mismatched_incorrect_qids"))
    format_compliance = hybrid_eval.get("answer_type_format_compliance")
    citation_coverage = hybrid_eval.get("citation_coverage")
    grounding_g_score = hybrid_eval.get("grounding_g_score_beta_2_5")
    lineage_confidence = infer_lineage_confidence(
        candidate_row=candidate_row,
        equivalence_report=equivalence_report,
    )

    extra_penalty = 0.0
    no_submit_reasons: list[str] = []

    if lineage_confidence != "high":
        extra_penalty += 0.004
        no_submit_reasons.append(f"lineage_confidence={lineage_confidence}")
    if unresolved_qids:
        extra_penalty += 0.0035 * len(unresolved_qids)
        no_submit_reasons.append("known incorrect scaffold cases remain unresolved")
    if format_compliance is not None and _as_float(format_compliance) < 1.0:
        extra_penalty += 0.008 * (1.0 - _as_float(format_compliance))
        no_submit_reasons.append("format compliance below strict local bar")
    if citation_coverage is not None and _as_float(citation_coverage) < 1.0:
        extra_penalty += 0.008 * (1.0 - _as_float(citation_coverage))
        no_submit_reasons.append("citation coverage below strict local bar")
    if grounding_g_score is not None and _as_float(grounding_g_score) < 0.8:
        extra_penalty += 0.004 * (0.8 - _as_float(grounding_g_score))
        no_submit_reasons.append("grounding score below strict local bar")

    pass_rate = hybrid_judge.get("pass_rate")
    avg_grounding = hybrid_judge.get("avg_grounding")
    avg_accuracy = hybrid_judge.get("avg_accuracy")
    judge_failures = _as_int(hybrid_judge.get("judge_failures"))

    if pass_rate is not None and _as_float(pass_rate) < 1.0:
        extra_penalty += 0.01 * (1.0 - _as_float(pass_rate))
        no_submit_reasons.append("judge pass rate below perfect")
    if avg_grounding is not None and _as_float(avg_grounding) < 5.0:
        extra_penalty += 0.002 * (5.0 - _as_float(avg_grounding))
        no_submit_reasons.append("judge grounding below perfect")
    if avg_accuracy is not None and _as_float(avg_accuracy) < 5.0:
        extra_penalty += 0.0015 * (5.0 - _as_float(avg_accuracy))
        no_submit_reasons.append("judge accuracy below perfect")
    if _as_bool(hybrid_judge.get("disagreement")):
        extra_penalty += 0.002
        no_submit_reasons.append("cheap/strict judge disagree")
    if _as_bool(hybrid_judge.get("judge_timeout_or_failure")) or judge_failures > 0:
        extra_penalty += 0.004
        no_submit_reasons.append("judge timeout/failure present")

    if _as_int(candidate_row.get("page_drift")) > 0 and _as_float(candidate_row.get("hidden_g_trusted_delta")) <= 0.0:
        extra_penalty += 0.0025
        no_submit_reasons.append("page drift without trusted hidden-G gain")

    strict_offset = _as_float(calibration_block.get("strict_offset"))
    platform_like_offset = _as_float(calibration_block.get("platform_like_offset"))
    paranoid_offset = _as_float(calibration_block.get("paranoid_offset"))

    strict_total_estimate = max(0.0, strict_total - strict_offset)
    platform_like_total_estimate = max(
        0.0,
        strict_total_estimate - platform_like_offset - (extra_penalty * 0.5),
    )
    paranoid_total_estimate = max(
        0.0,
        min(paranoid_total - paranoid_offset, platform_like_total_estimate - (extra_penalty * 0.5)),
    )

    current_total = _as_float(subject_summary.get("total"))
    submit_eligibility = (
        platform_like_total_estimate > current_total + 1e-9
        and _as_float(candidate_row.get("hidden_g_trusted_delta")) >= 0.0
        and lineage_confidence == "high"
        and not unresolved_qids
        and not _as_bool(hybrid_judge.get("judge_timeout_or_failure"))
        and (pass_rate is None or _as_float(pass_rate) >= 1.0)
    )
    if not submit_eligibility and not no_submit_reasons:
        no_submit_reasons.append("platform-like estimate does not clear the current public baseline under strict governance")

    candidate_class = str(candidate_row.get("branch_class") or candidate_row.get("label") or "unknown").strip() or "unknown"
    return {
        "candidate_class": candidate_class,
        "lineage_confidence": lineage_confidence,
        "hidden_g_trusted": {
            "delta": _as_float(candidate_row.get("hidden_g_trusted_delta")),
            "baseline": candidate_row.get("benchmark_trusted_baseline"),
            "candidate": candidate_row.get("benchmark_trusted_candidate"),
        },
        "hidden_g_all": {
            "delta": _as_float(candidate_row.get("hidden_g_all_delta")),
            "baseline": candidate_row.get("benchmark_all_baseline"),
            "candidate": candidate_row.get("benchmark_all_candidate"),
        },
        "exactness": {
            "resolved_incorrect_qids": _as_str_list((exactness_report or {}).get("resolved_incorrect_qids")),
            "still_mismatched_incorrect_qids": unresolved_qids,
        },
        "judge": hybrid_judge,
        "eval": {
            "citation_coverage": citation_coverage,
            "answer_type_format_compliance": format_compliance,
            "grounding_g_score_beta_2_5": grounding_g_score,
        },
        "platform_like_total_estimate": platform_like_total_estimate,
        "strict_total_estimate": strict_total_estimate,
        "paranoid_total_estimate": paranoid_total_estimate,
        "upper_total_estimate": upper_total,
        "extra_paranoid_penalty": extra_penalty,
        "submit_eligibility": submit_eligibility,
        "no_submit_reason": "; ".join(no_submit_reasons) if no_submit_reasons else "",
    }
