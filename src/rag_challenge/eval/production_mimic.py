from __future__ import annotations

import hashlib
import json
from typing import cast

JsonDict = dict[str, object]
JsonList = list[JsonDict]

_CITATION_FLOORS_BY_ANSWER_TYPE: dict[str, float] = {
    "boolean": 0.80,
    "name": 0.85,
    "names": 0.85,
    "date": 0.85,
    "number": 0.85,
    "free_text": 0.60,
}


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


def _as_float_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for raw_key, raw_value in cast("dict[object, object]", value).items():
        key = str(raw_key).strip()
        if not key:
            continue
        out[key] = _as_float(raw_value)
    return out


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _page_doc(page_id: str) -> str:
    if "_" not in page_id:
        return page_id
    return page_id.rsplit("_", 1)[0]


def _page_num(page_id: str) -> int | None:
    if "_" not in page_id:
        return None
    suffix = page_id.rsplit("_", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _index_scaffold_records(scaffold_payload: JsonDict | None) -> dict[str, JsonDict]:
    if scaffold_payload is None:
        return {}
    records_obj = scaffold_payload.get("records")
    if not isinstance(records_obj, list):
        return {}
    out: dict[str, JsonDict] = {}
    for raw in cast("list[object]", records_obj):
        if not isinstance(raw, dict):
            continue
        record = cast("JsonDict", raw)
        question_id = str(record.get("question_id") or "").strip()
        if question_id:
            out[question_id] = record
    return out


def _requires_title_anchor(record: JsonDict, *, question: str) -> bool:
    requirements = cast("JsonDict", record.get("support_shape_requirements") or {})
    if _as_bool(requirements.get("requires_title_anchor")):
        return True
    shape = str(record.get("support_shape_class") or "").strip().lower()
    if shape in {"comparison", "named_metadata"}:
        return True
    q = question.lower()
    return any(term in q for term in ("title page", "cover page", "first page", "header", "caption"))


def _requires_page_two(record: JsonDict, *, question: str) -> bool:
    anchor = _page_num(str(record.get("required_page_anchor") or "").strip())
    if anchor == 2:
        return True
    q = question.lower()
    return "page 2" in q or "second page" in q


def _requires_outcome_anchor(record: JsonDict, *, question: str) -> bool:
    shape = str(record.get("support_shape_class") or "").strip().lower()
    if shape in {"case_outcome", "outcome_plus_costs"}:
        return True
    q = question.lower()
    return any(term in q for term in ("outcome", "order", "judgment", "disposed", "dismissed", "allowed"))


def build_support_shape_report(
    *,
    raw_results_payload: JsonList | None,
    scaffold_payload: JsonDict | None,
) -> JsonDict:
    if not raw_results_payload or scaffold_payload is None:
        return {
            "cases_scored": 0,
            "page_budget_case_count": 0,
            "citation_overbreadth_case_count": 0,
            "weak_same_doc_anchor_case_count": 0,
            "weak_same_doc_anchor_qids": [],
            "page_budget_qids": [],
            "citation_overbreadth_qids": [],
        }

    records_by_qid = _index_scaffold_records(scaffold_payload)
    page_budget_qids: list[str] = []
    citation_overbreadth_qids: list[str] = []
    weak_anchor_qids: list[str] = []
    cases_scored = 0

    for raw in raw_results_payload:
        telemetry = cast("JsonDict", raw.get("telemetry") or {})
        question_id = str(telemetry.get("question_id") or "").strip()
        if not question_id:
            continue
        record = records_by_qid.get(question_id)
        if record is None:
            continue
        cases_scored += 1

        question = str(record.get("question") or cast("JsonDict", raw.get("case") or {}).get("question") or "").strip()
        used_page_ids = _as_str_list(telemetry.get("used_page_ids"))
        retrieved_page_ids = _as_str_list(telemetry.get("retrieved_page_ids"))
        if not used_page_ids:
            continue

        used_by_doc: dict[str, list[str]] = {}
        retrieved_by_doc: dict[str, list[str]] = {}
        for page_id in used_page_ids:
            used_by_doc.setdefault(_page_doc(page_id), []).append(page_id)
        for page_id in retrieved_page_ids:
            retrieved_by_doc.setdefault(_page_doc(page_id), []).append(page_id)

        minimal_pages = _as_str_list(record.get("minimal_required_support_pages"))
        minimal_set = set(minimal_pages)
        weak_anchor = False

        for gold_page in minimal_pages:
            doc = _page_doc(gold_page)
            if doc in used_by_doc and gold_page in retrieved_page_ids and gold_page not in used_page_ids:
                weak_anchor = True
                break

        if not weak_anchor:
            wants_title = _requires_title_anchor(record, question=question)
            wants_page_two = _requires_page_two(record, question=question)
            wants_outcome = _requires_outcome_anchor(record, question=question)

            for doc, used_pages in used_by_doc.items():
                retrieved_pages = retrieved_by_doc.get(doc, [])
                retrieved_nums = [num for page in retrieved_pages if (num := _page_num(page)) is not None]
                used_nums = [num for page in used_pages if (num := _page_num(page)) is not None]
                if not retrieved_nums or not used_nums:
                    continue
                if wants_title and 1 in retrieved_nums and 1 not in used_nums and doc not in {_page_doc(page) for page in minimal_set if _page_num(page) == 1}:
                    weak_anchor = True
                    break
                if wants_page_two and 2 in retrieved_nums and 2 not in used_nums and doc not in {_page_doc(page) for page in minimal_set if _page_num(page) == 2}:
                    weak_anchor = True
                    break
                if wants_outcome:
                    last_page = max(retrieved_nums)
                    if last_page not in used_nums and doc not in {_page_doc(page) for page in minimal_set if _page_num(page) == last_page}:
                        weak_anchor = True
                        break

        if weak_anchor:
            weak_anchor_qids.append(question_id)

        max_pages_per_doc = max((len(pages) for pages in used_by_doc.values()), default=0)
        if max_pages_per_doc > 2:
            page_budget_qids.append(question_id)

        allowed_total = min(4, max(2, len(minimal_pages) + 1))
        if len(used_page_ids) > allowed_total:
            citation_overbreadth_qids.append(question_id)

    return {
        "cases_scored": cases_scored,
        "page_budget_case_count": len(page_budget_qids),
        "citation_overbreadth_case_count": len(citation_overbreadth_qids),
        "weak_same_doc_anchor_case_count": len(weak_anchor_qids),
        "page_budget_qids": page_budget_qids,
        "citation_overbreadth_qids": citation_overbreadth_qids,
        "weak_same_doc_anchor_qids": weak_anchor_qids,
    }


def _eval_summary(payload: JsonDict | None) -> JsonDict:
    if payload is None:
        return {}
    summary_obj = payload.get("summary")
    if isinstance(summary_obj, dict):
        return cast("JsonDict", summary_obj)
    nested_obj = payload.get("production_mimic")
    if isinstance(nested_obj, dict):
        nested = cast("JsonDict", nested_obj)
        eval_block = cast("JsonDict", nested.get("eval") or {})
        judge_block = cast("JsonDict", nested.get("judge") or {})
        summary: JsonDict = dict(eval_block)
        if judge_block:
            summary["judge"] = judge_block
        return summary
    return payload


def _support_shape_penalty(report: JsonDict) -> tuple[float, list[str]]:
    cases_scored = max(1, _as_int(report.get("cases_scored")))
    weak_cases = _as_int(report.get("weak_same_doc_anchor_case_count"))
    page_budget_cases = _as_int(report.get("page_budget_case_count"))
    citation_overbreadth_cases = _as_int(report.get("citation_overbreadth_case_count"))

    penalty = 0.0
    reasons: list[str] = []

    if weak_cases > 0:
        penalty += min(0.015, 0.02 * (weak_cases / cases_scored))
        reasons.append(
            f"weak same-doc page choices where stronger anchors were already available ({weak_cases}/{cases_scored})"
        )
    if page_budget_cases > 0:
        penalty += min(0.006, 0.01 * (page_budget_cases / cases_scored))
        reasons.append(f"pages-per-doc exceeded strict local budget ({page_budget_cases}/{cases_scored})")
    if citation_overbreadth_cases > 0:
        penalty += min(0.008, 0.01 * (citation_overbreadth_cases / cases_scored))
        reasons.append(
            f"citation overbreadth exceeded strict local budget ({citation_overbreadth_cases}/{cases_scored})"
        )

    return penalty, reasons


def extract_judge_summary(payload: JsonDict | None) -> JsonDict:
    summary = _eval_summary(payload)
    judge_obj = summary.get("judge")
    if isinstance(judge_obj, dict):
        return cast("JsonDict", judge_obj)
    return {}


def _judge_case_rows(payload: JsonDict | None) -> list[JsonDict]:
    if payload is None:
        return []
    for key in ("cases", "judge_cases", "records", "judge_records"):
        rows_obj = payload.get(key)
        if isinstance(rows_obj, list):
            return [cast("JsonDict", item) for item in cast("list[object]", rows_obj) if isinstance(item, dict)]
    summary = _eval_summary(payload)
    judge = cast("JsonDict", summary.get("judge") or {})
    for key in ("cases", "judge_cases", "records", "judge_records"):
        rows_obj = judge.get(key)
        if isinstance(rows_obj, list):
            return [cast("JsonDict", item) for item in cast("list[object]", rows_obj) if isinstance(item, dict)]
    return []


def _judge_context_fingerprint(case: JsonDict) -> str:
    explicit = str(case.get("context_fingerprint") or "").strip()
    if explicit:
        return explicit
    for key in ("context_ids", "context_page_ids", "used_page_ids", "retrieved_page_ids"):
        values = _as_str_list(case.get(key))
        if values:
            return _sha256_text(_canonical_json(values))
    context = case.get("context")
    if context is not None:
        return _sha256_text(_canonical_json(context))
    return ""


def _judge_cache_key(case: JsonDict) -> str:
    answer = str(case.get("answer") or case.get("answer_text") or "").strip()
    cited_ids = _as_str_list(case.get("cited_ids") or case.get("cited_chunk_ids"))
    prompt_version = str(case.get("judge_prompt_version") or case.get("prompt_version") or "").strip()
    payload = {
        "answer": answer,
        "cited_ids": cited_ids,
        "context_fingerprint": _judge_context_fingerprint(case),
        "judge_prompt_version": prompt_version,
    }
    return _sha256_text(_canonical_json(payload))


def _judge_cache_summary(
    *,
    cheap_payload: JsonDict | None,
    strict_payload: JsonDict | None,
    use_strict_judge: bool,
    strict_skip_reason: str | None,
) -> JsonDict:
    cheap_keys = {_judge_cache_key(case) for case in _judge_case_rows(cheap_payload)}
    strict_requested_keys = {_judge_cache_key(case) for case in _judge_case_rows(strict_payload)}
    strict_used_keys: set[str] = strict_requested_keys if use_strict_judge else set()
    shared = cheap_keys.intersection(strict_used_keys)
    return {
        "cheap_case_count": len(cheap_keys),
        "strict_case_count_requested": len(strict_requested_keys),
        "strict_case_count_used": len(strict_used_keys),
        "cheap_cache_key_count": len(cheap_keys),
        "strict_cache_key_count": len(strict_used_keys),
        "shared_cache_key_count": len(shared),
        "cache_hit_count": len(shared),
        "cache_miss_count": len(cheap_keys.union(strict_used_keys) - shared),
        "strict_requested": bool(strict_requested_keys) or strict_payload is not None,
        "strict_used": use_strict_judge and bool(strict_requested_keys or strict_payload is not None),
        "strict_skip_reason": strict_skip_reason or "",
    }


def extract_eval_metrics(payload: JsonDict | None) -> JsonDict:
    summary = _eval_summary(payload)
    return {
        "citation_coverage": summary.get("citation_coverage"),
        "citation_coverage_by_answer_type": _as_float_dict(summary.get("citation_coverage_by_answer_type")),
        "answer_type_format_compliance": summary.get("answer_type_format_compliance"),
        "grounding_g_score_beta_2_5": summary.get("grounding_g_score_beta_2_5"),
        "judge": extract_judge_summary(payload),
    }


def aggregate_hybrid_strict_judge(
    *,
    cheap_payload: JsonDict | None,
    strict_payload: JsonDict | None,
    use_strict_judge: bool = True,
    strict_skip_reason: str | None = None,
) -> JsonDict:
    cheap = extract_judge_summary(cheap_payload)
    strict = extract_judge_summary(strict_payload) if use_strict_judge else {}
    cache = _judge_cache_summary(
        cheap_payload=cheap_payload,
        strict_payload=strict_payload,
        use_strict_judge=use_strict_judge,
        strict_skip_reason=strict_skip_reason,
    )

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
        "strict_requested": bool(cache.get("strict_requested")),
        "strict_used": bool(cache.get("strict_used")),
        "strict_skip_reason": str(cache.get("strict_skip_reason") or ""),
        "disagreement": disagreement,
        "judge_timeout_or_failure": failures > 0 or not cheap_present,
        "cache": cache,
        "top_fails": strict.get("top_fails") or cheap.get("top_fails") or [],
    }


def aggregate_hybrid_strict_eval(
    *,
    cheap_payload: JsonDict | None,
    strict_payload: JsonDict | None,
    use_strict_judge: bool = True,
    strict_skip_reason: str | None = None,
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

    def _pick_min_by_answer_type(metric: str) -> dict[str, float]:
        merged: dict[str, float] = {}
        for block in (cheap, strict):
            values = _as_float_dict(block.get(metric))
            for answer_type, value in values.items():
                current = merged.get(answer_type)
                merged[answer_type] = value if current is None else min(current, value)
        return merged

    return {
        "citation_coverage": _pick_min("citation_coverage"),
        "citation_coverage_by_answer_type": _pick_min_by_answer_type("citation_coverage_by_answer_type"),
        "answer_type_format_compliance": _pick_min("answer_type_format_compliance"),
        "grounding_g_score_beta_2_5": _pick_min("grounding_g_score_beta_2_5"),
        "judge": aggregate_hybrid_strict_judge(
            cheap_payload=cheap_payload,
            strict_payload=strict_payload,
            use_strict_judge=use_strict_judge,
            strict_skip_reason=strict_skip_reason,
        ),
    }


def should_run_strict_judge(*, subject_summary: JsonDict, candidate_row: JsonDict) -> tuple[bool, str | None]:
    current_total = _as_float(subject_summary.get("total"))
    upper_total = _as_float(candidate_row.get("upper_total_estimate"), default=_as_float(candidate_row.get("strict_total_estimate")))
    if upper_total <= current_total + 1e-9:
        return False, "candidate_not_near_promote"
    return True, None


def build_page_trace_summary(page_trace_payload: JsonDict | None) -> JsonDict:
    empty_summary: JsonDict = {
        "cases_scored": 0,
        "trusted_case_count": 0,
        "gold_in_retrieved_count": 0,
        "gold_in_reranked_count": 0,
        "gold_in_used_count": 0,
        "false_positive_case_count": 0,
        "failure_stage_counts": {},
        "stage_examples": {},
        "explained_ratio": 0.0,
        "page_true_positive_count": 0,
        "page_used_count": 0,
        "page_gold_count": 0,
        "page_precision": 0.0,
        "page_recall": 0.0,
        "trusted_page_true_positive_count": 0,
        "trusted_page_used_count": 0,
        "trusted_page_gold_count": 0,
        "trusted_page_precision": 0.0,
        "trusted_page_recall": 0.0,
    }
    if page_trace_payload is None:
        return empty_summary
    summary_obj = page_trace_payload.get("summary")
    if not isinstance(summary_obj, dict):
        return empty_summary
    summary = cast("JsonDict", summary_obj)
    records_obj = page_trace_payload.get("records")
    page_true_positive_count = 0
    page_used_count = 0
    page_gold_count = 0
    trusted_page_true_positive_count = 0
    trusted_page_used_count = 0
    trusted_page_gold_count = 0
    if isinstance(records_obj, list):
        for raw_record in cast("list[object]", records_obj):
            if not isinstance(raw_record, dict):
                continue
            record = cast("JsonDict", raw_record)
            gold_pages = set(_as_str_list(record.get("gold_pages")))
            used_pages = set(_as_str_list(record.get("used_pages")))
            true_positive_count = len(gold_pages.intersection(used_pages))
            page_true_positive_count += true_positive_count
            page_used_count += len(used_pages)
            page_gold_count += len(gold_pages)
            if str(record.get("trust_tier") or "").strip().lower() == "trusted":
                trusted_page_true_positive_count += true_positive_count
                trusted_page_used_count += len(used_pages)
                trusted_page_gold_count += len(gold_pages)
    return {
        "cases_scored": _as_int(summary.get("cases_scored")),
        "trusted_case_count": _as_int(summary.get("trusted_case_count")),
        "gold_in_retrieved_count": _as_int(summary.get("gold_in_retrieved_count")),
        "gold_in_reranked_count": _as_int(summary.get("gold_in_reranked_count")),
        "gold_in_used_count": _as_int(summary.get("gold_in_used_count")),
        "false_positive_case_count": _as_int(summary.get("false_positive_case_count")),
        "failure_stage_counts": cast("dict[str, object]", summary.get("failure_stage_counts") or {}),
        "stage_examples": cast("dict[str, object]", summary.get("stage_examples") or {}),
        "explained_ratio": _as_float(summary.get("explained_ratio")),
        "page_true_positive_count": page_true_positive_count,
        "page_used_count": page_used_count,
        "page_gold_count": page_gold_count,
        "page_precision": 0.0 if page_used_count <= 0 else page_true_positive_count / page_used_count,
        "page_recall": 0.0 if page_gold_count <= 0 else page_true_positive_count / page_gold_count,
        "trusted_page_true_positive_count": trusted_page_true_positive_count,
        "trusted_page_used_count": trusted_page_used_count,
        "trusted_page_gold_count": trusted_page_gold_count,
        "trusted_page_precision": (
            0.0 if trusted_page_used_count <= 0 else trusted_page_true_positive_count / trusted_page_used_count
        ),
        "trusted_page_recall": (
            0.0 if trusted_page_gold_count <= 0 else trusted_page_true_positive_count / trusted_page_gold_count
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


def _citation_floor_failures(citation_coverage_by_answer_type: dict[str, float]) -> list[JsonDict]:
    failures: list[JsonDict] = []
    for answer_type, floor in _CITATION_FLOORS_BY_ANSWER_TYPE.items():
        observed = citation_coverage_by_answer_type.get(answer_type)
        if observed is None or observed + 1e-9 >= floor:
            continue
        failures.append(
            {
                "answer_type": answer_type,
                "observed": round(observed, 4),
                "floor": floor,
                "gap": round(floor - observed, 4),
            }
        )
    return failures


def estimate_production_mimic(
    *,
    subject_summary: JsonDict,
    candidate_row: JsonDict,
    exactness_report: JsonDict | None,
    equivalence_report: JsonDict | None,
    cheap_eval_payload: JsonDict | None,
    strict_eval_payload: JsonDict | None,
    calibration: JsonDict | None,
    raw_results_payload: JsonList | None = None,
    scaffold_payload: JsonDict | None = None,
    page_trace_payload: JsonDict | None = None,
) -> JsonDict:
    calibration_block = calibration or {}
    use_strict_judge, strict_skip_reason = should_run_strict_judge(
        subject_summary=subject_summary,
        candidate_row=candidate_row,
    )
    hybrid_eval = aggregate_hybrid_strict_eval(
        cheap_payload=cheap_eval_payload,
        strict_payload=strict_eval_payload,
        use_strict_judge=use_strict_judge,
        strict_skip_reason=strict_skip_reason,
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
    citation_coverage_by_answer_type = _as_float_dict(hybrid_eval.get("citation_coverage_by_answer_type"))
    grounding_g_score = hybrid_eval.get("grounding_g_score_beta_2_5")
    lineage_confidence = infer_lineage_confidence(
        candidate_row=candidate_row,
        equivalence_report=equivalence_report,
    )
    page_trace = build_page_trace_summary(page_trace_payload)
    support_shape = build_support_shape_report(
        raw_results_payload=raw_results_payload,
        scaffold_payload=scaffold_payload,
    )
    citation_floor_failures = _citation_floor_failures(citation_coverage_by_answer_type)

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
    if citation_floor_failures:
        extra_penalty += min(0.012, 0.02 * sum(_as_float(item.get("gap")) for item in citation_floor_failures))
        no_submit_reasons.append(
            "citation floor miss: "
            + ", ".join(f"{item.get('answer_type') or ''}<{_as_float(item.get('floor')):.2f}" for item in citation_floor_failures)
        )
    if grounding_g_score is not None and _as_float(grounding_g_score) < 0.8:
        extra_penalty += 0.004 * (0.8 - _as_float(grounding_g_score))
        no_submit_reasons.append("grounding score below strict local bar")

    pass_rate = hybrid_judge.get("pass_rate")
    avg_grounding = hybrid_judge.get("avg_grounding")
    avg_accuracy = hybrid_judge.get("avg_accuracy")
    judge_failures = _as_int(hybrid_judge.get("judge_failures"))

    judge_pass_rate_penalty = 0.0
    judge_grounding_penalty = 0.0
    judge_accuracy_penalty = 0.0
    judge_disagreement_penalty = 0.0
    judge_timeout_penalty = 0.0

    if pass_rate is not None and _as_float(pass_rate) < 1.0:
        judge_pass_rate_penalty = 0.01 * (1.0 - _as_float(pass_rate))
        extra_penalty += judge_pass_rate_penalty
        no_submit_reasons.append("judge pass rate below perfect")
    if avg_grounding is not None and _as_float(avg_grounding) < 5.0:
        judge_grounding_penalty = 0.002 * (5.0 - _as_float(avg_grounding))
        extra_penalty += judge_grounding_penalty
        no_submit_reasons.append("judge grounding below perfect")
    if avg_accuracy is not None and _as_float(avg_accuracy) < 5.0:
        judge_accuracy_penalty = 0.0015 * (5.0 - _as_float(avg_accuracy))
        extra_penalty += judge_accuracy_penalty
        no_submit_reasons.append("judge accuracy below perfect")
    if _as_bool(hybrid_judge.get("disagreement")):
        judge_disagreement_penalty = 0.002
        extra_penalty += judge_disagreement_penalty
        no_submit_reasons.append("cheap/strict judge disagree")
    if _as_bool(hybrid_judge.get("judge_timeout_or_failure")) or judge_failures > 0:
        judge_timeout_penalty = 0.004
        extra_penalty += judge_timeout_penalty
        no_submit_reasons.append("judge timeout/failure present")

    if _as_int(candidate_row.get("page_drift")) > 0 and _as_float(candidate_row.get("hidden_g_trusted_delta")) <= 0.0:
        extra_penalty += 0.0025
        no_submit_reasons.append("page drift without trusted hidden-G gain")
    cases_scored = _as_int(page_trace.get("cases_scored"))
    page_precision = _as_float(page_trace.get("page_precision"))
    page_recall = _as_float(page_trace.get("page_recall"))
    trusted_case_count = _as_int(page_trace.get("trusted_case_count"))
    if cases_scored >= 5 and page_precision < 0.35:
        extra_penalty += 0.02 * (0.35 - page_precision)
        no_submit_reasons.append("page-id precision below strict local floor")
    if cases_scored >= 5 and page_recall < 0.50:
        extra_penalty += 0.015 * (0.50 - page_recall)
        no_submit_reasons.append("page-id recall below strict local floor")
    if cases_scored > 0 and trusted_case_count == 0:
        extra_penalty += 0.003
        no_submit_reasons.append("changed-set page trace has no trusted page-id cases")
    elif 0 < trusted_case_count < 5:
        extra_penalty += 0.0015
        no_submit_reasons.append("trusted page-id slice still narrow")
    if cases_scored > 0 and _as_float(page_trace.get("explained_ratio")) < 0.95:
        extra_penalty += 0.004 * (0.95 - _as_float(page_trace.get("explained_ratio")))
        no_submit_reasons.append("page-trace stage explanation below strict local bar")
    support_shape_penalty, support_shape_reasons = _support_shape_penalty(support_shape)
    extra_penalty += support_shape_penalty
    no_submit_reasons.extend(support_shape_reasons)

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
        and not citation_floor_failures
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
        "judge_penalties": {
            "pass_rate_penalty": judge_pass_rate_penalty,
            "grounding_penalty": judge_grounding_penalty,
            "accuracy_penalty": judge_accuracy_penalty,
            "disagreement_penalty": judge_disagreement_penalty,
            "timeout_penalty": judge_timeout_penalty,
            "total": (
                judge_pass_rate_penalty
                + judge_grounding_penalty
                + judge_accuracy_penalty
                + judge_disagreement_penalty
                + judge_timeout_penalty
            ),
        },
        "eval": {
            "citation_coverage": citation_coverage,
            "citation_coverage_by_answer_type": citation_coverage_by_answer_type,
            "citation_floor_failures": citation_floor_failures,
            "answer_type_format_compliance": format_compliance,
            "grounding_g_score_beta_2_5": grounding_g_score,
        },
        "page_trace": page_trace,
        "support_shape": support_shape,
        "platform_like_total_estimate": platform_like_total_estimate,
        "strict_total_estimate": strict_total_estimate,
        "paranoid_total_estimate": paranoid_total_estimate,
        "upper_total_estimate": upper_total,
        "extra_paranoid_penalty": extra_penalty,
        "submit_eligibility": submit_eligibility,
        "no_submit_reason": "; ".join(no_submit_reasons) if no_submit_reasons else "",
    }
