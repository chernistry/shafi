# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]
ConfigDict = dict[str, float | int]


def _f_beta(*, predicted: set[str], gold: set[str], beta: float) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    true_positive = len(predicted.intersection(gold))
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


def _load_rows(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object row in {path}")
        rows.append(cast("JsonDict", row))
    return rows


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _rows_by_qid(rows: list[JsonDict]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        if qid:
            grouped[qid].append(row)
    return dict(grouped)


def _as_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _baseline_predictions(case_rows: list[JsonDict]) -> list[str]:
    ranked = sorted(
        [row for row in case_rows if _coerce_bool(row.get("is_baseline_predicted"))],
        key=lambda row: (_coerce_int(row.get("baseline_rank")) or 9999, str(row.get("page_id") or "")),
    )
    return [str(row.get("page_id") or "").strip() for row in ranked if str(row.get("page_id") or "").strip()]


def _score_row(row: JsonDict, *, config: ConfigDict) -> float:
    baseline_rank = _coerce_int(row.get("baseline_rank")) or 999
    page_number = _coerce_int(row.get("page_number")) or 999
    requested_page_match = _coerce_bool(row.get("requested_page_match"))
    is_page_one = _coerce_bool(row.get("is_page_one"))
    is_page_two = _coerce_bool(row.get("is_page_two"))
    query_has_title_page = _coerce_bool(row.get("query_has_title_page"))
    query_has_cover_page = _coerce_bool(row.get("query_has_cover_page"))
    query_has_first_page = _coerce_bool(row.get("query_has_first_page"))
    query_has_second_page = _coerce_bool(row.get("query_has_second_page"))
    query_has_article = _coerce_bool(row.get("query_has_article"))
    query_has_section = _coerce_bool(row.get("query_has_section"))

    score = 0.0
    score += _as_float(config["baseline_rank_weight"]) * (100.0 - float(baseline_rank))
    if requested_page_match:
        score += _as_float(config["requested_page_bonus"])
    if (query_has_title_page or query_has_cover_page or query_has_first_page) and is_page_one:
        score += _as_float(config["page_one_title_bonus"])
    if query_has_second_page and is_page_two:
        score += _as_float(config["page_two_bonus"])
    if (query_has_title_page or query_has_cover_page or query_has_first_page) and is_page_one:
        score += _as_float(config["title_page_decay_weight"]) * max(0.0, 10.0 - float(page_number))
    if query_has_article or query_has_section:
        score += _as_float(config["article_page_bias"]) * float(page_number)
    return score


def _select_candidate_pages(case_rows: list[JsonDict], *, config: ConfigDict) -> list[str]:
    observed_rows = [row for row in case_rows if _coerce_bool(row.get("candidate_observed"))]
    scored = sorted(
        observed_rows,
        key=lambda row: (
            -_score_row(row, config=config),
            _coerce_int(row.get("baseline_rank")) or 9999,
            _coerce_int(row.get("page_number")) or 9999,
            str(row.get("page_id") or ""),
        ),
    )
    keep_top_k = max(1, _coerce_int(config.get("keep_top_k")) or 1)
    return [str(row.get("page_id") or "").strip() for row in scored[:keep_top_k] if str(row.get("page_id") or "").strip()]


def _case_metrics(predicted_pages: list[str], gold_pages: list[str], *, beta: float) -> JsonDict:
    predicted = set(predicted_pages)
    gold = set(gold_pages)
    precision = len(predicted.intersection(gold)) / len(predicted) if predicted else 0.0
    recall = len(predicted.intersection(gold)) / len(gold) if gold else (1.0 if not predicted else 0.0)
    return {
        "predicted_pages": predicted_pages,
        "gold_pages": gold_pages,
        "f_beta": _f_beta(predicted=predicted, gold=gold, beta=beta),
        "precision": precision,
        "recall": recall,
        "gold_hit": bool(predicted.intersection(gold)),
        "predicted_page_count": len(predicted_pages),
    }


def _aggregate_metrics(case_metrics: list[JsonDict]) -> JsonDict:
    if not case_metrics:
        return {
            "case_count": 0,
            "mean_f_beta": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "gold_hit_rate": 0.0,
            "avg_pages_per_case": 0.0,
        }
    return {
        "case_count": len(case_metrics),
        "mean_f_beta": sum(_as_float(metric["f_beta"]) for metric in case_metrics) / len(case_metrics),
        "mean_precision": sum(_as_float(metric["precision"]) for metric in case_metrics) / len(case_metrics),
        "mean_recall": sum(_as_float(metric["recall"]) for metric in case_metrics) / len(case_metrics),
        "gold_hit_rate": sum(1 for metric in case_metrics if bool(metric["gold_hit"])) / len(case_metrics),
        "avg_pages_per_case": sum(_coerce_int(metric["predicted_page_count"]) or 0 for metric in case_metrics)
        / len(case_metrics),
    }


def run_falsifier(rows: list[JsonDict], *, beta: float = 2.5) -> JsonDict:
    grouped = _rows_by_qid(rows)
    configs = [
        {
            "keep_top_k": keep_top_k,
            "baseline_rank_weight": baseline_weight,
            "requested_page_bonus": requested_bonus,
            "page_one_title_bonus": title_bonus,
            "page_two_bonus": page_two_bonus,
            "title_page_decay_weight": title_decay,
            "article_page_bias": article_bias,
        }
        for keep_top_k, baseline_weight, requested_bonus, title_bonus, page_two_bonus, title_decay, article_bias in itertools.product(
            [1, 2],
            [1.0],
            [0.0, 10.0, 25.0],
            [0.0, 10.0, 25.0],
            [0.0, 10.0, 25.0],
            [0.0, 0.5],
            [0.0, 0.05],
        )
    ]

    baseline_case_metrics: list[JsonDict] = []
    best_payload: JsonDict | None = None

    for qid, case_rows in sorted(grouped.items()):
        gold_pages = [str(row.get("page_id") or "").strip() for row in case_rows if _coerce_bool(row.get("is_gold"))]
        gold_pages = sorted({page_id for page_id in gold_pages if page_id})
        baseline_prediction = _baseline_predictions(case_rows)
        baseline_case_metrics.append(
            {
                "qid": qid,
                **_case_metrics(baseline_prediction, gold_pages, beta=beta),
            }
        )

    baseline_summary = _aggregate_metrics(baseline_case_metrics)

    for config in configs:
        candidate_case_metrics: list[JsonDict] = []
        improved_qids: list[str] = []
        regressed_qids: list[str] = []
        for qid, case_rows in sorted(grouped.items()):
            gold_pages = [str(row.get("page_id") or "").strip() for row in case_rows if _coerce_bool(row.get("is_gold"))]
            gold_pages = sorted({page_id for page_id in gold_pages if page_id})
            baseline_prediction = _baseline_predictions(case_rows)
            candidate_prediction = _select_candidate_pages(case_rows, config=config)
            baseline_metric = _case_metrics(baseline_prediction, gold_pages, beta=beta)
            candidate_metric = _case_metrics(candidate_prediction, gold_pages, beta=beta)
            if _as_float(candidate_metric["f_beta"]) > _as_float(baseline_metric["f_beta"]) + 1e-9:
                improved_qids.append(qid)
            elif _as_float(candidate_metric["f_beta"]) + 1e-9 < _as_float(baseline_metric["f_beta"]):
                regressed_qids.append(qid)
            candidate_case_metrics.append(
                {
                    "qid": qid,
                    "baseline_predicted_pages": baseline_prediction,
                    "candidate_predicted_pages": candidate_prediction,
                    "gold_pages": gold_pages,
                    "baseline_f_beta": _as_float(baseline_metric["f_beta"]),
                    "candidate_f_beta": _as_float(candidate_metric["f_beta"]),
                }
            )

        candidate_summary = _aggregate_metrics(
            [
                {
                    "f_beta": metric["candidate_f_beta"],
                    "precision": _case_metrics(cast("list[str]", metric["candidate_predicted_pages"]), cast("list[str]", metric["gold_pages"]), beta=beta)["precision"],
                    "recall": _case_metrics(cast("list[str]", metric["candidate_predicted_pages"]), cast("list[str]", metric["gold_pages"]), beta=beta)["recall"],
                    "gold_hit": _case_metrics(cast("list[str]", metric["candidate_predicted_pages"]), cast("list[str]", metric["gold_pages"]), beta=beta)["gold_hit"],
                    "predicted_page_count": len(cast("list[str]", metric["candidate_predicted_pages"])),
                }
                for metric in candidate_case_metrics
            ]
        )
        delta_f_beta = _as_float(candidate_summary["mean_f_beta"]) - _as_float(baseline_summary["mean_f_beta"])
        delta_gold_hit_rate = _as_float(candidate_summary["gold_hit_rate"]) - _as_float(baseline_summary["gold_hit_rate"])
        delta_avg_pages = _as_float(candidate_summary["avg_pages_per_case"]) - _as_float(
            baseline_summary["avg_pages_per_case"]
        )
        verdict = (
            "win"
            if delta_f_beta > 0.0 and delta_gold_hit_rate >= 0.0 and delta_avg_pages <= 0.0
            else "no_win"
        )
        payload: JsonDict = {
            "config": config,
            "baseline_summary": baseline_summary,
            "candidate_summary": candidate_summary,
            "delta_f_beta": delta_f_beta,
            "delta_gold_hit_rate": delta_gold_hit_rate,
            "delta_avg_pages_per_case": delta_avg_pages,
            "improved_qids": improved_qids,
            "regressed_qids": regressed_qids,
            "cases": candidate_case_metrics,
            "verdict": verdict,
        }
        if best_payload is None:
            best_payload = payload
            continue
        best_delta = _as_float(best_payload["delta_f_beta"])
        if delta_f_beta > best_delta + 1e-9:
            best_payload = payload
            continue
        if abs(delta_f_beta - best_delta) <= 1e-9 and delta_avg_pages < _as_float(
            best_payload["delta_avg_pages_per_case"]
        ):
            best_payload = payload

    assert best_payload is not None
    return {
        "ticket": 73,
        "created_at": "2026-03-13",
        "policy": (
            "Offline-only falsifier over observed same-doc candidate pages. "
            "Gold-only rows are excluded from selection to avoid leakage."
        ),
        **best_payload,
    }


def _render_markdown(payload: JsonDict) -> str:
    baseline = cast("JsonDict", payload["baseline_summary"])
    candidate = cast("JsonDict", payload["candidate_summary"])
    lines = [
        "# Same-Doc Page Selector Falsifier",
        "",
        f"- verdict: `{payload['verdict']}`",
        f"- delta_f_beta: `{_as_float(payload['delta_f_beta']):.6f}`",
        f"- delta_gold_hit_rate: `{_as_float(payload['delta_gold_hit_rate']):.6f}`",
        f"- delta_avg_pages_per_case: `{_as_float(payload['delta_avg_pages_per_case']):.6f}`",
        f"- improved_qids: `{len(cast('list[str]', payload['improved_qids']))}`",
        f"- regressed_qids: `{len(cast('list[str]', payload['regressed_qids']))}`",
        "",
        "## Baseline",
        "",
        f"- mean_f_beta: `{_as_float(baseline['mean_f_beta']):.6f}`",
        f"- gold_hit_rate: `{_as_float(baseline['gold_hit_rate']):.6f}`",
        f"- avg_pages_per_case: `{_as_float(baseline['avg_pages_per_case']):.6f}`",
        "",
        "## Candidate",
        "",
        f"- mean_f_beta: `{_as_float(candidate['mean_f_beta']):.6f}`",
        f"- gold_hit_rate: `{_as_float(candidate['gold_hit_rate']):.6f}`",
        f"- avg_pages_per_case: `{_as_float(candidate['avg_pages_per_case']):.6f}`",
        "",
        "## Best Config",
        "",
    ]
    for key, value in cast("JsonDict", payload["config"]).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an offline same-doc page-selector falsifier over exported features.")
    parser.add_argument("--features-jsonl", required=True, help="Path to ticket72 feature JSONL")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output Markdown summary path")
    args = parser.parse_args(argv)

    rows = _load_rows(Path(args.features_jsonl))
    payload = run_falsifier(rows)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
