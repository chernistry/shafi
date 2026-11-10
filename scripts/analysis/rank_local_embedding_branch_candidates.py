from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _build_rows(*, relevance_payload: JsonDict, latency_payload: JsonDict) -> list[JsonDict]:
    relevance_rows_obj = relevance_payload.get("summaries")
    latency_rows_obj = latency_payload.get("results")
    if not isinstance(relevance_rows_obj, list):
        raise ValueError("Relevance payload missing summaries[]")
    if not isinstance(latency_rows_obj, list):
        raise ValueError("Latency payload missing results[]")

    relevance_rows = {
        str(cast("JsonDict", row).get("model") or "").strip(): cast("JsonDict", row)
        for row in cast("list[object]", relevance_rows_obj)
        if isinstance(row, dict)
    }
    latency_rows = {
        str(cast("JsonDict", row).get("model") or "").strip(): cast("JsonDict", row)
        for row in cast("list[object]", latency_rows_obj)
        if isinstance(row, dict)
    }

    models = sorted(set(relevance_rows) & set(latency_rows))
    out: list[JsonDict] = []
    for model in models:
        relevance = relevance_rows[model]
        latency = latency_rows[model]
        out.append(
            {
                "model": model,
                "evaluated_cases": _coerce_int(relevance.get("evaluated_cases")),
                "skipped_cases": _coerce_int(relevance.get("skipped_cases")),
                "gold_top1_rate": _coerce_float(relevance.get("gold_top1_rate")),
                "gold_top3_rate": _coerce_float(relevance.get("gold_top3_rate")),
                "mean_gold_margin": _coerce_float(relevance.get("mean_gold_margin")),
                "mean_best_gold_rank": _coerce_float(relevance.get("mean_best_gold_rank")),
                "latency_ms_p50": _coerce_float(latency.get("latency_ms_p50")),
                "latency_ms_p95": _coerce_float(latency.get("latency_ms_p95")),
                "docs_per_second_mean": _coerce_float(latency.get("docs_per_second_mean")),
            }
        )
    return out


def _score(row: JsonDict) -> tuple[float, float, float, float, float, str]:
    return (
        _coerce_float(row.get("gold_top1_rate")),
        _coerce_float(row.get("gold_top3_rate")),
        _coerce_float(row.get("mean_gold_margin")),
        -_coerce_float(row.get("latency_ms_p50")),
        _coerce_float(row.get("docs_per_second_mean")),
        str(row.get("model") or ""),
    )


def _render_md(*, rows: list[JsonDict], relevance_path: Path, latency_path: Path) -> str:
    ranked = sorted(rows, key=_score, reverse=True)
    lines = [
        "# Local Embedding Branch Shortlist",
        "",
        f"- relevance_source: `{relevance_path}`",
        f"- latency_source: `{latency_path}`",
        f"- compared_models: `{len(ranked)}`",
        f"- recommended_model: `{ranked[0]['model']}`" if ranked else "- recommended_model: `n/a`",
        "- branch_policy: `OFFLINE_ONLY_NO_MAINLINE_PROMOTION_WITHOUT_CLEAR_WIN`",
        "",
        "| Rank | Model | Cases | Skipped | Top1 | Top3 | Mean margin | Mean rank | p50 ms | p95 ms | Docs/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | `{row['model']}` | {_coerce_int(row.get('evaluated_cases'))} | {_coerce_int(row.get('skipped_cases'))} | "
            f"{_coerce_float(row.get('gold_top1_rate')):.3f} | {_coerce_float(row.get('gold_top3_rate')):.3f} | "
            f"{_coerce_float(row.get('mean_gold_margin')):.4f} | {_coerce_float(row.get('mean_best_gold_rank')):.2f} | "
            f"{_coerce_float(row.get('latency_ms_p50')):.2f} | {_coerce_float(row.get('latency_ms_p95')):.2f} | {_coerce_float(row.get('docs_per_second_mean')):.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank local embedding branch candidates from relevance and latency probes.")
    parser.add_argument("--relevance-json", type=Path, required=True)
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    relevance = _load_json(args.relevance_json)
    latency = _load_json(args.latency_json)
    rows = _build_rows(relevance_payload=relevance, latency_payload=latency)
    ranked = sorted(rows, key=_score, reverse=True)
    payload = {
        "recommended_model": ranked[0]["model"] if ranked else None,
        "ranked_models": ranked,
        "branch_policy": "OFFLINE_ONLY_NO_MAINLINE_PROMOTION_WITHOUT_CLEAR_WIN",
    }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(
        _render_md(rows=rows, relevance_path=args.relevance_json, latency_path=args.latency_json),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
