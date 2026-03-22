from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


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


def _recommendation_rank(value: str) -> int:
    return {"PROMISING": 2, "EXPERIMENTAL_NO_SUBMIT": 1}.get(value.strip().upper(), 0)


def _load_results(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    results_obj = cast("JsonDict", obj).get("results")
    if not isinstance(results_obj, list):
        raise ValueError(f"Expected results[] in {path}")
    rows = cast("list[object]", results_obj)
    return [cast("JsonDict", row) for row in rows if isinstance(row, dict)]


def _judge_pass_delta(row: JsonDict) -> float:
    return _coerce_float(row.get("judge_pass_rate_candidate")) - _coerce_float(row.get("judge_pass_rate_baseline"))


def _judge_grounding_delta(row: JsonDict) -> float:
    return _coerce_float(row.get("judge_grounding_candidate")) - _coerce_float(row.get("judge_grounding_baseline"))


def _hidden_all_delta(row: JsonDict) -> float:
    return _coerce_float(row.get("benchmark_all_candidate")) - _coerce_float(row.get("benchmark_all_baseline"))


def _hidden_trusted_delta(row: JsonDict) -> float:
    return _coerce_float(row.get("benchmark_trusted_candidate")) - _coerce_float(row.get("benchmark_trusted_baseline"))


def _page_drift(row: JsonDict) -> int:
    return _coerce_int(row.get("retrieval_page_projection_changed_count"))


def _answer_drift(row: JsonDict) -> int:
    return _coerce_int(row.get("answer_changed_count"))


def _page_p95(row: JsonDict) -> int:
    return _coerce_int(row.get("candidate_page_p95"))


def _seed_improved_count(row: JsonDict) -> int:
    value = row.get("improved_seed_cases")
    if not isinstance(value, list):
        return 0
    return len(cast("list[object]", value))


def _blindspot_improved_count(row: JsonDict) -> int:
    value = row.get("blindspot_improved_cases")
    if not isinstance(value, list):
        return 0
    return len(cast("list[object]", value))


def _undercoverage_count(row: JsonDict) -> int:
    value = row.get("blindspot_support_undercoverage_cases")
    if not isinstance(value, list):
        return 0
    return len(cast("list[object]", value))


def _portfolio_role(row: JsonDict) -> str:
    trusted_delta = _hidden_trusted_delta(row)
    all_delta = _hidden_all_delta(row)
    if trusted_delta > 0.0 or all_delta > 0.0:
        return "promotion_candidate"
    return "defensive_only"


def _budget_filter(
    row: JsonDict,
    *,
    max_answer_drift: int,
    max_page_drift: int,
    max_page_p95: int,
) -> bool:
    return (
        _answer_drift(row) <= max_answer_drift
        and _page_drift(row) <= max_page_drift
        and _page_p95(row) <= max_page_p95
    )


def _combined_score(row: JsonDict) -> tuple[float, float, float, float, int, int, int, int]:
    seed_improved_count = _seed_improved_count(row)
    blindspot_improved_count = _blindspot_improved_count(row)
    undercoverage_count = _undercoverage_count(row)
    return (
        _hidden_trusted_delta(row) * 1000.0
        + _hidden_all_delta(row) * 300.0
        + _judge_pass_delta(row) * 10.0
        + _judge_grounding_delta(row)
        + seed_improved_count * 5.0
        + blindspot_improved_count * 2.0
        - undercoverage_count * 6.0,
        _hidden_trusted_delta(row),
        _hidden_all_delta(row),
        float(seed_improved_count),
        -undercoverage_count,
        blindspot_improved_count,
        -_page_drift(row),
        -_answer_drift(row),
    )


def _top_rows(rows: list[JsonDict], *, limit: int) -> list[JsonDict]:
    ranked = sorted(
        rows,
        key=lambda row: (
            _recommendation_rank(str(row.get("recommendation") or "")),
            *_combined_score(row),
            tuple(cast("list[str]", row.get("qids") or [])),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _render_markdown(*, source_path: Path, rows: list[JsonDict], limit: int) -> str:
    lines = [
        "# Candidate Portfolio Ranking",
        "",
        f"- source: `{source_path}`",
        f"- candidates: `{len(rows)}`",
        "",
        "| Rank | QIDs | Recommendation | Hidden-G Trusted Δ | Hidden-G All Δ | Judge Pass Δ | Judge Grounding Δ | Answer Drift | Page Drift | Page p95 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(_top_rows(rows, limit=limit), start=1):
        qids = cast("list[str]", row.get("qids") or [])
        lines.append(
            "| "
            f"{index} | `{','.join(qids)}` | `{row.get('recommendation') or ''}` | "
            f"{_hidden_trusted_delta(row):.4f} | {_hidden_all_delta(row):.4f} | "
            f"{_judge_pass_delta(row):.4f} | {_judge_grounding_delta(row):.4f} | "
            f"{_answer_drift(row)} | {_page_drift(row)} | {_page_p95(row)} |"
        )
        lines.append(
            f"  role=`{_portfolio_role(row)}` seed_improved=`{_seed_improved_count(row)}` "
            f"blindspot_improved=`{_blindspot_improved_count(row)}` "
            f"undercoverage_blindspots=`{_undercoverage_count(row)}`"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidate portfolio search results under a bounded drift budget.")
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-answer-drift", type=int, default=0)
    parser.add_argument("--max-page-drift", type=int, default=6)
    parser.add_argument("--max-page-p95", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_results(args.source_json.resolve())
    filtered = [
        row
        for row in rows
        if _budget_filter(
            row,
            max_answer_drift=int(args.max_answer_drift),
            max_page_drift=int(args.max_page_drift),
            max_page_p95=int(args.max_page_p95),
        )
    ]
    ranked = [
        {
            **row,
            "portfolio_role": _portfolio_role(row),
            "seed_improved_count": _seed_improved_count(row),
            "blindspot_improved_count": _blindspot_improved_count(row),
            "undercoverage_count": _undercoverage_count(row),
            "serious_candidate_eligible": (
                (_hidden_trusted_delta(row) > 0.0 or _hidden_all_delta(row) > 0.0)
                and _undercoverage_count(row) == 0
            ),
        }
        for row in _top_rows(filtered, limit=int(args.top_k))
    ]
    payload = {
        "source_json": str(args.source_json.resolve()),
        "candidate_count": len(rows),
        "filtered_count": len(filtered),
        "max_answer_drift": int(args.max_answer_drift),
        "max_page_drift": int(args.max_page_drift),
        "max_page_p95": int(args.max_page_p95),
        "ranked_candidates": ranked,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(source_path=args.source_json.resolve(), rows=filtered, limit=int(args.top_k)), encoding="utf-8")


if __name__ == "__main__":
    main()
