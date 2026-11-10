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


def _load_results(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    results_obj = cast("JsonDict", obj).get("results")
    if not isinstance(results_obj, list):
        raise ValueError(f"Expected results[] in {path}")
    rows = cast("list[object]", results_obj)
    return [cast("JsonDict", row) for row in rows if isinstance(row, dict)]


def _hidden_trusted_delta(row: JsonDict) -> float:
    return _coerce_float(row.get("benchmark_trusted_candidate")) - _coerce_float(row.get("benchmark_trusted_baseline"))


def _hidden_all_delta(row: JsonDict) -> float:
    return _coerce_float(row.get("benchmark_all_candidate")) - _coerce_float(row.get("benchmark_all_baseline"))


def _judge_pass_delta(row: JsonDict) -> float:
    candidate = row.get("judge_pass_rate_candidate")
    baseline = row.get("judge_pass_rate_baseline")
    if candidate is None or baseline is None:
        return 0.0
    return _coerce_float(candidate) - _coerce_float(baseline)


def _judge_grounding_delta(row: JsonDict) -> float:
    candidate = row.get("judge_grounding_candidate")
    baseline = row.get("judge_grounding_baseline")
    if candidate is None or baseline is None:
        return 0.0
    return _coerce_float(candidate) - _coerce_float(baseline)


def _recommendation_rank(row: JsonDict) -> int:
    value = str(row.get("recommendation") or "").strip().upper()
    return {"PROMISING": 2, "EXPERIMENTAL_NO_SUBMIT": 1}.get(value, 0)


def _score_tuple(row: JsonDict) -> tuple[float, float, float, float, int, int]:
    qids = cast("list[str]", row.get("qids") or [])
    changed_count = len(qids)
    return (
        _hidden_trusted_delta(row) * 1000.0
        + _hidden_all_delta(row) * 300.0
        + _judge_pass_delta(row) * 10.0
        + _judge_grounding_delta(row)
        + changed_count,
        _hidden_trusted_delta(row),
        _hidden_all_delta(row),
        _judge_pass_delta(row),
        -_coerce_int(row.get("retrieval_page_projection_changed_count")),
        -_coerce_int(row.get("answer_changed_count")),
    )


def _budget_filter(row: JsonDict, *, max_answer_drift: int, max_page_drift: int, max_page_p95: int) -> bool:
    return (
        _coerce_int(row.get("answer_changed_count")) <= max_answer_drift
        and _coerce_int(row.get("retrieval_page_projection_changed_count")) <= max_page_drift
        and _coerce_int(row.get("candidate_page_p95")) <= max_page_p95
    )


def _best_row(rows: list[JsonDict]) -> JsonDict | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            _recommendation_rank(row),
            *_score_tuple(row),
            tuple(cast("list[str]", row.get("qids") or [])),
        ),
    )


def _row_summary(row: JsonDict | None) -> JsonDict | None:
    if row is None:
        return None
    return {
        "qids": cast("list[str]", row.get("qids") or []),
        "recommendation": str(row.get("recommendation") or ""),
        "answer_drift": _coerce_int(row.get("answer_changed_count")),
        "page_drift": _coerce_int(row.get("retrieval_page_projection_changed_count")),
        "page_p95": _coerce_int(row.get("candidate_page_p95")),
        "hidden_g_trusted_delta": round(_hidden_trusted_delta(row), 4),
        "hidden_g_all_delta": round(_hidden_all_delta(row), 4),
        "judge_pass_delta": round(_judge_pass_delta(row), 4),
        "judge_grounding_delta": round(_judge_grounding_delta(row), 4),
    }


def _render_markdown(*, source_json: Path, filtered_rows: list[JsonDict], item_reports: list[JsonDict]) -> str:
    lines = [
        "# Portfolio Marginal Contribution",
        "",
        f"- source: `{source_json}`",
        f"- filtered_candidates: `{len(filtered_rows)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| QID | Best With Item | Best Without Item | Trusted Δ Gain | All Δ Gain | Judge Pass Gain | Judge Grounding Gain | Plateau |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in item_reports:
        lines.append(
            "| "
            f"`{row['qid']}` | "
            f"`{','.join(cast('list[str]', row['best_with_item_qids']))}` | "
            f"`{','.join(cast('list[str]', row['best_without_item_qids']))}` | "
            f"{row['trusted_delta_gain']:.4f} | "
            f"{row['all_delta_gain']:.4f} | "
            f"{row['judge_pass_gain']:.4f} | "
            f"{row['judge_grounding_gain']:.4f} | "
            f"`{'yes' if row['is_plateau_item'] else 'no'}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether individual portfolio items expand or merely match the current bounded frontier.")
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-answer-drift", type=int, default=0)
    parser.add_argument("--max-page-drift", type=int, default=6)
    parser.add_argument("--max-page-p95", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_results(args.source_json.resolve())
    filtered_rows = [
        row
        for row in rows
        if _budget_filter(
            row,
            max_answer_drift=int(args.max_answer_drift),
            max_page_drift=int(args.max_page_drift),
            max_page_p95=int(args.max_page_p95),
        )
    ]
    qids = sorted({qid for row in filtered_rows for qid in cast("list[str]", row.get("qids") or [])})
    item_reports: list[JsonDict] = []
    for qid in qids:
        with_item = [row for row in filtered_rows if qid in cast("list[str]", row.get("qids") or [])]
        without_item = [row for row in filtered_rows if qid not in cast("list[str]", row.get("qids") or [])]
        best_with_item = _best_row(with_item)
        best_without_item = _best_row(without_item)
        trusted_gain = (
            _hidden_trusted_delta(best_with_item) - _hidden_trusted_delta(best_without_item)
            if best_with_item is not None and best_without_item is not None
            else 0.0
        )
        all_gain = (
            _hidden_all_delta(best_with_item) - _hidden_all_delta(best_without_item)
            if best_with_item is not None and best_without_item is not None
            else 0.0
        )
        judge_pass_gain = (
            _judge_pass_delta(best_with_item) - _judge_pass_delta(best_without_item)
            if best_with_item is not None and best_without_item is not None
            else 0.0
        )
        judge_grounding_gain = (
            _judge_grounding_delta(best_with_item) - _judge_grounding_delta(best_without_item)
            if best_with_item is not None and best_without_item is not None
            else 0.0
        )
        item_reports.append(
            {
                "qid": qid,
                "best_with_item_qids": cast("list[str]", best_with_item.get("qids") or []) if best_with_item is not None else [],
                "best_without_item_qids": cast("list[str]", best_without_item.get("qids") or []) if best_without_item is not None else [],
                "trusted_delta_gain": round(trusted_gain, 4),
                "all_delta_gain": round(all_gain, 4),
                "judge_pass_gain": round(judge_pass_gain, 4),
                "judge_grounding_gain": round(judge_grounding_gain, 4),
                "is_plateau_item": abs(trusted_gain) < 1e-9 and abs(all_gain) < 1e-9 and abs(judge_pass_gain) < 1e-9 and abs(judge_grounding_gain) < 1e-9,
                "best_with_item": _row_summary(best_with_item),
                "best_without_item": _row_summary(best_without_item),
            }
        )

    payload = {
        "source_json": str(args.source_json.resolve()),
        "filtered_candidate_count": len(filtered_rows),
        "max_answer_drift": int(args.max_answer_drift),
        "max_page_drift": int(args.max_page_drift),
        "max_page_p95": int(args.max_page_p95),
        "item_reports": item_reports,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(
        _render_markdown(
            source_json=args.source_json.resolve(),
            filtered_rows=filtered_rows,
            item_reports=item_reports,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
