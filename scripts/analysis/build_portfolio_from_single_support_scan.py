from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _load_rows(path: Path) -> list[JsonDict]:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    rows_obj = cast("JsonDict", obj).get("results")
    if not isinstance(rows_obj, list):
        raise ValueError(f"Missing results[] in {path}")
    rows = cast("list[object]", rows_obj)
    return [cast("JsonDict", row) for row in rows if isinstance(row, dict)]


def _sort_key(row: JsonDict) -> tuple[int, float, float, str]:
    recommendation = str(row.get("recommendation") or "").strip().upper()
    rank = {"PROMISING": 2, "EXPERIMENTAL_NO_SUBMIT": 1}.get(recommendation, 0)
    judge_pass_delta = (_coerce_float(row.get("judge_pass_rate_candidate")) or 0.0) - (
        _coerce_float(row.get("judge_pass_rate_baseline")) or 0.0
    )
    judge_grounding_delta = (_coerce_float(row.get("judge_grounding_candidate")) or 0.0) - (
        _coerce_float(row.get("judge_grounding_baseline")) or 0.0
    )
    return (rank, judge_pass_delta, judge_grounding_delta, str(row.get("question_id") or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a support portfolio JSON from a filtered single-support-swap scan.")
    parser.add_argument("--scan-json", type=Path, required=True)
    parser.add_argument("--single-swap-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-recommendation", action="append", default=["PROMISING"])
    parser.add_argument("--require-judge-pass-improvement", action="store_true")
    parser.add_argument("--require-judge-grounding-improvement", action="store_true")
    parser.add_argument("--max-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_recommendations = {
        str(value).strip().upper()
        for value in args.include_recommendation
        if str(value).strip()
    }

    out_rows: list[JsonDict] = []
    for row in sorted(_load_rows(args.scan_json.resolve()), key=_sort_key, reverse=True):
        qid = str(row.get("question_id") or "").strip()
        recommendation = str(row.get("recommendation") or "").strip().upper()
        if not qid or recommendation not in allowed_recommendations:
            continue

        judge_pass_delta = (_coerce_float(row.get("judge_pass_rate_candidate")) or 0.0) - (
            _coerce_float(row.get("judge_pass_rate_baseline")) or 0.0
        )
        judge_grounding_delta = (_coerce_float(row.get("judge_grounding_candidate")) or 0.0) - (
            _coerce_float(row.get("judge_grounding_baseline")) or 0.0
        )
        if args.require_judge_pass_improvement and judge_pass_delta <= 0.0:
            continue
        if args.require_judge_grounding_improvement and judge_grounding_delta <= 0.0:
            continue

        submission = args.single_swap_dir.resolve() / f"submission_single_swap_{qid}.json"
        raw_results = args.single_swap_dir.resolve() / f"raw_results_single_swap_{qid}.json"
        preflight = args.single_swap_dir.resolve() / f"preflight_summary_single_swap_{qid}.json"
        if not submission.exists() or not raw_results.exists() or not preflight.exists():
            continue

        out_rows.append(
            {
                "qid": qid,
                "label": f"{qid[:6]}_single_swap",
                "submission_path": str(submission),
                "raw_results_path": str(raw_results),
                "preflight_path": str(preflight),
                "notes": "; ".join(
                    [
                        f"recommendation={recommendation.lower()}",
                        f"judge_pass_delta={judge_pass_delta:+.4f}",
                        f"judge_grounding_delta={judge_grounding_delta:+.4f}",
                        f"page_drift={int(row.get('retrieval_page_projection_changed_count') or 0)}",
                    ]
                ),
            }
        )
        max_items = int(args.max_items)
        if max_items > 0 and len(out_rows) >= max_items:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
