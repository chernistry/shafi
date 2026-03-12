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
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _load_base_portfolio(path: Path | None) -> list[JsonDict]:
    if path is None:
        return []
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    rows_obj = cast("list[object]", obj)
    rows = [cast("JsonDict", row) for row in rows_obj if isinstance(row, dict)]
    return rows


def _load_audit_records(path: Path) -> list[JsonDict]:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    obj_dict = cast("JsonDict", obj)
    records_obj = obj_dict.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Missing records[] in {path}")
    rows = cast("list[object]", records_obj)
    return [cast("JsonDict", row) for row in rows if isinstance(row, dict)]


def _summary_judge(path: Path) -> tuple[float | None, float | None]:
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return None, None
    obj_dict = cast("JsonDict", obj)
    summary_obj = obj_dict.get("summary")
    if not isinstance(summary_obj, dict):
        return None, None
    summary = cast("JsonDict", summary_obj)
    judge_obj = summary.get("judge")
    if not isinstance(judge_obj, dict):
        return None, None
    judge = cast("JsonDict", judge_obj)
    return (
        _coerce_float(judge.get("pass_rate")),
        _coerce_float(judge.get("avg_grounding")),
    )


def _single_swap_artifacts(single_swap_dir: Path, qid: str) -> tuple[Path, Path, Path] | None:
    submission = single_swap_dir / f"submission_single_swap_{qid}.json"
    raw_results = single_swap_dir / f"raw_results_single_swap_{qid}.json"
    preflight = single_swap_dir / f"preflight_summary_single_swap_{qid}.json"
    if submission.exists() and raw_results.exists() and preflight.exists():
        return submission, raw_results, preflight
    return None


def _eval_path(single_swap_dir: Path, qid: str) -> Path | None:
    path = single_swap_dir / f"eval_candidate_debug_single_swap_{qid}.json"
    return path if path.exists() else None


def _record_sort_key(record: JsonDict) -> tuple[int, float, str]:
    recommendation = str(record.get("recommendation") or "").strip().upper()
    rank = {"PROMISING": 2, "WATCH": 1}.get(recommendation, 0)
    return (
        rank,
        float(record.get("opportunity_score") or 0.0),
        str(record.get("question_id") or ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a support-portfolio JSON from comparison/title-page audit records and single-swap artifacts.")
    parser.add_argument("--comparison-audit-json", type=Path, required=True)
    parser.add_argument("--single-swap-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-portfolio-json", type=Path, default=None)
    parser.add_argument("--baseline-eval-json", type=Path, default=None)
    parser.add_argument("--include-recommendation", action="append", default=["PROMISING"])
    parser.add_argument("--max-new-items", type=int, default=0)
    parser.add_argument("--require-judge-non-inferior", action="store_true")
    parser.add_argument("--require-judge-pass-improvement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = _load_base_portfolio(args.base_portfolio_json.resolve() if args.base_portfolio_json else None)
    baseline_pass_rate: float | None = None
    baseline_grounding: float | None = None
    if args.baseline_eval_json is not None:
        baseline_pass_rate, baseline_grounding = _summary_judge(args.baseline_eval_json.resolve())

    allowed_recommendations = {
        str(value).strip().upper()
        for value in args.include_recommendation
        if str(value).strip()
    }
    existing_qids = {str(row.get("qid") or "").strip() for row in base_rows if str(row.get("qid") or "").strip()}

    added_rows: list[JsonDict] = []
    audit_records = sorted(_load_audit_records(args.comparison_audit_json.resolve()), key=_record_sort_key, reverse=True)
    for record in audit_records:
        qid = str(record.get("question_id") or "").strip()
        if not qid or qid in existing_qids:
            continue
        recommendation = str(record.get("recommendation") or "").strip().upper()
        if allowed_recommendations and recommendation not in allowed_recommendations:
            continue
        artifacts = _single_swap_artifacts(args.single_swap_dir.resolve(), qid)
        if artifacts is None:
            continue

        candidate_pass_rate: float | None = None
        candidate_grounding: float | None = None
        eval_path = _eval_path(args.single_swap_dir.resolve(), qid)
        if eval_path is not None:
            candidate_pass_rate, candidate_grounding = _summary_judge(eval_path)
        if (
            args.require_judge_non_inferior
            and baseline_pass_rate is not None
            and candidate_pass_rate is not None
            and candidate_pass_rate < baseline_pass_rate
        ):
            continue
        if (
            args.require_judge_non_inferior
            and baseline_grounding is not None
            and candidate_grounding is not None
            and candidate_grounding < baseline_grounding
        ):
            continue
        if (
            args.require_judge_pass_improvement
            and baseline_pass_rate is not None
            and candidate_pass_rate is not None
            and candidate_pass_rate <= baseline_pass_rate
        ):
            continue

        submission, raw_results, preflight = artifacts
        notes_parts = [
            f"compare_kind={str(record.get('compare_kind') or '').strip() or 'unknown'}",
            f"recommendation={recommendation.lower()}",
        ]
        missing_count = int(record.get("minimal_required_page1_count") or 0) - int(record.get("baseline_used_page1_doc_hits") or 0)
        if missing_count > 0:
            notes_parts.append(f"missing_used_page1_docs={missing_count}")
        if candidate_pass_rate is not None and baseline_pass_rate is not None:
            notes_parts.append(
                f"judge_pass_delta={candidate_pass_rate - baseline_pass_rate:+.4f}"
            )
        if candidate_grounding is not None and baseline_grounding is not None:
            notes_parts.append(
                f"judge_grounding_delta={candidate_grounding - baseline_grounding:+.4f}"
            )
        added_rows.append(
            {
                "qid": qid,
                "label": f"{qid[:6]}_{str(record.get('compare_kind') or 'compare').strip()}_single_swap",
                "submission_path": str(submission),
                "raw_results_path": str(raw_results),
                "preflight_path": str(preflight),
                "notes": "; ".join(notes_parts),
            }
        )
        existing_qids.add(qid)
        max_new = int(args.max_new_items)
        if max_new > 0 and len(added_rows) >= max_new:
            break

    output = [*base_rows, *added_rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
