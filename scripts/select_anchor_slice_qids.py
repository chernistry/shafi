from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_payload(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _rows(payload: JsonDict) -> list[JsonDict]:
    rows_obj = payload.get("rows")
    if not isinstance(rows_obj, list):
        raise ValueError("Anchor-slice JSON is missing 'rows'")
    rows_list = cast("list[object]", rows_obj)
    return [cast("JsonDict", row) for row in rows_list if isinstance(row, dict)]


def load_anchor_slice_rows(path: Path) -> list[JsonDict]:
    return _rows(_load_payload(path))


def _as_bool(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else False


def select_qids(
    *,
    rows: list[JsonDict],
    include_statuses: set[str],
    exclude_statuses: set[str],
    require_no_answer_change: bool,
    require_used_support: bool,
    excluded_qids: set[str],
) -> tuple[list[str], JsonDict]:
    selected: list[str] = []
    reasons: dict[str, str] = {}

    for row in rows:
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            continue

        status = str(row.get("status") or "").strip()
        answer_changed = _as_bool(row.get("answer_changed"))
        candidate_used_hit = _as_bool(row.get("candidate_used_hit"))
        candidate_used_equivalent_hit = _as_bool(row.get("candidate_used_equivalent_hit"))

        if qid in excluded_qids:
            reasons[qid] = "excluded_qid"
            continue
        if include_statuses and status not in include_statuses:
            reasons[qid] = f"status_not_included:{status or '(blank)'}"
            continue
        if exclude_statuses and status in exclude_statuses:
            reasons[qid] = f"status_excluded:{status}"
            continue
        if require_no_answer_change and answer_changed:
            reasons[qid] = "answer_changed"
            continue
        if require_used_support and not (candidate_used_hit or candidate_used_equivalent_hit):
            reasons[qid] = "missing_used_support_hit"
            continue

        selected.append(qid)

    report: JsonDict = {
        "selected_qids": selected,
        "selected_count": len(selected),
        "rejection_reasons_by_qid": reasons,
        "selection_policy": {
            "include_statuses": sorted(include_statuses),
            "exclude_statuses": sorted(exclude_statuses),
            "require_no_answer_change": require_no_answer_change,
            "require_used_support": require_used_support,
            "excluded_qids": sorted(excluded_qids),
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        },
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select QIDs from an anchor-slice JSON report for bounded counterfactual experiments.")
    parser.add_argument("--anchor-slice-json", type=Path, required=True)
    parser.add_argument("--include-status", action="append", default=[])
    parser.add_argument("--exclude-status", action="append", default=[])
    parser.add_argument("--exclude-qid", action="append", default=[])
    parser.add_argument("--require-no-answer-change", action="store_true")
    parser.add_argument("--require-used-support", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    payload = _load_payload(args.anchor_slice_json)
    rows = _rows(payload)
    selected, report = select_qids(
        rows=rows,
        include_statuses={str(status).strip() for status in args.include_status if str(status).strip()},
        exclude_statuses={str(status).strip() for status in args.exclude_status if str(status).strip()},
        require_no_answer_change=bool(args.require_no_answer_change),
        require_used_support=bool(args.require_used_support),
        excluded_qids={str(qid).strip() for qid in args.exclude_qid if str(qid).strip()},
    )

    args.out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
