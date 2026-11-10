from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

JsonDict = dict[str, Any]


def load_anchor_slice_rows(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_obj: list[object]
    if isinstance(payload, list):
        rows_obj = cast("list[object]", payload)
    elif isinstance(payload, dict):
        payload_dict = cast("dict[str, object]", payload)
        rows_value = payload_dict.get("rows", [])
        if not isinstance(rows_value, list):
            raise ValueError(f"Expected 'rows' array in {path}")
        rows_obj = cast("list[object]", rows_value)
    else:
        raise ValueError(f"Expected JSON object or array in {path}")
    return [cast("JsonDict", row) for row in rows_obj if isinstance(row, dict)]


def select_qids(
    *,
    rows: list[JsonDict],
    include_statuses: set[str] | None = None,
    exclude_statuses: set[str] | None = None,
    require_no_answer_change: bool = False,
    require_used_support: bool = False,
    excluded_qids: set[str] | None = None,
) -> tuple[list[str], JsonDict]:
    include = {status.strip() for status in (include_statuses or set()) if status.strip()}
    exclude = {status.strip() for status in (exclude_statuses or set()) if status.strip()}
    excluded = {qid.strip() for qid in (excluded_qids or set()) if qid.strip()}

    selected_qids: list[str] = []
    seen_qids: set[str] = set()
    counters = {
        "total_rows": len(rows),
        "included_rows": 0,
        "skipped_missing_qid": 0,
        "skipped_excluded_qid": 0,
        "skipped_include_status": 0,
        "skipped_exclude_status": 0,
        "skipped_answer_changed": 0,
        "skipped_missing_used_support": 0,
        "duplicate_qids": 0,
    }

    for row in rows:
        qid = str(row.get("question_id") or row.get("qid") or "").strip()
        if not qid:
            counters["skipped_missing_qid"] += 1
            continue
        if qid in excluded:
            counters["skipped_excluded_qid"] += 1
            continue

        status = str(row.get("status") or "").strip()
        if include and status not in include:
            counters["skipped_include_status"] += 1
            continue
        if exclude and status in exclude:
            counters["skipped_exclude_status"] += 1
            continue
        if require_no_answer_change and bool(row.get("answer_changed")):
            counters["skipped_answer_changed"] += 1
            continue
        if require_used_support and not (
            bool(row.get("candidate_used_hit")) or bool(row.get("candidate_used_equivalent_hit"))
        ):
            counters["skipped_missing_used_support"] += 1
            continue

        counters["included_rows"] += 1
        if qid in seen_qids:
            counters["duplicate_qids"] += 1
            continue
        seen_qids.add(qid)
        selected_qids.append(qid)

    report: JsonDict = {
        **counters,
        "selected_qids": list(selected_qids),
        "selected_count": len(selected_qids),
        "include_statuses": sorted(include),
        "exclude_statuses": sorted(exclude),
        "excluded_qids": sorted(excluded),
        "require_no_answer_change": require_no_answer_change,
        "require_used_support": require_used_support,
    }
    return selected_qids, report
