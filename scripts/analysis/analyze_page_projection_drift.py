from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class DriftRecord:
    qid: str
    doc_family: str
    route: str
    failure_stage: str
    false_positive_page_count: int
    mixed_doc_page_count: int
    orphan_page_count: int
    low_confidence_page_count: int
    dominant_noise_class: str
    page_budget_overrun: bool
    wrong_document_risk: bool


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _doc_id(page_id: str) -> str:
    text = str(page_id).strip()
    if "_" not in text:
        return text
    return text.rsplit("_", 1)[0]


def _pages(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in cast("list[object]", value) if str(item).strip()]


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    weight = rank - lower
    return (1.0 - weight) * lower_value + weight * upper_value


def _classify_record(record: JsonDict) -> DriftRecord:
    gold_pages = set(_pages(record.get("gold_pages")))
    used_pages = _pages(record.get("used_pages"))
    false_positive_pages = _pages(record.get("false_positive_pages"))

    gold_doc_ids = {_doc_id(page_id) for page_id in gold_pages}
    true_positive_doc_ids = {_doc_id(page_id) for page_id in used_pages if page_id in gold_pages}

    mixed_doc_pages = [page_id for page_id in false_positive_pages if _doc_id(page_id) not in gold_doc_ids]
    orphan_pages = [
        page_id
        for page_id in false_positive_pages
        if _doc_id(page_id) in gold_doc_ids and _doc_id(page_id) not in true_positive_doc_ids
    ]
    low_confidence_pages = [
        page_id
        for page_id in false_positive_pages
        if page_id not in mixed_doc_pages and page_id not in orphan_pages
    ]

    counts = {
        "mixed_doc": len(mixed_doc_pages),
        "orphan": len(orphan_pages),
        "low_confidence": len(low_confidence_pages),
    }
    dominant_noise_class = max(counts.items(), key=lambda item: (item[1], item[0]))[0] if any(counts.values()) else "none"

    return DriftRecord(
        qid=str(record.get("qid") or "").strip(),
        doc_family=str(record.get("doc_family") or "unknown").strip() or "unknown",
        route=str(record.get("route") or "unknown").strip() or "unknown",
        failure_stage=str(record.get("failure_stage") or "unknown").strip() or "unknown",
        false_positive_page_count=len(false_positive_pages),
        mixed_doc_page_count=len(mixed_doc_pages),
        orphan_page_count=len(orphan_pages),
        low_confidence_page_count=len(low_confidence_pages),
        dominant_noise_class=dominant_noise_class,
        page_budget_overrun=bool(record.get("page_budget_overrun")),
        wrong_document_risk=bool(record.get("wrong_document_risk")),
    )


def build_report(*, page_trace_ledger: JsonDict) -> JsonDict:
    raw_records = page_trace_ledger.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Page-trace ledger is missing records[]")
    ledger_records = cast("list[object]", raw_records)
    records: list[DriftRecord] = []
    for item in ledger_records:
        if isinstance(item, dict):
            records.append(_classify_record(cast("JsonDict", item)))

    affected = [record for record in records if record.false_positive_page_count > 0]
    family_rollup: dict[str, Counter[str]] = defaultdict(Counter)
    route_rollup: dict[str, Counter[str]] = defaultdict(Counter)
    for record in affected:
        family_rollup[record.doc_family]["cases"] += 1
        family_rollup[record.doc_family]["false_positive_pages"] += record.false_positive_page_count
        family_rollup[record.doc_family]["mixed_doc_pages"] += record.mixed_doc_page_count
        family_rollup[record.doc_family]["orphan_pages"] += record.orphan_page_count
        family_rollup[record.doc_family]["low_confidence_pages"] += record.low_confidence_page_count
        family_rollup[record.doc_family]["page_budget_overrun_cases"] += int(record.page_budget_overrun)

        route_rollup[record.route]["cases"] += 1
        route_rollup[record.route]["false_positive_pages"] += record.false_positive_page_count

    false_positive_counts = [record.false_positive_page_count for record in affected]
    noise_counts = Counter(
        {
            "mixed_doc": sum(record.mixed_doc_page_count for record in affected),
            "orphan": sum(record.orphan_page_count for record in affected),
            "low_confidence": sum(record.low_confidence_page_count for record in affected),
        }
    )
    focus_order = [label for label, count in noise_counts.most_common() if count > 0]
    recommended_max_page_drift = 0
    if false_positive_counts:
        recommended_max_page_drift = min(4, max(2, math.ceil(_percentile(false_positive_counts, 0.75))))

    high_false_positive = sorted(
        affected,
        key=lambda record: (
            -record.false_positive_page_count,
            -record.mixed_doc_page_count,
            -record.orphan_page_count,
            record.qid,
        ),
    )

    return {
        "source_page_trace_ledger": page_trace_ledger.get("source_raw_results"),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "summary": {
            "cases_scored": len(records),
            "false_positive_case_count": len(affected),
            "mixed_doc_case_count": sum(1 for record in affected if record.mixed_doc_page_count > 0),
            "orphan_case_count": sum(1 for record in affected if record.orphan_page_count > 0),
            "low_confidence_case_count": sum(1 for record in affected if record.low_confidence_page_count > 0),
            "wrong_document_risk_case_count": sum(1 for record in affected if record.wrong_document_risk),
            "page_budget_overrun_case_count": sum(1 for record in affected if record.page_budget_overrun),
            "false_positive_page_count": sum(record.false_positive_page_count for record in affected),
            "mixed_doc_page_count": noise_counts["mixed_doc"],
            "orphan_page_count": noise_counts["orphan"],
            "low_confidence_page_count": noise_counts["low_confidence"],
            "p50_false_positive_pages": _percentile(false_positive_counts, 0.50),
            "p75_false_positive_pages": _percentile(false_positive_counts, 0.75),
            "p90_false_positive_pages": _percentile(false_positive_counts, 0.90),
            "recommended_max_page_drift": recommended_max_page_drift,
            "focus_order": focus_order,
        },
        "family_rollup": [
            {
                "doc_family": family,
                **dict(counter),
            }
            for family, counter in sorted(
                family_rollup.items(),
                key=lambda item: (-item[1]["false_positive_pages"], -item[1]["cases"], item[0]),
            )
        ],
        "route_rollup": [
            {
                "route": route,
                **dict(counter),
            }
            for route, counter in sorted(
                route_rollup.items(),
                key=lambda item: (-item[1]["false_positive_pages"], -item[1]["cases"], item[0]),
            )
        ],
        "high_false_positive_qids": [asdict(record) for record in high_false_positive[:15]],
    }


def _render_markdown(report: JsonDict) -> str:
    summary = cast("JsonDict", report.get("summary") or {})
    family_rollup = cast("list[JsonDict]", report.get("family_rollup") or [])
    qids = cast("list[JsonDict]", report.get("high_false_positive_qids") or [])

    lines = [
        "# Page Projection Drift Audit",
        "",
        f"- false_positive_case_count: `{summary.get('false_positive_case_count')}`",
        f"- mixed_doc_case_count: `{summary.get('mixed_doc_case_count')}`",
        f"- orphan_case_count: `{summary.get('orphan_case_count')}`",
        f"- low_confidence_case_count: `{summary.get('low_confidence_case_count')}`",
        f"- false_positive_page_count: `{summary.get('false_positive_page_count')}`",
        f"- recommended_max_page_drift: `{summary.get('recommended_max_page_drift')}`",
        f"- focus_order: `{summary.get('focus_order')}`",
        "",
        "## Families",
        "",
        "| Family | Cases | FP Pages | Mixed-Doc | Orphan | Low-Confidence | Budget Overrun |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in family_rollup[:12]:
        lines.append(
            "| {doc_family} | {cases} | {false_positive_pages} | {mixed_doc_pages} | {orphan_pages} | {low_confidence_pages} | {page_budget_overrun_cases} |".format(
                doc_family=row.get("doc_family"),
                cases=row.get("cases", 0),
                false_positive_pages=row.get("false_positive_pages", 0),
                mixed_doc_pages=row.get("mixed_doc_pages", 0),
                orphan_pages=row.get("orphan_pages", 0),
                low_confidence_pages=row.get("low_confidence_pages", 0),
                page_budget_overrun_cases=row.get("page_budget_overrun_cases", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Top QIDs",
            "",
            "| QID | Family | FP Pages | Mixed-Doc | Orphan | Low-Confidence | Failure Stage |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in qids:
        lines.append(
            "| {qid} | {doc_family} | {false_positive_page_count} | {mixed_doc_page_count} | {orphan_page_count} | {low_confidence_page_count} | {failure_stage} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze false-positive support-page drift from a page-trace ledger.")
    parser.add_argument("--page-trace-ledger", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(page_trace_ledger=_load_json(args.page_trace_ledger))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
