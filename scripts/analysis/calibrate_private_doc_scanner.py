from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, cast

try:
    from scripts.scan_private_doc_anomalies import build_summary_markdown, build_top20_report_markdown, scan_pdf_corpus
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.scan_private_doc_anomalies import build_summary_markdown, build_top20_report_markdown, scan_pdf_corpus

JsonDict = dict[str, Any]


def _percentile(values: list[int], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return float(ordered[index])


def _summarize_numeric_series(values: list[float]) -> JsonDict:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p50": round(_percentile([round(value * 10000) for value in ordered], 0.50) / 10000, 4),
        "p75": round(_percentile([round(value * 10000) for value in ordered], 0.75) / 10000, 4),
        "p90": round(_percentile([round(value * 10000) for value in ordered], 0.90) / 10000, 4),
        "max": round(ordered[-1], 4),
    }


def _signal_baselines(records: list[JsonDict]) -> JsonDict:
    doc_numeric: dict[str, list[float]] = defaultdict(list)
    doc_boolean_active: dict[str, int] = defaultdict(int)
    doc_boolean_total: dict[str, int] = defaultdict(int)
    page_numeric: dict[str, list[float]] = defaultdict(list)
    page_boolean_active: dict[str, int] = defaultdict(int)
    page_boolean_total: dict[str, int] = defaultdict(int)

    def _collect(signals: JsonDict, *, numeric_target: dict[str, list[float]], bool_active: dict[str, int], bool_total: dict[str, int]) -> None:
        for name, value in signals.items():
            if isinstance(value, bool):
                bool_total[name] += 1
                if value:
                    bool_active[name] += 1
            elif isinstance(value, int | float):
                numeric_target[name].append(float(value))

    for record in records:
        _collect(cast("JsonDict", record.get("signals") or {}), numeric_target=doc_numeric, bool_active=doc_boolean_active, bool_total=doc_boolean_total)
        for page_record in cast("list[JsonDict]", record.get("per_page") or []):
            _collect(
                cast("JsonDict", page_record.get("signals") or {}),
                numeric_target=page_numeric,
                bool_active=page_boolean_active,
                bool_total=page_boolean_total,
            )

    def _serialize_scope(
        numeric_target: dict[str, list[float]],
        bool_active: dict[str, int],
        bool_total: dict[str, int],
    ) -> JsonDict:
        return {
            "numeric": {name: _summarize_numeric_series(values) for name, values in sorted(numeric_target.items())},
            "boolean": {
                name: {
                    "count": bool_total[name],
                    "active_count": bool_active[name],
                    "activation_rate": round(bool_active[name] / max(bool_total[name], 1), 4),
                }
                for name in sorted(bool_total)
            },
        }

    return {
        "doc_signals": _serialize_scope(doc_numeric, doc_boolean_active, doc_boolean_total),
        "page_signals": _serialize_scope(page_numeric, page_boolean_active, page_boolean_total),
    }


def _fixture_candidates(records: list[JsonDict]) -> JsonDict:
    candidate_rules = {
        "contents_linked_docs": lambda record: (
            float(cast("JsonDict", record.get("signals") or {}).get("contents_internal_link_density") or 0.0) > 0.0
            or float(record.get("contents_link_count") or 0.0) > 0.0
        ),
        "schedule_annex_docs": lambda record: (
            any(tag in cast("list[str]", record.get("doc_family_tags") or []) for tag in ("schedule_annex_heavy", "annex_copy"))
            or any(
                bool(cast("JsonDict", page_record.get("signals") or {}).get("schedule_signature"))
                or bool(cast("JsonDict", page_record.get("signals") or {}).get("annex_signature"))
                for page_record in cast("list[JsonDict]", record.get("per_page") or [])
            )
        ),
        "enactment_notice_docs": lambda record: "enactment_notice" in cast("list[str]", record.get("doc_family_tags") or []),
        "table_heavy_docs": lambda record: (
            "table_heavy" in cast("list[str]", record.get("doc_family_tags") or [])
            or any(bool(cast("JsonDict", page_record.get("signals") or {}).get("table_heavy_signature")) for page_record in cast("list[JsonDict]", record.get("per_page") or []))
        ),
        "tracked_changes_docs": lambda record: bool(record.get("tracked_changes_detected")),
        "translation_caveat_docs": lambda record: bool(record.get("translation_caveat")),
        "image_heavy_docs": lambda record: float(record.get("image_only_page_fraction") or 0.0) > 0.0,
    }

    candidates: JsonDict = {}
    for label, rule in candidate_rules.items():
        matching = [
            {
                "doc_id": record["doc_id"],
                "filename": record["filename"],
                "score": record["suspicion_score"],
                "reason_tags": cast("list[str]", record.get("reason_tags") or [])[:5],
            }
            for record in records
            if rule(record)
        ]
        candidates[label] = matching[:5]
    return candidates


def build_scanner_baseline(*, docs_dir: Path, coverage_priors: JsonDict | None = None) -> JsonDict:
    records = scan_pdf_corpus(input_dir=docs_dir, mode="raw-pdf-corpus", coverage_priors=coverage_priors or {})
    scores = [int(record["suspicion_score"]) for record in records]
    top_docs = [
        {
            "doc_id": record["doc_id"],
            "filename": record["filename"],
            "score": record["suspicion_score"],
            "reason_tags": cast("list[str]", record.get("reason_tags") or []),
        }
        for record in records[:5]
    ]
    score_distribution = {
        "min": min(scores, default=0),
        "max": max(scores, default=0),
        "median": float(median(scores)) if scores else 0.0,
        "p75": _percentile(scores, 0.75),
        "p90": _percentile(scores, 0.90),
        "p10": _percentile(scores, 0.10),
    }
    p10 = score_distribution["p10"]
    p90 = score_distribution["p90"]
    discrimination_ratio = p90 / max(p10, 1.0)
    return {
        "docs_scanned": len(records),
        "score_distribution": score_distribution,
        "signal_baselines": _signal_baselines(records),
        "fixture_candidates": _fixture_candidates(records),
        "discrimination_ratio": round(discrimination_ratio, 4),
        "discrimination_ok": discrimination_ratio >= 2.0 or p90 > p10,
        "top_docs": top_docs,
        "records": records,
        "summary_md": build_summary_markdown(records),
        "top20_md": build_top20_report_markdown(records),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate scanner thresholds on the public corpus.")
    parser.add_argument("--docs-dir", type=Path, default=Path("dataset/dataset_documents"))
    parser.add_argument("--coverage-priors-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    coverage_priors = None
    if args.coverage_priors_json:
        payload = json.loads(args.coverage_priors_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            coverage_priors = cast("JsonDict", payload)
    report = build_scanner_baseline(docs_dir=args.docs_dir, coverage_priors=coverage_priors)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "public_baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "summary.md").write_text(str(report["summary_md"]), encoding="utf-8")
    (args.out_dir / "top20_report.md").write_text(str(report["top20_md"]), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
