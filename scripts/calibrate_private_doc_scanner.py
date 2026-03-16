from __future__ import annotations

import argparse
import json
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
