from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

try:
    from scripts.scan_private_doc_anomalies import scan_pdf_corpus
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.scan_private_doc_anomalies import scan_pdf_corpus

JsonDict = dict[str, Any]


def _load_questions(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Questions JSON must be a list: {path}")
    rows: list[JsonDict] = []
    for item in cast("list[object]", payload):
        if isinstance(item, dict):
            rows.append(cast("JsonDict", item))
    return rows


def _load_scaffold(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scaffold must contain records[]: {path}")
    payload_dict = cast("JsonDict", payload)
    records_obj = payload_dict.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold must contain records[]: {path}")
    return [cast("JsonDict", item) for item in cast("list[object]", records_obj) if isinstance(item, dict)]


def build_question_surface_gap_report(
    *,
    docs_dir: Path,
    questions_path: Path,
    scaffold_path: Path,
) -> JsonDict:
    records = scan_pdf_corpus(input_dir=docs_dir, mode="raw-pdf-corpus", coverage_priors={})
    doc_to_families = {
        str(record["doc_id"]): cast("list[str]", record.get("doc_family_tags") or [])
        for record in records
    }

    docs_in_family: dict[str, int] = defaultdict(int)
    for families in doc_to_families.values():
        for family in families:
            docs_in_family[family] += 1

    questions = _load_questions(questions_path)
    question_ids = {str(question.get("id") or "") for question in questions}
    scaffold_records = _load_scaffold(scaffold_path)

    questions_targeting_family: dict[str, set[str]] = defaultdict(set)
    qid_to_doc_ids: dict[str, list[str]] = {}
    for record in scaffold_records:
        question_id = str(record.get("question_id") or "")
        if not question_id or question_id not in question_ids:
            continue
        retrieved_chunk_pages = cast("list[JsonDict]", record.get("retrieved_chunk_pages") or [])
        doc_ids = sorted(
            {
                str(page_ref.get("doc_id") or "").strip()
                for page_ref in retrieved_chunk_pages
                if str(page_ref.get("doc_id") or "").strip()
            }
        )
        qid_to_doc_ids[question_id] = doc_ids
        targeted_families = {
            family for doc_id in doc_ids for family in doc_to_families.get(doc_id, [])
        }
        for family in targeted_families:
            questions_targeting_family[family].add(question_id)

    family_coverage_report: list[JsonDict] = []
    family_buckets: dict[str, str] = {}
    family_scores: dict[str, int] = {}
    for family in sorted(docs_in_family):
        question_count = len(questions_targeting_family.get(family, set()))
        doc_count = docs_in_family[family]
        coverage_ratio = round(question_count / doc_count, 4) if doc_count else 0.0
        if question_count == 0:
            bucket = "zero-hit"
        elif question_count == 1:
            bucket = "one-hit"
        else:
            bucket = "exercised"
        family_buckets[family] = bucket
        family_scores[family] = 6 if bucket == "zero-hit" else 3 if bucket == "one-hit" else 0
        family_coverage_report.append(
            {
                "family_tag": family,
                "doc_count": doc_count,
                "question_count": question_count,
                "coverage_ratio": coverage_ratio,
                "coverage_bucket": bucket,
            }
        )

    return {
        "docs_scanned": len(records),
        "questions_scanned": len(question_ids),
        "family_coverage_report": family_coverage_report,
        "family_buckets": family_buckets,
        "family_scores": family_scores,
        "qid_to_doc_ids": qid_to_doc_ids,
    }


def render_question_surface_gap_markdown(report: JsonDict) -> str:
    lines = [
        "# Public Question-Surface Gap Report",
        "",
        f"- Docs scanned: {report['docs_scanned']}",
        f"- Questions scanned: {report['questions_scanned']}",
        "",
        "| family_tag | doc_count | question_count | coverage_ratio | coverage_bucket |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    rows = cast("list[JsonDict]", report["family_coverage_report"])
    if not rows:
        lines.append("| - | 0 | 0 | 0.0 | zero-hit |")
    else:
        for row in rows:
            lines.append(
                f"| {row['family_tag']} | {row['doc_count']} | {row['question_count']} | "
                f"{row['coverage_ratio']} | {row['coverage_bucket']} |"
            )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit public question-surface coverage against document families.")
    parser.add_argument("--docs-dir", type=Path, default=Path("dataset/dataset_documents"))
    parser.add_argument("--questions", type=Path, default=Path("dataset/public_dataset.json"))
    parser.add_argument("--scaffold", type=Path, default=Path("platform_runs/warmup/truth_audit_scaffold_v6_context_seed.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = build_question_surface_gap_report(
        docs_dir=args.docs_dir,
        questions_path=args.questions,
        scaffold_path=args.scaffold,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_question_surface_gap_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
