from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(cast("JsonDict", payload))
    return rows


def _load_submission(path: Path) -> dict[str, JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answers = cast("list[JsonDict]", payload.get("answers") or [])
    return {str(answer.get("question_id") or ""): answer for answer in answers if str(answer.get("question_id") or "")}


def _load_scaffold(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [cast("JsonDict", item) for item in cast("list[object]", payload.get("records") or []) if isinstance(item, dict)]


def build_duplicate_family_audit(
    *,
    scan_results_path: Path,
    scaffold_path: Path,
    submission_path: Path,
) -> JsonDict:
    records = _load_jsonl(scan_results_path)
    scaffold_records = _load_scaffold(scaffold_path)
    submission_by_qid = _load_submission(submission_path)
    by_sha = {str(record["sha256"]): record for record in records}

    seen_pairs: set[tuple[str, str]] = set()
    pair_reports: list[JsonDict] = []
    for record in records:
        current_sha = str(record["sha256"])
        sibling_hashes = sorted(
            set(cast("list[str]", record.get("collision_doc_ids") or []))
            | set(cast("list[str]", record.get("duplicate_same_family_doc_ids") or []))
        )
        for sibling_sha in sibling_hashes:
            pair: tuple[str, str] = (
                (current_sha, sibling_sha) if current_sha <= sibling_sha else (sibling_sha, current_sha)
            )
            if pair in seen_pairs or sibling_sha not in by_sha:
                continue
            seen_pairs.add(pair)
            doc_ids = {str(record["doc_id"]), str(by_sha[sibling_sha]["doc_id"])}
            targeted_qids: list[str] = []
            partner_retrieved = 0
            for scaffold_record in scaffold_records:
                question_id = str(scaffold_record.get("question_id") or "")
                retrieved_pages = cast("list[JsonDict]", scaffold_record.get("retrieved_chunk_pages") or [])
                scaffold_doc_ids = {
                    str(page_ref.get("doc_id") or "")
                    for page_ref in retrieved_pages
                    if str(page_ref.get("doc_id") or "")
                }
                if not scaffold_doc_ids.intersection(doc_ids):
                    continue
                targeted_qids.append(question_id)
                submission_answer = submission_by_qid.get(question_id)
                if submission_answer is None:
                    continue
                telemetry = cast("JsonDict", submission_answer.get("telemetry") or {})
                retrieval = cast("JsonDict", telemetry.get("retrieval") or {})
                runtime_pages = cast("list[JsonDict]", retrieval.get("retrieved_chunk_pages") or [])
                runtime_doc_ids = {
                    str(page_ref.get("doc_id") or "")
                    for page_ref in runtime_pages
                    if str(page_ref.get("doc_id") or "")
                }
                if len(runtime_doc_ids.intersection(doc_ids)) > 1:
                    partner_retrieved += 1
            recommendation = "safe"
            if partner_retrieved > 0:
                recommendation = "risky"
            elif cast("list[str]", record.get("duplicate_same_family_doc_ids") or []):
                recommendation = "dedup_candidate"
            pair_reports.append(
                {
                    "doc_pair": list(pair),
                    "doc_ids": sorted(doc_ids),
                    "targeted_qids": sorted(set(targeted_qids)),
                    "collision_partner_retrieved_count": partner_retrieved,
                    "recommendation": recommendation,
                }
            )

    return {
        "pair_count": len(pair_reports),
        "pairs": pair_reports,
    }


def render_duplicate_family_markdown(report: JsonDict) -> str:
    lines = [
        "# Duplicate Family Audit",
        "",
        f"- Pairs audited: {report['pair_count']}",
        "",
        "| doc_ids | targeted_qids | collision_partner_retrieved_count | recommendation |",
        "| --- | --- | ---: | --- |",
    ]
    for pair in cast("list[JsonDict]", report["pairs"]):
        lines.append(
            f"| {', '.join(cast('list[str]', pair['doc_ids']))} | "
            f"{', '.join(cast('list[str]', pair['targeted_qids'])) or '-'} | "
            f"{pair['collision_partner_retrieved_count']} | {pair['recommendation']} |"
        )
    if not cast("list[JsonDict]", report["pairs"]):
        lines.append("| - | - | 0 | safe |")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit same-title and duplicate-family retrieval risk.")
    parser.add_argument("--scan-results-jsonl", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, default=Path("platform_runs/warmup/truth_audit_scaffold_v6_context_seed.json"))
    parser.add_argument("--submission", type=Path, default=Path("platform_runs/warmup/submission_v6_context_seed.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = build_duplicate_family_audit(
        scan_results_path=args.scan_results_jsonl,
        scaffold_path=args.scaffold,
        submission_path=args.submission,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_duplicate_family_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
