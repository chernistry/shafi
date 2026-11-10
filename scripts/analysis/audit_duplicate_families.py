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


def _page_ids_from_refs(page_refs: list[JsonDict]) -> set[str]:
    page_ids: set[str] = set()
    for page_ref in page_refs:
        doc_id = str(page_ref.get("doc_id") or "").strip()
        if not doc_id:
            continue
        page_numbers = cast("list[object]", page_ref.get("page_numbers") or [])
        for page_number in page_numbers:
            if isinstance(page_number, int):
                page_ids.add(f"{doc_id}_{page_number}")
    return page_ids


def _doc_ids_from_refs(page_refs: list[JsonDict]) -> set[str]:
    return {
        str(page_ref.get("doc_id") or "").strip()
        for page_ref in page_refs
        if str(page_ref.get("doc_id") or "").strip()
    }


def _page_f_beta(*, true_positive: int, used: int, gold: int, beta: float = 2.5) -> float:
    precision = 0.0 if used <= 0 else true_positive / used
    recall = 0.0 if gold <= 0 else true_positive / gold
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom


def _score_pages(gold_pages: set[str], used_pages: set[str]) -> JsonDict:
    true_positive = len(gold_pages.intersection(used_pages))
    return {
        "true_positive": true_positive,
        "used_count": len(used_pages),
        "gold_count": len(gold_pages),
        "page_precision": 0.0 if not used_pages else true_positive / len(used_pages),
        "page_recall": 0.0 if not gold_pages else true_positive / len(gold_pages),
        "grounding_g_score_beta_2_5": _page_f_beta(true_positive=true_positive, used=len(used_pages), gold=len(gold_pages)),
    }


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
            sibling_record = by_sha[sibling_sha]
            doc_ids = {str(record["doc_id"]), str(sibling_record["doc_id"])}
            targeted_qids: list[str] = []
            partner_retrieved = 0
            current_scores: list[float] = []
            without_partner_scores: list[float] = []
            improved_qids: list[str] = []
            for scaffold_record in scaffold_records:
                question_id = str(scaffold_record.get("question_id") or "")
                retrieved_pages = cast("list[JsonDict]", scaffold_record.get("retrieved_chunk_pages") or [])
                scaffold_doc_ids = _doc_ids_from_refs(retrieved_pages)
                if not scaffold_doc_ids.intersection(doc_ids):
                    continue
                targeted_qids.append(question_id)
                gold_pages = _page_ids_from_refs(retrieved_pages)
                submission_answer = submission_by_qid.get(question_id)
                if submission_answer is None:
                    continue
                telemetry = cast("JsonDict", submission_answer.get("telemetry") or {})
                retrieval = cast("JsonDict", telemetry.get("retrieval") or {})
                runtime_pages = cast("list[JsonDict]", retrieval.get("retrieved_chunk_pages") or [])
                runtime_doc_ids = _doc_ids_from_refs(runtime_pages)
                if len(runtime_doc_ids.intersection(doc_ids)) > 1:
                    partner_retrieved += 1
                used_pages = _page_ids_from_refs(runtime_pages)
                partner_doc_ids = doc_ids - scaffold_doc_ids
                filtered_used_pages = {page_id for page_id in used_pages if page_id.rsplit("_", 1)[0] not in partner_doc_ids}
                current_score = _score_pages(gold_pages, used_pages)
                filtered_score = _score_pages(gold_pages, filtered_used_pages)
                current_scores.append(float(current_score["grounding_g_score_beta_2_5"]))
                without_partner_scores.append(float(filtered_score["grounding_g_score_beta_2_5"]))
                if float(filtered_score["grounding_g_score_beta_2_5"]) > float(current_score["grounding_g_score_beta_2_5"]) + 1e-9:
                    improved_qids.append(question_id)
            recommendation = "safe"
            avg_current = sum(current_scores) / len(current_scores) if current_scores else 0.0
            avg_without_partner = sum(without_partner_scores) / len(without_partner_scores) if without_partner_scores else avg_current
            g_delta = avg_without_partner - avg_current
            exact_duplicate = bool(record.get("exact_duplicate_cluster_id")) and record.get("exact_duplicate_cluster_id") == sibling_record.get(
                "exact_duplicate_cluster_id"
            )
            if partner_retrieved > 0:
                recommendation = "risky"
            elif g_delta > 0.05 or cast("list[str]", record.get("duplicate_same_family_doc_ids") or []) or exact_duplicate:
                recommendation = "dedup_candidate"
            pair_reports.append(
                {
                    "doc_pair": list(pair),
                    "doc_ids": sorted(doc_ids),
                    "sha256_pair": list(pair),
                    "normalized_titles": sorted(
                        {str(record.get("normalized_title") or ""), str(sibling_record.get("normalized_title") or "")}
                    ),
                    "doc_family_tags": {
                        str(record["doc_id"]): cast("list[str]", record.get("doc_family_tags") or []),
                        str(sibling_record["doc_id"]): cast("list[str]", sibling_record.get("doc_family_tags") or []),
                    },
                    "targeted_qids": sorted(set(targeted_qids)),
                    "collision_partner_retrieved_count": partner_retrieved,
                    "grounding_g_score_beta_2_5_current": round(avg_current, 4),
                    "grounding_g_score_beta_2_5_without_partner": round(avg_without_partner, 4),
                    "grounding_g_score_beta_2_5_delta": round(g_delta, 4),
                    "improved_qids": sorted(set(improved_qids)),
                    "exact_duplicate_cluster_id": (
                        record.get("exact_duplicate_cluster_id")
                        if record.get("exact_duplicate_cluster_id") == sibling_record.get("exact_duplicate_cluster_id")
                        else None
                    ),
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
        "| doc_ids | targeted_qids | partner_retrieved | g_beta_current | g_beta_without_partner | g_delta | recommendation |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in cast("list[JsonDict]", report["pairs"]):
        lines.append(
            f"| {', '.join(cast('list[str]', pair['doc_ids']))} | "
            f"{', '.join(cast('list[str]', pair['targeted_qids'])) or '-'} | "
            f"{pair['collision_partner_retrieved_count']} | "
            f"{pair['grounding_g_score_beta_2_5_current']} | "
            f"{pair['grounding_g_score_beta_2_5_without_partner']} | "
            f"{pair['grounding_g_score_beta_2_5_delta']} | {pair['recommendation']} |"
        )
    if not cast("list[JsonDict]", report["pairs"]):
        lines.append("| - | - | 0 | 0.0 | 0.0 | 0.0 | safe |")
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
