from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_legalbenchrag_mini_bootstrap import (
    BenchmarkCase,
    BenchmarkSnippet,
    RetrievedSnippet,
    build_chunk_index,
    compute_precision_recall,
    score_chunk,
)

from shafi.core.pipeline import RAGPipelineBuilder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit named-ref and provision-query lanes on the public legal pack.")
    parser.add_argument("--pack-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--top-k-per-query", type=int, default=1)
    return parser.parse_args()


def _load_pack(path: Path) -> list[dict[str, object]]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError("Pack JSON must be an object")
    payload = cast("dict[str, object]", payload_obj)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Pack JSON must contain a `cases` array")
    raw_cases = cast("list[object]", cases)
    return [cast("dict[str, object]", case) for case in raw_cases if isinstance(case, dict)]


def _derive_benchmark_case(case: dict[str, object], repo_root: Path) -> tuple[dict[str, str], BenchmarkCase] | None:
    source_fixtures = case.get("source_fixtures", [])
    gold_texts = case.get("gold_texts", [])
    if not isinstance(source_fixtures, list) or not isinstance(gold_texts, list):
        return None

    text_paths = [repo_root / fixture for fixture in cast("list[object]", source_fixtures) if isinstance(fixture, str) and fixture.endswith(".txt")]
    if len(text_paths) != len(cast("list[object]", source_fixtures)) or not text_paths:
        return None

    corpus: dict[str, str] = {}
    for path in text_paths:
        corpus[path.name] = path.read_text(encoding="utf-8")

    snippets: list[BenchmarkSnippet] = []
    for raw_gold in cast("list[object]", gold_texts):
        gold_text = str(raw_gold).strip()
        if not gold_text:
            continue
        matches: list[BenchmarkSnippet] = []
        for file_name, content in corpus.items():
            start = content.find(gold_text)
            if start >= 0:
                matches.append(BenchmarkSnippet(file_path=file_name, span=(start, start + len(gold_text))))
        if len(matches) != 1:
            return None
        snippets.append(matches[0])
    if not snippets:
        return None

    benchmark_case = BenchmarkCase(
        query=str(case.get("question", "")).strip(),
        snippets=tuple(snippets),
        tags=(
            str(case.get("family", "")).strip(),
            str(case.get("subtype", "")).strip(),
        ),
    )
    return corpus, benchmark_case


def _rank_query(
    *,
    query: str,
    chunk_index: dict[str, list[tuple[tuple[int, int], str]]],
    top_k: int,
) -> list[RetrievedSnippet]:
    ranked: list[RetrievedSnippet] = []
    for file_path, chunks in chunk_index.items():
        for span, chunk_text in chunks:
            score = score_chunk(query, chunk_text)
            if score <= 0:
                continue
            ranked.append(
                RetrievedSnippet(
                    file_path=file_path,
                    span=span,
                    score=score,
                    text=chunk_text,
                )
            )
    ranked.sort(key=lambda item: (-item.score, item.file_path, item.span[0], item.span[1]))
    return ranked[: max(1, int(top_k))]


def _merge_ranked_rows(rows: list[list[RetrievedSnippet]], *, top_k: int) -> list[RetrievedSnippet]:
    merged: dict[tuple[str, int, int], RetrievedSnippet] = {}
    for row in rows:
        for snippet in row:
            key = (snippet.file_path, snippet.span[0], snippet.span[1])
            existing = merged.get(key)
            if existing is None or snippet.score > existing.score:
                merged[key] = snippet
    ranked = sorted(merged.values(), key=lambda item: (-item.score, item.file_path, item.span[0], item.span[1]))
    return ranked[: max(1, int(top_k))]


def _targeted_queries(case: dict[str, object]) -> tuple[str, list[str]]:
    query = str(case.get("question", "")).strip()
    doc_refs = [str(ref).strip() for ref in cast("list[object]", case.get("doc_refs", [])) if str(ref).strip()]
    provision_refs = [str(ref).strip() for ref in cast("list[object]", case.get("provision_refs", [])) if str(ref).strip()]
    named_ref_builder = getattr(RAGPipelineBuilder, "_targeted_named_ref_query", None)
    provision_builder = getattr(RAGPipelineBuilder, "_targeted_provision_ref_query", None)
    if not callable(named_ref_builder) or not callable(provision_builder):
        raise TypeError("Expected named-ref and provision query builders on RAGPipelineBuilder")

    if provision_refs and doc_refs:
        return "targeted_provision", [
            str(provision_builder(query=query, ref=doc_refs[0], refs=doc_refs))
        ]
    if doc_refs:
        return "targeted_named_ref", [
            str(named_ref_builder(query=query, ref=ref, refs=doc_refs))
            for ref in doc_refs[: max(1, min(2, len(doc_refs)))]
        ]
    return "baseline_only", [query]


def _render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Named-Ref Public Pack Audit",
        "",
        f"- runnable_case_count: `{summary['runnable_case_count']}`",
        f"- skipped_case_count: `{summary['skipped_case_count']}`",
        f"- baseline_avg_precision: `{summary['baseline_avg_precision']}`",
        f"- baseline_avg_recall: `{summary['baseline_avg_recall']}`",
        f"- targeted_avg_precision: `{summary['targeted_avg_precision']}`",
        f"- targeted_avg_recall: `{summary['targeted_avg_recall']}`",
        f"- actionable_miss_count: `{summary['actionable_miss_count']}`",
        f"- overall_verdict: `{summary['overall_verdict']}`",
        "",
        "## Cases",
        "",
    ]
    case_rows = summary.get("cases", [])
    if isinstance(case_rows, list):
        for item in cast("list[object]", case_rows):
            if not isinstance(item, dict):
                continue
            row = cast("dict[str, object]", item)
            lines.append(f"### {row.get('case_id', '')}")
            lines.append(f"- family: `{row.get('family')}`")
            lines.append(f"- baseline_precision: `{row.get('baseline_precision')}`")
            lines.append(f"- baseline_recall: `{row.get('baseline_recall')}`")
            lines.append(f"- targeted_precision: `{row.get('targeted_precision')}`")
            lines.append(f"- targeted_recall: `{row.get('targeted_recall')}`")
            lines.append(f"- recommendation: `{row.get('recommendation')}`")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_audit(
    *,
    pack_json: Path,
    out_json: Path,
    out_md: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k_per_query: int,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    pack_cases = _load_pack(pack_json)

    case_rows: list[dict[str, object]] = []
    skipped_case_count = 0
    for case in pack_cases:
        derived = _derive_benchmark_case(case, repo_root)
        if derived is None:
            skipped_case_count += 1
            continue
        corpus, benchmark_case = derived
        chunk_index = build_chunk_index(corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        baseline_top_k = max(1, len(benchmark_case.snippets))
        baseline_retrieved = _rank_query(
            query=benchmark_case.query,
            chunk_index=chunk_index,
            top_k=baseline_top_k,
        )
        baseline_precision, baseline_recall = compute_precision_recall(baseline_retrieved, benchmark_case.snippets)

        targeted_variant, targeted_queries = _targeted_queries(case)
        targeted_rows = [
            _rank_query(query=query, chunk_index=chunk_index, top_k=top_k_per_query)
            for query in targeted_queries
        ]
        targeted_retrieved = _merge_ranked_rows(targeted_rows, top_k=baseline_top_k)
        targeted_precision, targeted_recall = compute_precision_recall(targeted_retrieved, benchmark_case.snippets)

        recommendation = "NO_CHANGE"
        if targeted_recall > baseline_recall or (
            targeted_recall == baseline_recall and targeted_precision > baseline_precision + 1e-9
        ):
            recommendation = "PROMISING"
        elif targeted_recall < baseline_recall or targeted_precision + 1e-9 < baseline_precision:
            recommendation = "REGRESSION"

        case_rows.append(
            {
                "case_id": str(case.get("case_id", "")).strip(),
                "family": str(case.get("family", "")).strip(),
                "targeted_variant": targeted_variant,
                "targeted_queries": targeted_queries,
                "baseline_precision": round(baseline_precision, 6),
                "baseline_recall": round(baseline_recall, 6),
                "targeted_precision": round(targeted_precision, 6),
                "targeted_recall": round(targeted_recall, 6),
                "recommendation": recommendation,
            }
        )

    runnable_case_count = len(case_rows)
    actionable_miss_count = sum(1 for row in case_rows if row["recommendation"] == "PROMISING")
    overall_verdict = "ACTIONABLE_MISS_FOUND" if actionable_miss_count > 0 else "TRANSFER_CONFIDENCE_ONLY"
    summary: dict[str, object] = {
        "pack_json": str(pack_json),
        "runnable_case_count": runnable_case_count,
        "skipped_case_count": skipped_case_count,
        "baseline_avg_precision": round(
            sum(float(cast("float", row["baseline_precision"])) for row in case_rows) / max(1, runnable_case_count),
            6,
        ),
        "baseline_avg_recall": round(
            sum(float(cast("float", row["baseline_recall"])) for row in case_rows) / max(1, runnable_case_count),
            6,
        ),
        "targeted_avg_precision": round(
            sum(float(cast("float", row["targeted_precision"])) for row in case_rows) / max(1, runnable_case_count),
            6,
        ),
        "targeted_avg_recall": round(
            sum(float(cast("float", row["targeted_recall"])) for row in case_rows) / max(1, runnable_case_count),
            6,
        ),
        "actionable_miss_count": actionable_miss_count,
        "overall_verdict": overall_verdict,
        "cases": case_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    run_audit(
        pack_json=args.pack_json,
        out_json=args.out_json,
        out_md=args.out_md,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k_per_query=args.top_k_per_query,
    )


if __name__ == "__main__":
    main()
