from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rag_challenge.core.classifier import QueryClassifier

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:\([^)]+\))?")
_PROVISION_RE = re.compile(
    r"\b(?:Article|Section|Schedule|Part|Chapter)\s+\d+(?:\s*\(\s*[^)]+\s*\))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkSnippet:
    file_path: str
    span: tuple[int, int]


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    snippets: tuple[BenchmarkSnippet, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedSnippet:
    file_path: str
    span: tuple[int, int]
    score: float
    text: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight LegalBench-RAG-style retrieval bootstrap.")
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-tests", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _load_benchmark(path: Path, *, max_tests: int) -> list[BenchmarkCase]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError("Benchmark JSON must be an object")
    payload = cast("dict[str, object]", payload_obj)
    tests = payload.get("tests")
    if not isinstance(tests, list):
        raise ValueError("Benchmark JSON must contain a top-level `tests` array")

    out: list[BenchmarkCase] = []
    raw_cases = cast("list[object]", tests)
    for raw_case in raw_cases[: max(0, int(max_tests))]:
        if not isinstance(raw_case, dict):
            continue
        case_dict = cast("dict[str, object]", raw_case)
        query = str(case_dict.get("query", "")).strip()
        raw_snippets = case_dict.get("snippets")
        if not query or not isinstance(raw_snippets, list):
            continue
        snippets: list[BenchmarkSnippet] = []
        snippet_rows = cast("list[object]", raw_snippets)
        for raw_snippet in snippet_rows:
            if not isinstance(raw_snippet, dict):
                continue
            snippet_dict = cast("dict[str, object]", raw_snippet)
            file_path = str(snippet_dict.get("file_path", "")).strip()
            span = snippet_dict.get("span")
            if not file_path or not isinstance(span, (list, tuple)):
                continue
            span_parts = cast("list[object] | tuple[object, ...]", span)
            if len(span_parts) != 2:
                continue
            start = _coerce_int(span_parts[0])
            end = _coerce_int(span_parts[1])
            if start is None or end is None:
                continue
            if end <= start:
                continue
            snippets.append(BenchmarkSnippet(file_path=file_path, span=(start, end)))
        if not snippets:
            continue
        raw_tags = case_dict.get("tags", [])
        tags = (
            tuple(str(tag).strip() for tag in cast("list[object]", raw_tags) if str(tag).strip())
            if isinstance(raw_tags, list)
            else ()
        )
        out.append(BenchmarkCase(query=query, snippets=tuple(snippets), tags=tags))
    return out


def _load_corpus(corpus_dir: Path, *, file_paths: set[str]) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for file_path in sorted(file_paths):
        full_path = corpus_dir / file_path
        corpus[file_path] = full_path.read_text(encoding="utf-8")
    return corpus


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _extract_exact_features(query: str) -> tuple[list[str], list[str]]:
    exact_refs = [ref.strip().lower() for ref in QueryClassifier.extract_exact_legal_refs(query) if ref.strip()]
    provision_refs: list[str] = []
    seen: set[str] = set()
    for match in _PROVISION_RE.finditer(query or ""):
        normalized = re.sub(r"\s+", " ", match.group(0)).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        provision_refs.append(normalized)
    return exact_refs, provision_refs


def _chunk_text(content: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    size = max(1, int(chunk_size))
    overlap = max(0, int(chunk_overlap))
    if overlap >= size:
        overlap = max(0, size - 1)
    step = max(1, size - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(content)
    while start < length:
        end = min(length, start + size)
        spans.append((start, end))
        if end >= length:
            break
        start += step
    return spans


def _score_chunk(query: str, chunk_text: str) -> float:
    chunk_lower = (chunk_text or "").lower()
    query_tokens = _tokenize(query)
    if not query_tokens or not chunk_lower:
        return 0.0

    token_weights: dict[str, float] = {}
    for token in query_tokens:
        token_weights[token] = max(token_weights.get(token, 0.0), 2.0 if any(ch.isdigit() for ch in token) else 1.0)

    score = 0.0
    for token, weight in token_weights.items():
        if token in chunk_lower:
            score += weight

    exact_refs, provision_refs = _extract_exact_features(query)
    for ref in exact_refs:
        if ref and ref in chunk_lower:
            score += 8.0
    for provision_ref in provision_refs:
        if provision_ref and provision_ref in chunk_lower:
            score += 6.0
            short = provision_ref.split(" ", maxsplit=1)[-1].strip()
            if short and short in chunk_lower:
                score += 1.5

    if chunk_lower.startswith(tuple(filter(None, exact_refs + provision_refs))):
        score += 1.0

    return score


def _build_chunk_index(
    corpus: dict[str, str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, list[tuple[tuple[int, int], str]]]:
    index: dict[str, list[tuple[tuple[int, int], str]]] = {}
    for file_path, content in corpus.items():
        spans = _chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        index[file_path] = [((start, end), content[start:end]) for start, end in spans]
    return index


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _overlap_len(spans_a: list[tuple[int, int]], spans_b: list[tuple[int, int]]) -> int:
    overlap = 0
    for start_a, end_a in spans_a:
        for start_b, end_b in spans_b:
            common_start = max(start_a, start_b)
            common_end = min(end_a, end_b)
            if common_end > common_start:
                overlap += common_end - common_start
    return overlap


def _precision_recall(retrieved: list[RetrievedSnippet], gold: tuple[BenchmarkSnippet, ...]) -> tuple[float, float]:
    retrieved_by_file: dict[str, list[tuple[int, int]]] = {}
    gold_by_file: dict[str, list[tuple[int, int]]] = {}

    for snippet in retrieved:
        retrieved_by_file.setdefault(snippet.file_path, []).append(snippet.span)
    for snippet in gold:
        gold_by_file.setdefault(snippet.file_path, []).append(snippet.span)

    total_retrieved = 0
    total_relevant = 0
    relevant_retrieved = 0

    for file_path, spans in retrieved_by_file.items():
        merged_retrieved = _merge_spans(spans)
        total_retrieved += sum(end - start for start, end in merged_retrieved)
        merged_gold = _merge_spans(gold_by_file.get(file_path, []))
        relevant_retrieved += _overlap_len(merged_retrieved, merged_gold)

    for spans in gold_by_file.values():
        merged_gold = _merge_spans(spans)
        total_relevant += sum(end - start for start, end in merged_gold)

    precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    return precision, recall


def build_chunk_index(
    corpus: dict[str, str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, list[tuple[tuple[int, int], str]]]:
    return _build_chunk_index(corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def score_chunk(query: str, chunk_text: str) -> float:
    return _score_chunk(query, chunk_text)


def compute_precision_recall(
    retrieved: list[RetrievedSnippet],
    gold: tuple[BenchmarkSnippet, ...],
) -> tuple[float, float]:
    return _precision_recall(retrieved, gold)


def _run_case(case: BenchmarkCase, chunk_index: dict[str, list[tuple[tuple[int, int], str]]], *, top_k: int) -> dict[str, object]:
    ranked: list[RetrievedSnippet] = []
    for file_path, chunks in chunk_index.items():
        for span, chunk_text in chunks:
            score = _score_chunk(case.query, chunk_text)
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
    retrieved = ranked[: max(1, int(top_k))]
    precision, recall = _precision_recall(retrieved, case.snippets)
    return {
        "query": case.query,
        "tags": list(case.tags),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "gold": [
            {
                "file_path": snippet.file_path,
                "span": [snippet.span[0], snippet.span[1]],
            }
            for snippet in case.snippets
        ],
        "retrieved": [
            {
                "file_path": snippet.file_path,
                "span": [snippet.span[0], snippet.span[1]],
                "score": round(snippet.score, 4),
                "text_preview": snippet.text[:160],
            }
            for snippet in retrieved
        ],
    }


def _render_markdown(summary: dict[str, object]) -> str:
    cases_obj = summary.get("cases", [])
    cases = cast("list[object]", cases_obj) if isinstance(cases_obj, list) else []
    lines = [
        "# LegalBench-RAG Mini Bootstrap",
        "",
        f"- case_count: `{summary['case_count']}`",
        f"- doc_count: `{summary['doc_count']}`",
        f"- avg_precision: `{summary['avg_precision']}`",
        f"- avg_recall: `{summary['avg_recall']}`",
        f"- chunk_size: `{summary['chunk_size']}`",
        f"- chunk_overlap: `{summary['chunk_overlap']}`",
        f"- top_k: `{summary['top_k']}`",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        record = cast("dict[str, object]", case) if isinstance(case, dict) else {}
        lines.append(f"### {record.get('query', '')}")
        lines.append(f"- precision: `{record.get('precision')}`")
        lines.append(f"- recall: `{record.get('recall')}`")
        retrieved = record.get("retrieved", [])
        if isinstance(retrieved, list):
            for item in cast("list[object]", retrieved)[:3]:
                if not isinstance(item, dict):
                    continue
                item_dict = cast("dict[str, object]", item)
                lines.append(
                    f"- retrieved: `{item_dict.get('file_path')}:{item_dict.get('span')}` score=`{item_dict.get('score')}`"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_bootstrap(
    *,
    benchmark_json: Path,
    corpus_dir: Path,
    out_json: Path,
    out_md: Path,
    max_tests: int,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> dict[str, object]:
    cases = _load_benchmark(benchmark_json, max_tests=max_tests)
    file_paths = {snippet.file_path for case in cases for snippet in case.snippets}
    corpus = _load_corpus(corpus_dir, file_paths=file_paths)
    chunk_index = _build_chunk_index(corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    case_rows = [_run_case(case, chunk_index, top_k=top_k) for case in cases]

    avg_precision = round(sum(float(cast("float", row["precision"])) for row in case_rows) / max(1, len(case_rows)), 6)
    avg_recall = round(sum(float(cast("float", row["recall"])) for row in case_rows) / max(1, len(case_rows)), 6)
    summary: dict[str, object] = {
        "benchmark_json": str(benchmark_json),
        "corpus_dir": str(corpus_dir),
        "case_count": len(case_rows),
        "doc_count": len(corpus),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "top_k": int(top_k),
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "cases": case_rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = _parse_args()
    run_bootstrap(
        benchmark_json=args.benchmark_json,
        corpus_dir=args.corpus_dir,
        out_json=args.out_json,
        out_md=args.out_md,
        max_tests=args.max_tests,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
