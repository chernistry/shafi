from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

try:
    from build_embedding_doc_family_candidate import _select_pages
    from probe_local_embedding_relevance import (
        _coerce_str_list,
        _cosine_similarity,
        _embed_texts,
        _extract_page_text,
        _load_questions,
        _load_raw_results,
        _load_scaffold_cases,
        _parse_page_id,
        _same_doc_neighbor_page_ids,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_embedding_doc_family_candidate import _select_pages
    from scripts.probe_local_embedding_relevance import (
        _coerce_str_list,
        _cosine_similarity,
        _embed_texts,
        _extract_page_text,
        _load_questions,
        _load_raw_results,
        _load_scaffold_cases,
        _parse_page_id,
        _same_doc_neighbor_page_ids,
    )

JsonDict = dict[str, object]

@dataclass(frozen=True)
class OpportunityPageScore:
    page_id: str
    score: float
    is_gold: bool


@dataclass(frozen=True)
class _PageScoreCompat:
    page_id: str
    score: float


@dataclass(frozen=True)
class SupportOpportunity:
    question_id: str
    question: str
    failure_class: str
    manual_verdict: str
    gold_page_ids: list[str]
    current_page_ids: list[str]
    selected_page_ids: list[str]
    gold_top1: bool
    gold_top3: bool
    current_has_gold: bool
    selected_has_gold: bool
    new_gold_gain: bool
    best_gold_rank: int
    top_page_id: str
    gold_margin: float
    scored_pages: list[OpportunityPageScore]


def _raw_results_current_pages(path: Path) -> dict[str, list[str]]:
    payload = _load_raw_results(path)
    out: dict[str, list[str]] = {}
    for qid, row in payload.items():
        telemetry_obj = row.get("telemetry")
        telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        page_ids: list[str] = []
        for key in ("used_page_ids", "context_page_ids", "retrieved_page_ids"):
            for page_id in _coerce_str_list(telemetry.get(key)):
                if page_id not in page_ids:
                    page_ids.append(page_id)
        out[qid] = page_ids
    return out


def _page_count(pdf_path: Path, cache: dict[str, int]) -> int:
    key = str(pdf_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    import fitz

    doc = cast("Any", fitz.open(str(pdf_path)))
    try:
        count = int(doc.page_count)
    finally:
        doc.close()
    cache[key] = count
    return count


async def _score_pages(
    *,
    question: str,
    page_ids: list[str],
    dataset_dir: Path,
    model: str,
    base_url: str,
    text_cache: dict[str, str],
) -> list[OpportunityPageScore]:
    page_texts: list[str] = []
    kept_page_ids: list[str] = []
    for page_id in page_ids:
        try:
            text = _extract_page_text(page_id, dataset_dir=dataset_dir, cache=text_cache)
        except (FileNotFoundError, ValueError):
            continue
        if not text:
            continue
        kept_page_ids.append(page_id)
        page_texts.append(text)
    if not page_texts:
        return []
    embeddings = await _embed_texts(base_url, model=model, texts=[question, *page_texts])
    query_embedding = embeddings[0]
    scored = [
        OpportunityPageScore(
            page_id=page_id,
            score=_cosine_similarity(query_embedding, embedding),
            is_gold=False,
        )
        for page_id, embedding in zip(kept_page_ids, embeddings[1:], strict=True)
    ]
    return sorted(scored, key=lambda item: (-item.score, item.page_id))


def _render_markdown(*, model: str, opportunities: list[SupportOpportunity]) -> str:
    lines = [
        "# Embedding Support Opportunity Mining",
        "",
        f"- model: `{model}`",
        f"- opportunities: `{len(opportunities)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | QID | Gold Top1 | Gold Top3 | Current Has Gold | Selected Has Gold | New Gold Gain | Best Gold Rank | Top Page | Margin | Selected Pages |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for index, item in enumerate(opportunities, start=1):
        lines.append(
            "| "
            f"{index} | `{item.question_id}` | `{item.gold_top1}` | `{item.gold_top3}` | `{item.current_has_gold}` | "
            f"`{item.selected_has_gold}` | `{item.new_gold_gain}` | `{item.best_gold_rank}` | "
            f"`{item.top_page_id}` | {item.gold_margin:.4f} | `{item.selected_page_ids}` |"
        )
    lines.append("")
    for item in opportunities:
        lines.extend(
            [
                f"## {item.question_id}",
                "",
                f"- question: {item.question}",
                f"- failure_class: `{item.failure_class}`",
                f"- manual_verdict: `{item.manual_verdict}`",
                f"- gold_page_ids: `{item.gold_page_ids}`",
                f"- current_page_ids: `{item.current_page_ids}`",
                f"- selected_page_ids: `{item.selected_page_ids}`",
                "",
                "| Rank | Page | Score | Gold |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for rank, score in enumerate(item.scored_pages, start=1):
            lines.append(f"| {rank} | `{score.page_id}` | {score.score:.4f} | `{score.is_gold}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _opportunity_sort_key(item: SupportOpportunity) -> tuple[int, int, int, float, str]:
    return (
        1 if item.new_gold_gain else 0,
        1 if item.gold_top1 else 0,
        1 if item.gold_top3 else 0,
        item.gold_margin,
        item.question_id,
    )


async def _mine(args: argparse.Namespace) -> list[SupportOpportunity]:
    current_pages_by_qid = _raw_results_current_pages(args.current_raw_results)
    questions = _load_questions(args.questions)
    scaffold_cases = _load_scaffold_cases(
        args.scaffold,
        question_ids=None,
        manual_verdicts={text.strip().lower() for text in args.manual_verdict} if args.manual_verdict else None,
        failure_classes={text.strip() for text in args.failure_class} if args.failure_class else None,
        max_cases=args.max_cases,
    )
    text_cache: dict[str, str] = {}
    page_count_cache: dict[str, int] = {}
    out: list[SupportOpportunity] = []

    for case in scaffold_cases:
        qid = str(case["question_id"])
        question = questions.get(qid, "").strip()
        if not question:
            continue
        gold_page_ids = _coerce_str_list(case.get("gold_page_ids"))
        if not gold_page_ids:
            continue
        current_page_ids = current_pages_by_qid.get(qid, [])
        candidate_page_ids: list[str] = []
        seen: set[str] = set()
        for gold_page_id in gold_page_ids:
            doc_id, _page = _parse_page_id(gold_page_id)
            pdf_path = args.dataset_documents / f"{doc_id}.pdf"
            if not pdf_path.exists():
                continue
            count = _page_count(pdf_path, page_count_cache)
            for page_number in range(1, count + 1):
                page_id = f"{doc_id}_{page_number}"
                if page_id not in seen:
                    seen.add(page_id)
                    candidate_page_ids.append(page_id)
        for neighbor_page_id in _same_doc_neighbor_page_ids(
            gold_page_ids=gold_page_ids,
            dataset_dir=args.dataset_documents,
            page_count_cache={},
        ):
            if neighbor_page_id not in seen:
                seen.add(neighbor_page_id)
                candidate_page_ids.append(neighbor_page_id)
        scored = await _score_pages(
            question=question,
            page_ids=candidate_page_ids,
            dataset_dir=args.dataset_documents,
            model=args.model,
            base_url=args.base_url,
            text_cache=text_cache,
        )
        if not scored:
            continue
        gold_set = set(gold_page_ids)
        scored_marked = [
            OpportunityPageScore(page_id=row.page_id, score=row.score, is_gold=row.page_id in gold_set) for row in scored
        ]
        selected_page_ids = _select_pages(
            scored_pages=cast("Any", [_PageScoreCompat(page_id=row.page_id, score=row.score) for row in scored_marked]),
            per_doc_pages=args.per_doc_pages,
            extra_global_pages=args.extra_global_pages,
        )
        best_gold_rank = next((index for index, row in enumerate(scored_marked, start=1) if row.is_gold), len(scored_marked) + 1)
        best_gold_score = next((row.score for row in scored_marked if row.is_gold), 0.0)
        best_non_gold_score = next((row.score for row in scored_marked if not row.is_gold), 0.0)
        selected_has_gold = any(page_id in gold_set for page_id in selected_page_ids)
        current_has_gold = any(page_id in gold_set for page_id in current_page_ids)
        out.append(
            SupportOpportunity(
                question_id=qid,
                question=question,
                failure_class=str(case.get("source_failure_class") or ""),
                manual_verdict=str(case.get("source_manual_verdict") or ""),
                gold_page_ids=gold_page_ids,
                current_page_ids=current_page_ids,
                selected_page_ids=selected_page_ids,
                gold_top1=best_gold_rank == 1,
                gold_top3=best_gold_rank <= 3,
                current_has_gold=current_has_gold,
                selected_has_gold=selected_has_gold,
                new_gold_gain=selected_has_gold and not current_has_gold,
                best_gold_rank=best_gold_rank,
                top_page_id=scored_marked[0].page_id,
                gold_margin=best_gold_score - best_non_gold_score,
                scored_pages=scored_marked[: args.report_top_k],
            )
        )
    return sorted(out, key=_opportunity_sort_key, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine scaffold support-undercoverage cases for embedding-based page opportunities within gold doc families.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--current-raw-results", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dataset-documents", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--manual-verdict", action="append", default=["correct"])
    parser.add_argument("--failure-class", action="append", default=["support_undercoverage"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--per-doc-pages", type=int, default=1)
    parser.add_argument("--extra-global-pages", type=int, default=1)
    parser.add_argument("--report-top-k", type=int, default=5)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opportunities = asyncio.run(_mine(args))
    payload = {
        "model": args.model,
        "opportunity_count": len(opportunities),
        "opportunities": [asdict(item) for item in opportunities],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(model=args.model, opportunities=opportunities), encoding="utf-8")


if __name__ == "__main__":
    main()
