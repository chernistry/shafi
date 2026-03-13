from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_challenge.eval.sources import PdfPageTextProvider

from score_page_benchmark import CaseScore, build_report, build_scores, load_benchmark

JsonDict = dict[str, object]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _dedupe(items: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def _page_sort_key(page_id: str) -> tuple[str, int]:
    doc_id, _, page_text = page_id.rpartition("_")
    if doc_id and page_text.isdigit():
        return doc_id, int(page_text)
    return page_id, 0


def _summary_from_scores(scores: Sequence[CaseScore]) -> JsonDict:
    typed_scores = list(scores)
    if not typed_scores:
        return {
            "case_count": 0,
            "mean_f_beta": 0.0,
            "gold_hit_rate": 0.0,
            "orphan_case_rate": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
        }

    gold_hit_rate = sum(1 for score in typed_scores if set(score.predicted_pages).intersection(score.gold_pages)) / len(
        typed_scores
    )
    orphan_case_rate = sum(1 for score in typed_scores if score.orphan_pages) / len(typed_scores)

    precisions: list[float] = []
    recalls: list[float] = []
    for score in typed_scores:
        predicted = set(score.predicted_pages)
        gold = set(score.gold_pages)
        if not predicted:
            precisions.append(0.0)
        else:
            precisions.append(len(predicted.intersection(gold)) / len(predicted))
        if not gold:
            recalls.append(1.0)
        else:
            recalls.append(len(predicted.intersection(gold)) / len(gold))

    return {
        "case_count": len(typed_scores),
        "mean_f_beta": sum(score.f_beta for score in typed_scores) / len(typed_scores),
        "gold_hit_rate": gold_hit_rate,
        "orphan_case_rate": orphan_case_rate,
        "mean_precision": sum(precisions) / len(precisions),
        "mean_recall": sum(recalls) / len(recalls),
    }


def _load_stub_scores(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = _load_json(path)
    cases = _coerce_dict_list(payload.get("cases"))
    out: dict[str, dict[str, float]] = {}
    for case in cases:
        qid = str(case.get("question_id") or case.get("qid") or "").strip()
        if not qid:
            continue
        scores_obj = case.get("scores")
        if not isinstance(scores_obj, dict):
            continue
        page_scores: dict[str, float] = {}
        typed_scores_obj = cast("dict[object, object]", scores_obj)
        for raw_page_id, raw_score in typed_scores_obj.items():
            page_id = str(raw_page_id).strip()
            if not page_id:
                continue
            if not isinstance(raw_score, (int, float, str)):
                continue
            try:
                page_scores[page_id] = float(raw_score)
            except ValueError:
                continue
        out[qid] = page_scores
    return out


def _candidate_pages(record: JsonDict, *, max_candidate_pages: int | None) -> list[str]:
    pages: list[str] = []
    for key in ("used_pages", "context_pages", "retrieved_pages"):
        pages.extend(_coerce_str_list(record.get(key)))
    deduped = _dedupe(pages)
    if max_candidate_pages is None:
        return deduped
    return deduped[: max(0, max_candidate_pages)]


def _prediction_payload(label: str, cases: list[JsonDict]) -> JsonDict:
    return {
        "label": label,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "cases": cases,
    }


def _resolve_api_key(explicit_api_key: str | None, api_key_env: str) -> str | None:
    if explicit_api_key:
        return explicit_api_key.strip() or None
    env_names = [api_key_env, "COHERE_API_KEY", "CO_API_KEY"]
    for env_name in env_names:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


async def _cohere_score_documents(
    *,
    query: str,
    documents: list[str],
    model: str,
    api_key: str,
    timeout: float,
) -> list[float]:
    import cohere

    client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout)
    response = await client.rerank(model=model, query=query, documents=documents, top_n=len(documents))
    results_obj = getattr(response, "results", None)
    if not isinstance(results_obj, list):
        raise ValueError("Cohere response missing results")
    scores: list[float] = [0.0] * len(documents)
    for item in cast("list[object]", results_obj):
        index_obj = getattr(item, "index", None)
        score_obj = getattr(item, "relevance_score", None)
        if not isinstance(index_obj, int) or not (0 <= index_obj < len(documents)):
            raise ValueError("Invalid Cohere result index")
        if not isinstance(score_obj, (int, float, str)):
            raise ValueError("Invalid Cohere result score")
        scores[index_obj] = float(score_obj)
    return scores


async def _score_case_pages(
    *,
    query: str,
    candidate_pages: list[str],
    pdf_provider: PdfPageTextProvider,
    model: str,
    api_key: str | None,
    timeout: float,
    stub_scores: dict[str, float] | None,
) -> tuple[list[JsonDict], list[str]]:
    page_texts: list[tuple[str, str]] = []
    missing_text_pages: list[str] = []
    for page_id in candidate_pages:
        doc_id, _, page_text = page_id.rpartition("_")
        if not doc_id or not page_text.isdigit():
            continue
        text = pdf_provider.get_page_text(doc_id=doc_id, page=int(page_text))
        if not text:
            missing_text_pages.append(page_id)
            continue
        page_texts.append((page_id, text))

    if not page_texts:
        return [], missing_text_pages

    if stub_scores is not None:
        scores = [float(stub_scores.get(page_id, 0.0)) for page_id, _ in page_texts]
    else:
        if api_key is None:
            raise RuntimeError("Cohere API key is required when stub scores are not provided")
        scores = await _cohere_score_documents(
            query=query,
            documents=[text for _, text in page_texts],
            model=model,
            api_key=api_key,
            timeout=timeout,
        )

    scored_pages: list[JsonDict] = [
        {
            "page_id": page_id,
            "score": score,
        }
        for (page_id, _text), score in zip(page_texts, scores, strict=True)
    ]
    scored_pages.sort(key=lambda item: (-cast("float", item["score"]), _page_sort_key(str(item["page_id"]))))
    return scored_pages, missing_text_pages


def _render_markdown(payload: JsonDict, *, baseline_report: str, cohere_report: str | None) -> str:
    summary = cast("JsonDict", payload.get("summary") or {})
    cases = _coerce_dict_list(payload.get("cases"))
    lines = [
        "# Cohere Fast Page Falsifier",
        "",
        f"- status: `{payload.get('status')}`",
        f"- model: `{payload.get('model')}`",
        f"- submission_policy: `{payload.get('submission_policy')}`",
        f"- source_miss_pack: `{payload.get('source_miss_pack')}`",
        f"- source_page_trace_ledger: `{payload.get('source_page_trace_ledger')}`",
        f"- source_benchmark: `{payload.get('source_benchmark')}`",
        "",
        "## Summary",
        "",
        f"- verdict: `{summary.get('verdict')}`",
        f"- continue_local_page_reranker: `{summary.get('continue_local_page_reranker')}`",
        f"- case_count: `{summary.get('case_count')}`",
        f"- candidate_contains_gold_rate: `{summary.get('candidate_contains_gold_rate')}`",
        f"- candidate_undercoverage_rate: `{summary.get('candidate_undercoverage_rate')}`",
        f"- skipped_case_count: `{summary.get('skipped_case_count')}`",
    ]
    if summary.get("blocked_reason"):
        lines.append(f"- blocked_reason: `{summary.get('blocked_reason')}`")
    baseline_summary = cast("JsonDict", summary.get("baseline") or {})
    lines.extend(
        [
            "",
            "## Baseline Used-Page Score",
            "",
            f"- mean_f_beta: `{baseline_summary.get('mean_f_beta')}`",
            f"- gold_hit_rate: `{baseline_summary.get('gold_hit_rate')}`",
            f"- orphan_case_rate: `{baseline_summary.get('orphan_case_rate')}`",
            f"- mean_precision: `{baseline_summary.get('mean_precision')}`",
            f"- mean_recall: `{baseline_summary.get('mean_recall')}`",
        ]
    )
    cohere_summary = cast("JsonDict", summary.get("cohere") or {})
    if cohere_summary:
        lines.extend(
            [
                "",
                "## Cohere Score",
                "",
                f"- mean_f_beta: `{cohere_summary.get('mean_f_beta')}`",
                f"- gold_hit_rate: `{cohere_summary.get('gold_hit_rate')}`",
                f"- orphan_case_rate: `{cohere_summary.get('orphan_case_rate')}`",
                f"- mean_precision: `{cohere_summary.get('mean_precision')}`",
                f"- mean_recall: `{cohere_summary.get('mean_recall')}`",
                f"- delta_f_beta: `{summary.get('delta_f_beta')}`",
                f"- delta_gold_hit_rate: `{summary.get('delta_gold_hit_rate')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| qid | candidates | candidate_has_gold | baseline_pages | cohere_pages |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("question_id") or "")[:12],
                    str(case.get("candidate_count") or 0),
                    str(case.get("candidate_contains_gold") or False),
                    ", ".join(_coerce_str_list(case.get("baseline_page_ids"))),
                    ", ".join(_coerce_str_list(case.get("cohere_page_ids"))),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Baseline Benchmark Report", "", baseline_report.strip(), ""])
    if cohere_report is not None:
        lines.extend(["## Cohere Benchmark Report", "", cohere_report.strip(), ""])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Falsify a learned page scorer on the bounded miss pack.")
    parser.add_argument("--miss-pack", type=Path, required=True)
    parser.add_argument("--page-trace-ledger", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--dataset-documents", type=Path, required=True)
    parser.add_argument("--model", default="rerank-v4.0-fast")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="COHERE_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-candidate-pages", type=int, default=0)
    parser.add_argument("--top-pages", type=int, default=0)
    parser.add_argument("--stub-scores-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-baseline-predictions", type=Path, default=None)
    parser.add_argument("--out-cohere-predictions", type=Path, default=None)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> JsonDict:
    miss_pack = _load_json(args.miss_pack)
    ledger = _load_json(args.page_trace_ledger)
    benchmark_cases = load_benchmark(args.benchmark)
    benchmark_by_qid = {case.question_id: case for case in benchmark_cases}
    miss_pack_cases = _coerce_dict_list(miss_pack.get("cases"))
    ledger_by_qid = {
        str(record.get("qid") or "").strip(): record for record in _coerce_dict_list(ledger.get("records")) if str(record.get("qid") or "").strip()
    }
    stub_scores = _load_stub_scores(args.stub_scores_json)

    api_key = _resolve_api_key(args.api_key, args.api_key_env)
    blocked_reason = ""
    if api_key is None and args.stub_scores_json is None:
        blocked_reason = "missing_cohere_api_key"

    pdf_provider = PdfPageTextProvider(args.dataset_documents, max_chars_per_page=4000)
    try:
        baseline_prediction_cases: list[JsonDict] = []
        cohere_prediction_cases: list[JsonDict] = []
        case_rows: list[JsonDict] = []
        scored_qids: list[str] = []
        skipped_qids: list[str] = []

        max_candidate_pages = args.max_candidate_pages if int(args.max_candidate_pages) > 0 else None

        for miss_case in miss_pack_cases:
            qid = str(miss_case.get("qid") or "").strip()
            if not qid:
                continue
            ledger_record = ledger_by_qid.get(qid)
            benchmark_case = benchmark_by_qid.get(qid)
            if ledger_record is None or benchmark_case is None:
                skipped_qids.append(qid)
                continue

            candidate_pages = _candidate_pages(ledger_record, max_candidate_pages=max_candidate_pages)
            baseline_page_ids = _coerce_str_list(miss_case.get("used_pages"))
            gold_page_ids = _coerce_str_list(miss_case.get("gold_pages"))
            question = str(ledger_record.get("question") or "").strip()
            candidate_contains_gold = bool(set(candidate_pages).intersection(gold_page_ids))
            top_pages = int(args.top_pages) if int(args.top_pages) > 0 else max(1, len(baseline_page_ids))

            baseline_prediction_cases.append(
                {
                    "question_id": qid,
                    "predicted_page_ids": baseline_page_ids,
                    "answer": "",
                }
            )

            scored_pages: list[dict[str, object]] = []
            missing_text_pages: list[str] = []
            if blocked_reason:
                cohere_page_ids: list[str] = []
            else:
                scored_pages, missing_text_pages = await _score_case_pages(
                    query=question,
                    candidate_pages=candidate_pages,
                    pdf_provider=pdf_provider,
                    model=args.model,
                    api_key=api_key,
                    timeout=float(args.timeout),
                    stub_scores=stub_scores.get(qid) if stub_scores else None,
                )
                cohere_page_ids = [str(item.get("page_id") or "") for item in scored_pages[:top_pages]]
                cohere_prediction_cases.append(
                    {
                        "question_id": qid,
                        "predicted_page_ids": cohere_page_ids,
                        "answer": "",
                    }
                )
                scored_qids.append(qid)

            case_rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "miss_family": str(miss_case.get("miss_family") or ""),
                    "candidate_page_ids": candidate_pages,
                    "candidate_count": len(candidate_pages),
                    "candidate_contains_gold": candidate_contains_gold,
                    "gold_page_ids": gold_page_ids,
                    "baseline_page_ids": baseline_page_ids,
                    "cohere_page_ids": cohere_page_ids,
                    "missing_candidate_page_ids": sorted(set(gold_page_ids).difference(candidate_pages)),
                    "missing_text_page_ids": missing_text_pages,
                    "top_scored_pages": scored_pages[:5],
                }
            )

        include_qids = [str(case.get("question_id") or "") for case in case_rows if str(case.get("question_id") or "")]
        baseline_scores = build_scores(
            cases=[benchmark_by_qid[qid] for qid in include_qids if qid in benchmark_by_qid],
            eval_cases={
                str(case.get("question_id") or ""): {
                    "question_id": str(case.get("question_id") or ""),
                    "answer": str(case.get("answer") or ""),
                    "telemetry": {"used_page_ids": _coerce_str_list(case.get("predicted_page_ids"))},
                }
                for case in baseline_prediction_cases
            },
            beta=2.5,
        )
        baseline_summary = _summary_from_scores(baseline_scores)
        baseline_report = build_report(scores=baseline_scores, beta=2.5)

        if blocked_reason:
            cohere_scores: list[CaseScore] = []
            cohere_summary: JsonDict | None = None
            cohere_report: str | None = None
            delta_f_beta = None
            delta_gold_hit_rate = None
            verdict = "blocked_missing_api_key"
            continue_local_page_reranker = None
        else:
            cohere_scores = build_scores(
                cases=[benchmark_by_qid[qid] for qid in scored_qids if qid in benchmark_by_qid],
                eval_cases={
                    str(case.get("question_id") or ""): {
                        "question_id": str(case.get("question_id") or ""),
                        "answer": str(case.get("answer") or ""),
                        "telemetry": {"used_page_ids": _coerce_str_list(case.get("predicted_page_ids"))},
                    }
                    for case in cohere_prediction_cases
                },
                beta=2.5,
            )
            cohere_summary = _summary_from_scores(cohere_scores)
            cohere_report = build_report(scores=cohere_scores, beta=2.5)
            delta_f_beta = float(cast("float", cohere_summary["mean_f_beta"])) - float(
                cast("float", baseline_summary["mean_f_beta"])
            )
            delta_gold_hit_rate = float(cast("float", cohere_summary["gold_hit_rate"])) - float(
                cast("float", baseline_summary["gold_hit_rate"])
            )
            continue_local_page_reranker = delta_f_beta > 0.0 and delta_gold_hit_rate >= 0.0
            verdict = "continue_local_page_reranker" if continue_local_page_reranker else "deprioritize"

        candidate_contains_gold_rate = (
            sum(1 for case in case_rows if bool(case.get("candidate_contains_gold"))) / len(case_rows) if case_rows else 0.0
        )
        payload: JsonDict = {
            "status": "blocked" if blocked_reason else "complete",
            "model": args.model,
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            "source_miss_pack": str(args.miss_pack),
            "source_page_trace_ledger": str(args.page_trace_ledger),
            "source_benchmark": str(args.benchmark),
            "source_dataset_documents": str(args.dataset_documents),
            "summary": {
                "verdict": verdict,
                "continue_local_page_reranker": continue_local_page_reranker,
                "blocked_reason": blocked_reason,
                "case_count": len(case_rows),
                "skipped_case_count": len(skipped_qids),
                "candidate_contains_gold_rate": candidate_contains_gold_rate,
                "candidate_undercoverage_rate": 1.0 - candidate_contains_gold_rate,
                "baseline": baseline_summary,
                "cohere": cohere_summary,
                "delta_f_beta": delta_f_beta,
                "delta_gold_hit_rate": delta_gold_hit_rate,
                "scored_qids": scored_qids,
                "skipped_qids": skipped_qids,
            },
            "cases": case_rows,
        }

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            _render_markdown(payload, baseline_report=baseline_report, cohere_report=cohere_report) + "\n",
            encoding="utf-8",
        )
        if args.out_baseline_predictions is not None:
            args.out_baseline_predictions.parent.mkdir(parents=True, exist_ok=True)
            args.out_baseline_predictions.write_text(
                json.dumps(_prediction_payload("miss_pack_baseline", baseline_prediction_cases), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.out_cohere_predictions is not None and not blocked_reason:
            args.out_cohere_predictions.parent.mkdir(parents=True, exist_ok=True)
            args.out_cohere_predictions.write_text(
                json.dumps(_prediction_payload("miss_pack_cohere", cohere_prediction_cases), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return payload
    finally:
        pdf_provider.close()


def main() -> int:
    args = parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
