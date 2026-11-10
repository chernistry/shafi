from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import fitz
import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


JsonDict = dict[str, object]


@dataclass(frozen=True)
class CandidatePage:
    page_id: str
    text: str
    is_gold: bool


@dataclass(frozen=True)
class CaseProbeResult:
    question_id: str
    question: str
    gold_page_ids: list[str]
    distractor_page_ids: list[str]
    candidate_count: int
    gold_top1: bool
    gold_top3: bool
    best_gold_rank: int
    best_gold_score: float
    best_distractor_score: float
    gold_margin: float
    top_page_id: str


@dataclass(frozen=True)
class ModelProbeSummary:
    model: str
    evaluated_cases: int
    skipped_cases: int
    gold_top1_rate: float
    gold_top3_rate: float
    mean_best_gold_rank: float
    mean_gold_margin: float
    median_gold_margin: float
    failed_top1_question_ids: list[str]
    skipped_question_ids: list[str]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_raw_results(path: Path) -> dict[str, JsonDict]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, list):
        raise ValueError(f"Expected raw results list in {path}")
    payload = cast("list[object]", payload_obj)
    out: dict[str, JsonDict] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        entry = cast("JsonDict", raw)
        telemetry_obj = entry.get("telemetry")
        if not isinstance(telemetry_obj, dict):
            continue
        telemetry = cast("JsonDict", telemetry_obj)
        question_id = str(telemetry.get("question_id") or "").strip()
        if not question_id:
            continue
        out[question_id] = entry
    return out


def _load_questions(path: Path) -> dict[str, str]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, list):
        raise ValueError(f"Expected questions list in {path}")
    payload = cast("list[object]", payload_obj)
    out: dict[str, str] = {}
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            continue
        item = cast("JsonDict", raw_item)
        question_id = str(item.get("id") or "").strip()
        if not question_id:
            continue
        out[question_id] = str(item.get("question") or "").strip()
    return out


def _parse_page_id(page_id: str) -> tuple[str, int]:
    doc_id, _, page_text = page_id.rpartition("_")
    if not doc_id or not page_text.isdigit():
        raise ValueError(f"Invalid page id: {page_id}")
    page_number = int(page_text)
    if page_number < 1:
        raise ValueError(f"Invalid page number in page id: {page_id}")
    return doc_id, page_number


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _load_benchmark_cases(path: Path, *, trust_tier: str, question_ids: set[str] | None) -> list[JsonDict]:
    payload = _load_json(path)
    cases_obj = payload.get("cases")
    if not isinstance(cases_obj, list):
        raise ValueError(f"Expected 'cases' list in {path}")
    out: list[JsonDict] = []
    for raw_case in cast("list[object]", cases_obj):
        if not isinstance(raw_case, dict):
            continue
        case = cast("JsonDict", raw_case)
        question_id = str(case.get("question_id") or "").strip()
        if not question_id:
            continue
        if question_ids is not None and question_id not in question_ids:
            continue
        if str(case.get("trust_tier") or "").strip().lower() != trust_tier:
            continue
        out.append(case)
    return out


def _load_scaffold_cases(
    path: Path,
    *,
    question_ids: set[str] | None,
    manual_verdicts: set[str] | None,
    failure_classes: set[str] | None,
    max_cases: int | None,
) -> list[JsonDict]:
    payload = _load_json(path)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Expected 'records' list in {path}")
    out: list[JsonDict] = []
    for raw_case in cast("list[object]", records_obj):
        if not isinstance(raw_case, dict):
            continue
        case = cast("JsonDict", raw_case)
        question_id = str(case.get("question_id") or "").strip()
        if not question_id:
            continue
        if question_ids is not None and question_id not in question_ids:
            continue
        verdict = str(case.get("manual_verdict") or "").strip().lower()
        if manual_verdicts is not None and verdict not in manual_verdicts:
            continue
        failure_class = str(case.get("failure_class") or "").strip()
        if failure_classes is not None and failure_class not in failure_classes:
            continue
        gold_page_ids = _coerce_str_list(case.get("minimal_required_support_pages"))
        if not gold_page_ids:
            continue
        out.append(
            {
                "question_id": question_id,
                "gold_page_ids": gold_page_ids,
                "source_manual_verdict": verdict,
                "source_failure_class": failure_class,
            }
        )
        if max_cases is not None and len(out) >= max_cases:
            break
    return out


def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("Embedding dimensions do not match")
    dot = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for a, b in zip(lhs, rhs, strict=True):
        dot += a * b
        lhs_norm += a * a
        rhs_norm += b * b
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(lhs_norm) * math.sqrt(rhs_norm))


def _pick_distractor_page_ids(
    *,
    gold_page_ids: list[str],
    raw_result: JsonDict,
    max_distractors: int,
) -> list[str]:
    telemetry_obj = raw_result.get("telemetry")
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    candidates: list[str] = []
    for key in ("used_page_ids", "context_page_ids", "retrieved_page_ids"):
        for page_id in _coerce_str_list(telemetry.get(key)):
            if page_id not in candidates:
                candidates.append(page_id)
    gold_set = set(gold_page_ids)
    out: list[str] = []
    for page_id in candidates:
        if page_id in gold_set:
            continue
        out.append(page_id)
        if len(out) >= max_distractors:
            break
    return out


def _same_doc_neighbor_page_ids(
    *,
    gold_page_ids: list[str],
    dataset_dir: Path,
    page_count_cache: dict[str, int],
) -> list[str]:
    gold_set = set(gold_page_ids)
    out: list[str] = []
    seen: set[str] = set()
    for gold_page_id in gold_page_ids:
        doc_id, page_number = _parse_page_id(gold_page_id)
        page_count = page_count_cache.get(doc_id)
        if page_count is None:
            pdf_path = dataset_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                continue
            doc = cast("Any", fitz.open(str(pdf_path)))
            try:
                page_count = int(doc.page_count)
            finally:
                doc.close()
            page_count_cache[doc_id] = page_count
        for candidate_page in (page_number - 1, page_number + 1):
            if candidate_page < 1 or candidate_page > page_count:
                continue
            candidate_page_id = f"{doc_id}_{candidate_page}"
            if candidate_page_id in gold_set or candidate_page_id in seen:
                continue
            seen.add(candidate_page_id)
            out.append(candidate_page_id)
    return out


def _extract_page_text(page_id: str, *, dataset_dir: Path, cache: dict[str, str]) -> str:
    cached = cache.get(page_id)
    if cached is not None:
        return cached
    doc_id, page_number = _parse_page_id(page_id)
    pdf_path = dataset_dir / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF for page id {page_id}: {pdf_path}")
    doc = cast("Any", fitz.open(str(pdf_path)))
    try:
        page_count = int(doc.page_count)
        if page_number > page_count:
            raise ValueError(f"Page {page_number} out of range for {pdf_path}")
        page = doc.load_page(page_number - 1)
        text_obj = cast("str", page.get_text("text"))
    finally:
        doc.close()
    text = text_obj
    normalized = " ".join(text.split())
    cache[page_id] = normalized
    return normalized


async def _embed_texts(base_url: str, *, model: str, texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        response = await client.post("/api/embed", json={"model": model, "input": texts})
        response.raise_for_status()
        payload = response.json()
        embeddings_obj = payload.get("embeddings")
        if isinstance(embeddings_obj, list) and embeddings_obj and isinstance(embeddings_obj[0], list):
            return cast("list[list[float]]", embeddings_obj)
    raise ValueError("Unexpected Ollama embed response shape")


def _score_case(
    *,
    question_id: str,
    question: str,
    candidates: list[CandidatePage],
    query_embedding: list[float],
    page_embeddings: list[list[float]],
) -> CaseProbeResult:
    scored = [
        (candidate, _cosine_similarity(query_embedding, embedding))
        for candidate, embedding in zip(candidates, page_embeddings, strict=True)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    gold_rank = 0
    best_gold_score = float("-inf")
    best_distractor_score = float("-inf")
    for index, (candidate, score) in enumerate(scored, start=1):
        if candidate.is_gold and gold_rank == 0:
            gold_rank = index
        if candidate.is_gold:
            best_gold_score = max(best_gold_score, score)
        else:
            best_distractor_score = max(best_distractor_score, score)
    if gold_rank == 0:
        raise ValueError(f"No gold candidate scored for {question_id}")
    if best_distractor_score == float("-inf"):
        best_distractor_score = -1.0
    top_candidate = scored[0][0]
    gold_page_ids = [candidate.page_id for candidate in candidates if candidate.is_gold]
    distractor_page_ids = [candidate.page_id for candidate in candidates if not candidate.is_gold]
    return CaseProbeResult(
        question_id=question_id,
        question=question,
        gold_page_ids=gold_page_ids,
        distractor_page_ids=distractor_page_ids,
        candidate_count=len(candidates),
        gold_top1=gold_rank == 1,
        gold_top3=gold_rank <= 3,
        best_gold_rank=gold_rank,
        best_gold_score=best_gold_score,
        best_distractor_score=best_distractor_score,
        gold_margin=best_gold_score - best_distractor_score,
        top_page_id=top_candidate.page_id,
    )


async def _probe_model(
    *,
    model: str,
    cases: list[JsonDict],
    question_map: dict[str, str],
    raw_results_by_qid: dict[str, JsonDict],
    dataset_dir: Path,
    base_url: str,
    max_distractors: int,
) -> tuple[ModelProbeSummary, list[CaseProbeResult]]:
    page_cache: dict[str, str] = {}
    page_count_cache: dict[str, int] = {}
    case_results: list[CaseProbeResult] = []
    skipped_question_ids: list[str] = []
    for case in cases:
        question_id = str(case.get("question_id") or "").strip()
        question = question_map.get(question_id, "").strip()
        if not question:
            raw_result = raw_results_by_qid.get(question_id, {})
            case_obj = raw_result.get("case")
            if isinstance(case_obj, dict):
                case_dict = cast("JsonDict", case_obj)
                question = str(case_dict.get("question") or "").strip()
        if not question:
            continue
        gold_page_ids = _coerce_str_list(case.get("gold_page_ids"))
        if not gold_page_ids:
            continue
        raw_result = raw_results_by_qid.get(question_id)
        if raw_result is None:
            continue
        distractor_page_ids = _pick_distractor_page_ids(
            gold_page_ids=gold_page_ids,
            raw_result=raw_result,
            max_distractors=max_distractors,
        )
        if len(distractor_page_ids) < max_distractors:
            for candidate_page_id in _same_doc_neighbor_page_ids(
                gold_page_ids=gold_page_ids,
                dataset_dir=dataset_dir,
                page_count_cache=page_count_cache,
            ):
                if candidate_page_id in distractor_page_ids:
                    continue
                distractor_page_ids.append(candidate_page_id)
                if len(distractor_page_ids) >= max_distractors:
                    break
        if not distractor_page_ids:
            continue
        candidates: list[CandidatePage] = []
        try:
            for page_id in gold_page_ids:
                text = _extract_page_text(page_id, dataset_dir=dataset_dir, cache=page_cache)
                if text:
                    candidates.append(CandidatePage(page_id=page_id, text=text, is_gold=True))
            for page_id in distractor_page_ids:
                text = _extract_page_text(page_id, dataset_dir=dataset_dir, cache=page_cache)
                if text:
                    candidates.append(CandidatePage(page_id=page_id, text=text, is_gold=False))
        except (FileNotFoundError, ValueError):
            skipped_question_ids.append(question_id)
            continue
        if len(candidates) <= len(gold_page_ids):
            continue
        embeddings = await _embed_texts(base_url, model=model, texts=[question, *[candidate.text for candidate in candidates]])
        query_embedding = embeddings[0]
        page_embeddings = embeddings[1:]
        case_results.append(
            _score_case(
                question_id=question_id,
                question=question,
                candidates=candidates,
                query_embedding=query_embedding,
                page_embeddings=page_embeddings,
            )
        )

    if not case_results:
        raise ValueError(f"No evaluable cases for model {model}")
    top1_count = sum(1 for result in case_results if result.gold_top1)
    top3_count = sum(1 for result in case_results if result.gold_top3)
    mean_rank = sum(result.best_gold_rank for result in case_results) / len(case_results)
    margins = sorted(result.gold_margin for result in case_results)
    median_margin = margins[len(margins) // 2] if len(margins) % 2 == 1 else (margins[len(margins) // 2 - 1] + margins[len(margins) // 2]) / 2.0
    summary = ModelProbeSummary(
        model=model,
        evaluated_cases=len(case_results),
        skipped_cases=len(skipped_question_ids),
        gold_top1_rate=top1_count / len(case_results),
        gold_top3_rate=top3_count / len(case_results),
        mean_best_gold_rank=mean_rank,
        mean_gold_margin=sum(result.gold_margin for result in case_results) / len(case_results),
        median_gold_margin=median_margin,
        failed_top1_question_ids=[result.question_id for result in case_results if not result.gold_top1],
        skipped_question_ids=skipped_question_ids,
    )
    return summary, case_results


def _render_report(
    *,
    case_source: str,
    case_source_path: Path,
    raw_results_path: Path,
    summaries: list[ModelProbeSummary],
    case_results_by_model: dict[str, list[CaseProbeResult]],
) -> str:
    ranked = sorted(
        summaries,
        key=lambda item: (
            -item.gold_top1_rate,
            -item.gold_top3_rate,
            -item.mean_gold_margin,
            item.mean_best_gold_rank,
            item.model,
        ),
    )
    lines = [
        "# Local Embedding Relevance Probe",
        "",
        f"- case_source: `{case_source}`",
        f"- case_source_path: `{case_source_path}`",
        f"- raw_results: `{raw_results_path}`",
        f"- models_compared: `{len(ranked)}`",
        f"- recommended_model: `{ranked[0].model}`",
        "",
        "| Rank | Model | Cases | Top1 | Top3 | Mean gold rank | Mean margin | Median margin |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, summary in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | `{summary.model}` | {summary.evaluated_cases} | "
            f"{summary.gold_top1_rate:.3f} | {summary.gold_top3_rate:.3f} | "
            f"{summary.mean_best_gold_rank:.2f} | {summary.mean_gold_margin:.4f} | {summary.median_gold_margin:.4f} |"
        )
    for summary in ranked:
        case_results = case_results_by_model[summary.model]
        failed = [result for result in case_results if not result.gold_top1][:5]
        lines.extend(
            [
                "",
                f"## {summary.model}",
                "",
                f"- evaluated_cases: `{summary.evaluated_cases}`",
                f"- skipped_cases: `{summary.skipped_cases}`",
                f"- top1_rate: `{summary.gold_top1_rate:.3f}`",
                f"- top3_rate: `{summary.gold_top3_rate:.3f}`",
                f"- mean_gold_margin: `{summary.mean_gold_margin:.4f}`",
            ]
        )
        if failed:
            lines.extend(["", "| QID | Gold rank | Margin | Top page |", "| --- | ---: | ---: | --- |"])
            for result in failed:
                lines.append(
                    f"| `{result.question_id}` | {result.best_gold_rank} | {result.gold_margin:.4f} | `{result.top_page_id}` |"
                )
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> tuple[list[ModelProbeSummary], dict[str, list[CaseProbeResult]]]:
    selected_qids = set(args.question_id or [])
    question_ids = selected_qids if selected_qids else None
    if args.case_source == "benchmark":
        cases = _load_benchmark_cases(args.benchmark, trust_tier=args.trust_tier, question_ids=question_ids)
    else:
        manual_verdicts = {value.strip().lower() for value in args.manual_verdict if value.strip()} if args.manual_verdict else None
        failure_classes = {value.strip() for value in args.failure_class if value.strip()} if args.failure_class else None
        cases = _load_scaffold_cases(
            args.scaffold,
            question_ids=question_ids,
            manual_verdicts=manual_verdicts,
            failure_classes=failure_classes,
            max_cases=args.max_cases,
        )
    question_map = _load_questions(args.questions)
    raw_results_by_qid = _load_raw_results(args.raw_results)
    summaries: list[ModelProbeSummary] = []
    case_results_by_model: dict[str, list[CaseProbeResult]] = {}
    for model in args.model:
        summary, case_results = await _probe_model(
            model=model,
            cases=cases,
            question_map=question_map,
            raw_results_by_qid=raw_results_by_qid,
            dataset_dir=args.dataset_dir,
            base_url=args.base_url,
            max_distractors=args.max_distractors,
        )
        summaries.append(summary)
        case_results_by_model[model] = case_results
    return summaries, case_results_by_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local embedding models on trusted gold-page relevance.")
    parser.add_argument("--model", action="append", required=True, help="Embedding model to evaluate. Repeat for multiple models.")
    parser.add_argument("--case-source", choices=("benchmark", "scaffold"), default="benchmark")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "tests/fixtures/internal_hidden_g_benchmark_seed.json")
    parser.add_argument("--scaffold", type=Path, default=ROOT / "platform_runs/warmup/truth_audit_scaffold.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "platform_runs/warmup/questions.json")
    parser.add_argument("--raw-results", type=Path, default=ROOT / "platform_runs/warmup/raw_results_v6_context_seed.json")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/dataset_documents")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--trust-tier", default="trusted")
    parser.add_argument("--manual-verdict", action="append", default=None, help="Optional scaffold manual verdict filter.")
    parser.add_argument("--failure-class", action="append", default=None, help="Optional scaffold failure_class filter.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--question-id", action="append", default=None, help="Optional question id filter. Repeat for multiple.")
    parser.add_argument("--max-distractors", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summaries, case_results_by_model = asyncio.run(_main_async(args))
    report = _render_report(
        case_source=args.case_source,
        case_source_path=args.benchmark if args.case_source == "benchmark" else args.scaffold,
        raw_results_path=args.raw_results,
        summaries=summaries,
        case_results_by_model=case_results_by_model,
    )
    payload: dict[str, Any] = {
        "summaries": [asdict(summary) for summary in summaries],
        "case_results_by_model": {
            model: [asdict(result) for result in results]
            for model, results in case_results_by_model.items()
        },
    }
    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
