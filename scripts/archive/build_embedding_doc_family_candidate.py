from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

try:
    from build_counterfactual_candidate import _build_preflight, _load_json_dict, _load_json_list
    from probe_local_embedding_relevance import _cosine_similarity, _embed_texts, _extract_page_text, _parse_page_id
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_counterfactual_candidate import _build_preflight, _load_json_dict, _load_json_list
    from scripts.probe_local_embedding_relevance import (
        _cosine_similarity,
        _embed_texts,
        _extract_page_text,
        _parse_page_id,
    )

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class PageScore:
    page_id: str
    score: float


@dataclass(frozen=True)
class QidSelection:
    question_id: str
    question: str
    source_doc_ids: list[str]
    baseline_page_ids: list[str]
    selected_page_ids: list[str]
    top_scored_pages: list[PageScore]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _load_submission_answers(path: Path) -> dict[str, JsonDict]:
    payload = _load_json_dict(path)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Expected answers[] in {path}")
    out: dict[str, JsonDict] = {}
    for raw in cast("list[object]", answers_obj):
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_raw_results_by_qid(path: Path) -> dict[str, JsonDict]:
    payload = _load_json_list(path)
    out: dict[str, JsonDict] = {}
    for row in payload:
        case = cast("JsonDict", row.get("case")) if isinstance(row.get("case"), dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _deepcopy_json(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _qid_set(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in args.qid:
        qid = str(raw).strip()
        if qid and qid not in seen:
            out.append(qid)
            seen.add(qid)
    if args.qids_file is not None:
        for line in args.qids_file.read_text(encoding="utf-8").splitlines():
            qid = line.strip()
            if qid and not qid.startswith("#") and qid not in seen:
                out.append(qid)
                seen.add(qid)
    if not out:
        raise ValueError("No QIDs provided")
    return out


def _page_ids_from_telemetry(telemetry: JsonDict, source: str) -> list[str]:
    page_ids = _coerce_str_list(telemetry.get(f"{source}_page_ids"))
    if page_ids:
        return page_ids
    if source != "retrieved":
        return _coerce_str_list(telemetry.get("retrieved_page_ids"))
    return []


def _question_text(raw_result: JsonDict) -> str:
    case = cast("JsonDict", raw_result.get("case")) if isinstance(raw_result.get("case"), dict) else {}
    return str(case.get("question") or "").strip()


def _source_doc_ids(page_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for page_id in page_ids:
        doc_id, _ = _parse_page_id(page_id)
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)
    return out


def _pdf_page_count(pdf_path: Path, cache: dict[str, int]) -> int:
    key = str(pdf_path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    import fitz

    doc = cast("Any", fitz.open(str(pdf_path)))
    try:
        page_count = int(doc.page_count)
    finally:
        doc.close()
    cache[key] = page_count
    return page_count


def _group_retrieved_chunk_pages(page_ids: list[str]) -> list[JsonDict]:
    grouped: dict[str, list[int]] = {}
    for page_id in page_ids:
        doc_id, page_number = _parse_page_id(page_id)
        grouped.setdefault(doc_id, [])
        if page_number not in grouped[doc_id]:
            grouped[doc_id].append(page_number)
    out: list[JsonDict] = []
    for doc_id in sorted(grouped):
        out.append({"doc_id": doc_id, "page_numbers": sorted(grouped[doc_id])})
    return out


def _select_pages(
    *,
    scored_pages: list[PageScore],
    per_doc_pages: int,
    extra_global_pages: int,
) -> list[str]:
    by_doc: dict[str, list[PageScore]] = {}
    for row in scored_pages:
        doc_id, _ = _parse_page_id(row.page_id)
        by_doc.setdefault(doc_id, []).append(row)
    selected: list[str] = []
    seen: set[str] = set()
    for doc_id in sorted(by_doc):
        doc_rows = sorted(by_doc[doc_id], key=lambda item: (-item.score, item.page_id))
        for row in doc_rows[:per_doc_pages]:
            if row.page_id not in seen:
                selected.append(row.page_id)
                seen.add(row.page_id)
    if extra_global_pages > 0:
        for row in sorted(scored_pages, key=lambda item: (-item.score, item.page_id)):
            if row.page_id in seen:
                continue
            selected.append(row.page_id)
            seen.add(row.page_id)
            if len(selected) >= len(by_doc) * per_doc_pages + extra_global_pages:
                break
    return selected


def _patch_submission_answer(answer_record: JsonDict, *, selected_page_ids: list[str]) -> JsonDict:
    cloned = cast("JsonDict", _deepcopy_json(answer_record))
    telemetry = cast("JsonDict", cloned.setdefault("telemetry", {}))
    retrieval = cast("JsonDict", telemetry.setdefault("retrieval", {}))
    retrieval["retrieved_chunk_pages"] = _group_retrieved_chunk_pages(selected_page_ids)
    telemetry["retrieval"] = retrieval
    cloned["telemetry"] = telemetry
    return cloned


def _patch_raw_result(raw_result: JsonDict, *, selected_page_ids: list[str]) -> JsonDict:
    cloned = cast("JsonDict", _deepcopy_json(raw_result))
    telemetry = cast("JsonDict", cloned.setdefault("telemetry", {}))
    for key in ("retrieved_page_ids", "context_page_ids", "cited_page_ids", "used_page_ids"):
        telemetry[key] = list(selected_page_ids)
    telemetry["context_chunk_count"] = len(selected_page_ids)
    cloned["telemetry"] = telemetry
    return cloned


async def _score_qid(
    *,
    question: str,
    page_ids: list[str],
    dataset_dir: Path,
    model: str,
    base_url: str,
    text_cache: dict[str, str],
) -> list[PageScore]:
    candidate_texts: list[str] = []
    candidate_page_ids: list[str] = []
    for page_id in page_ids:
        try:
            text = _extract_page_text(page_id, dataset_dir=dataset_dir, cache=text_cache)
        except (FileNotFoundError, ValueError):
            continue
        if not text:
            continue
        candidate_page_ids.append(page_id)
        candidate_texts.append(text)
    if not candidate_texts:
        return []
    embeddings = await _embed_texts(base_url, model=model, texts=[question, *candidate_texts])
    query_embedding = embeddings[0]
    page_embeddings = embeddings[1:]

    scored = [
        PageScore(page_id=page_id, score=_cosine_similarity(query_embedding, embedding))
        for page_id, embedding in zip(candidate_page_ids, page_embeddings, strict=True)
    ]
    return sorted(scored, key=lambda item: (-item.score, item.page_id))


async def _build_candidate(args: argparse.Namespace) -> tuple[list[QidSelection], JsonDict, list[JsonDict], JsonDict]:
    qids = _qid_set(args)
    baseline_submission_payload = _load_json_dict(args.baseline_submission)
    baseline_preflight_payload = _load_json_dict(args.baseline_preflight)
    baseline_submission = _load_submission_answers(args.baseline_submission)
    baseline_raw_results = _load_raw_results_by_qid(args.baseline_raw_results)
    merged_submission_payload = cast("JsonDict", _deepcopy_json(baseline_submission_payload))
    merged_answers_obj = merged_submission_payload.get("answers")
    if not isinstance(merged_answers_obj, list):
        raise ValueError(f"Expected answers[] in {args.baseline_submission}")
    merged_answers = cast("list[JsonDict]", merged_answers_obj)
    merged_answers_by_qid = {str(row.get("question_id") or ""): row for row in merged_answers}
    merged_raw_results_payload = _load_json_list(args.baseline_raw_results)
    dataset_dir = args.dataset_documents

    text_cache: dict[str, str] = {}
    page_count_cache: dict[str, int] = {}
    selections: list[QidSelection] = []

    for qid in qids:
        raw_result = baseline_raw_results.get(qid)
        answer_record = baseline_submission.get(qid)
        if raw_result is None or answer_record is None:
            raise ValueError(f"Missing baseline records for {qid}")
        telemetry = cast("JsonDict", raw_result.get("telemetry")) if isinstance(raw_result.get("telemetry"), dict) else {}
        baseline_page_ids = _page_ids_from_telemetry(telemetry, args.page_source)
        doc_ids = _source_doc_ids(baseline_page_ids)
        question = _question_text(raw_result)
        if not question:
            raise ValueError(f"Missing question text for {qid}")
        candidate_page_ids: list[str] = []
        for doc_id in doc_ids:
            pdf_path = dataset_dir / f"{doc_id}.pdf"
            if not pdf_path.exists():
                continue
            cached_count = page_count_cache.get(doc_id)
            if cached_count is None:
                cached_count = _pdf_page_count(pdf_path, {})
                page_count_cache[doc_id] = cached_count
            page_count = cached_count if args.max_pages_per_doc <= 0 else min(cached_count, args.max_pages_per_doc)
            candidate_page_ids.extend(f"{doc_id}_{page_number}" for page_number in range(1, page_count + 1))
        scored_pages = await _score_qid(
            question=question,
            page_ids=candidate_page_ids,
            dataset_dir=dataset_dir,
            model=args.model,
            base_url=args.base_url,
            text_cache=text_cache,
        )
        if not scored_pages:
            raise ValueError(f"No scored pages for {qid}")
        selected_page_ids = _select_pages(
            scored_pages=scored_pages,
            per_doc_pages=args.per_doc_pages,
            extra_global_pages=args.extra_global_pages,
        )
        selections.append(
            QidSelection(
                question_id=qid,
                question=question,
                source_doc_ids=doc_ids,
                baseline_page_ids=baseline_page_ids,
                selected_page_ids=selected_page_ids,
                top_scored_pages=scored_pages[: args.report_top_k],
            )
        )
        merged_answers_by_qid[qid].clear()
        merged_answers_by_qid[qid].update(_patch_submission_answer(answer_record, selected_page_ids=selected_page_ids))
        for row in merged_raw_results_payload:
            case = cast("JsonDict", row.get("case")) if isinstance(row.get("case"), dict) else {}
            row_qid = str(case.get("case_id") or case.get("question_id") or "").strip()
            if row_qid == qid:
                row.clear()
                row.update(_patch_raw_result(raw_result, selected_page_ids=selected_page_ids))
                break

    preflight = _build_preflight(
        merged_payload=merged_submission_payload,
        answer_source_preflight=baseline_preflight_payload,
        page_source_preflight=baseline_preflight_payload,
        answer_source_submission=args.baseline_submission,
        page_source_submission=args.baseline_submission,
        allowlisted_qids=set(),
        page_allowlisted_qids={selection.question_id for selection in selections},
    )
    preflight["counterfactual_projection"] = {
        "answer_source_submission": str(args.baseline_submission),
        "page_source_submission": str(args.baseline_submission),
        "page_source_answer_qids": [],
        "page_source_page_qids": [selection.question_id for selection in selections],
        "page_source_policy": {
            "kind": "embedding_doc_family_rerank",
            "model": args.model,
            "page_source": args.page_source,
            "per_doc_pages": args.per_doc_pages,
            "extra_global_pages": args.extra_global_pages,
            "max_pages_per_doc": args.max_pages_per_doc,
        },
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    return selections, merged_submission_payload, merged_raw_results_payload, preflight


def _render_markdown(
    *,
    label: str,
    model: str,
    selections: list[QidSelection],
) -> str:
    lines = [
        "# Embedding Doc-Family Candidate",
        "",
        f"- label: `{label}`",
        f"- model: `{model}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
    ]
    for selection in selections:
        lines.extend(
            [
                f"## {selection.question_id}",
                "",
                f"- question: {selection.question}",
                f"- source_doc_ids: `{selection.source_doc_ids}`",
                f"- baseline_page_ids: `{selection.baseline_page_ids}`",
                f"- selected_page_ids: `{selection.selected_page_ids}`",
                "",
                "| Rank | Page | Score |",
                "| --- | --- | ---: |",
            ]
        )
        for index, row in enumerate(selection.top_scored_pages, start=1):
            lines.append(f"| {index} | `{row.page_id}` | {row.score:.4f} |")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a support-only candidate by reranking pages within baseline doc families using a local embedding model.")
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--dataset-documents", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qids-file", type=Path)
    parser.add_argument("--page-source", choices=("retrieved", "context", "used"), default="retrieved")
    parser.add_argument("--per-doc-pages", type=int, default=1)
    parser.add_argument("--extra-global-pages", type=int, default=0)
    parser.add_argument("--max-pages-per-doc", type=int, default=0)
    parser.add_argument("--report-top-k", type=int, default=5)
    parser.add_argument("--out-submission", type=Path, required=True)
    parser.add_argument("--out-raw-results", type=Path, required=True)
    parser.add_argument("--out-preflight", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selections, submission_payload, raw_results_payload, preflight_payload = asyncio.run(_build_candidate(args))

    for path in (args.out_submission, args.out_raw_results, args.out_preflight, args.out_json, args.out_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.out_submission.write_text(json.dumps(submission_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_raw_results.write_text(json.dumps(raw_results_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_preflight.write_text(json.dumps(preflight_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_json.write_text(
        json.dumps(
            {
                "label": args.label,
                "model": args.model,
                "selection_count": len(selections),
                "selections": [asdict(selection) for selection in selections],
                "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(_render_markdown(label=args.label, model=args.model, selections=selections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
