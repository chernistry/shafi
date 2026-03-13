from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

try:
    from build_counterfactual_candidate import _build_preflight, _load_json_dict, _load_json_list
    from build_embedding_doc_family_candidate import (
        _load_raw_results_by_qid,
        _load_submission_answers,
        _page_ids_from_telemetry,
        _patch_raw_result,
        _patch_submission_answer,
        _question_text,
        _source_doc_ids,
    )
    from build_local_page_reranker_candidate import (
        QidSelection,
        _baseline_doc_pages,
        _candidate_pages_for_doc,
        _deepcopy_json,
        _qid_set,
        _render_markdown,
        _select_pages,
    )
    from probe_local_embedding_relevance import _extract_page_text
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_counterfactual_candidate import _build_preflight, _load_json_dict, _load_json_list
    from scripts.build_embedding_doc_family_candidate import (
        _load_raw_results_by_qid,
        _load_submission_answers,
        _page_ids_from_telemetry,
        _patch_raw_result,
        _patch_submission_answer,
        _question_text,
        _source_doc_ids,
    )
    from scripts.build_local_page_reranker_candidate import (
        QidSelection,
        _baseline_doc_pages,
        _candidate_pages_for_doc,
        _deepcopy_json,
        _qid_set,
        _render_markdown,
        _select_pages,
    )
    from scripts.probe_local_embedding_relevance import _extract_page_text

from rag_challenge.core.local_late_interaction_reranker import LocalLateInteractionReranker

JsonDict = dict[str, Any]


def _build_candidate(args: argparse.Namespace) -> tuple[list[QidSelection], JsonDict, list[JsonDict], JsonDict]:
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

    reranker = LocalLateInteractionReranker(
        model_name=args.model,
        max_chars=args.max_chars,
        max_query_chars=args.max_query_chars,
    )
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
        baseline_doc_pages = _baseline_doc_pages(baseline_page_ids)
        for doc_id in doc_ids:
            candidate_page_ids.extend(
                _candidate_pages_for_doc(
                    doc_id=doc_id,
                    baseline_pages=baseline_doc_pages.get(doc_id, []),
                    dataset_dir=dataset_dir,
                    page_count_cache=page_count_cache,
                    include_page_one=args.include_page_one,
                    include_page_two=args.include_page_two,
                    include_last_page=args.include_last_page,
                    neighbor_radius=args.neighbor_radius,
                    max_pages_per_doc=args.max_pages_per_doc,
                )
            )
        page_pairs: list[tuple[str, str]] = []
        for page_id in candidate_page_ids:
            try:
                text = _extract_page_text(page_id, dataset_dir=dataset_dir, cache=text_cache)
            except (FileNotFoundError, ValueError):
                continue
            if text:
                page_pairs.append((page_id, text))
        if not page_pairs:
            raise ValueError(f"No candidate pages with text for {qid}")
        scored_pages = reranker.score_pages(query=question, pages=page_pairs)
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
                candidate_page_ids=[page_id for page_id, _ in page_pairs],
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
            "kind": "local_late_interaction_reranker",
            "model": reranker.model_name,
            "page_source": args.page_source,
            "per_doc_pages": args.per_doc_pages,
            "extra_global_pages": args.extra_global_pages,
            "include_page_one": args.include_page_one,
            "include_page_two": args.include_page_two,
            "include_last_page": args.include_last_page,
            "neighbor_radius": args.neighbor_radius,
            "max_pages_per_doc": args.max_pages_per_doc,
        },
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    return selections, merged_submission_payload, merged_raw_results_payload, preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a support-only candidate by reranking bounded page candidates with a local late-interaction model."
    )
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--dataset-documents", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="answerdotai/answerai-colbert-small-v1")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--max-query-chars", type=int, default=1200)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qids-file", type=Path)
    parser.add_argument("--page-source", choices=("retrieved", "context", "used"), default="retrieved")
    parser.add_argument("--include-page-one", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-page-two", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-last-page", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-pages-per-doc", type=int, default=8)
    parser.add_argument("--per-doc-pages", type=int, default=1)
    parser.add_argument("--extra-global-pages", type=int, default=0)
    parser.add_argument("--report-top-k", type=int, default=5)
    parser.add_argument("--out-submission", type=Path, required=True)
    parser.add_argument("--out-raw-results", type=Path, required=True)
    parser.add_argument("--out-preflight", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selections, submission_payload, raw_results_payload, preflight_payload = _build_candidate(args)

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
