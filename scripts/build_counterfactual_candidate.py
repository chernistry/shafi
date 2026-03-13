from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from rag_challenge.submission.common import count_submission_sentences

JsonDict = dict[str, Any]


def _as_dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _as_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _load_json_dict(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_json_list(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return _as_dict_list(cast("object", obj))


def _answers_by_id(payload: JsonDict) -> dict[str, JsonDict]:
    answers = _as_dict_list(cast("object", payload.get("answers")))
    out: dict[str, JsonDict] = {}
    for raw in answers:
        qid = str(raw.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _raw_results_by_id(records: list[JsonDict]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for raw in records:
        case = _as_dict(raw.get("case"))
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _deepcopy_json(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _page_count(answer_record: JsonDict) -> int:
    telemetry = _as_dict(answer_record.get("telemetry"))
    retrieval = _as_dict(telemetry.get("retrieval"))
    pages = _as_dict_list(retrieval.get("retrieved_chunk_pages"))
    count = 0
    for raw in pages:
        page_numbers_obj = raw.get("page_numbers")
        if isinstance(page_numbers_obj, list):
            count += len([item for item in cast("list[object]", page_numbers_obj) if isinstance(item, int | float)])
    return count


def _percentile_int(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return int(ordered[idx])


def _qid_allowlist(args: argparse.Namespace) -> set[str]:
    out: set[str] = set()
    for raw in args.page_source_answer_qid:
        text = str(raw).strip()
        if text:
            out.add(text)
    if args.page_source_answer_qids_file is not None:
        for line in args.page_source_answer_qids_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                out.add(text)
    return out


def _page_qid_allowlist(args: argparse.Namespace) -> set[str]:
    out: set[str] = set()
    for raw in args.page_source_page_qid:
        text = str(raw).strip()
        if text:
            out.add(text)
    if args.page_source_page_qids_file is not None:
        for line in args.page_source_page_qids_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                out.add(text)
    return out


def _build_preflight(
    *,
    merged_payload: JsonDict,
    answer_source_preflight: JsonDict,
    page_source_preflight: JsonDict,
    answer_source_submission: Path,
    page_source_submission: Path,
    allowlisted_qids: set[str],
    page_allowlisted_qids: set[str],
) -> JsonDict:
    answers = _answers_by_id(merged_payload)
    answer_type_counts: Counter[str] = Counter()
    null_answer_counts: Counter[str] = Counter()
    empty_pages_counts: Counter[str] = Counter()
    page_counts: list[int] = []
    free_text_char_counts: list[int] = []
    free_text_sentence_counts: list[int] = []
    model_name_empty_count = 0

    answer_source_counts = answer_source_preflight.get("answer_type_counts")
    if not isinstance(answer_source_counts, dict):
        answer_source_counts = {}

    answer_type_by_qid: dict[str, str] = {}
    for qid, raw in answers.items():
        answer_type = ""
        telemetry = _as_dict(raw.get("telemetry"))
        answer_type = str(telemetry.get("answer_type") or "").strip().lower()
        if not answer_type:
            answer_type = "free_text"
        answer_type_by_qid[qid] = answer_type

    for qid, raw in answers.items():
        answer_type = answer_type_by_qid[qid]
        answer_type_counts[answer_type] += 1
        answer_value = raw.get("answer")
        if answer_value is None:
            null_answer_counts[answer_type] += 1
        pages = _page_count(raw)
        page_counts.append(pages)
        if pages == 0:
            empty_pages_counts[answer_type] += 1
        telemetry = _as_dict(raw.get("telemetry"))
        if not str(telemetry.get("model_name") or "").strip():
            model_name_empty_count += 1
        if answer_type == "free_text" and isinstance(answer_value, str):
            free_text_char_counts.append(len(answer_value))
            free_text_sentence_counts.append(count_submission_sentences(answer_value))

    return {
        "phase": page_source_preflight.get("phase") or answer_source_preflight.get("phase"),
        "questions_count": len(answers),
        "answer_type_counts": dict(answer_type_counts),
        "null_answer_counts_by_type": dict(null_answer_counts),
        "empty_retrieved_chunk_pages_counts_by_type": dict(empty_pages_counts),
        "page_count_distribution": {
            "min": min(page_counts, default=0),
            "p50": _percentile_int(page_counts, 0.50),
            "p95": _percentile_int(page_counts, 0.95),
            "max": max(page_counts, default=0),
        },
        "free_text_char_distribution": {
            "min": min(free_text_char_counts, default=0),
            "p50": _percentile_int(free_text_char_counts, 0.50),
            "p95": _percentile_int(free_text_char_counts, 0.95),
            "max": max(free_text_char_counts, default=0),
        },
        "free_text_sentence_distribution": {
            "min": min(free_text_sentence_counts, default=0),
            "p50": _percentile_int(free_text_sentence_counts, 0.50),
            "p95": _percentile_int(free_text_sentence_counts, 0.95),
            "max": max(free_text_sentence_counts, default=0),
        },
        "model_name_empty_count": model_name_empty_count,
        "submission_sha256": "",
        "code_archive_sha256": page_source_preflight.get("code_archive_sha256") or answer_source_preflight.get("code_archive_sha256") or "",
        "questions_sha256": page_source_preflight.get("questions_sha256") or answer_source_preflight.get("questions_sha256") or "",
        "documents_zip_sha256": page_source_preflight.get("documents_zip_sha256") or answer_source_preflight.get("documents_zip_sha256") or "",
        "pdf_count": page_source_preflight.get("pdf_count") or answer_source_preflight.get("pdf_count") or 0,
        "phase_collection_name": page_source_preflight.get("phase_collection_name") or answer_source_preflight.get("phase_collection_name"),
        "qdrant_point_count": page_source_preflight.get("qdrant_point_count") or answer_source_preflight.get("qdrant_point_count"),
        "truth_audit_workbook_path": page_source_preflight.get("truth_audit_workbook_path") or answer_source_preflight.get("truth_audit_workbook_path"),
        "raw_results_path": "",
        "counterfactual_projection": {
            "answer_source_submission": str(answer_source_submission),
            "page_source_submission": str(page_source_submission),
            "page_source_answer_qids": sorted(allowlisted_qids),
            "page_source_page_qids": sorted(page_allowlisted_qids),
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        },
    }


def _merge_records(
    *,
    answer_source_submission: JsonDict,
    answer_source_raw_results: list[JsonDict],
    page_source_submission: JsonDict,
    page_source_raw_results: list[JsonDict],
    allowlisted_qids: set[str],
    page_allowlisted_qids: set[str],
    page_source_pages_default: str,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    answer_submission_by_id = _answers_by_id(answer_source_submission)
    page_submission_by_id = _answers_by_id(page_source_submission)
    answer_raw_by_id = _raw_results_by_id(answer_source_raw_results)
    page_raw_by_id = _raw_results_by_id(page_source_raw_results)

    merged_answers: list[JsonDict] = []
    merged_raw_results: list[JsonDict] = []
    report: JsonDict = {
        "answer_source_count": len(answer_submission_by_id),
        "page_source_count": len(page_submission_by_id),
        "merged_count": 0,
        "answer_changed_count_vs_answer_source": 0,
        "page_projection_changed_count_vs_answer_source": 0,
        "page_source_answer_qids": sorted(allowlisted_qids),
        "page_source_page_qids": sorted(page_allowlisted_qids),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }

    changed_answer_qids: list[str] = []
    changed_page_qids: list[str] = []

    for qid, answer_record in answer_submission_by_id.items():
        page_record = page_submission_by_id.get(qid)
        answer_raw = answer_raw_by_id.get(qid)
        page_raw = page_raw_by_id.get(qid)
        if page_record is None or answer_raw is None or page_raw is None:
            raise ValueError(f"Missing matching records for question_id={qid}")

        use_page_source_answer = qid in allowlisted_qids
        use_page_source_pages = page_source_pages_default == "all" and not page_allowlisted_qids
        if page_allowlisted_qids:
            use_page_source_pages = qid in page_allowlisted_qids
        chosen_answer_record = page_record if use_page_source_answer else answer_record
        chosen_answer_raw = page_raw if use_page_source_answer else answer_raw

        merged_answer_record = cast("JsonDict", _deepcopy_json(chosen_answer_record))
        merged_answer_telemetry = _as_dict(merged_answer_record.get("telemetry"))
        page_telemetry = _as_dict(page_record.get("telemetry"))
        answer_telemetry = _as_dict(answer_record.get("telemetry"))
        answer_retrieval = _as_dict(answer_telemetry.get("retrieval"))
        page_retrieval = _as_dict(page_telemetry.get("retrieval"))
        retrieval_source = page_retrieval if use_page_source_pages else answer_retrieval
        merged_answer_telemetry["retrieval"] = _deepcopy_json(retrieval_source)
        merged_answer_record["telemetry"] = merged_answer_telemetry
        merged_answers.append(merged_answer_record)

        merged_raw_record = cast("JsonDict", _deepcopy_json(chosen_answer_raw))
        merged_raw_telemetry = _as_dict(merged_raw_record.get("telemetry"))
        page_raw_telemetry = _as_dict(page_raw.get("telemetry"))
        answer_raw_telemetry = _as_dict(answer_raw.get("telemetry"))
        page_fields = (
            "retrieved_chunk_ids",
            "retrieved_page_ids",
            "context_chunk_ids",
            "context_page_ids",
            "used_chunk_ids",
            "used_page_ids",
            "must_include_chunk_ids",
            "doc_shortlist",
            "support_shape_flags",
            "support_shape_class",
            "localized_support_chunk_ids",
            "localized_support_page_ids",
        )
        for field in page_fields:
            source_telemetry = page_raw_telemetry if use_page_source_pages else answer_raw_telemetry
            if field in source_telemetry:
                merged_raw_telemetry[field] = _deepcopy_json(source_telemetry[field])
        merged_raw_record["telemetry"] = merged_raw_telemetry
        merged_raw_results.append(merged_raw_record)

        if json.dumps(answer_record.get("answer"), ensure_ascii=False, sort_keys=True) != json.dumps(
            merged_answer_record.get("answer"), ensure_ascii=False, sort_keys=True
        ):
            changed_answer_qids.append(qid)
        base_pages = _as_dict(_as_dict(answer_record.get("telemetry")).get("retrieval")).get("retrieved_chunk_pages", [])
        merged_pages = _as_dict(_as_dict(merged_answer_record.get("telemetry")).get("retrieval")).get("retrieved_chunk_pages", [])
        if json.dumps(base_pages, ensure_ascii=False, sort_keys=True) != json.dumps(merged_pages, ensure_ascii=False, sort_keys=True):
            changed_page_qids.append(qid)

    report["merged_count"] = len(merged_answers)
    report["answer_changed_count_vs_answer_source"] = len(changed_answer_qids)
    report["page_projection_changed_count_vs_answer_source"] = len(changed_page_qids)
    report["answer_changed_qids"] = changed_answer_qids
    report["page_projection_changed_qids"] = changed_page_qids

    merged_submission = {
        "architecture_summary": _deepcopy_json(answer_source_submission.get("architecture_summary") or page_source_submission.get("architecture_summary") or {}),
        "answers": merged_answers,
    }
    return merged_submission, merged_raw_results, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline counterfactual candidate by mixing answer and page sources.")
    parser.add_argument("--answer-source-submission", type=Path, required=True)
    parser.add_argument("--answer-source-raw-results", type=Path, required=True)
    parser.add_argument("--answer-source-preflight", type=Path, required=True)
    parser.add_argument("--page-source-submission", type=Path, required=True)
    parser.add_argument("--page-source-raw-results", type=Path, required=True)
    parser.add_argument("--page-source-preflight", type=Path, required=True)
    parser.add_argument("--page-source-answer-qid", action="append", default=[])
    parser.add_argument("--page-source-answer-qids-file", type=Path, default=None)
    parser.add_argument("--page-source-page-qid", action="append", default=[])
    parser.add_argument("--page-source-page-qids-file", type=Path, default=None)
    parser.add_argument("--page-source-pages-default", choices=("all", "none"), default="all")
    parser.add_argument("--out-submission", type=Path, required=True)
    parser.add_argument("--out-raw-results", type=Path, required=True)
    parser.add_argument("--out-preflight", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    answer_source_submission = _load_json_dict(args.answer_source_submission)
    answer_source_raw_results = _load_json_list(args.answer_source_raw_results)
    answer_source_preflight = _load_json_dict(args.answer_source_preflight)
    page_source_submission = _load_json_dict(args.page_source_submission)
    page_source_raw_results = _load_json_list(args.page_source_raw_results)
    page_source_preflight = _load_json_dict(args.page_source_preflight)
    allowlisted_qids = _qid_allowlist(args)
    page_allowlisted_qids = _page_qid_allowlist(args)

    merged_submission, merged_raw_results, report = _merge_records(
        answer_source_submission=answer_source_submission,
        answer_source_raw_results=answer_source_raw_results,
        page_source_submission=page_source_submission,
        page_source_raw_results=page_source_raw_results,
        allowlisted_qids=allowlisted_qids,
        page_allowlisted_qids=page_allowlisted_qids,
        page_source_pages_default=str(args.page_source_pages_default),
    )
    merged_preflight = _build_preflight(
        merged_payload=merged_submission,
        answer_source_preflight=answer_source_preflight,
        page_source_preflight=page_source_preflight,
        answer_source_submission=args.answer_source_submission,
        page_source_submission=args.page_source_submission,
        allowlisted_qids=allowlisted_qids,
        page_allowlisted_qids=page_allowlisted_qids,
    )

    args.out_submission.write_text(json.dumps(merged_submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_raw_results.write_text(json.dumps(merged_raw_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    merged_preflight["submission_sha256"] = ""
    merged_preflight["raw_results_path"] = str(args.out_raw_results)
    args.out_preflight.write_text(json.dumps(merged_preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
