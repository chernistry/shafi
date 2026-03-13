from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_json_dict(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_json_list(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [cast("JsonDict", item) for item in cast("list[object]", obj) if isinstance(item, dict)]


def _as_dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _as_dict_list(value: object) -> list[JsonDict]:
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


def _deepcopy_json(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _answers_by_id(payload: JsonDict) -> dict[str, JsonDict]:
    answers = _as_dict_list(payload.get("answers"))
    out: dict[str, JsonDict] = {}
    for answer in answers:
        qid = str(answer.get("question_id") or "").strip()
        if qid:
            out[qid] = answer
    return out


def _raw_results_by_id(records: list[JsonDict]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for raw in records:
        case = _as_dict(raw.get("case"))
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _scaffold_records_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json_dict(path)
    out: dict[str, JsonDict] = {}
    for record in _as_dict_list(payload.get("records")):
        qid = str(record.get("question_id") or record.get("case_id") or "").strip()
        if qid:
            out[qid] = record
    return out


def _group_retrieved_chunk_pages(page_ids: list[str]) -> list[JsonDict]:
    pages_by_doc: dict[str, list[int]] = defaultdict(list)
    for page_id in page_ids:
        if "_" not in page_id:
            continue
        doc_id, _, page_part = page_id.rpartition("_")
        try:
            page_num = int(page_part)
        except ValueError:
            continue
        pages_by_doc[doc_id].append(page_num)
    grouped: list[JsonDict] = []
    for doc_id, page_numbers in sorted(pages_by_doc.items()):
        grouped.append({"doc_id": doc_id, "page_numbers": sorted(set(page_numbers))})
    return grouped


def _qid_allowlist(args: argparse.Namespace) -> list[str]:
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
    return out


def build_scaffold_support_surrogate(
    *,
    baseline_submission_path: Path,
    baseline_raw_results_path: Path,
    scaffold_path: Path,
    qids: list[str],
    out_submission_path: Path,
    out_raw_results_path: Path,
    out_report_path: Path,
) -> None:
    baseline_submission = _load_json_dict(baseline_submission_path)
    baseline_raw_results = _load_json_list(baseline_raw_results_path)
    scaffold_records = _scaffold_records_by_id(scaffold_path)

    answers_by_id = _answers_by_id(baseline_submission)
    raw_by_id = _raw_results_by_id(baseline_raw_results)

    patched_qids: list[str] = []
    missing_qids: list[str] = []

    submission_payload = cast("JsonDict", _deepcopy_json(baseline_submission))
    raw_payload = cast("list[JsonDict]", _deepcopy_json(baseline_raw_results))
    submission_answers_by_id = _answers_by_id(submission_payload)
    raw_payload_by_id = _raw_results_by_id(raw_payload)

    for qid in qids:
        scaffold_record = scaffold_records.get(qid)
        baseline_answer = answers_by_id.get(qid)
        baseline_raw = raw_by_id.get(qid)
        submission_answer = submission_answers_by_id.get(qid)
        raw_record = raw_payload_by_id.get(qid)
        if scaffold_record is None or baseline_answer is None or baseline_raw is None or submission_answer is None or raw_record is None:
            missing_qids.append(qid)
            continue

        gold_pages = _coerce_str_list(scaffold_record.get("minimal_required_support_pages"))
        if not gold_pages:
            missing_qids.append(qid)
            continue

        retrieved_chunk_pages = _group_retrieved_chunk_pages(gold_pages)
        doc_shortlist = sorted({page_id.rpartition("_")[0] for page_id in gold_pages if "_" in page_id})

        submission_telemetry = _as_dict(submission_answer.get("telemetry"))
        submission_retrieval = _as_dict(submission_telemetry.get("retrieval"))
        submission_retrieval["retrieved_chunk_pages"] = retrieved_chunk_pages
        submission_telemetry["retrieval"] = submission_retrieval
        submission_answer["telemetry"] = submission_telemetry

        raw_telemetry = _as_dict(raw_record.get("telemetry"))
        for field in (
            "retrieved_page_ids",
            "context_page_ids",
            "used_page_ids",
            "cited_page_ids",
            "localized_support_page_ids",
        ):
            raw_telemetry[field] = list(gold_pages)
        for field in (
            "retrieved_chunk_ids",
            "context_chunk_ids",
            "used_chunk_ids",
            "cited_chunk_ids",
            "localized_support_chunk_ids",
            "must_include_chunk_ids",
        ):
            raw_telemetry[field] = []
        raw_telemetry["doc_shortlist"] = doc_shortlist
        raw_record["telemetry"] = raw_telemetry
        patched_qids.append(qid)

    report: JsonDict = {
        "baseline_submission": str(baseline_submission_path),
        "baseline_raw_results": str(baseline_raw_results_path),
        "scaffold_path": str(scaffold_path),
        "requested_qids": qids,
        "patched_qids": patched_qids,
        "missing_qids": missing_qids,
        "answer_changed_count_vs_baseline": 0,
        "page_projection_changed_count_vs_baseline": len(patched_qids),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }

    out_submission_path.parent.mkdir(parents=True, exist_ok=True)
    out_raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_submission_path.write_text(json.dumps(submission_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_raw_results_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build support-only surrogate by swapping selected QIDs to scaffold gold pages.")
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qids-file", type=Path)
    parser.add_argument("--out-submission", type=Path, required=True)
    parser.add_argument("--out-raw-results", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    qids = _qid_allowlist(args)
    if not qids:
        raise SystemExit("No QIDs provided")

    build_scaffold_support_surrogate(
        baseline_submission_path=args.baseline_submission,
        baseline_raw_results_path=args.baseline_raw_results,
        scaffold_path=args.scaffold,
        qids=qids,
        out_submission_path=args.out_submission,
        out_raw_results_path=args.out_raw_results,
        out_report_path=args.out_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
