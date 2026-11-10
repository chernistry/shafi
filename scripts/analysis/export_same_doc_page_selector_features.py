# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]

_TITLE_PAGE_RE = re.compile(r"\btitle page\b", re.IGNORECASE)
_COVER_PAGE_RE = re.compile(r"\bcover page\b", re.IGNORECASE)
_FIRST_PAGE_RE = re.compile(r"\bfirst page\b", re.IGNORECASE)
_SECOND_PAGE_RE = re.compile(r"\bsecond page\b|\bpage 2\b", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\barticle\s+\d", re.IGNORECASE)
_SECTION_RE = re.compile(r"\bsection\s+\d", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bschedule\b", re.IGNORECASE)
_CAPTION_RE = re.compile(r"\bcaption\b", re.IGNORECASE)
_HEADER_RE = re.compile(r"\bheader\b", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\b(?:law number|law no\.?|difc law no\.?|regulation no\.?)\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(
    r"\b(?:claim number|case number|appeal number|appeal no\.?|enf-\d|ca-\d|arb-\d)\b",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^(?P<doc_id>.+)_(?P<page>\d+)$")
_REQUESTED_PAGE_RE = re.compile(r"\bpage\s+(?P<page>\d+)\b", re.IGNORECASE)


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", payload)


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


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _page_parts(page_id: str) -> tuple[str, int | None]:
    match = _PAGE_NUMBER_RE.match(page_id.strip())
    if not match:
        return page_id.strip(), None
    return match.group("doc_id"), int(match.group("page"))


def extract_query_flags(question: str) -> dict[str, object]:
    normalized = str(question or "").strip()
    requested_match = _REQUESTED_PAGE_RE.search(normalized)
    requested_page = int(requested_match.group("page")) if requested_match else None
    return {
        "query_has_title_page": bool(_TITLE_PAGE_RE.search(normalized)),
        "query_has_cover_page": bool(_COVER_PAGE_RE.search(normalized)),
        "query_has_first_page": bool(_FIRST_PAGE_RE.search(normalized)),
        "query_has_second_page": bool(_SECOND_PAGE_RE.search(normalized)),
        "query_has_article": bool(_ARTICLE_RE.search(normalized)),
        "query_has_section": bool(_SECTION_RE.search(normalized)),
        "query_has_schedule": bool(_SCHEDULE_RE.search(normalized)),
        "query_has_caption": bool(_CAPTION_RE.search(normalized)),
        "query_has_header": bool(_HEADER_RE.search(normalized)),
        "query_has_law_number": bool(_LAW_NUMBER_RE.search(normalized)),
        "query_has_case_ref": bool(_CASE_REF_RE.search(normalized)),
        "requested_page": requested_page,
    }


def _questions_by_id(path: Path) -> dict[str, JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    out: dict[str, JsonDict] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        question_id = str(raw.get("id") or raw.get("question_id") or "").strip()
        if question_id:
            out[question_id] = cast("JsonDict", raw)
    return out


def _baseline_cases_by_qid(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    cases = _coerce_dict_list(payload.get("cases"))
    out: dict[str, JsonDict] = {}
    for case in cases:
        qid = str(case.get("question_id") or case.get("qid") or "").strip()
        if qid:
            out[qid] = case
    return out


def build_rows(
    *,
    miss_pack_path: Path,
    baseline_predictions_path: Path,
    questions_path: Path,
) -> tuple[list[JsonDict], JsonDict]:
    pack = _load_json(miss_pack_path)
    cases = _coerce_dict_list(pack.get("cases"))
    baseline_by_qid = _baseline_cases_by_qid(baseline_predictions_path)
    questions_by_id = _questions_by_id(questions_path)

    rows: list[JsonDict] = []
    observed_gold_qids: list[str] = []
    observed_only_case_count = 0

    for case in cases:
        qid = str(case.get("qid") or case.get("question_id") or "").strip()
        if not qid:
            continue
        question_record = questions_by_id.get(qid, {})
        question = str(question_record.get("question") or case.get("question") or "").strip()
        answer_type = str(question_record.get("answer_type") or case.get("answer_type") or "").strip().lower() or "free_text"
        query_flags = extract_query_flags(question)

        gold_pages = _coerce_str_list(case.get("gold_pages"))
        used_pages = _coerce_str_list(case.get("used_pages"))
        false_positive_pages = _coerce_str_list(case.get("false_positive_pages"))
        baseline_case = baseline_by_qid.get(qid, {})
        baseline_pages = _coerce_str_list(baseline_case.get("predicted_page_ids"))
        target_doc_ids = _coerce_str_list(case.get("target_doc_ids"))

        observed_pages = _dedupe(used_pages + false_positive_pages + baseline_pages)
        candidate_pages = _dedupe(gold_pages + observed_pages)
        observed_gold_pages = [page_id for page_id in gold_pages if page_id in set(observed_pages)]
        if observed_gold_pages:
            observed_gold_qids.append(qid)
        if candidate_pages and candidate_pages == observed_pages:
            observed_only_case_count += 1

        for page_id in candidate_pages:
            doc_id, page_number = _page_parts(page_id)
            requested_page = cast("int | None", query_flags["requested_page"])
            baseline_rank = baseline_pages.index(page_id) + 1 if page_id in baseline_pages else None
            row: JsonDict = {
                "qid": qid,
                "question": question,
                "answer_type": answer_type,
                "question_family": str(case.get("question_family") or "").strip(),
                "miss_family": str(case.get("miss_family") or "").strip(),
                "route": str(case.get("route") or "").strip(),
                "trust_tier": str(case.get("trust_tier") or "").strip(),
                "ocr_risk": bool(case.get("ocr_risk", False)),
                "page_id": page_id,
                "doc_id": doc_id,
                "page_number": page_number,
                "baseline_rank": baseline_rank,
                "is_gold": page_id in gold_pages,
                "is_used_page": page_id in used_pages,
                "is_false_positive": page_id in false_positive_pages,
                "is_baseline_predicted": page_id in baseline_pages,
                "candidate_observed": page_id in observed_pages,
                "candidate_gold_only": page_id in gold_pages and page_id not in observed_pages,
                "is_target_doc": doc_id in target_doc_ids,
                "requested_page_match": requested_page is not None and page_number == requested_page,
                "page_distance_from_requested": (
                    abs(page_number - requested_page) if requested_page is not None and page_number is not None else None
                ),
                "is_page_one": page_number == 1,
                "is_page_two": page_number == 2,
                "is_page_leq_5": page_number is not None and page_number <= 5,
                "observed_gold_available_for_qid": bool(observed_gold_pages),
                "observed_gold_pages_for_qid": observed_gold_pages,
            }
            row.update(query_flags)
            rows.append(row)

    summary: JsonDict = {
        "ticket": 72,
        "created_at": "2026-03-13",
        "source_miss_pack": str(miss_pack_path),
        "source_baseline_predictions": str(baseline_predictions_path),
        "source_questions": str(questions_path),
        "summary": {
            "case_count": len(cases),
            "row_count": len(rows),
            "observed_row_count": sum(1 for row in rows if bool(row["candidate_observed"])),
            "gold_only_row_count": sum(1 for row in rows if bool(row["candidate_gold_only"])),
            "cases_with_observed_gold_candidate": len(set(observed_gold_qids)),
            "observed_only_case_count": observed_only_case_count,
            "miss_family_counts": dict(
                sorted(Counter(str(case.get("miss_family") or "").strip() for case in cases if str(case.get("miss_family") or "").strip()).items())
            ),
            "question_family_counts": dict(
                sorted(
                    Counter(
                        str(case.get("question_family") or "").strip()
                        for case in cases
                        if str(case.get("question_family") or "").strip()
                    ).items()
                )
            ),
        },
        "policy": (
            "Artifact-only feature export for offline same-doc page-selector falsification. "
            "Gold-only rows are exported for label coverage but must not be used as observed candidates."
        ),
    }
    return rows, summary


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, rows: list[JsonDict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export artifact-only features for the same-doc page-selector falsifier.")
    parser.add_argument("--miss-pack", required=True, help="Path to ticket65_single_doc_rerank_targets_miss_pack.json")
    parser.add_argument("--baseline-predictions", required=True, help="Path to baseline_predictions.json")
    parser.add_argument("--questions", required=True, help="Path to warm-up questions.json")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--out-summary", required=True, help="Output summary JSON path")
    args = parser.parse_args(argv)

    rows, summary = build_rows(
        miss_pack_path=Path(args.miss_pack),
        baseline_predictions_path=Path(args.baseline_predictions),
        questions_path=Path(args.questions),
    )
    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_jsonl, rows)
    _write_csv(out_csv, rows)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
