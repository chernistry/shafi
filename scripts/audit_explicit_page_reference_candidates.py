from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]
_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*\d{1,4}/\d{4}\b", re.IGNORECASE)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_questions(path: Path) -> dict[str, JsonDict]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected question array in {path}")
    rows = cast("list[object]", obj)
    out: dict[str, JsonDict] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("id") or row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_raw_results(path: Path) -> dict[str, JsonDict]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"Expected raw-results array in {path}")
    rows = cast("list[object]", obj)
    out: dict[str, JsonDict] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        case_obj = row.get("case")
        case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
        qid = str(case.get("case_id") or row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_scaffold(path: Path | None) -> dict[str, JsonDict]:
    if path is None:
        return {}
    obj = _load_json(path)
    if not isinstance(obj, dict):
        return {}
    records_obj = cast("JsonDict", obj).get("records")
    if not isinstance(records_obj, list):
        return {}
    rows = cast("list[object]", records_obj)
    out: dict[str, JsonDict] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast("JsonDict", row_obj)
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _coerce_page_ids(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    rows = cast("list[object]", value)
    return [text for item in rows if (text := str(item).strip())]


def _page_id_parts(page_id: str) -> tuple[str, int | None]:
    if "_" not in page_id:
        return page_id, None
    doc_id, suffix = page_id.rsplit("_", 1)
    try:
        return doc_id, int(suffix)
    except ValueError:
        return doc_id, None


def _page_hits(page_ids: list[str], *, target_page: int) -> tuple[int, list[str]]:
    doc_ids: list[str] = []
    for page_id in page_ids:
        doc_id, page_num = _page_id_parts(page_id)
        if page_num == target_page:
            doc_ids.append(doc_id)
    unique_doc_ids = list(dict.fromkeys(doc_ids))
    return len(unique_doc_ids), unique_doc_ids


def _answer_text(row: JsonDict) -> str:
    return str(row.get("answer_text") or "").strip()


def _question_target(question: str) -> tuple[str, int] | None:
    q = re.sub(r"\s+", " ", question).strip().lower()
    if not q:
        return None
    if "page 2" in q or "second page" in q:
        return ("page_2", 2)
    if any(term in q for term in ("title page", "cover page", "first page", "header", "caption")):
        return ("page_1_anchor", 1)
    return None


def _required_doc_count(question: str) -> int:
    refs = _CASE_REF_RE.findall(question or "")
    return max(1, len({ref.upper() for ref in refs}))


def _recommendation(*, baseline_used_hits: int, best_source_hits: int, answer_changed: bool) -> str:
    if not answer_changed and best_source_hits > baseline_used_hits:
        return "PROMISING"
    if not answer_changed and best_source_hits == baseline_used_hits and best_source_hits > 0:
        return "REPORT_ONLY"
    return "REPORT_ONLY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit explicit page-reference questions against baseline and source raw-results artifacts.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--scaffold", type=Path, default=None)
    parser.add_argument("--source", action="append", default=[], help="label=/abs/path/to/raw_results.json")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_by_id = _load_questions(args.questions.resolve())
    baseline_rows = _load_raw_results(args.baseline_raw_results.resolve())
    scaffold_by_id = _load_scaffold(args.scaffold.resolve() if args.scaffold else None)

    source_rows: list[tuple[str, dict[str, JsonDict]]] = []
    for raw in args.source:
        label, sep, path_text = str(raw).partition("=")
        if not sep:
            raise ValueError(f"Invalid --source value: {raw}")
        source_rows.append((label.strip(), _load_raw_results(Path(path_text).expanduser().resolve())))

    records: list[JsonDict] = []
    for qid, question_row in questions_by_id.items():
        question = str(question_row.get("question") or "").strip()
        target = _question_target(question)
        if target is None or qid not in baseline_rows:
            continue
        target_kind, target_page = target
        required_docs = _required_doc_count(question)
        baseline = baseline_rows[qid]
        baseline_answer = _answer_text(baseline)
        baseline_telemetry = cast("JsonDict", baseline.get("telemetry")) if isinstance(baseline.get("telemetry"), dict) else {}
        baseline_used = _coerce_page_ids(baseline_telemetry.get("used_page_ids"))
        baseline_context = _coerce_page_ids(baseline_telemetry.get("context_page_ids"))
        baseline_retrieved = _coerce_page_ids(baseline_telemetry.get("retrieved_page_ids"))
        baseline_used_hits, baseline_used_doc_ids = _page_hits(baseline_used, target_page=target_page)
        baseline_context_hits, _baseline_context_doc_ids = _page_hits(baseline_context, target_page=target_page)
        baseline_retrieved_hits, _baseline_retrieved_doc_ids = _page_hits(baseline_retrieved, target_page=target_page)

        source_signals: list[JsonDict] = []
        best_source_hits = baseline_used_hits
        any_answer_changed = False
        for label, rows in source_rows:
            candidate = rows.get(qid)
            if candidate is None:
                continue
            candidate_answer = _answer_text(candidate)
            candidate_telemetry = cast("JsonDict", candidate.get("telemetry")) if isinstance(candidate.get("telemetry"), dict) else {}
            candidate_used = _coerce_page_ids(candidate_telemetry.get("used_page_ids"))
            used_hits, used_doc_ids = _page_hits(candidate_used, target_page=target_page)
            answer_changed = candidate_answer != baseline_answer
            any_answer_changed = any_answer_changed or answer_changed
            best_source_hits = max(best_source_hits, used_hits)
            source_signals.append(
                {
                    "label": label,
                    "answer_changed": answer_changed,
                    "used_page_hits": used_hits,
                    "used_page_doc_ids": used_doc_ids,
                    "used_page_ids": candidate_used,
                }
            )

        scaffold = scaffold_by_id.get(qid, {})
        records.append(
            {
                "question_id": qid,
                "question": question,
                "answer_type": str(question_row.get("answer_type") or "").strip(),
                "target_kind": target_kind,
                "target_page": target_page,
                "required_doc_count": required_docs,
                "route_family": str(scaffold.get("route_family") or "").strip(),
                "support_shape_class": str(scaffold.get("support_shape_class") or "").strip(),
                "baseline_used_page_hits": baseline_used_hits,
                "baseline_context_page_hits": baseline_context_hits,
                "baseline_retrieved_page_hits": baseline_retrieved_hits,
                "baseline_used_page_doc_ids": baseline_used_doc_ids,
                "baseline_used_page_ids": baseline_used,
                "source_signals": source_signals,
                "recommendation": _recommendation(
                    baseline_used_hits=baseline_used_hits,
                    best_source_hits=best_source_hits,
                    answer_changed=any_answer_changed,
                ),
            }
        )

    ranked = sorted(
        records,
        key=lambda row: (
            {"PROMISING": 1, "REPORT_ONLY": 0}.get(str(row.get("recommendation") or ""), 0),
            int(row.get("required_doc_count") or 0) - int(row.get("baseline_used_page_hits") or 0),
            -int(row.get("target_page") or 0),
            str(row.get("question_id") or ""),
        ),
        reverse=True,
    )

    md_lines = [
        "# Explicit Page-Reference Candidate Audit",
        "",
        f"- baseline_label: `{args.baseline_label}`",
        f"- records: `{len(ranked)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | QID | Kind | Recommendation | Required Docs | Baseline Used Hits | Baseline Context Hits | Source Rescue |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(ranked, start=1):
        rescue = ", ".join(
            f"{signal['label']}:{signal['used_page_hits']}"
            for signal in cast("list[JsonDict]", row.get("source_signals") or [])
        ) or "n/a"
        md_lines.append(
            f"| {index} | `{row['question_id']}` | `{row['target_kind']}` | `{row['recommendation']}` | "
            f"{row['required_doc_count']} | {row['baseline_used_page_hits']} | {row['baseline_context_page_hits']} | {rescue} |"
        )

    payload = {
        "baseline_label": args.baseline_label,
        "records": ranked,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
