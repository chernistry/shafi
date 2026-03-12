from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _as_dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _as_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_json_list(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return _as_dict_list(cast("object", obj))


def _answers_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    answers = _as_dict_list(cast("object", payload.get("answers")))
    out: dict[str, JsonDict] = {}
    for raw in answers:
        qid = str(raw.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _raw_results_by_id(path: Path) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for raw in _load_json_list(path):
        case = _as_dict(raw.get("case"))
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _scaffold_by_id(path: Path | None) -> dict[str, JsonDict]:
    if path is None:
        return {}
    payload = _load_json(path)
    records = _as_dict_list(cast("object", payload.get("records")))
    out: dict[str, JsonDict] = {}
    for raw in records:
        qid = str(raw.get("question_id") or raw.get("case_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _retrieved_chunk_pages(answer_record: JsonDict) -> list[JsonDict]:
    telemetry = _as_dict(answer_record.get("telemetry"))
    retrieval = _as_dict(telemetry.get("retrieval"))
    return _as_dict_list(cast("object", retrieval.get("retrieved_chunk_pages")))


def _question_flags(question: str) -> list[str]:
    q = re.sub(r"\s+", " ", question).strip().casefold()
    flags: list[str] = []
    if "page 2" in q or "second page" in q:
        flags.append("page2")
    if any(term in q for term in ("title page", "cover page", "first page", "header", "caption")):
        flags.append("explicit_anchor")
    if any(term in q for term in ("both cases", "common to both", "across all documents", "identify whether any")):
        flags.append("comparison")
    if any(term in q for term in ("monetary claim", "higher monetary claim", "claim amount", "financial limit")):
        flags.append("monetary")
    if any(term in q for term in ("outcome", "result", "costs", "it is hereby ordered", "order")):
        flags.append("outcome")
    if "article " in q or "schedule " in q or "definitions" in q:
        flags.append("heading")
    if not flags:
        flags.append("other")
    return flags


def render_report(
    *,
    baseline_label: str,
    candidate_label: str,
    changed_rows: list[dict[str, object]],
) -> str:
    by_answer_type = Counter(str(row["answer_type"]) for row in changed_rows)
    by_route_family = Counter(str(row["route_family"]) for row in changed_rows)
    by_flag = Counter(flag for row in changed_rows for flag in cast("list[str]", row["flags"]))

    lines = [
        "# Page Projection Drift Report",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Candidate: `{candidate_label}`",
        f"- Changed page projections: `{len(changed_rows)}`",
        "",
        "## Grouping",
        "",
        f"- by_answer_type: `{dict(by_answer_type)}`",
        f"- by_route_family: `{dict(by_route_family)}`",
        f"- by_question_flag: `{dict(by_flag)}`",
    ]

    for row in changed_rows:
        lines.extend(
            [
                "",
                f"## {row['question_id']}",
                f"- answer_type: `{row['answer_type']}`",
                f"- route_family: `{row['route_family']}`",
                f"- question: {row['question']}",
                f"- flags: `{', '.join(cast('list[str]', row['flags']))}`",
                f"- failure_class: `{row['failure_class']}`",
                f"- baseline_pages: `{row['baseline_pages']}`",
                f"- candidate_pages: `{row['candidate_pages']}`",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Group changed retrieved_chunk_pages projections by question family.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--candidate-raw-results", type=Path, required=True)
    parser.add_argument("--candidate-scaffold", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    baseline = _answers_by_id(args.baseline_submission)
    candidate = _answers_by_id(args.candidate_submission)
    raw_results = _raw_results_by_id(args.candidate_raw_results)
    scaffold = _scaffold_by_id(args.candidate_scaffold)

    changed_rows: list[dict[str, object]] = []
    for qid, baseline_answer in baseline.items():
        candidate_answer = candidate.get(qid)
        raw = raw_results.get(qid)
        if candidate_answer is None or raw is None:
            continue
        baseline_pages = _retrieved_chunk_pages(baseline_answer)
        candidate_pages = _retrieved_chunk_pages(candidate_answer)
        if json.dumps(baseline_pages, ensure_ascii=False, sort_keys=True) == json.dumps(
            candidate_pages,
            ensure_ascii=False,
            sort_keys=True,
        ):
            continue
        case = _as_dict(raw.get("case"))
        question = str(case.get("question") or "").strip()
        answer_type = str(case.get("answer_type") or "").strip().lower() or "free_text"
        record = scaffold.get(qid, {})
        changed_rows.append(
            {
                "question_id": qid,
                "answer_type": answer_type,
                "route_family": str(record.get("route_family") or "(unknown)"),
                "failure_class": str(record.get("failure_class") or "(blank)"),
                "question": question,
                "flags": _question_flags(question),
                "baseline_pages": baseline_pages,
                "candidate_pages": candidate_pages,
            }
        )

    changed_rows.sort(key=lambda row: (str(row["answer_type"]), str(row["question_id"])))
    report = render_report(
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        changed_rows=changed_rows,
    )
    args.out.write_text(report + "\n", encoding="utf-8")
    args.json_out.write_text(
        json.dumps(
            {
                "baseline_label": args.baseline_label,
                "candidate_label": args.candidate_label,
                "changed_count": len(changed_rows),
                "rows": changed_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
