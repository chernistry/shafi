from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_json_dict(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _answers_by_id(payload: JsonDict) -> dict[str, JsonDict]:
    answers_obj = payload.get("answers")
    answers = cast("list[object]", answers_obj) if isinstance(answers_obj, list) else []
    out: dict[str, JsonDict] = {}
    for row in answers:
        if not isinstance(row, dict):
            continue
        record = cast("JsonDict", row)
        qid = str(record.get("question_id") or "").strip()
        if qid:
            out[qid] = record
    return out


def _load_qid_set(*, values: list[str], file_path: Path | None) -> set[str]:
    out = {text for raw in values if (text := str(raw).strip())}
    if file_path is not None:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                out.add(text)
    return out


def _answer_value(record: JsonDict) -> str:
    return json.dumps(record.get("answer"), ensure_ascii=False, sort_keys=True)


def _page_projection(record: JsonDict) -> str:
    telemetry = cast("JsonDict", record.get("telemetry")) if isinstance(record.get("telemetry"), dict) else {}
    retrieval = cast("JsonDict", telemetry.get("retrieval")) if isinstance(telemetry.get("retrieval"), dict) else {}
    return json.dumps(retrieval.get("retrieved_chunk_pages", []), ensure_ascii=False, sort_keys=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_md(*, baseline_path: Path, candidate_path: Path, report: JsonDict) -> str:
    lines = [
        "# Candidate Lineage Verification",
        "",
        f"- baseline: `{baseline_path}`",
        f"- candidate: `{candidate_path}`",
        f"- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "## Summary",
        "",
        f"- baseline_sha256: `{report['baseline_sha256']}`",
        f"- candidate_sha256: `{report['candidate_sha256']}`",
        f"- answer_changed_count: `{report['answer_changed_count']}`",
        f"- page_changed_count: `{report['page_changed_count']}`",
        f"- unexpected_answer_qids: `{len(report['unexpected_answer_qids'])}`",
        f"- unexpected_page_qids: `{len(report['unexpected_page_qids'])}`",
        f"- lineage_ok: `{report['lineage_ok']}`",
        "",
    ]
    if report["answer_changed_qids"]:
        lines.extend(
            [
                "## Answer Changes",
                "",
                *(f"- `{qid}`" for qid in report["answer_changed_qids"]),
                "",
            ]
        )
    if report["page_changed_qids"]:
        lines.extend(
            [
                "## Page Projection Changes",
                "",
                *(f"- `{qid}`" for qid in report["page_changed_qids"]),
                "",
            ]
        )
    if report["unexpected_answer_qids"] or report["unexpected_page_qids"]:
        lines.extend(
            [
                "## Unexpected Drift",
                "",
                *(f"- unexpected_answer: `{qid}`" for qid in report["unexpected_answer_qids"]),
                *(f"- unexpected_page: `{qid}`" for qid in report["unexpected_page_qids"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a candidate artifact matches baseline except for intended answer/page qids.")
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--allowed-answer-qid", action="append", default=[])
    parser.add_argument("--allowed-answer-qids-file", type=Path, default=None)
    parser.add_argument("--allowed-page-qid", action="append", default=[])
    parser.add_argument("--allowed-page-qids-file", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline_submission.resolve()
    candidate_path = args.candidate_submission.resolve()
    baseline = _answers_by_id(_load_json_dict(baseline_path))
    candidate = _answers_by_id(_load_json_dict(candidate_path))
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate question_id sets do not match")

    allowed_answer_qids = _load_qid_set(values=args.allowed_answer_qid, file_path=args.allowed_answer_qids_file.resolve() if args.allowed_answer_qids_file else None)
    allowed_page_qids = _load_qid_set(values=args.allowed_page_qid, file_path=args.allowed_page_qids_file.resolve() if args.allowed_page_qids_file else None)

    answer_changed_qids: list[str] = []
    page_changed_qids: list[str] = []
    for qid in sorted(baseline):
        if _answer_value(baseline[qid]) != _answer_value(candidate[qid]):
            answer_changed_qids.append(qid)
        if _page_projection(baseline[qid]) != _page_projection(candidate[qid]):
            page_changed_qids.append(qid)

    unexpected_answer_qids = [qid for qid in answer_changed_qids if qid not in allowed_answer_qids]
    unexpected_page_qids = [qid for qid in page_changed_qids if qid not in allowed_page_qids]

    report: JsonDict = {
        "baseline_submission": str(baseline_path),
        "candidate_submission": str(candidate_path),
        "baseline_sha256": _sha256(baseline_path),
        "candidate_sha256": _sha256(candidate_path),
        "allowed_answer_qids": sorted(allowed_answer_qids),
        "allowed_page_qids": sorted(allowed_page_qids),
        "answer_changed_count": len(answer_changed_qids),
        "page_changed_count": len(page_changed_qids),
        "answer_changed_qids": answer_changed_qids,
        "page_changed_qids": page_changed_qids,
        "unexpected_answer_qids": unexpected_answer_qids,
        "unexpected_page_qids": unexpected_page_qids,
        "lineage_ok": not unexpected_answer_qids and not unexpected_page_qids,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_md(baseline_path=baseline_path, candidate_path=candidate_path, report=report), encoding="utf-8")


if __name__ == "__main__":
    main()
