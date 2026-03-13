from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ExactnessRow:
    question_id: str
    manual_verdict: str
    expected_answers: list[str]
    baseline_answer: str
    candidate_answer: str
    baseline_matches_expected: bool
    candidate_matches_expected: bool
    labels: list[str]
    failure_class: str


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _submission_answers_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Submission at {path} is missing 'answers'")
    answers = cast("list[object]", answers_obj)
    out: dict[str, JsonDict] = {}
    for raw in answers:
        if not isinstance(raw, dict):
            continue
        record = cast("JsonDict", raw)
        qid = str(record.get("question_id") or "").strip()
        if qid:
            out[qid] = record
    return out


def _scaffold_records_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {path} is missing 'records'")
    records = cast("list[object]", records_obj)
    out: dict[str, JsonDict] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        record = cast("JsonDict", raw)
        qid = str(record.get("question_id") or record.get("case_id") or "").strip()
        if qid:
            out[qid] = record
    return out


def _normalize_answer_text(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        parts = _coerce_str_list(cast("list[object]", value))
        return "null" if not parts else " | ".join(parts)
    text = str(value).strip()
    return text if text else "null"


def _answer_variants(value: object) -> list[str]:
    if value is None:
        return ["null"]
    if isinstance(value, list):
        parts = _coerce_str_list(cast("list[object]", value))
        return parts or ["null"]
    text = str(value).strip()
    return [text if text else "null"]


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [text for item in items if (text := str(item).strip())]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _changed_answer_qids(
    baseline_submission: dict[str, JsonDict],
    candidate_submission: dict[str, JsonDict],
) -> list[str]:
    out: list[str] = []
    for qid, baseline in baseline_submission.items():
        candidate = candidate_submission.get(qid)
        if candidate is None:
            continue
        baseline_answer = _normalize_answer_text(baseline.get("answer"))
        candidate_answer = _normalize_answer_text(candidate.get("answer"))
        if baseline_answer != candidate_answer:
            out.append(qid)
    return sorted(out)


def _matches_expected(answer: str, expected_answers: list[str]) -> bool:
    normalized = answer.strip()
    return any(normalized == expected.strip() for expected in expected_answers)


def _build_exactness_rows(
    *,
    changed_qids: list[str],
    baseline_submission: dict[str, JsonDict],
    candidate_submission: dict[str, JsonDict],
    scaffold_records: dict[str, JsonDict],
) -> list[ExactnessRow]:
    rows: list[ExactnessRow] = []
    for qid in changed_qids:
        record = scaffold_records.get(qid, {})
        expected_answers = _coerce_str_list(record.get("expected_answer"))
        baseline_value = baseline_submission[qid].get("answer")
        candidate_value = candidate_submission[qid].get("answer")
        baseline_answer = _normalize_answer_text(baseline_value)
        candidate_answer = _normalize_answer_text(candidate_value)
        baseline_variants = _answer_variants(baseline_value)
        candidate_variants = _answer_variants(candidate_value)
        rows.append(
            ExactnessRow(
                question_id=qid,
                manual_verdict=str(record.get("manual_verdict") or "").strip() or "unknown",
                expected_answers=expected_answers,
                baseline_answer=baseline_answer,
                candidate_answer=candidate_answer,
                baseline_matches_expected=any(_matches_expected(answer, expected_answers) for answer in baseline_variants),
                candidate_matches_expected=any(_matches_expected(answer, expected_answers) for answer in candidate_variants),
                labels=_coerce_str_list(record.get("manual_exactness_labels")),
                failure_class=str(record.get("failure_class") or "").strip(),
            )
        )
    return rows


def _write_qids_file(path: Path, qids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")


def _run_debug_eval(
    *,
    root: Path,
    baseline_label: str,
    baseline_raw_results: Path,
    candidate_label: str,
    candidate_raw_results: Path,
    questions: Path,
    docs_dir: Path,
    include_qids_file: Path,
    out_dir: Path,
    judge_scope: str,
) -> JsonDict:
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_candidate_debug_signal.py",
            "--baseline-label",
            baseline_label,
            "--baseline-raw-results",
            str(baseline_raw_results),
            "--candidate-label",
            candidate_label,
            "--candidate-raw-results",
            str(candidate_raw_results),
            "--questions",
            str(questions),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
            "--case-scope",
            "all",
            "--judge-scope",
            judge_scope,
            "--include-qids-file",
            str(include_qids_file),
        ],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
    eval_path = out_dir / f"eval_candidate_debug_{candidate_label}.json"
    return _load_json(eval_path)


def _judge_case_by_id(eval_payload: JsonDict) -> dict[str, JsonDict]:
    cases_obj = eval_payload.get("cases")
    if not isinstance(cases_obj, list):
        return {}
    cases = cast("list[object]", cases_obj)
    out: dict[str, JsonDict] = {}
    for raw in cases:
        if not isinstance(raw, dict):
            continue
        case = cast("JsonDict", raw)
        qid = str(case.get("question_id") or case.get("case_id") or "").strip()
        if qid:
            out[qid] = case
    return out


def _render_markdown(
    *,
    label: str,
    baseline_label: str,
    rows: list[ExactnessRow],
    debug_eval: JsonDict | None,
) -> str:
    lines = [
        "# Exactness Candidate Audit",
        "",
        f"- `baseline`: `{baseline_label}`",
        f"- `candidate`: `{label}`",
        f"- `answer_changed_qids`: `{len(rows)}`",
        "- `submission_policy`: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
    ]
    if debug_eval is not None:
        summary_obj = debug_eval.get("summary")
        summary = cast("JsonDict", summary_obj) if isinstance(summary_obj, dict) else {}
        judge_obj = summary.get("judge")
        judge = cast("JsonDict", judge_obj) if isinstance(judge_obj, dict) else {}
        lines.extend(
            [
                "## Debug Summary",
                "",
                f"- `judge_pass_rate`: `{judge.get('pass_rate', 'n/a')}`",
                f"- `judge_avg_grounding`: `{judge.get('avg_grounding', 'n/a')}`",
                f"- `citation_coverage`: `{summary.get('citation_coverage', 'n/a')}`",
                f"- `format_compliance`: `{summary.get('answer_type_format_compliance', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Changed Exactness Cases",
            "",
            "| QID | Manual Verdict | Expected | Baseline | Candidate | Baseline Match | Candidate Match | Labels | Failure Class | Judge | Grounding |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: |",
        ]
    )
    judge_cases = _judge_case_by_id(debug_eval) if debug_eval is not None else {}
    for row in rows:
        judge_case = judge_cases.get(row.question_id, {})
        judge_obj = judge_case.get("judge")
        judge = cast("JsonDict", judge_obj) if isinstance(judge_obj, dict) else {}
        scores_obj = judge.get("scores")
        scores = cast("JsonDict", scores_obj) if isinstance(scores_obj, dict) else {}
        lines.append(
            "| "
            f"`{row.question_id}` | `{row.manual_verdict}` | "
            f"`{row.expected_answers or ['<none>']}` | "
            f"`{row.baseline_answer}` | `{row.candidate_answer}` | "
            f"{int(row.baseline_matches_expected)} | {int(row.candidate_matches_expected)} | "
            f"`{row.labels}` | `{row.failure_class or 'n/a'}` | "
            f"`{judge.get('verdict', 'n/a')}` | `{scores.get('grounding', 'n/a')}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exactness-only deltas against scaffold truth and local debug judge.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--truth-audit-scaffold", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path)
    parser.add_argument("--candidate-raw-results", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--docs-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--judge-scope", choices=("all", "free_text", "none"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_submission = _submission_answers_by_id(args.baseline_submission.resolve())
    candidate_submission = _submission_answers_by_id(args.candidate_submission.resolve())
    scaffold_records = _scaffold_records_by_id(args.truth_audit_scaffold.resolve())
    changed_qids = _changed_answer_qids(baseline_submission, candidate_submission)
    rows = _build_exactness_rows(
        changed_qids=changed_qids,
        baseline_submission=baseline_submission,
        candidate_submission=candidate_submission,
        scaffold_records=scaffold_records,
    )

    debug_eval: JsonDict | None = None
    if (
        args.baseline_raw_results is not None
        and args.candidate_raw_results is not None
        and args.questions is not None
        and args.docs_dir is not None
        and args.out_dir is not None
        and changed_qids
    ):
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        include_qids_file = out_dir / f"exactness_qids_{args.candidate_label}.txt"
        _write_qids_file(include_qids_file, changed_qids)
        debug_eval = _run_debug_eval(
            root=root,
            baseline_label=str(args.baseline_label),
            baseline_raw_results=args.baseline_raw_results.resolve(),
            candidate_label=str(args.candidate_label),
            candidate_raw_results=args.candidate_raw_results.resolve(),
            questions=args.questions.resolve(),
            docs_dir=args.docs_dir.resolve(),
            include_qids_file=include_qids_file,
            out_dir=out_dir,
            judge_scope=str(args.judge_scope),
        )

    payload: JsonDict = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "answer_changed_qids": changed_qids,
        "resolved_incorrect_qids": [
            row.question_id
            for row in rows
            if row.manual_verdict == "incorrect" and not row.baseline_matches_expected and row.candidate_matches_expected
        ],
        "still_mismatched_incorrect_qids": [
            row.question_id
            for row in rows
            if row.manual_verdict == "incorrect" and not row.candidate_matches_expected
        ],
        "rows": [
            {
                "question_id": row.question_id,
                "manual_verdict": row.manual_verdict,
                "expected_answers": row.expected_answers,
                "baseline_answer": row.baseline_answer,
                "candidate_answer": row.candidate_answer,
                "baseline_matches_expected": row.baseline_matches_expected,
                "candidate_matches_expected": row.candidate_matches_expected,
                "labels": row.labels,
                "failure_class": row.failure_class,
            }
            for row in rows
        ],
        "debug_eval_path": None if debug_eval is None else str(args.out_dir.resolve() / f"eval_candidate_debug_{args.candidate_label}.json"),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(
        _render_markdown(
            label=str(args.candidate_label),
            baseline_label=str(args.baseline_label),
            rows=rows,
            debug_eval=debug_eval,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
