# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

try:
    from scripts.run_experiment_gate import (
        _coerce_str_list,
        _scaffold_records_by_id,
        _seed_case_deltas,
        _submission_answers_by_id,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution uses a different import root
    from run_experiment_gate import (
        _coerce_str_list,
        _scaffold_records_by_id,
        _seed_case_deltas,
        _submission_answers_by_id,
    )

JsonDict = dict[str, object]


@dataclass(frozen=True)
class AnchorSliceRow:
    question_id: str
    question: str
    answer_type: str
    route_family: str
    baseline_answer: str
    candidate_answer: str
    expected_answer: str
    manual_verdict: str
    failure_class: str
    status: str
    gold_page_ids: list[str]
    baseline_used_page_ids: list[str]
    candidate_used_page_ids: list[str]
    baseline_context_page_ids: list[str]
    candidate_context_page_ids: list[str]
    baseline_used_hit: bool
    candidate_used_hit: bool
    baseline_context_hit: bool
    candidate_context_hit: bool
    candidate_used_equivalent_hit: bool
    candidate_context_equivalent_hit: bool
    answer_changed: bool
    required_page_anchor: JsonDict
    manual_exactness_labels: list[str]
    exactness_review_flags: list[str]


def _load_qids(args: argparse.Namespace) -> list[str]:
    qids: list[str] = []
    for raw in args.qid:
        text = str(raw).strip()
        if text:
            qids.append(text)
    if args.qids_file is not None:
        for line in args.qids_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                qids.append(text)
    # preserve order while removing duplicates
    return list(dict.fromkeys(qids))


def _answer_text(answer_record: JsonDict | None) -> str:
    if answer_record is None:
        return ""
    value = answer_record.get("answer")
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row_status(*, answer_changed: bool, delta: JsonDict) -> str:
    baseline_support = bool(delta["baseline_used_hit"] or delta["baseline_context_hit"])
    candidate_support = bool(
        delta["candidate_used_hit"]
        or delta["candidate_context_hit"]
        or delta["candidate_used_equivalent_hit"]
        or delta["candidate_context_equivalent_hit"]
    )
    if candidate_support and not baseline_support:
        return "support_improved_answer_changed" if answer_changed else "support_improved"
    if baseline_support and not candidate_support:
        return "support_regressed_answer_changed" if answer_changed else "support_regressed"
    if answer_changed:
        return "answer_changed_only"
    if (
        delta["candidate_used_equivalent_hit"]
        or delta["candidate_context_equivalent_hit"]
        or delta["candidate_used_hit"]
        or delta["candidate_context_hit"]
    ):
        return "support_equivalent_or_held"
    return "mixed_or_no_hit"


def build_rows(
    *,
    baseline_submission_path: Path,
    candidate_submission_path: Path,
    baseline_scaffold_path: Path,
    candidate_scaffold_path: Path | None,
    baseline_raw_results_path: Path,
    candidate_raw_results_path: Path,
    qids: list[str],
) -> list[AnchorSliceRow]:
    baseline_submission = _submission_answers_by_id(baseline_submission_path)
    candidate_submission = _submission_answers_by_id(candidate_submission_path)
    baseline_scaffold = _scaffold_records_by_id(baseline_scaffold_path)
    candidate_scaffold = _scaffold_records_by_id(candidate_scaffold_path) if candidate_scaffold_path is not None else {}
    deltas = _seed_case_deltas(
        baseline_scaffold_path=baseline_scaffold_path,
        candidate_scaffold_path=candidate_scaffold_path,
        baseline_raw_results_path=baseline_raw_results_path,
        candidate_raw_results_path=candidate_raw_results_path,
        seed_qids=qids,
    )

    rows: list[AnchorSliceRow] = []
    for delta in deltas:
        baseline_record = baseline_scaffold.get(delta.question_id, {})
        candidate_record = candidate_scaffold.get(delta.question_id, baseline_record)
        baseline_answer = _answer_text(baseline_submission.get(delta.question_id))
        candidate_answer = _answer_text(candidate_submission.get(delta.question_id))
        rows.append(
            AnchorSliceRow(
                question_id=delta.question_id,
                question=str(baseline_record.get("question") or "").strip(),
                answer_type=str(baseline_record.get("answer_type") or "").strip(),
                route_family=str(baseline_record.get("route_family") or "").strip(),
                baseline_answer=baseline_answer,
                candidate_answer=candidate_answer,
                expected_answer=str(baseline_record.get("expected_answer") or "").strip(),
                manual_verdict=str(baseline_record.get("manual_verdict") or "").strip(),
                failure_class=str(baseline_record.get("failure_class") or "").strip(),
                status=_row_status(
                    answer_changed=baseline_answer != candidate_answer,
                    delta=asdict(delta),
                ),
                gold_page_ids=delta.gold_page_ids,
                baseline_used_page_ids=delta.baseline_used_page_ids,
                candidate_used_page_ids=delta.candidate_used_page_ids,
                baseline_context_page_ids=delta.baseline_context_page_ids,
                candidate_context_page_ids=delta.candidate_context_page_ids,
                baseline_used_hit=delta.baseline_used_hit,
                candidate_used_hit=delta.candidate_used_hit,
                baseline_context_hit=delta.baseline_context_hit,
                candidate_context_hit=delta.candidate_context_hit,
                candidate_used_equivalent_hit=delta.candidate_used_equivalent_hit,
                candidate_context_equivalent_hit=delta.candidate_context_equivalent_hit,
                answer_changed=baseline_answer != candidate_answer,
                required_page_anchor=cast(
                    "JsonDict",
                    candidate_record.get("required_page_anchor")
                    if isinstance(candidate_record.get("required_page_anchor"), dict)
                    else baseline_record.get("required_page_anchor")
                    if isinstance(baseline_record.get("required_page_anchor"), dict)
                    else {},
                ),
                manual_exactness_labels=_coerce_str_list(baseline_record.get("manual_exactness_labels")),
                exactness_review_flags=_coerce_str_list(baseline_record.get("exactness_review_flags")),
            )
        )
    return rows


def _status_counts(rows: list[AnchorSliceRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row.status] = out.get(row.status, 0) + 1
    return dict(sorted(out.items()))


def render_report(rows: list[AnchorSliceRow], *, baseline_label: str, candidate_label: str) -> str:
    lines = [
        "# Anchor Slice Diff Report",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Candidate: `{candidate_label}`",
        f"- Cases: `{len(rows)}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in _status_counts(rows).items():
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Case Details", ""])
    for row in rows:
        lines.extend(
            [
                f"## {row.question_id}",
                f"- status: `{row.status}`",
                f"- answer_type: `{row.answer_type}`",
                f"- route_family: `{row.route_family}`",
                f"- question: {row.question}",
                f"- baseline_answer: `{row.baseline_answer}`",
                f"- candidate_answer: `{row.candidate_answer}`",
                f"- expected_answer: `{row.expected_answer or 'None'}`",
                f"- manual_verdict: `{row.manual_verdict or '(blank)'}`",
                f"- failure_class: `{row.failure_class or '(blank)'}`",
                f"- manual_exactness_labels: `{', '.join(row.manual_exactness_labels) if row.manual_exactness_labels else '(none)'}`",
                f"- exactness_review_flags: `{', '.join(row.exactness_review_flags) if row.exactness_review_flags else '(none)'}`",
                f"- answer_changed: `{row.answer_changed}`",
                f"- gold_page_ids: `{row.gold_page_ids}`",
                f"- baseline_used: `{row.baseline_used_page_ids}`",
                f"- candidate_used: `{row.candidate_used_page_ids}`",
                f"- baseline_context: `{row.baseline_context_page_ids}`",
                f"- candidate_context: `{row.candidate_context_page_ids}`",
                f"- baseline_hits: `used={row.baseline_used_hit} context={row.baseline_context_hit}`",
                f"- candidate_hits: `used={row.candidate_used_hit} context={row.candidate_context_hit}`",
                f"- candidate_equivalent_hits: `used={row.candidate_used_equivalent_hit} context={row.candidate_context_equivalent_hit}`",
                f"- required_page_anchor: `{json.dumps(row.required_page_anchor, ensure_ascii=False, sort_keys=True) if row.required_page_anchor else '{}'} `",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a case-by-case anchor-sensitive diff report.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--candidate-raw-results", type=Path, required=True)
    parser.add_argument("--baseline-scaffold", type=Path, required=True)
    parser.add_argument("--candidate-scaffold", type=Path, default=None)
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--qids-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    qids = _load_qids(args)
    rows = build_rows(
        baseline_submission_path=args.baseline_submission,
        candidate_submission_path=args.candidate_submission,
        baseline_scaffold_path=args.baseline_scaffold,
        candidate_scaffold_path=args.candidate_scaffold,
        baseline_raw_results_path=args.baseline_raw_results,
        candidate_raw_results_path=args.candidate_raw_results,
        qids=qids,
    )
    report = render_report(rows, baseline_label=args.baseline_label, candidate_label=args.candidate_label)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)
    if args.json_out is not None:
        payload = {
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "status_counts": _status_counts(rows),
            "rows": [asdict(row) for row in rows],
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
