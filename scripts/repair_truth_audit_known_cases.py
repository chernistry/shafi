from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from build_platform_truth_audit import render_truth_audit_workbook

JsonDict = dict[str, object]

_QID_43 = "43f77ed8a37c7af9b3e52b0532c593c768f8f1159db9b9ca717a700d6b0a47f9"
_QID_F950 = "f950917f9b85f687161b1022a11c3ce31e4f6ab459af69dfea311c20893fc8a7"


@dataclass(frozen=True)
class RepairRule:
    question_id: str
    stale_surface: str
    corrected_answer: object
    manual_exactness_labels: list[str]
    failure_class: str
    notes: str


@dataclass(frozen=True)
class RepairResult:
    scaffold_path: str
    changed_question_ids: list[str]
    workbook_path: str | None


_REPAIR_RULES: tuple[RepairRule, ...] = (
    RepairRule(
        question_id=_QID_43,
        stale_surface="Architeriors Interior Design (LLC)",
        corrected_answer=["Architeriors Interior Design (L.L.C)"],
        manual_exactness_labels=["platform_exact_risk", "suffix_risk"],
        failure_class="wrong_strict_extraction",
        notes=(
            "Direct title-page source span shows the claimant as Architeriors Interior Design (L.L.C). "
            "Platform exactness already validated that the undotted LLC normalization was too lenient."
        ),
    ),
    RepairRule(
        question_id=_QID_F950,
        stale_surface="Coinmena BSC (C)",
        corrected_answer=["Coinmena B.S.C. (C)"],
        manual_exactness_labels=["platform_exact_risk", "suffix_risk"],
        failure_class="wrong_strict_extraction",
        notes=(
            "Direct title-page source span shows the claimant as Coinmena B.S.C. (C). "
            "Platform exactness already validated that the undotted BSC normalization was too lenient."
        ),
    ),
)


def _load_scaffold(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", payload)


def _answer_surface_text(record: JsonDict) -> str:
    current_answer = record.get("current_answer")
    if isinstance(current_answer, list):
        return json.dumps(current_answer, ensure_ascii=False)
    if isinstance(current_answer, str):
        return current_answer
    return str(record.get("current_answer_text") or "")


def _recompute_summary(scaffold: JsonDict) -> None:
    summary_obj = scaffold.get("summary")
    if not isinstance(summary_obj, dict):
        return
    summary = cast("JsonDict", summary_obj)
    records_obj = scaffold.get("records")
    if not isinstance(records_obj, list):
        return
    raw_records = cast("list[object]", records_obj)
    records = [cast("JsonDict", item) for item in raw_records if isinstance(item, dict)]
    deterministic_cases = [
        record
        for record in records
        if str(record.get("answer_type") or "").strip().lower() != "free_text"
    ]
    free_text_cases = [
        record
        for record in records
        if str(record.get("answer_type") or "").strip().lower() == "free_text"
    ]
    manual_exactness_label_counts: dict[str, int] = {}
    manual_verdict_counts = {
        "deterministic_complete": 0,
        "deterministic_incomplete": 0,
        "free_text_complete": 0,
        "free_text_incomplete": 0,
    }
    for record in records:
        answer_type = str(record.get("answer_type") or "").strip().lower()
        complete = bool(str(record.get("manual_verdict") or "").strip())
        for label in cast("list[object]", record.get("manual_exactness_labels") or []):
            text = str(label).strip()
            if text:
                manual_exactness_label_counts[text] = manual_exactness_label_counts.get(text, 0) + 1
        if answer_type == "free_text":
            manual_verdict_counts["free_text_complete" if complete else "free_text_incomplete"] += 1
        else:
            manual_verdict_counts["deterministic_complete" if complete else "deterministic_incomplete"] += 1
    scaffold["deterministic_cases"] = deterministic_cases
    scaffold["free_text_cases"] = free_text_cases
    summary["deterministic_count"] = len(deterministic_cases)
    summary["free_text_count"] = len(free_text_cases)
    summary["manual_exactness_label_counts"] = manual_exactness_label_counts
    summary["manual_verdict_counts"] = manual_verdict_counts


def repair_scaffold(scaffold: JsonDict) -> list[str]:
    changed_question_ids: list[str] = []
    records_obj = scaffold.get("records")
    if not isinstance(records_obj, list):
        raise ValueError("Scaffold is missing 'records'")
    raw_records = cast("list[object]", records_obj)
    records = [cast("JsonDict", item) for item in raw_records if isinstance(item, dict)]
    rules_by_id = {rule.question_id: rule for rule in _REPAIR_RULES}
    for record in records:
        question_id = str(record.get("question_id") or "").strip()
        rule = rules_by_id.get(question_id)
        if rule is None:
            continue
        if rule.stale_surface not in _answer_surface_text(record):
            continue
        record["manual_verdict"] = "incorrect"
        record["expected_answer"] = rule.corrected_answer
        record["manual_exactness_labels"] = list(rule.manual_exactness_labels)
        record["failure_class"] = rule.failure_class
        record["notes"] = rule.notes
        review_packet_obj = record.get("review_packet")
        if isinstance(review_packet_obj, dict):
            review_packet = cast("JsonDict", review_packet_obj)
            review_packet["manual_exactness_labels"] = list(rule.manual_exactness_labels)
        changed_question_ids.append(question_id)
    _recompute_summary(scaffold)
    return changed_question_ids


def _default_workbook_path(scaffold_path: Path) -> Path:
    name = scaffold_path.name.replace("truth_audit_scaffold", "truth_audit_workbook")
    return scaffold_path.with_name(name).with_suffix(".md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair known exactness-truth mismatches in truth-audit scaffolds.")
    parser.add_argument("--scaffold", type=Path, action="append", required=True)
    parser.add_argument("--rewrite-workbook", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results: list[RepairResult] = []
    for scaffold_path in args.scaffold:
        scaffold = _load_scaffold(scaffold_path)
        changed_question_ids = repair_scaffold(scaffold)
        scaffold_path.write_text(json.dumps(scaffold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        workbook_path: Path | None = None
        if args.rewrite_workbook:
            workbook_path = _default_workbook_path(scaffold_path)
            workbook_path.write_text(render_truth_audit_workbook(scaffold) + "\n", encoding="utf-8")
        results.append(
            RepairResult(
                scaffold_path=str(scaffold_path),
                changed_question_ids=changed_question_ids,
                workbook_path=str(workbook_path) if workbook_path is not None else None,
            )
        )

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps({"results": [asdict(result) for result in results]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
