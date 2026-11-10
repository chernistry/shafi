#!/usr/bin/env python3
"""Pre-submission sanity checks for a pipeline submission artifact.

Validates:
  1. Format: type coercion, 280-char limit, names format
  2. Page IDs: valid {doc_hash}_{page_num} format
  3. Coverage: no null answers where reference had answers
  4. Regression guard: flags if >5% questions regressed vs reference
  5. Confidence distribution: flags unusual patterns

Usage:
    PYTHONPATH=src python scripts/pre_submit_sanity_check.py \
        --submission submission.json \
        --reference best_known_submission.json \
        --output sanity_report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

_PAGE_ID_RE = re.compile(r"^[0-9a-f]{64}_\d+$")
_FREE_TEXT_LIMIT = 280


def _g_score(predicted: set[str], gold: set[str], beta: float = 2.5) -> float:
    """Compute grounding F-beta score (mirrors eval/harness.py)."""
    if not gold:
        return 1.0 if not predicted else 0.0
    true_positives = len(predicted & gold)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _answers_by_id(submission: JsonDict) -> dict[str, JsonDict]:
    return {a["question_id"]: a for a in submission.get("answers", [])}


def _used_pages(answer: JsonDict) -> list[str]:
    """Extract used page IDs from answer telemetry."""
    tel = answer.get("telemetry", {})
    used = tel.get("used_page_ids")
    if isinstance(used, list) and used:
        return [str(p) for p in used if p]
    retrieval = tel.get("retrieval", {})
    pages_list = retrieval.get("retrieved_chunk_pages", [])
    page_ids: list[str] = []
    for entry in pages_list:
        if isinstance(entry, dict):
            doc_id = str(entry.get("doc_id", ""))
            for pn in entry.get("page_numbers", []):
                page_ids.append(f"{doc_id}_{pn}")
    return page_ids


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def check_format(
    answers: dict[str, JsonDict],
    questions: list[JsonDict] | None = None,
) -> list[JsonDict]:
    """Check answer format compliance.

    If *questions* is provided (list of {id, answer_type, question}),
    uses ground-truth answer_type instead of telemetry-embedded type.
    """
    issues: list[JsonDict] = []
    q_type_map: dict[str, str] = {}
    if questions:
        q_type_map = {q["id"]: q.get("answer_type", "free_text") for q in questions}

    for qid, answer in answers.items():
        val = answer.get("answer")
        tel = answer.get("telemetry", {})
        answer_type = q_type_map.get(qid) or str(
            tel.get("answer_type", tel.get("retrieval", {}).get("answer_type", "unknown"))
        )

        if val is None:
            continue

        # Boolean check
        if answer_type == "boolean":
            if not isinstance(val, bool):
                issues.append({"question_id": qid, "check": "format", "issue": f"boolean answer is {type(val).__name__}: {val!r}"})

        # Number check
        elif answer_type == "number":
            if not isinstance(val, (int, float)):
                issues.append({"question_id": qid, "check": "format", "issue": f"number answer is {type(val).__name__}: {val!r}"})

        # Names check — should be list[str], alphabetically sorted
        elif answer_type == "names":
            if not isinstance(val, list):
                issues.append({"question_id": qid, "check": "format", "issue": f"names answer is {type(val).__name__}, expected list"})
            elif any(not isinstance(n, str) for n in val):
                issues.append({"question_id": qid, "check": "format", "issue": "names list contains non-string elements"})
            else:
                str_list: list[str] = [str(n) for n in val]
                if len(str_list) > 1 and str_list != sorted(str_list, key=str.lower):
                    issues.append({"question_id": qid, "check": "format", "issue": "names list not alphabetically sorted"})

        # Date — should be ISO YYYY-MM-DD
        elif answer_type == "date":
            if isinstance(val, str) and not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
                issues.append({"question_id": qid, "check": "format", "issue": f"date not ISO format: {val!r}"})

        # Free text checks
        elif answer_type == "free_text":
            if isinstance(val, str):
                if len(val) > _FREE_TEXT_LIMIT:
                    issues.append({"question_id": qid, "check": "format", "issue": f"exceeds {_FREE_TEXT_LIMIT} chars ({len(val)})"})
                if "(cite:" in val:
                    issues.append({"question_id": qid, "check": "format", "issue": "residual (cite:) marker"})
                if val.rstrip() and val.rstrip()[-1] not in '.!?)"\'':
                    issues.append({"question_id": qid, "check": "format", "issue": f"missing terminal punctuation, ends with '{val[-3:]}'"})
                stripped = val.rstrip(".")
                if len(val) < 30 and "no information" not in val.lower() and stripped.endswith(" No"):
                    issues.append({"question_id": qid, "check": "format", "issue": f"garbage fragment: {val!r}"})
                if re.search(r"(?:Law|Order|Paper|Resolution)\s+No\.\s*$", val):
                    issues.append({"question_id": qid, "check": "format", "issue": "truncated law number at end"})
                if re.search(r"\n\d+\.\s*$", val):
                    issues.append({"question_id": qid, "check": "format", "issue": "truncated numbered list at end"})

        # Name — should be str
        elif answer_type == "name":
            if isinstance(val, str) and len(val) > _FREE_TEXT_LIMIT:
                issues.append({"question_id": qid, "check": "format", "issue": f"exceeds {_FREE_TEXT_LIMIT} chars ({len(val)})"})

    return issues


def check_page_ids(answers: dict[str, JsonDict]) -> list[JsonDict]:
    """Validate page ID format."""
    issues: list[JsonDict] = []
    all_doc_hashes: set[str] = set()

    for qid, answer in answers.items():
        pages = _used_pages(answer)
        for page_id in pages:
            if not _PAGE_ID_RE.match(page_id):
                issues.append({
                    "question_id": qid,
                    "check": "page_id_format",
                    "issue": f"invalid page ID format: {page_id!r}",
                })
            else:
                doc_hash = page_id.rsplit("_", 1)[0]
                all_doc_hashes.add(doc_hash)

    return issues


def check_coverage(
    answers: dict[str, JsonDict],
    reference_answers: dict[str, JsonDict],
) -> list[JsonDict]:
    """Flag questions where submission has null but reference had an answer."""
    issues: list[JsonDict] = []

    for qid in reference_answers:
        ref_val = reference_answers[qid].get("answer")
        sub_val = answers.get(qid, {}).get("answer")

        if ref_val is not None and sub_val is None:
            issues.append({
                "question_id": qid,
                "check": "coverage",
                "issue": f"null answer where reference had: {json.dumps(ref_val)[:80]}",
            })

    missing_qids = set(reference_answers) - set(answers)
    for qid in sorted(missing_qids):
        issues.append({
            "question_id": qid,
            "check": "coverage",
            "issue": "question missing entirely from submission",
        })

    return issues


def check_regression(
    answers: dict[str, JsonDict],
    reference_answers: dict[str, JsonDict],
    golden_map: dict[str, set[str]] | None = None,
) -> list[JsonDict]:
    """Flag if >5% of questions regressed vs reference (answer changed + G dropped)."""
    issues: list[JsonDict] = []

    if golden_map is None:
        # Without golden labels, just flag answer changes
        changed = 0
        common = set(answers) & set(reference_answers)
        for qid in common:
            ref_ans = json.dumps(reference_answers[qid].get("answer"), sort_keys=True)
            sub_ans = json.dumps(answers[qid].get("answer"), sort_keys=True)
            if ref_ans != sub_ans:
                changed += 1
        if common and changed / len(common) > 0.20:
            issues.append({
                "question_id": "*",
                "check": "regression",
                "issue": f"{changed}/{len(common)} answers changed ({100*changed/len(common):.1f}%) — review carefully",
            })
        return issues

    # With golden labels, check G-score regressions
    regressions = 0
    common = set(answers) & set(reference_answers) & set(golden_map)
    for qid in common:
        gold = golden_map[qid]
        ref_pages = set(_used_pages(reference_answers[qid]))
        sub_pages = set(_used_pages(answers[qid]))
        ref_g = _g_score(ref_pages, gold)
        sub_g = _g_score(sub_pages, gold)
        if sub_g < ref_g - 0.001:
            regressions += 1

    if common and regressions / len(common) > 0.05:
        issues.append({
            "question_id": "*",
            "check": "regression",
            "issue": f"{regressions}/{len(common)} G-score regressions ({100*regressions/len(common):.1f}%) — exceeds 5% threshold",
        })

    return issues


def check_confidence(answers: dict[str, JsonDict]) -> list[JsonDict]:
    """Check confidence score distribution for unusual patterns."""
    issues: list[JsonDict] = []

    # Count answers by type
    type_counts: dict[str, int] = {}
    null_counts: dict[str, int] = {}
    total = len(answers)

    for answer in answers.values():
        tel = answer.get("telemetry", {})
        at = str(tel.get("answer_type", "unknown"))
        type_counts[at] = type_counts.get(at, 0) + 1
        if answer.get("answer") is None:
            null_counts[at] = null_counts.get(at, 0) + 1

    # Flag high null rates
    total_nulls = sum(null_counts.values())
    if total > 0 and total_nulls / total > 0.15:
        issues.append({
            "question_id": "*",
            "check": "confidence",
            "issue": f"high null answer rate: {total_nulls}/{total} ({100*total_nulls/total:.1f}%)",
        })

    # Flag zero-page answers
    zero_page_count = 0
    for answer in answers.values():
        if answer.get("answer") is not None and not _used_pages(answer):
            zero_page_count += 1
    if zero_page_count > 0:
        issues.append({
            "question_id": "*",
            "check": "confidence",
            "issue": f"{zero_page_count} non-null answers with zero used pages",
        })

    return issues


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(
    submission_path: Path,
    reference_path: Path | None,
    all_issues: list[JsonDict],
    answer_count: int,
) -> str:
    """Render sanity check results as markdown."""
    lines: list[str] = ["# Pre-Submission Sanity Check", ""]
    lines.append(f"- **Submission**: `{submission_path}`")
    if reference_path:
        lines.append(f"- **Reference**: `{reference_path}`")
    lines.append(f"- **Answers**: {answer_count}")
    lines.append("")

    # Group by check type
    by_check: dict[str, list[JsonDict]] = {}
    for issue in all_issues:
        check = issue["check"]
        by_check.setdefault(check, []).append(issue)

    if not all_issues:
        lines.append("## Result: PASS")
        lines.append("")
        lines.append("All checks passed. Safe to submit.")
        return "\n".join(lines) + "\n"

    # Summary
    lines.append(f"## Result: {'FAIL' if len(all_issues) > 5 else 'WARNING'} — {len(all_issues)} issues found")
    lines.append("")

    check_labels = {
        "format": "Format Validation",
        "page_id_format": "Page ID Validation",
        "coverage": "Coverage Check",
        "regression": "Regression Guard",
        "confidence": "Confidence Distribution",
    }

    for check_type in ("format", "page_id_format", "coverage", "regression", "confidence"):
        issues = by_check.get(check_type, [])
        label = check_labels.get(check_type, check_type)
        status = f"PASS ({len(issues)} issues)" if not issues else f"**{len(issues)} issues**"
        lines.append(f"### {label}: {status}")
        lines.append("")
        if issues:
            for issue in issues[:20]:  # cap display
                qid = issue["question_id"]
                qid_display = f"`{qid[:12]}...`" if len(qid) > 12 else f"`{qid}`"
                lines.append(f"- {qid_display}: {issue['issue']}")
            if len(issues) > 20:
                lines.append(f"- ... and {len(issues) - 20} more")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--submission", type=Path, required=True,
                        help="Submission JSON to validate")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Best known submission for regression comparison")
    parser.add_argument("--questions", type=Path, default=None,
                        help="Questions JSON for answer_type validation")
    parser.add_argument("--golden", type=Path, default=None,
                        help="Golden labels for G-score regression check")
    parser.add_argument("--output", type=str, default="sanity_report",
                        help="Output base path (writes .md and .json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.submission.exists():
        print(f"ERROR: Submission not found: {args.submission}", file=sys.stderr)
        return 1

    submission = _load_json(args.submission)
    answers = _answers_by_id(submission)

    all_issues: list[JsonDict] = []

    # Load questions for answer_type mapping
    questions: list[JsonDict] | None = None
    if args.questions and args.questions.exists():
        questions = _load_json(args.questions)
    else:
        # Try default location
        default_q = Path("dataset/private/questions.json")
        if default_q.exists():
            questions = _load_json(default_q)

    # 1. Format validation
    all_issues.extend(check_format(answers, questions))

    # 2. Page ID validation
    all_issues.extend(check_page_ids(answers))

    # 3-4. Coverage and regression (need reference)
    reference_answers: dict[str, JsonDict] = {}
    if args.reference and args.reference.exists():
        reference = _load_json(args.reference)
        reference_answers = _answers_by_id(reference)
        all_issues.extend(check_coverage(answers, reference_answers))

        golden_map = None
        if args.golden and args.golden.exists():
            golden_data = _load_json(args.golden)
            golden_map = {
                case["id"]: set(str(g) for g in case.get("gold_chunk_ids", []) if g)
                for case in golden_data
            }
        all_issues.extend(check_regression(answers, reference_answers, golden_map))

    # 5. Confidence distribution
    all_issues.extend(check_confidence(answers))

    # Render report
    report = render_report(args.submission, args.reference, all_issues, len(answers))

    out_md = Path(args.output + ".md")
    out_json = Path(args.output + ".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(report, encoding="utf-8")
    out_json.write_text(
        json.dumps({"issues": all_issues, "issue_count": len(all_issues), "answer_count": len(answers)},
                    indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Print summary
    print(report)
    print(f"Written: {out_md}, {out_json}")

    # Non-zero exit if critical issues
    critical = sum(1 for i in all_issues if i["check"] in ("format", "page_id_format"))
    return 1 if critical > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
