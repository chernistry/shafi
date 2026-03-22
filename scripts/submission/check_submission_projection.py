from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import cast

from rag_challenge.submission.common import count_submission_sentences
from rag_challenge.submission.generate import _project_submission_result

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PAGE_ID_RE = re.compile(r"^.+_(\d+)$")


@dataclass(frozen=True)
class ProjectionIssue:
    question_id: str
    answer_type: str
    issue: str
    detail: str


def _percentile_int(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return int(ordered[idx])


def _classify_projection_change(raw_answer: object, projected_answer: object, answer_type: str) -> str:
    kind = answer_type.strip().lower()
    if json.dumps(raw_answer, ensure_ascii=False, sort_keys=True) == json.dumps(
        projected_answer,
        ensure_ascii=False,
        sort_keys=True,
    ):
        return "unchanged"

    if kind == "boolean" and isinstance(projected_answer, bool):
        return "boolean_coercion"
    if kind == "date" and isinstance(projected_answer, str):
        return "date_normalization"
    if kind == "number" and isinstance(projected_answer, int | float):
        return "number_coercion"
    if kind == "free_text":
        raw_text = "" if raw_answer is None else str(raw_answer)
        projected_text = projected_answer if isinstance(projected_answer, str) else ""
        raw_stripped = raw_text.strip()
        projected_stripped = projected_text.strip()
        if projected_stripped.lower().startswith("there is no information on this question"):
            return "free_text_unanswerable_text"
        if "(cite:" in raw_stripped.lower() and "(cite:" not in projected_stripped.lower():
            if len(projected_stripped) == len(strip_inline_citations(raw_stripped).strip()):
                return "citation_stripping"
            return "citation_stripping_plus_trim"
        if count_submission_sentences(projected_stripped) < count_submission_sentences(raw_stripped):
            return "free_text_sentence_trimming"
        if len(projected_stripped) < len(raw_stripped):
            return "free_text_length_trimming"
    return "other"


def strip_inline_citations(text: str) -> str:
    return re.sub(r"\s*\(cite:[^)]+\)", "", text)


def _classify_free_text_fragment(projected_answer: str) -> str | None:
    text = projected_answer.strip()
    if not text:
        return "empty_fragment"
    if text.endswith("..."):
        return "ellipsis_truncated_tail"
    if re.search(r"(?:^|[; ])\d+\.$", text):
        return "dangling_list_marker"
    if text.endswith("No.") or text.endswith("No") or text.endswith(":"):
        return "heading_like_fragment"
    return None


def _coerce_cases(eval_obj: object) -> list[dict[str, object]]:
    if isinstance(eval_obj, dict):
        cases_obj = eval_obj.get("cases")
        if isinstance(cases_obj, list):
            return [cast("dict[str, object]", item) for item in cases_obj if isinstance(item, dict)]
    if isinstance(eval_obj, list):
        return [cast("dict[str, object]", item) for item in eval_obj if isinstance(item, dict)]
    raise ValueError("Eval artifact must contain a top-level 'cases' list or be a list of case objects")


def _is_submission_answer_compliant(answer: object, answer_type: str) -> tuple[bool, str]:
    kind = answer_type.strip().lower()
    if kind == "boolean":
        return isinstance(answer, bool) or answer is None, "boolean must be JSON true/false/null"
    if kind == "number":
        return ((isinstance(answer, int) and not isinstance(answer, bool)) or isinstance(answer, float) or answer is None), (
            "number must be JSON number/null"
        )
    if kind == "date":
        return (answer is None) or (isinstance(answer, str) and _ISO_DATE_RE.fullmatch(answer.strip()) is not None), (
            "date must be ISO YYYY-MM-DD or null"
        )
    if kind == "name":
        return answer is None or isinstance(answer, str), "name must be string/null"
    if kind == "names":
        return (
            answer is None
            or (
                isinstance(answer, list)
                and all(isinstance(item, str) and item.strip() for item in cast("list[object]", answer))
            )
        ), "names must be list[str]/null"
    if kind == "free_text":
        sentence_count = count_submission_sentences(answer) if isinstance(answer, str) else 0
        return (
            isinstance(answer, str)
            and bool(answer.strip())
            and len(answer) <= 280
            and 1 <= sentence_count <= 3
            and "(cite:" not in answer.lower()
        ), "free_text must be 1-3 sentences, non-empty, <=280 chars, without inline citations"
    return False, f"unsupported answer_type={answer_type}"


def _is_page_list_compliant(page_ids: list[str], *, answer: object, answer_type: str) -> tuple[bool, str]:
    for page_id in page_ids:
        match = _PAGE_ID_RE.fullmatch(page_id.strip())
        if match is None:
            return False, f"invalid page id format: {page_id}"
        if int(match.group(1)) < 1:
            return False, f"page id must be 1-based: {page_id}"

    answer_type_key = answer_type.strip().lower()
    if answer is None and answer_type_key in {"boolean", "number", "date", "name", "names"} and page_ids:
        return False, "strict null answer must have empty retrieved_chunk_ids"
    if (
        answer_type_key == "free_text"
        and isinstance(answer, str)
        and answer.lower().startswith("there is no information on this question")
        and page_ids
    ):
        return False, "free_text unanswerable answer must have empty retrieved_chunk_ids"
    return True, ""


def _project_case(case: dict[str, object]) -> dict[str, object]:
    question_id = str(case.get("question_id") or case.get("case_id") or "").strip()
    answer_type = str(case.get("answer_type") or "free_text").strip() or "free_text"
    answer_obj = case.get("answer")
    # For list answers (e.g. "names" type), join elements rather than
    # using str() which produces "['Alice']" Python repr.
    if answer_obj is None:
        answer_text = ""
    elif isinstance(answer_obj, list):
        answer_text = ", ".join(str(item) for item in answer_obj)
    else:
        answer_text = str(answer_obj)
    telemetry_obj = case.get("telemetry")
    telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    total_ms_fallback = int(float(case.get("ttft_ms") or 0))
    return _project_submission_result(
        case_id=question_id,
        answer_type=answer_type,
        answer_text=answer_text,
        telemetry=telemetry,
        total_ms_fallback=total_ms_fallback,
    )


def _build_report(eval_path: Path, cases: list[dict[str, object]]) -> str:
    issues: list[ProjectionIssue] = []
    total = len(cases)
    changed_answers = 0
    changed_page_sets = 0
    by_type: dict[str, int] = {}
    answer_changes_by_type: dict[str, int] = {}
    change_classes: dict[str, int] = {}
    null_flip_count = 0
    char_deltas: list[int] = []
    sentence_deltas: list[int] = []
    free_text_sentence_loss_cases = 0
    free_text_hit_limit_cases = 0
    free_text_unanswerable_projection_cases = 0
    free_text_fragment_buckets: dict[str, int] = {}

    for case in cases:
        answer_type = str(case.get("answer_type") or "free_text").strip().lower() or "free_text"
        by_type[answer_type] = by_type.get(answer_type, 0) + 1
        projected = _project_case(case)

        raw_answer = case.get("answer")
        projected_answer = projected.get("answer")
        if json.dumps(raw_answer, ensure_ascii=False, sort_keys=True) != json.dumps(
            projected_answer,
            ensure_ascii=False,
            sort_keys=True,
        ):
            changed_answers += 1
            answer_changes_by_type[answer_type] = answer_changes_by_type.get(answer_type, 0) + 1
            change_class = _classify_projection_change(raw_answer, projected_answer, answer_type)
            change_classes[change_class] = change_classes.get(change_class, 0) + 1
            if (raw_answer is None) != (projected_answer is None):
                null_flip_count += 1
            if answer_type == "free_text":
                raw_text = "" if raw_answer is None else str(raw_answer)
                projected_text = projected_answer if isinstance(projected_answer, str) else ""
                char_deltas.append(len(projected_text) - len(raw_text))
                sentence_deltas.append(
                    count_submission_sentences(projected_text) - count_submission_sentences(raw_text)
                )
                if count_submission_sentences(projected_text) < count_submission_sentences(raw_text):
                    free_text_sentence_loss_cases += 1
                if isinstance(projected_answer, str) and len(projected_answer) >= 280:
                    free_text_hit_limit_cases += 1
                if isinstance(projected_answer, str) and projected_answer.lower().startswith(
                    "there is no information on this question"
                ):
                    free_text_unanswerable_projection_cases += 1

        telemetry_obj = case.get("telemetry")
        telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        raw_used_pages = telemetry.get("used_page_ids")
        raw_pages = [str(item).strip() for item in cast("list[object]", raw_used_pages)] if isinstance(raw_used_pages, list) else []
        projected_pages = cast("list[str]", projected.get("retrieved_chunk_ids") or [])
        if raw_pages != projected_pages:
            changed_page_sets += 1

        answer_ok, answer_detail = _is_submission_answer_compliant(projected_answer, answer_type)
        if not answer_ok:
            issues.append(
                ProjectionIssue(
                    question_id=str(case.get("question_id") or case.get("case_id") or ""),
                    answer_type=answer_type,
                    issue="answer",
                    detail=answer_detail,
                )
            )
        if answer_type == "free_text":
            fragment_source = projected_answer if isinstance(projected_answer, str) else (str(raw_answer) if raw_answer is not None else "")
            fragment_issue = _classify_free_text_fragment(fragment_source) if fragment_source else None
            if fragment_issue is not None:
                free_text_fragment_buckets[fragment_issue] = free_text_fragment_buckets.get(fragment_issue, 0) + 1
                issues.append(
                    ProjectionIssue(
                        question_id=str(case.get("question_id") or case.get("case_id") or ""),
                        answer_type=answer_type,
                        issue="artifact_fragment",
                        detail=fragment_issue,
                    )
                )

        pages_ok, page_detail = _is_page_list_compliant(projected_pages, answer=projected_answer, answer_type=answer_type)
        if not pages_ok:
            issues.append(
                ProjectionIssue(
                    question_id=str(case.get("question_id") or case.get("case_id") or ""),
                    answer_type=answer_type,
                    issue="pages",
                    detail=page_detail,
                )
            )

    free_text_projected = [
        cast("str", _project_case(case).get("answer"))
        for case in cases
        if str(case.get("answer_type") or "").strip().lower() == "free_text"
        and isinstance(_project_case(case).get("answer"), str)
    ]
    max_free_text_len = max((len(answer) for answer in free_text_projected), default=0)
    max_free_text_sentences = max((count_submission_sentences(answer) for answer in free_text_projected), default=0)
    bool_cases = sum(1 for case in cases if str(case.get("answer_type") or "").strip().lower() == "boolean")
    projected_bool_answers = [_project_case(case).get("answer") for case in cases if str(case.get("answer_type") or "").strip().lower() == "boolean"]
    boolean_json_safe = sum(1 for answer in projected_bool_answers if isinstance(answer, bool) or answer is None)

    lines = [
        "# Submission Projection Check",
        "",
        f"- Eval artifact: `{eval_path.name}`",
        f"- Total cases: `{total}`",
        f"- Submission-compliant cases: `{total - len(issues)}/{total}`",
        f"- Cases with projected answer changes vs eval artifact: `{changed_answers}`",
        f"- Cases with projected retrieved pages changes vs eval telemetry: `{changed_page_sets}`",
        f"- Null flips during projection: `{null_flip_count}`",
        f"- Free-text projected max length: `{max_free_text_len}`",
        f"- Free-text projected max sentences: `{max_free_text_sentences}`",
        f"- Boolean JSON-safe after projection: `{boolean_json_safe}/{bool_cases}`",
        "",
        "## By Answer Type",
        "",
    ]
    for answer_type in sorted(by_type):
        lines.append(f"- `{answer_type}`: `{by_type[answer_type]}`")

    lines.extend(["", "## Projection Change Severity", ""])
    if not change_classes:
        lines.append("- No answer changes")
    else:
        for label in sorted(change_classes):
            lines.append(f"- `{label}`: `{change_classes[label]}`")
        lines.extend(
            [
                "",
                "## Changed Answers By Type",
                "",
            ]
        )
        for answer_type in sorted(answer_changes_by_type):
            lines.append(f"- `{answer_type}`: `{answer_changes_by_type[answer_type]}`")
        lines.extend(
            [
                "",
                "## Free-Text Change Deltas",
                "",
                f"- Projected char delta p50: `{int(median(char_deltas)) if char_deltas else 0}`",
                f"- Projected char delta p95: `{_percentile_int(char_deltas, 0.95)}`",
                f"- Projected char delta max: `{max(char_deltas, default=0)}`",
                f"- Projected sentence delta p50: `{int(median(sentence_deltas)) if sentence_deltas else 0}`",
                f"- Projected sentence delta p95: `{_percentile_int(sentence_deltas, 0.95)}`",
                f"- Projected sentence delta max: `{max(sentence_deltas, default=0)}`",
                f"- Free-text cases losing 1+ sentence: `{free_text_sentence_loss_cases}`",
                f"- Free-text cases clipped at 280 chars: `{free_text_hit_limit_cases}`",
                f"- Free-text cases projected to unanswerable text: `{free_text_unanswerable_projection_cases}`",
            ]
        )
    lines.extend(["", "## Free-Text Artifact Fragments", ""])
    if free_text_fragment_buckets:
        for issue_name in sorted(free_text_fragment_buckets):
            lines.append(f"- `{issue_name}`: `{free_text_fragment_buckets[issue_name]}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Issues", ""])
    if not issues:
        lines.append("- None")
    else:
        for issue in issues:
            lines.append(
                f"- `{issue.question_id}` [{issue.answer_type}] `{issue.issue}`: {issue.detail}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Project an eval artifact into submission format and report compliance gaps."""
    parser = argparse.ArgumentParser(description="Check submission-format compliance by projecting an eval artifact.")
    parser.add_argument("--eval", required=True, help="Path to eval JSON artifact")
    parser.add_argument("--out", help="Optional markdown report output path")
    args = parser.parse_args(argv)

    eval_path = Path(args.eval)
    eval_obj = json.loads(eval_path.read_text(encoding="utf-8"))
    cases = _coerce_cases(eval_obj)
    report = _build_report(eval_path, cases)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
