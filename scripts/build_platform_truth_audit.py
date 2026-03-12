from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

from rag_challenge.submission.common import classify_unanswerable_answer

JsonObject = dict[str, object]
JsonList = list[object]

_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*\d{1,4}\s*[/-]\s*\d{4}\b", re.IGNORECASE)
_LAW_REF_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_TITLE_REF_RE = re.compile(r"\b[A-Z][A-Za-z0-9&().,'/-]+(?:\s+[A-Z][A-Za-z0-9&().,'/-]+)*\s+Law(?:\s+\d{4})?\b")


def _route_family(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    if normalized == "strict-extractor":
        return "strict"
    if normalized == "structured-extractor":
        return "structured"
    if normalized == "premise-guard":
        return "premise_guard"
    return "model"


def _question_refs(question: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in _CASE_REF_RE.finditer(question or ""):
        ref = re.sub(r"\s+", " ", match.group(0)).strip()
        key = ref.casefold()
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    for match in _LAW_REF_RE.finditer(question or ""):
        ref = re.sub(r"\s+", " ", match.group(0)).strip()
        key = ref.casefold()
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    for match in _TITLE_REF_RE.finditer(question or ""):
        ref = re.sub(r"\s+", " ", match.group(0)).strip()
        key = ref.casefold()
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


def _support_shape_class(*, question: str, answer_type: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    refs = _question_refs(question)
    if answer_type != "free_text" and len(_CASE_REF_RE.findall(question or "")) == 2:
        return "comparison"
    if answer_type == "free_text" and (
        "how did the court of appeal rule" in q
        or "it is hereby ordered that" in q
        or "result of the application" in q
        or "outcome of the specific order or application" in q
        or (("outcome" in q or "result" in q) and any(term in q for term in ("application", "appeal", "order")))
    ):
        return "outcome_plus_costs" if ("cost" in q or "final ruling" in q) else "case_outcome"
    if refs and any(
        term in q
        for term in (
            "title",
            "citation title",
            "updated",
            "consolidated version",
            "published",
            "enact",
            "effective date",
            "commencement",
            "administ",
            "made by",
            "who made",
        )
    ):
        return "named_metadata"
    return "generic"


def _support_shape_flags(
    *,
    support_shape_class: str,
    retrieved_chunk_pages: list[dict[str, object]],
    answer_text: str,
    answer_type: str,
) -> tuple[int, int, list[str]]:
    doc_ids = [
        str(item.get("doc_id") or "").strip()
        for item in retrieved_chunk_pages
        if isinstance(item, dict) and str(item.get("doc_id") or "").strip()
    ]
    page_count = sum(
        len(
            [
                page
                for page in cast("list[object]", item.get("page_numbers"))
                if isinstance(page, int)
            ]
        )
        for item in retrieved_chunk_pages
        if isinstance(item, dict) and isinstance(item.get("page_numbers"), list)
    )
    doc_count = len(dict.fromkeys(doc_ids))
    flags: list[str] = []
    if support_shape_class == "comparison" and doc_count < 2:
        flags.append("comparison_missing_side")
    if support_shape_class == "outcome_plus_costs" and page_count < 2:
        flags.append("multi_slot_support_maybe_undercovered")
    if support_shape_class == "named_metadata" and page_count < 2:
        flags.append("metadata_support_maybe_undercovered")
    normalized_answer = answer_text.strip().lower()
    if answer_type == "free_text" and normalized_answer.startswith("there is no information") and page_count > 0:
        flags.append("unsupported_with_support_pages")
    return doc_count, page_count, flags


def _load_questions(path: Path) -> dict[str, JsonObject]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    payload = cast("JsonList", payload_obj) if isinstance(payload_obj, list) else None
    if payload is None:
        raise ValueError("questions.json must be a list")
    questions: dict[str, JsonObject] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = cast("JsonObject", item)
        question_id = str(question.get("id") or "").strip()
        if question_id:
            questions[question_id] = question
    return questions


def _load_answers(path: Path) -> list[JsonObject]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload_obj, dict):
        payload = cast("JsonObject", payload_obj)
        answers_obj = payload.get("answers")
        if isinstance(answers_obj, list):
            answers = cast("JsonList", answers_obj)
            return [cast("JsonObject", item) for item in answers if isinstance(item, dict)]
    if isinstance(payload_obj, list):
        payload = cast("JsonList", payload_obj)
        return [cast("JsonObject", item) for item in payload if isinstance(item, dict)]
    raise ValueError("submission.json must be a platform payload or list of answer objects")


def _build_case_record(
    *,
    question: JsonObject,
    answer_payload: JsonObject,
) -> JsonObject:
    answer_type = str(question.get("answer_type") or "free_text").strip().lower() or "free_text"
    answer_value = answer_payload.get("answer")
    telemetry_obj = answer_payload.get("telemetry")
    telemetry = cast("dict[str, object]", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    retrieval_obj = telemetry.get("retrieval")
    retrieval = cast("dict[str, object]", retrieval_obj) if isinstance(retrieval_obj, dict) else {}
    refs_obj = retrieval.get("retrieved_chunk_pages")
    refs = cast("list[dict[str, object]]", refs_obj) if isinstance(refs_obj, list) else []

    answer_text = answer_value if isinstance(answer_value, str) else json.dumps(answer_value, ensure_ascii=False)
    is_unanswerable_strict, is_unanswerable_free_text = classify_unanswerable_answer(answer_text, answer_type)
    model_name = str(telemetry.get("model_name") or "").strip()
    route_family = _route_family(model_name)
    shape_class = _support_shape_class(question=str(question.get("question") or ""), answer_type=answer_type)
    support_doc_count, support_page_count, support_shape_flags = _support_shape_flags(
        support_shape_class=shape_class,
        retrieved_chunk_pages=refs,
        answer_text=answer_text,
        answer_type=answer_type,
    )

    return {
        "question_id": str(question.get("id") or "").strip(),
        "question": str(question.get("question") or "").strip(),
        "question_refs": _question_refs(str(question.get("question") or "")),
        "answer_type": answer_type,
        "current_answer": answer_value,
        "current_answer_text": answer_text,
        "model_route": model_name,
        "route_family": route_family,
        "retrieved_chunk_pages": refs,
        "support_shape_class": shape_class,
        "support_doc_count": support_doc_count,
        "support_page_count": support_page_count,
        "support_shape_flags": support_shape_flags,
        "manual_verdict": "",
        "expected_answer": None,
        "minimal_required_support_pages": [],
        "failure_class": "",
        "notes": "",
        "flags": {
            "unanswerable_strict": is_unanswerable_strict,
            "unanswerable_free_text": is_unanswerable_free_text,
            "empty_pages": not refs,
            "weak_path": route_family == "model",
        },
    }


def build_truth_audit_scaffold(
    *,
    questions_path: Path,
    submission_path: Path,
) -> dict[str, object]:
    questions = _load_questions(questions_path)
    answers = _load_answers(submission_path)

    deterministic_cases: list[dict[str, object]] = []
    free_text_cases: list[dict[str, object]] = []
    missing_questions: list[str] = []

    for answer_payload in answers:
        question_id = str(answer_payload.get("question_id") or "").strip()
        question = questions.get(question_id)
        if question is None:
            missing_questions.append(question_id)
            continue
        record = _build_case_record(question=question, answer_payload=answer_payload)
        if str(record["answer_type"]) == "free_text":
            free_text_cases.append(record)
        else:
            deterministic_cases.append(record)

    deterministic_cases.sort(key=lambda item: (str(item["answer_type"]), str(item["question_id"])))
    free_text_cases.sort(key=lambda item: str(item["question_id"]))

    route_family_counts: dict[str, int] = {}
    support_shape_counts: dict[str, int] = {}
    for record in [*deterministic_cases, *free_text_cases]:
        route_family = str(record.get("route_family") or "unknown")
        support_shape_class = str(record.get("support_shape_class") or "generic")
        route_family_counts[route_family] = route_family_counts.get(route_family, 0) + 1
        support_shape_counts[support_shape_class] = support_shape_counts.get(support_shape_class, 0) + 1

    return {
        "summary": {
            "questions_count": len(questions),
            "answers_count": len(answers),
            "deterministic_count": len(deterministic_cases),
            "free_text_count": len(free_text_cases),
            "missing_questions": missing_questions,
            "route_family_counts": route_family_counts,
            "support_shape_class_counts": support_shape_counts,
        },
        "deterministic_cases": deterministic_cases,
        "free_text_cases": free_text_cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a manual truth-audit scaffold from platform questions and submission artifacts.")
    parser.add_argument("--questions", required=True, help="Path to platform questions.json")
    parser.add_argument("--submission", required=True, help="Path to platform submission.json")
    parser.add_argument("--out", help="Output path for the generated truth-audit scaffold JSON")
    args = parser.parse_args(argv)

    questions_path = Path(args.questions)
    submission_path = Path(args.submission)
    out_path = Path(args.out) if args.out else submission_path.with_name("truth_audit_scaffold.json")

    scaffold = build_truth_audit_scaffold(
        questions_path=questions_path,
        submission_path=submission_path,
    )
    out_path.write_text(json.dumps(scaffold, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
