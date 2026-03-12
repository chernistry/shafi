from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from rag_challenge.submission.common import classify_unanswerable_answer

JsonObject = dict[str, object]
JsonList = list[object]


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

    return {
        "question_id": str(question.get("id") or "").strip(),
        "question": str(question.get("question") or "").strip(),
        "answer_type": answer_type,
        "current_answer": answer_value,
        "current_answer_text": answer_text,
        "model_route": model_name,
        "retrieved_chunk_pages": refs,
        "manual_verdict": "",
        "expected_answer": None,
        "notes": "",
        "flags": {
            "unanswerable_strict": is_unanswerable_strict,
            "unanswerable_free_text": is_unanswerable_free_text,
            "empty_pages": not refs,
            "weak_path": model_name not in {"strict-extractor", "structured-extractor", "premise-guard"},
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

    return {
        "summary": {
            "questions_count": len(questions),
            "answers_count": len(answers),
            "deterministic_count": len(deterministic_cases),
            "free_text_count": len(free_text_cases),
            "missing_questions": missing_questions,
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
