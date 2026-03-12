from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

from rag_challenge.eval.sources import PdfPageTextProvider
from rag_challenge.submission.common import classify_unanswerable_answer

JsonObject = dict[str, object]
JsonList = list[object]

_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*\d{1,4}\s*[/-]\s*\d{4}\b", re.IGNORECASE)
_LAW_REF_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_TITLE_REF_RE = re.compile(r"\b[A-Z][A-Za-z0-9&().,'/-]+(?:\s+[A-Z][A-Za-z0-9&().,'/-]+)*\s+Law(?:\s+\d{4})?\b")
_WHITESPACE_RE = re.compile(r"\s+")
_DISPOSITION_RE = re.compile(
    r"\b(refused|dismissed|granted|allowed|discharged|set aside|struck out|denied|upheld|granted in part)\b",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
_FIRST_PAGE_RE = re.compile(r"\bfirst page\b", re.IGNORECASE)
_SECOND_PAGE_RE = re.compile(r"\bsecond page\b", re.IGNORECASE)
_TITLE_COVER_PAGE_RE = re.compile(r"\btitle(?:/cover)? page(?:s)?\b|\bcover page(?:s)?\b", re.IGNORECASE)
_LAST_PAGE_RE = re.compile(r"\blast page\b", re.IGNORECASE)

_MANUAL_FIELDS = (
    "manual_verdict",
    "expected_answer",
    "minimal_required_support_pages",
    "failure_class",
    "notes",
)

_WORKBOOK_PREVIEW_LIMIT = 3


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


def _metadata_support_requirements(question: str) -> dict[str, object]:
    q = re.sub(r"\s+", " ", (question or "").strip()).lower()
    refs = _question_refs(question)
    multiple_named_refs = (
        " and " in q
        and (
            len(_LAW_REF_RE.findall(question or "")) >= 2
            or len(_TITLE_REF_RE.findall(question or "")) >= 2
            or len(_CASE_REF_RE.findall(question or "")) >= 2
        )
    )
    atoms: list[str] = []

    if "citation title" in q or "what is the title" in q:
        atoms.append("title")
    if "official law number" in q or "official difc law number" in q:
        atoms.append("law_number")
    if any(term in q for term in ("updated", "consolidated version", "published")):
        atoms.append("publication")
    if any(term in q for term in ("enact", "effective date", "commencement")):
        atoms.append("date")
    if "administ" in q:
        atoms.append("administration")
    if "made by" in q or "who made" in q:
        atoms.append("maker")

    deduped_atoms = list(dict.fromkeys(atoms))
    ref_count = len(refs)
    requires_multi_page = len(deduped_atoms) >= 2 or (multiple_named_refs and len(deduped_atoms) >= 1)

    if "and any regulations made under it" in q:
        requires_multi_page = False

    requires_title_anchor = any(
        term in q for term in ("title page", "citation title", "official law number", "official difc law number")
    )

    return {
        "atom_count": len(deduped_atoms),
        "atoms": deduped_atoms,
        "ref_count": ref_count,
        "requires_multi_page": requires_multi_page,
        "requires_title_anchor": requires_title_anchor,
    }


def _required_page_anchor(question: str) -> dict[str, object]:
    normalized = re.sub(r"\s+", " ", (question or "").strip())
    if not normalized:
        return {}
    if _TITLE_COVER_PAGE_RE.search(normalized):
        return {"kind": "title_or_cover_page", "pages": [1]}
    if _FIRST_PAGE_RE.search(normalized):
        return {"kind": "explicit_page", "pages": [1]}
    if _SECOND_PAGE_RE.search(normalized):
        return {"kind": "explicit_page", "pages": [2]}
    page_match = _PAGE_NUMBER_RE.search(normalized)
    if page_match is not None:
        return {"kind": "explicit_page", "pages": [int(page_match.group(1))]}
    if _LAST_PAGE_RE.search(normalized):
        return {"kind": "last_page", "pages": []}
    return {}


def _support_shape_flags(
    *,
    support_shape_class: str,
    support_shape_requirements: dict[str, object],
    retrieved_chunk_pages: list[dict[str, object]],
    answer_text: str,
    answer_type: str,
) -> tuple[int, int, list[str]]:
    doc_ids = [
        str(item.get("doc_id") or "").strip()
        for item in retrieved_chunk_pages
        if str(item.get("doc_id") or "").strip()
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
        if isinstance(item.get("page_numbers"), list)
    )
    doc_count = len(dict.fromkeys(doc_ids))
    flags: list[str] = []
    normalized_answer = answer_text.strip().lower()
    if support_shape_class == "comparison" and doc_count < 2:
        flags.append("comparison_missing_side")
    if support_shape_class == "case_outcome" and not _DISPOSITION_RE.search(answer_text):
        flags.append("case_outcome_disposition_maybe_missing")
    if support_shape_class == "outcome_plus_costs" and page_count < 2:
        flags.append("multi_slot_support_maybe_undercovered")
    if support_shape_class == "outcome_plus_costs" and "cost" not in normalized_answer:
        flags.append("outcome_costs_text_missing")
    if (
        support_shape_class == "named_metadata"
        and bool(support_shape_requirements.get("requires_multi_page"))
        and page_count < 2
    ):
        flags.append("metadata_multi_atom_maybe_undercovered")
    if (
        support_shape_class == "named_metadata"
        and bool(support_shape_requirements.get("requires_title_anchor"))
        and not any(
            1 in cast("list[object]", item.get("page_numbers"))
            for item in retrieved_chunk_pages
            if isinstance(item.get("page_numbers"), list)
        )
    ):
        flags.append("metadata_title_anchor_maybe_missing")
    if answer_type == "free_text" and normalized_answer.startswith("there is no information") and page_count > 0:
        flags.append("unsupported_with_support_pages")
    return doc_count, page_count, flags


def _load_existing_manual_fields(path: Path | None) -> dict[str, JsonObject]:
    if path is None or not path.exists():
        return {}
    try:
        payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload_obj, dict):
        return {}
    payload = cast("JsonObject", payload_obj)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        return {}
    preserved: dict[str, JsonObject] = {}
    for item_obj in cast("list[object]", records_obj):
        if not isinstance(item_obj, dict):
            continue
        record = cast("JsonObject", item_obj)
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            continue
        preserved[question_id] = {
            field: record.get(field)
            for field in _MANUAL_FIELDS
            if field in record
        }
    return preserved


def _normalize_preview_text(text: str, *, limit: int = 220) -> str:
    normalized = _WHITESPACE_RE.sub(" ", (text or "").strip())
    if len(normalized) <= limit:
        return normalized
    cutoff = normalized.rfind(" ", 0, limit + 1)
    if cutoff <= 0:
        cutoff = limit
    return normalized[:cutoff].rstrip(" ,;:.") + "..."


def _page_title_guess(provider: PdfPageTextProvider, *, doc_id: str) -> str:
    first_page = provider.get_page_text(doc_id=doc_id, page=1) or ""
    if not first_page.strip():
        return doc_id
    for line in first_page.splitlines():
        candidate = _normalize_preview_text(line, limit=120)
        if candidate:
            return candidate
    return doc_id


def _support_page_previews(
    refs: list[dict[str, object]],
    *,
    provider: PdfPageTextProvider | None,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    previews: list[dict[str, object]] = []
    resolved_titles: dict[str, str] = {}
    if provider is None:
        return previews, resolved_titles

    for ref in refs:
        doc_id = str(ref.get("doc_id") or "").strip()
        page_numbers_obj = ref.get("page_numbers")
        if not doc_id or not isinstance(page_numbers_obj, list):
            continue
        title_guess = resolved_titles.get(doc_id)
        if title_guess is None:
            title_guess = _page_title_guess(provider, doc_id=doc_id)
            resolved_titles[doc_id] = title_guess
        for raw_page in cast("list[object]", page_numbers_obj):
            if not isinstance(raw_page, int | float):
                continue
            page = int(raw_page)
            page_text = provider.get_page_text(doc_id=doc_id, page=page) or ""
            previews.append(
                {
                    "doc_id": doc_id,
                    "page": page,
                    "doc_title": title_guess,
                    "snippet": _normalize_preview_text(page_text),
                }
            )
    return previews, resolved_titles


def _exact_span_candidates(
    refs: list[dict[str, object]],
    *,
    provider: PdfPageTextProvider | None,
    current_answer_text: str,
    question_text: str,
    required_page_anchor: dict[str, object],
    resolved_doc_titles: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    if provider is None:
        return []

    answer_text = (current_answer_text or "").strip()
    answer_token = answer_text.casefold() if answer_text.casefold() not in {"", "yes", "no", "true", "false", "null"} else ""
    question_refs = [ref.casefold() for ref in _question_refs(question_text)]
    anchor_pages = {
        int(page)
        for page in cast("list[object]", required_page_anchor.get("pages") or [])
        if isinstance(page, int | float)
    }
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()

    for ref in refs:
        doc_id = str(ref.get("doc_id") or "").strip()
        page_numbers_obj = ref.get("page_numbers")
        if not doc_id or not isinstance(page_numbers_obj, list):
            continue
        doc_title = (resolved_doc_titles or {}).get(doc_id) or _page_title_guess(provider, doc_id=doc_id)
        for raw_page in cast("list[object]", page_numbers_obj):
            if not isinstance(raw_page, int | float):
                continue
            page = int(raw_page)
            page_text = provider.get_page_text(doc_id=doc_id, page=page) or ""
            if not page_text.strip():
                continue
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            matched_lines: list[str] = []
            for line in lines:
                normalized = line.casefold()
                if answer_token and answer_token in normalized:
                    matched_lines.append(line)
                    continue
                if any(ref_token in normalized for ref_token in question_refs):
                    matched_lines.append(line)
            if not matched_lines and (page in anchor_pages or page == 1):
                matched_lines.extend(lines[:2])
            for line in matched_lines[:3]:
                snippet = _normalize_preview_text(line, limit=180)
                key = (doc_id, page, snippet)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "doc_id": doc_id,
                        "page": page,
                        "doc_title": doc_title,
                        "text": snippet,
                    }
                )
    return candidates[:8]


def _exactness_review_flags(
    *,
    answer_type: str,
    current_answer_text: str,
    exact_span_candidates: list[dict[str, object]],
    required_page_anchor: dict[str, object],
) -> list[str]:
    flags: list[str] = []
    normalized_answer = (current_answer_text or "").strip().casefold()
    candidate_texts = [str(candidate.get("text") or "").casefold() for candidate in exact_span_candidates]
    anchor_pages = {
        int(page)
        for page in cast("list[object]", required_page_anchor.get("pages") or [])
        if isinstance(page, int | float)
    }
    candidate_pages = {
        int(candidate.get("page"))
        for candidate in exact_span_candidates
        if isinstance(candidate.get("page"), int | float)
    }

    if anchor_pages and not (anchor_pages & candidate_pages):
        flags.append("required_page_anchor_missing_in_candidates")

    answer_type_key = answer_type.strip().lower()
    if answer_type_key in {"name", "number"} and normalized_answer:
        if not exact_span_candidates:
            flags.append("no_exact_span_candidates")
        elif not any(normalized_answer in text for text in candidate_texts):
            flags.append("exact_answer_not_in_candidate_spans")

    if answer_type_key == "name" and normalized_answer and _CASE_REF_RE.search(current_answer_text or ""):
        if not any(normalized_answer in text for text in candidate_texts):
            flags.append("case_ref_exactness_risk")

    return list(dict.fromkeys(flags))


def _audit_priority(
    *,
    answer_type: str,
    route_family: str,
    support_shape_class: str,
    required_page_anchor: dict[str, object],
) -> int:
    if answer_type != "free_text" and route_family == "model":
        return 10
    if answer_type != "free_text" and support_shape_class in {"comparison", "named_metadata"}:
        return 20
    if answer_type != "free_text" and (answer_type == "name" or bool(required_page_anchor)):
        return 25
    if answer_type != "free_text":
        return 30
    if route_family == "model":
        return 40
    if support_shape_class in {"case_outcome", "outcome_plus_costs"}:
        return 50
    if route_family == "premise_guard":
        return 70
    return 60


def _audit_priority_key(record: dict[str, object]) -> int:
    raw = record.get("audit_priority")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return 999


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
    provider: PdfPageTextProvider | None,
    preserved_manual_fields: JsonObject | None,
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
    shape_requirements = (
        _metadata_support_requirements(str(question.get("question") or ""))
        if shape_class == "named_metadata"
        else {}
    )
    support_doc_count, support_page_count, support_shape_flags = _support_shape_flags(
        support_shape_class=shape_class,
        support_shape_requirements=shape_requirements,
        retrieved_chunk_pages=refs,
        answer_text=answer_text,
        answer_type=answer_type,
    )
    support_page_previews, resolved_doc_titles = _support_page_previews(refs, provider=provider)
    required_page_anchor = _required_page_anchor(str(question.get("question") or ""))
    exact_span_candidates = _exact_span_candidates(
        refs,
        provider=provider,
        current_answer_text=answer_text,
        question_text=str(question.get("question") or ""),
        required_page_anchor=required_page_anchor,
        resolved_doc_titles=resolved_doc_titles,
    )
    exactness_review_flags = _exactness_review_flags(
        answer_type=answer_type,
        current_answer_text=answer_text,
        exact_span_candidates=exact_span_candidates,
        required_page_anchor=required_page_anchor,
    )
    manual_fields = preserved_manual_fields or {}
    review_packet = {
        "question_id": str(question.get("id") or "").strip(),
        "answer_type": answer_type,
        "route_family": route_family,
        "question_refs": _question_refs(str(question.get("question") or "")),
        "required_page_anchor": required_page_anchor,
        "current_answer_text": answer_text,
        "support_doc_count": support_doc_count,
        "support_page_count": support_page_count,
        "retrieved_chunk_pages": refs,
        "resolved_doc_titles": resolved_doc_titles,
        "support_page_previews": support_page_previews[:6],
        "exact_span_candidates": exact_span_candidates,
        "exactness_review_flags": exactness_review_flags,
    }

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
        "support_shape_requirements": shape_requirements,
        "support_doc_count": support_doc_count,
        "support_page_count": support_page_count,
        "support_shape_flags": support_shape_flags,
        "required_page_anchor": required_page_anchor,
        "resolved_doc_titles": resolved_doc_titles,
        "support_page_previews": support_page_previews,
        "exact_span_candidates": exact_span_candidates,
        "exactness_review_flags": exactness_review_flags,
        "review_packet": review_packet,
        "audit_priority": _audit_priority(
            answer_type=answer_type,
            route_family=route_family,
            support_shape_class=shape_class,
            required_page_anchor=required_page_anchor,
        ),
        "manual_verdict": str(manual_fields.get("manual_verdict") or ""),
        "expected_answer": manual_fields.get("expected_answer"),
        "minimal_required_support_pages": cast("list[object]", manual_fields.get("minimal_required_support_pages") or []),
        "failure_class": str(manual_fields.get("failure_class") or ""),
        "notes": str(manual_fields.get("notes") or ""),
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
    docs_dir: Path | None = None,
    existing_scaffold_path: Path | None = None,
) -> dict[str, object]:
    questions = _load_questions(questions_path)
    answers = _load_answers(submission_path)
    preserved_manual_fields = _load_existing_manual_fields(existing_scaffold_path)
    provider = PdfPageTextProvider(docs_dir, max_chars_per_page=600) if docs_dir is not None else None

    deterministic_cases: list[dict[str, object]] = []
    free_text_cases: list[dict[str, object]] = []
    missing_questions: list[str] = []

    try:
        for answer_payload in answers:
            question_id = str(answer_payload.get("question_id") or "").strip()
            question = questions.get(question_id)
            if question is None:
                missing_questions.append(question_id)
                continue
            record = _build_case_record(
                question=question,
                answer_payload=answer_payload,
                provider=provider,
                preserved_manual_fields=preserved_manual_fields.get(question_id),
            )
            if str(record["answer_type"]) == "free_text":
                free_text_cases.append(record)
            else:
                deterministic_cases.append(record)
    finally:
        if provider is not None:
            provider.close()

    deterministic_cases.sort(key=lambda item: (_audit_priority_key(item), str(item["answer_type"]), str(item["question_id"])))
    free_text_cases.sort(key=lambda item: (_audit_priority_key(item), str(item["question_id"])))

    route_family_counts: dict[str, int] = {}
    support_shape_counts: dict[str, int] = {}
    support_shape_flag_counts: dict[str, int] = {}
    exactness_review_flag_counts: dict[str, int] = {}
    manual_verdict_counts = {
        "deterministic_complete": 0,
        "deterministic_incomplete": 0,
        "free_text_complete": 0,
        "free_text_incomplete": 0,
    }
    for record in [*deterministic_cases, *free_text_cases]:
        route_family = str(record.get("route_family") or "unknown")
        support_shape_class = str(record.get("support_shape_class") or "generic")
        route_family_counts[route_family] = route_family_counts.get(route_family, 0) + 1
        support_shape_counts[support_shape_class] = support_shape_counts.get(support_shape_class, 0) + 1
        for flag in cast("list[str]", record.get("support_shape_flags") or []):
            support_shape_flag_counts[flag] = support_shape_flag_counts.get(flag, 0) + 1
        for flag in cast("list[str]", record.get("exactness_review_flags") or []):
            exactness_review_flag_counts[flag] = exactness_review_flag_counts.get(flag, 0) + 1
        complete = bool(str(record.get("manual_verdict") or "").strip())
        if str(record.get("answer_type") or "") == "free_text":
            manual_verdict_counts["free_text_complete" if complete else "free_text_incomplete"] += 1
        else:
            manual_verdict_counts["deterministic_complete" if complete else "deterministic_incomplete"] += 1

    return {
        "summary": {
            "questions_count": len(questions),
            "answers_count": len(answers),
            "deterministic_count": len(deterministic_cases),
            "free_text_count": len(free_text_cases),
            "missing_questions": missing_questions,
            "route_family_counts": route_family_counts,
            "support_shape_class_counts": support_shape_counts,
            "support_shape_flag_counts": support_shape_flag_counts,
            "exactness_review_flag_counts": exactness_review_flag_counts,
            "manual_verdict_counts": manual_verdict_counts,
        },
        "records": [*deterministic_cases, *free_text_cases],
        "deterministic_cases": deterministic_cases,
        "free_text_cases": free_text_cases,
    }


def render_truth_audit_workbook(scaffold: dict[str, object]) -> str:
    summary_obj = scaffold.get("summary")
    summary = cast("dict[str, object]", summary_obj) if isinstance(summary_obj, dict) else {}
    deterministic_cases = cast("list[dict[str, object]]", scaffold.get("deterministic_cases") or [])
    free_text_cases = cast("list[dict[str, object]]", scaffold.get("free_text_cases") or [])

    def _render_case_block(record: dict[str, object]) -> list[str]:
        refs = cast("list[object]", record.get("question_refs") or [])
        minimal_pages = cast("list[object]", record.get("minimal_required_support_pages") or [])
        previews = cast("list[dict[str, object]]", record.get("support_page_previews") or [])
        lines = [
            f"### {record.get('question_id', '')}",
            f"- answer_type: `{record.get('answer_type', '')}`",
            f"- route_family: `{record.get('route_family', '')}`",
            f"- support_shape_class: `{record.get('support_shape_class', '')}`",
            f"- audit_priority: `{record.get('audit_priority', '')}`",
            f"- question: {record.get('question', '')}",
            f"- current_answer: `{record.get('current_answer_text', '')}`",
            f"- question_refs: {', '.join(str(item) for item in refs) if refs else '(none)'}",
            f"- required_page_anchor: `{record.get('required_page_anchor')}`",
            f"- manual_verdict: `{record.get('manual_verdict', '') or '(blank)'}`",
            f"- expected_answer: `{record.get('expected_answer')}`",
            f"- failure_class: `{record.get('failure_class', '') or '(blank)'}`",
            f"- minimal_required_support_pages: {', '.join(str(item) for item in minimal_pages) if minimal_pages else '(blank)'}",
            f"- support_shape_flags: {', '.join(cast('list[str]', record.get('support_shape_flags') or [])) or '(none)'}",
            f"- exactness_review_flags: {', '.join(cast('list[str]', record.get('exactness_review_flags') or [])) or '(none)'}",
        ]
        if record.get("notes"):
            lines.append(f"- notes: {record.get('notes', '')}")
        exact_candidates = cast("list[dict[str, object]]", record.get("exact_span_candidates") or [])
        lines.append("- exact_span_candidates:")
        if not exact_candidates:
            lines.append("  - (none)")
        else:
            for candidate in exact_candidates[:_WORKBOOK_PREVIEW_LIMIT]:
                lines.append(
                    "  - "
                    f"{candidate.get('doc_title', candidate.get('doc_id', 'unknown'))} "
                    f"p{candidate.get('page', '?')}: {candidate.get('text', '')}"
                )
        lines.append("- support_page_previews:")
        if not previews:
            lines.append("  - (none)")
        else:
            for preview in previews[:_WORKBOOK_PREVIEW_LIMIT]:
                lines.append(
                    "  - "
                    f"{preview.get('doc_title', preview.get('doc_id', 'unknown'))} "
                    f"p{preview.get('page', '?')}: {preview.get('snippet', '')}"
                )
        return lines

    lines = [
        "# Truth Audit Workbook",
        "",
        "## Summary",
        f"- deterministic_count: {summary.get('deterministic_count', 0)}",
        f"- free_text_count: {summary.get('free_text_count', 0)}",
        f"- deterministic_complete: {cast('dict[str, object]', summary.get('manual_verdict_counts') or {}).get('deterministic_complete', 0)}",
        f"- deterministic_incomplete: {cast('dict[str, object]', summary.get('manual_verdict_counts') or {}).get('deterministic_incomplete', 0)}",
        f"- free_text_complete: {cast('dict[str, object]', summary.get('manual_verdict_counts') or {}).get('free_text_complete', 0)}",
        f"- free_text_incomplete: {cast('dict[str, object]', summary.get('manual_verdict_counts') or {}).get('free_text_incomplete', 0)}",
        "",
        "## Deterministic Cases",
        "",
    ]
    for record in deterministic_cases:
        lines.extend(_render_case_block(record))
        lines.append("")

    lines.extend(["## Free Text Cases", ""])
    for record in free_text_cases:
        lines.extend(_render_case_block(record))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a manual truth-audit scaffold from platform questions and submission artifacts.")
    parser.add_argument("--questions", required=True, help="Path to platform questions.json")
    parser.add_argument("--submission", required=True, help="Path to platform submission.json")
    parser.add_argument("--docs-dir", help="Optional directory containing source PDFs for support previews")
    parser.add_argument("--existing-scaffold", help="Optional existing scaffold path to preserve manual audit fields")
    parser.add_argument("--out", help="Output path for the generated truth-audit scaffold JSON")
    parser.add_argument("--workbook-out", help="Optional output path for a human-readable markdown workbook")
    args = parser.parse_args(argv)

    questions_path = Path(args.questions)
    submission_path = Path(args.submission)
    out_path = Path(args.out) if args.out else submission_path.with_name("truth_audit_scaffold.json")
    existing_scaffold_path = Path(args.existing_scaffold) if args.existing_scaffold else out_path
    docs_dir = Path(args.docs_dir) if args.docs_dir else None

    scaffold = build_truth_audit_scaffold(
        questions_path=questions_path,
        submission_path=submission_path,
        docs_dir=docs_dir,
        existing_scaffold_path=existing_scaffold_path,
    )
    out_path.write_text(json.dumps(scaffold, ensure_ascii=False, indent=2), encoding="utf-8")
    workbook_path = Path(args.workbook_out) if args.workbook_out else out_path.with_name("truth_audit_workbook.md")
    workbook_path.write_text(render_truth_audit_workbook(scaffold), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
