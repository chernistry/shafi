from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]
_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|ARB|SCT|CMC)\s*\d{1,4}/\d{2,4}\b", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\b(?:Article|Section)\s+[A-Za-z0-9()./-]+\s+of\s+", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s*([0-9]+(?:\s*of\s*[0-9]{4})?)\b", re.IGNORECASE)
_TITLE_PLUS_LAW_NUMBER_RE = re.compile(
    r"\b([A-Z][A-Za-z'&/()\- ]+?Law(?:\s+\d{4})?)\s*,?\s+((?:DIFC\s+)?Law\s+No\.?\s*[0-9]+(?:\s*of\s*[0-9]{4})?)\b"
)


def _load_questions(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Questions JSON must be a list: {path}")
    rows: list[JsonDict] = []
    for item in cast("list[object]", payload):
        if isinstance(item, dict):
            rows.append(cast("JsonDict", item))
    return rows


def _load_scaffold(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = cast("list[JsonDict]", payload.get("records") or [])
    result: dict[str, list[str]] = {}
    for record in records:
        question_id = str(record.get("question_id") or "")
        if not question_id:
            continue
        doc_ids = sorted(
            {
                str(page_ref.get("doc_id") or "").strip()
                for page_ref in cast("list[JsonDict]", record.get("retrieved_chunk_pages") or [])
                if str(page_ref.get("doc_id") or "").strip()
            }
        )
        if doc_ids:
            result[question_id] = doc_ids
    return result


def _unicode_variant(question: str) -> str:
    updated = question.replace("No. ", "No.\u00a0")
    updated = updated.replace("-", "\u2014")
    digits = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    return updated.translate(digits)


def _law_number_only_variant(question: str) -> str | None:
    match = _TITLE_PLUS_LAW_NUMBER_RE.search(question)
    if match is None:
        return None
    return question[: match.start()] + match.group(2) + question[match.end() :]


def _ordinal_label(index: int) -> str:
    labels = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
    }
    return labels.get(index, f"{index}th")


def _indirect_case_variant(question: str) -> str | None:
    matches = list(_CASE_REF_RE.finditer(question))
    if not matches:
        return None
    if len(matches) == 1:
        return _CASE_REF_RE.sub("the relevant DIFC court case", question)

    replacement_index = 0

    def _replacement(_match: re.Match[str]) -> str:
        nonlocal replacement_index
        replacement_index += 1
        return f"the {_ordinal_label(replacement_index)} relevant DIFC court case"

    return _CASE_REF_RE.sub(_replacement, question)


def build_variants(*, questions: list[JsonDict], qid_to_doc_ids: dict[str, list[str]], limit: int = 30) -> list[JsonDict]:
    candidates_by_type: dict[str, list[JsonDict]] = {
        "title_only_framing": [],
        "law_number_only_framing": [],
        "indirect_case_framing": [],
        "schedule_annex_framing": [],
        "unicode_variant": [],
    }
    for question in questions:
        qid = str(question.get("id") or "")
        if not qid or qid not in qid_to_doc_ids:
            continue
        text = str(question.get("question") or "")
        answer_type = str(question.get("answer_type") or "free_text")
        candidates: list[tuple[str, str]] = []
        title_only = _ARTICLE_RE.sub("", text)
        if title_only != text:
            candidates.append(("title_only_framing", title_only))
        law_number_only = _law_number_only_variant(_ARTICLE_RE.sub("", text))
        if law_number_only is not None:
            candidates.append(("law_number_only_framing", law_number_only))
        indirect_case = _indirect_case_variant(text)
        if indirect_case is not None:
            candidates.append(("indirect_case_framing", indirect_case))
        if re.search(r"\b(?:schedule|annex|appendix)\b", text, re.IGNORECASE):
            candidates.append(
                (
                    "schedule_annex_framing",
                    re.sub(r"\b(?:schedule|annex|appendix)\b", "appendix", text, flags=re.IGNORECASE),
                )
            )
        candidates.append(("unicode_variant", _unicode_variant(text)))
        for variant_type, variant_question in candidates:
            if variant_question == text:
                continue
            candidates_by_type.setdefault(variant_type, []).append(
                {
                    "id": f"{qid}__{variant_type}",
                    "question": variant_question,
                    "answer_type": answer_type,
                    "original_question_id": qid,
                    "variant_type": variant_type,
                    "expected_gold_doc_ids": qid_to_doc_ids[qid],
                }
            )
    variants: list[JsonDict] = []
    seen_ids: set[str] = set()
    variant_order = [
        "title_only_framing",
        "law_number_only_framing",
        "indirect_case_framing",
        "schedule_annex_framing",
        "unicode_variant",
    ]
    while len(variants) < limit and any(candidates_by_type.get(variant_type) for variant_type in variant_order):
        for variant_type in variant_order:
            queue = candidates_by_type.get(variant_type) or []
            while queue:
                candidate = queue.pop(0)
                candidate_id = str(candidate["id"])
                if candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                variants.append(candidate)
                break
            if len(variants) >= limit:
                break
    return variants


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate public anti-overfit question variants.")
    parser.add_argument("--questions", type=Path, default=Path("dataset/public_dataset.json"))
    parser.add_argument("--scaffold", type=Path, default=Path("platform_runs/warmup/truth_audit_scaffold_v6_context_seed.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    questions = _load_questions(args.questions)
    qid_to_doc_ids = _load_scaffold(args.scaffold)
    variants = build_variants(questions=questions, qid_to_doc_ids=qid_to_doc_ids, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(variants, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
