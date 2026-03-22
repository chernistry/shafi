from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s+\d{1,4}/\d{4}\b", re.IGNORECASE)

_PARTY_TERMS = (
    "claimant",
    "claimants",
    "defendant",
    "defendants",
    "respondent",
    "respondents",
    "appellant",
    "appellants",
    "party",
    "parties",
    "legal entities",
    "individuals",
    "main party",
)

_EXPLICIT_PAGE_TERMS = (
    "page 2",
    "second page",
    "title page",
    "cover page",
    "first page",
    "header",
    "caption",
)


def _load_questions(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    rows: list[JsonDict] = []
    raw_rows = cast("list[object]", obj)
    for raw in raw_rows:
        if isinstance(raw, dict):
            rows.append(cast("JsonDict", raw))
    return rows


def _matches_family(*, family: str, question: str, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", question).strip().lower()
    ref_count = len(_CASE_REF_RE.findall(question))
    if family == "party_title_metadata":
        return ref_count >= 1 and any(term in q for term in _PARTY_TERMS)
    if family == "comparison_party_title_metadata":
        return ref_count >= 2 and any(term in q for term in _PARTY_TERMS)
    if family == "single_case_party_title_metadata":
        return ref_count == 1 and any(term in q for term in _PARTY_TERMS)
    if family == "explicit_page_reference":
        return any(term in q for term in _EXPLICIT_PAGE_TERMS)
    if family == "strict_only":
        return answer_type.strip().lower() in {"boolean", "number", "date", "name", "names"}
    raise ValueError(f"Unsupported family: {family}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select question IDs for a reusable question family.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--family",
        required=True,
        choices=(
            "party_title_metadata",
            "comparison_party_title_metadata",
            "single_case_party_title_metadata",
            "explicit_page_reference",
            "strict_only",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_questions(args.questions.resolve())
    selected: list[str] = []
    summary_rows: list[JsonDict] = []
    for row in rows:
        qid = str(row.get("id") or "").strip()
        question = str(row.get("question") or "")
        answer_type = str(row.get("answer_type") or "").strip().lower()
        if not qid or not _matches_family(family=args.family, question=question, answer_type=answer_type):
            continue
        selected.append(qid)
        summary_rows.append(
            {
                "id": qid,
                "answer_type": answer_type,
                "question": question,
                "case_ref_count": len(_CASE_REF_RE.findall(question)),
            }
        )

    args.out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    json_out = args.json_out or args.out.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "family": args.family,
                "count": len(selected),
                "question_ids": selected,
                "rows": summary_rows,
                "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
