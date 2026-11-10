# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

_SUBTYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("title_page", re.compile(r"\b(title page|cover page|first page)\b", re.IGNORECASE)),
    ("page_specific", re.compile(r"\b(page 2|second page|page \d+|pages \d+)\b", re.IGNORECASE)),
    ("citation_title", re.compile(r"\bcitation title(?:s)?\b", re.IGNORECASE)),
    ("schedule", re.compile(r"\bschedule\b", re.IGNORECASE)),
    ("annex", re.compile(r"\bannex\b", re.IGNORECASE)),
    ("table", re.compile(r"\btable\b", re.IGNORECASE)),
    ("article_provision", re.compile(r"\b(article|section|rule|part|chapter)\b", re.IGNORECASE)),
)


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def classify_subtypes(question: str) -> list[str]:
    normalized = str(question or "").strip()
    if not normalized:
        return []
    return [name for name, pattern in _SUBTYPE_PATTERNS if pattern.search(normalized)]


def expected_eval_slices(subtypes: list[str]) -> list[str]:
    return [f"structured_{subtype}" for subtype in subtypes]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at {path}")
    cases = raw.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected 'cases' list at {path}")
    return [cast("dict[str, Any]", case) for case in cases if isinstance(case, dict)]


def _build_case(case: dict[str, Any]) -> dict[str, Any] | None:
    question = str(case.get("question") or "").strip()
    subtypes = classify_subtypes(question)
    if not subtypes:
        return None

    telemetry = cast("dict[str, Any]", case.get("telemetry") or {}) if isinstance(case.get("telemetry"), dict) else {}
    used_pages = _coerce_str_list(case.get("used_pages"))
    if not used_pages:
        used_pages = _coerce_str_list(telemetry.get("used_page_ids"))

    answer = str(case.get("answer") or "").strip()
    answer_type = str(case.get("answer_type") or "free_text").strip().lower() or "free_text"

    return {
        "qid": str(case.get("question_id") or case.get("case_id") or "").strip(),
        "question": question,
        "answer_type": answer_type,
        "subtypes": subtypes,
        "eval_slices": expected_eval_slices(subtypes),
        "current_answer": answer,
        "current_used_pages": used_pages,
        "null_answer": answer_type != "free_text" and answer.lower() == "null",
        "empty_used_pages": not used_pages,
    }


def build_pack(source_path: Path) -> dict[str, Any]:
    cases = _load_cases(source_path)
    structured_cases = [built for case in cases if (built := _build_case(case)) is not None]
    subtype_counts = Counter(subtype for case in structured_cases for subtype in cast("list[str]", case["subtypes"]))
    zero_count_subtypes = [name for name, _ in _SUBTYPE_PATTERNS if subtype_counts.get(name, 0) == 0]

    return {
        "ticket": 69,
        "created_at": "2026-03-13",
        "source_artifact": str(source_path),
        "policy": (
            "Champion-backed structural slice only. This pack is derived from the current public champion debug "
            "artifact and is intended for private-safe structure-gap audits rather than warm-up rider mining."
        ),
        "cases": structured_cases,
        "summary": {
            "total_cases": len(structured_cases),
            "subtype_counts": dict(sorted(subtype_counts.items())),
            "zero_count_subtypes": zero_count_subtypes,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a structural-stress pack from a champion debug artifact.")
    parser.add_argument("--source", required=True, help="Path to eval_candidate_debug_*.json")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    out_path = Path(args.out)
    pack = build_pack(source_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
