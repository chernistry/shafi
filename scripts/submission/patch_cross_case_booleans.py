#!/usr/bin/env python3
"""Patch cross-case boolean answers using corpus registry ground truth.

For cross-case party/judge overlap questions, the DB answerer produces
false positives (says Yes when there's no party/judge overlap). This script
uses the corpus registry as ground truth to correct those answers.

Expected impact: +1.2pp Det from fixing ~15 wrong booleans.

Usage:
    uv run python scripts/patch_cross_case_booleans.py
    uv run python scripts/patch_cross_case_booleans.py \
        --input data/private_submission_V15_PATCHED.json \
        --output data/private_submission_V15_PATCHED_v2.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "data" / "private_corpus_registry.json"
QUESTIONS_PATH = REPO / "dataset" / "private" / "questions.json"
DEFAULT_INPUT = REPO / "data" / "private_submission_V15_PATCHED.json"

_CASE_REF_RE = re.compile(
    r"\b(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*[-\s]*0*(\d{1,4})\s*[/-]\s*(\d{4})\b",
    re.IGNORECASE,
)


def _normalize_case_ref(prefix: str, num: str, year: str) -> str:
    return f"{prefix.upper()} {int(num):03d}/{year}"


def _build_case_lookup(
    registry: dict,
) -> dict[str, dict[str, set[str]]]:
    """Build case_number → {parties, judges} mapping from registry."""
    cases = registry.get("cases", {})
    lookup: dict[str, dict[str, set[str]]] = {}
    for _cid, case in cases.items():
        cn = case.get("case_number", "")
        if not cn:
            continue
        parties: set[str] = set()
        for p in case.get("parties", []):
            name = p.get("name", "").strip().lower()
            if name and len(name) > 2:
                parties.add(name)
        judges: set[str] = set()
        for j in case.get("judges", []):
            jname = j.strip().lower()
            if jname:
                judges.add(jname)
                # Extract surname for fuzzy matching
                parts = jname.split()
                if parts:
                    judges.add(parts[-1])
        lookup[cn] = {"parties": parties, "judges": judges}
    return lookup


def _has_overlap(
    set1: set[str], set2: set[str], *, fuzzy_surname: bool = False
) -> bool:
    """Check if two name sets have any overlap."""
    if set1 & set2:
        return True
    if fuzzy_surname:
        # Check if any surname in set1 matches any surname in set2
        surnames1 = {name.split()[-1] for name in set1 if len(name.split()) > 1}
        surnames2 = {name.split()[-1] for name in set2 if len(name.split()) > 1}
        if surnames1 & surnames2:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    registry = json.loads(REGISTRY_PATH.read_text())
    questions = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text())}
    case_lookup = _build_case_lookup(registry)
    submission = json.loads(args.input.read_text())
    answers = submission["answers"]

    corrections = 0
    for ans in answers:
        qid = ans.get("question_id", "")
        q = questions.get(qid)
        if not q or q.get("answer_type") != "boolean":
            continue

        current = ans.get("answer")
        if current is not True:  # Only fix True → False (false positives)
            continue

        ql = q["question"].lower()
        is_party = "main party" in ql or ("party" in ql and ("both" in ql or "common" in ql))
        is_judge = "judge" in ql and ("both" in ql or "common" in ql or "presided" in ql)
        if not is_party and not is_judge:
            continue

        case_refs = _CASE_REF_RE.findall(q["question"])
        if len(case_refs) < 2:
            continue

        cn1 = _normalize_case_ref(*case_refs[0])
        cn2 = _normalize_case_ref(*case_refs[1])
        r1 = case_lookup.get(cn1)
        r2 = case_lookup.get(cn2)

        if not r1 or not r2:
            continue  # Can't verify — skip

        field = "judges" if is_judge else "parties"
        overlap = _has_overlap(
            r1[field], r2[field],
            fuzzy_surname=(field == "judges"),
        )

        if not overlap:
            corrections += 1
            if args.dry_run:
                print(f"  WOULD FIX: {qid[:12]} {cn1} vs {cn2} ({field}): True → False")
            else:
                ans["answer"] = False

    print(f"\nTotal corrections: {corrections}")

    if not args.dry_run and corrections > 0:
        output = args.output or args.input.with_name(args.input.stem + "_boolfix.json")
        output.write_text(json.dumps(submission, indent=None, ensure_ascii=False))
        print(f"Written to: {output}")
    elif args.dry_run:
        print("(dry run — no changes written)")


if __name__ == "__main__":
    main()
