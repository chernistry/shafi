#!/usr/bin/env python3
"""Add grounding prefixes to free_text answers that lack explicit legal references.

Ticket 5002: Adds "Under [Law Name], " prefix to free_text answers that discuss
a regulation but don't cite it by name. Conservative — only prefixes when the
law title is cleanly extractable from the question text.

Usage:
    python scripts/add_freetext_prefixes.py [--input FILE] [--output FILE] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --- Law title extraction patterns ---

# Clean law title: "Companies Law 2018", "Employment Law", "Leasing Law 2020"
_CLEAN_LAW_TITLE_RE = re.compile(
    r"\b("
    r"(?:(?:Companies|Employment|Leasing|Arbitration|Insolvency|Contract|Trust|"
    r"Strata Title|Real Property|Data Protection|Electronic Transactions|"
    r"Personal Property|Obligations|Netting|Securities|Investment|"
    r"Limited Partnership|General Partnership|Foundations|"
    r"Body Corporate|Operating|Hotel Operating|Venture Studio|"
    r"Prescribed Company|Family Arrangements|Intellectual Property|"
    r"Common Reporting Standard|Implied Terms|Unfair Contract Terms)"
    r"\s+(?:Law|Regulations?))"
    r"(?:\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)

# "DIFC Law No. X of YYYY" pattern
_DIFC_LAW_NO_RE = re.compile(
    r"\bDIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4}\b",
    re.IGNORECASE,
)

# Case reference in question — skip these (case law answers should start with case ref)
_CASE_REF_RE = re.compile(r"\b(?:CFI|SCT|CA|ARB|ENF|TCD)\s+\d+/\d+\b", re.IGNORECASE)

# Evidence-first openings that are already good
_EVIDENCE_STARTS = (
    "Article", "Section", "Rule ", "Regulation", "Part ", "Schedule",
    "Clause", "Chapter", "Under ", "Pursuant", "In accordance", "Per ",
    "Paragraph", "DRA ", "DIFC ",
)
_CASE_START_RE = re.compile(r"^(?:CFI|SCT|CA|ARB|ENF|TCD)\s+\d+")

# Consultation paper pattern
_CP_RE = re.compile(r"\bConsultation Paper\s+No\.?\s*\d+", re.IGNORECASE)


def extract_law_title(question: str) -> str:
    """Extract the primary law title from a question.

    Returns the cleanest, shortest law title found, or empty string if none.
    """
    # Skip case-law questions
    if _CASE_REF_RE.search(question):
        return ""

    # Try clean law titles first
    matches = _CLEAN_LAW_TITLE_RE.findall(question)
    if not matches:
        return ""

    # Deduplicate (case-insensitive)
    unique = list({m.strip().lower(): m.strip() for m in matches}.values())

    # Skip multi-law comparison questions — can't tell which law the answer is about
    if len(unique) >= 2:
        return ""

    title = unique[0]
    # Normalize: ensure title case
    if title[0].islower():
        title = title[0].upper() + title[1:]
    return title


def _answer_already_mentions_law(answer: str, law_title: str) -> bool:
    """Check if the answer already references the law by name."""
    ans_lower = answer.lower()
    # Strip year from title for matching
    title_core = re.sub(r"\s+\d{4}$", "", law_title).strip().lower()
    return title_core in ans_lower


def should_prefix(answer: str, law_title: str = "") -> bool:
    """Check whether an answer would benefit from a law-reference prefix."""
    if not answer or not answer.strip():
        return False
    # Already evidence-first
    if any(answer.startswith(p) for p in _EVIDENCE_STARTS):
        return False
    if _CASE_START_RE.match(answer):
        return False
    # No-info
    if "no information" in answer.lower():
        return False
    # Consultation paper answers — already contextual
    if _CP_RE.match(answer):
        return False
    # Numbered list answers — structure is fine
    if answer.startswith("1.") or answer.startswith("1)"):
        return False
    # Answer already mentions the law — prefix would be redundant
    if law_title and _answer_already_mentions_law(answer, law_title):
        return False
    return True


def add_prefix(answer: str, law_title: str) -> str:
    """Add 'Under [law_title], ' prefix to answer.

    Lowercases the first character of the original answer to flow naturally.
    Trims to 280 chars at last complete sentence if needed.
    """
    # Only keep uppercase for ALL-CAPS acronyms (e.g. "DIFC", "ICC")
    first_word = answer.split()[0] if answer.split() else ""
    is_acronym = len(first_word) >= 2 and first_word.isupper()
    first_char = answer[0]
    if is_acronym:
        new_answer = f"Under the {law_title}, {answer}"
    elif first_char.isupper():
        new_answer = f"Under the {law_title}, {first_char.lower()}{answer[1:]}"
    else:
        new_answer = f"Under the {law_title}, {answer}"

    # Trim to 280 chars at sentence boundary
    if len(new_answer) > 280:
        # Find last sentence boundary before 280
        truncated = new_answer[:280]
        last_period = truncated.rfind(".")
        last_semicolon = truncated.rfind(";")
        boundary = max(last_period, last_semicolon)
        if boundary > len(f"Under the {law_title}, ") + 20:
            new_answer = truncated[: boundary + 1].rstrip()
        else:
            # Can't find good boundary — don't prefix
            return ""

    # Clean trailing punctuation and ensure ends with period
    new_answer = new_answer.rstrip().rstrip(";,:")
    if new_answer and not new_answer.rstrip().endswith((".", "!", "?")):
        new_answer = new_answer.rstrip() + "."

    return new_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Add grounding prefixes to free_text answers")
    parser.add_argument("--input", default="data/private_submission_FINAL_SUBMISSION.json")
    parser.add_argument("--output", default="data/private_submission_V2_prefixed.json")
    parser.add_argument("--questions", default="dataset/private/questions.json")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    with open(args.input) as f:
        submission = json.load(f)

    with open(args.questions) as f:
        questions = {q["id"]: q for q in json.load(f)}

    modified = 0
    skipped_already_good = 0
    skipped_no_law = 0
    skipped_noinfo = 0
    skipped_too_long = 0
    skipped_not_ft = 0
    total_chars_before = 0
    total_chars_after = 0

    for answer_entry in submission["answers"]:
        q = questions.get(answer_entry["question_id"], {})
        if q.get("answer_type") != "free_text":
            skipped_not_ft += 1
            continue

        ans = answer_entry["answer"]
        if ans is None:
            skipped_noinfo += 1
            continue
        if not isinstance(ans, str):
            continue

        if "no information" in ans.lower():
            skipped_noinfo += 1
            continue

        law_title = extract_law_title(q.get("question", ""))
        if not law_title:
            skipped_no_law += 1
            continue

        if not should_prefix(ans, law_title):
            skipped_already_good += 1
            continue

        new_ans = add_prefix(ans, law_title)
        if not new_ans:
            skipped_too_long += 1
            continue

        if len(new_ans) > 280:
            skipped_too_long += 1
            continue

        total_chars_before += len(ans)
        total_chars_after += len(new_ans)
        modified += 1

        if args.dry_run:
            print(f"  {answer_entry['question_id'][:12]}: [{law_title}]")
            print(f"    OLD ({len(ans):3d}): {ans[:70]}")
            print(f"    NEW ({len(new_ans):3d}): {new_ans[:70]}")
            print()
        else:
            answer_entry["answer"] = new_ans

    print(f"\n=== STATS ===")
    print(f"Modified: {modified}")
    print(f"Skipped (not free_text): {skipped_not_ft}")
    print(f"Skipped (already good): {skipped_already_good}")
    print(f"Skipped (no-info/null): {skipped_noinfo}")
    print(f"Skipped (no law in Q): {skipped_no_law}")
    print(f"Skipped (too long): {skipped_too_long}")
    if modified:
        print(f"Avg chars: {total_chars_before // modified} → {total_chars_after // modified}")

    if not args.dry_run and modified > 0:
        with open(args.output, "w") as f:
            json.dump(submission, f, indent=2)
        print(f"\nWritten: {args.output}")
    elif args.dry_run:
        print("\n(dry run — no file written)")


if __name__ == "__main__":
    main()
