#!/usr/bin/env python3
"""Build V16_SUPER_HYBRID: V15_SUPER (all fixes) + V16 free_text quality upgrades.

Strategy:
- Base: V15_SUPER (null=1, nopg=3, 14 number fixes, 2 trunc fixes, 5 page fixes)
- For free_text only: upgrade when V16 is substantively better
- ALWAYS keep V15 pages (V16 has nopg regression ~43)
- ALWAYS keep V15 typed answers (boolean/number/date/name/names)

Upgrade criteria for free_text:
1. V16 is non-null and non-no-info
2. V16 starts with strong evidence pattern (Under Article/Section/Rule/Part...) OR
   V16 is >20 chars longer than V15 AND V15 is below 200 chars OR
   V15 is no-info (any V16 substantive answer is better)
3. V16 answer is not obviously truncated (ends with proper punctuation)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def strip_cite(text: str) -> str:
    cleaned = re.sub(r"\s*\(cite:[^)]+\)", "", text or "").strip()
    return re.sub(r"  +", " ", cleaned)


def is_evidence_first(text: str) -> bool:
    """Check if answer starts with an evidence-grounded pattern."""
    t = text.strip()
    # Strong evidence-first patterns
    patterns = [
        r"^Under\s+(Article|Section|Rule|Part|Schedule|Clause|Paragraph)",
        r"^Article\s+\d",
        r"^Section\s+\d",
        r"^Rule\s+\d",
        r"^Schedule\s+\d",
        r"^Pursuant\s+to",
        r"^In\s+accordance\s+with",
        r"^According\s+to\s+(Article|Section|Rule|the)",
        r"^H\.E\.\s+Justice",  # Names of judges
    ]
    for p in patterns:
        if re.match(p, t, re.IGNORECASE):
            return True
    return False


def is_no_info(text: str | None) -> bool:
    if text is None:
        return True
    return "no information" in str(text).lower()


def is_truncated(text: str) -> bool:
    t = text.strip()
    if re.search(r"\bNo\.\s*$", t):
        return True
    if re.search(r"\b(of|and|the|in|by)\s*$", t, re.I) and len(t) < 300:
        return True
    return False


def add_period_if_needed(text: str) -> str:
    t = text.strip()
    if t and not re.search(r"[.!?]$", t):
        return t.rstrip(",:;") + "."
    return t


def load_checkpoint() -> dict[str, dict]:
    # Prefer full900 JSON if available (eval completed), else fall back to checkpoint
    full_path = REPO / "data" / "tzuf_private1_full900.json"
    if full_path.exists():
        data = json.loads(full_path.read_text())
        return {r["id"]: r for r in data.get("results", [])}
    cp_path = REPO / "data" / "tzuf_private1_checkpoint.jsonl"
    if not cp_path.exists():
        return {}
    results: dict[str, dict] = {}
    for line in cp_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id")
            if qid and qid not in results:
                results[qid] = r
        except json.JSONDecodeError:
            pass
    return results


def main() -> None:
    # Load sources
    print("Loading V15_SUPER (base)...")
    base_path = REPO / "data" / "private_submission_V15_SUPER.json"
    if not base_path.exists():
        print("ERROR: V15_SUPER not found. Build it first.")
        return
    submission = json.loads(base_path.read_text())

    print("Loading V16 checkpoint...")
    v16_by_id = load_checkpoint()
    print(f"  V16: {len(v16_by_id)} answers available")

    questions = json.loads((REPO / "dataset" / "private" / "questions.json").read_text())
    q_map = {q["id"]: q for q in questions}

    # Build hybrid
    upgraded = 0
    kept_v15 = 0
    v16_is_noinfo = 0
    v16_missing = 0

    new_answers = []
    for ans in submission["answers"]:
        qid = ans["question_id"]
        q = q_map.get(qid, {})

        # Only upgrade free_text
        if q.get("answer_type") != "free_text":
            new_answers.append(ans)
            continue

        v16 = v16_by_id.get(qid)
        if not v16:
            v16_missing += 1
            new_answers.append(ans)
            continue

        v16_raw = v16.get("answer")
        v15_ans = ans.get("answer")

        # Skip if V16 is null or no-info (keep V15)
        if is_no_info(v16_raw):
            v16_is_noinfo += 1
            new_answers.append(ans)
            continue

        v16_clean = strip_cite(str(v16_raw))
        if is_truncated(v16_clean):
            kept_v15 += 1
            new_answers.append(ans)
            continue

        # Normalize newlines to "; " for clean formatting
        v16_clean = re.sub(r"\n+", "; ", v16_clean).strip()
        v16_clean = add_period_if_needed(v16_clean)
        v15_str = str(v15_ans) if v15_ans is not None else ""

        # Determine if V16 is better
        upgrade = False

        # Case 1: V15 is no-info → take any substantive V16 answer
        if is_no_info(v15_ans):
            upgrade = True

        # Case 2: V16 starts with evidence-first pattern AND V15 doesn't
        elif is_evidence_first(v16_clean) and not is_evidence_first(v15_str):
            upgrade = True

        # Case 3: V16 is substantially longer and V15 is short
        elif len(v16_clean) > len(v15_str) + 30 and len(v15_str) < 200:
            upgrade = True

        if upgrade:
            new_ans = dict(ans)
            new_ans["answer"] = v16_clean
            # KEEP V15 PAGES (critical: V16 has nopg regression)
            new_answers.append(new_ans)
            upgraded += 1
        else:
            kept_v15 += 1
            new_answers.append(ans)

    result = dict(submission)
    result["answers"] = new_answers

    # Verify
    nulls = sum(1 for a in new_answers if a["answer"] is None)
    nopg = sum(1 for a in new_answers if a["answer"] is not None and not a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []))
    cite_leaks = sum(1 for a in new_answers if a["answer"] and re.search(r"\(cite:", str(a["answer"])))
    no_info_count = sum(1 for a in new_answers if isinstance(a["answer"], str) and "no information" in a["answer"].lower())

    print(f"\nUpgrade stats:")
    print(f"  Upgraded to V16: {upgraded}")
    print(f"  Kept V15: {kept_v15}")
    print(f"  V16 was no-info (kept V15): {v16_is_noinfo}")
    print(f"  V16 missing (kept V15): {v16_missing}")
    print(f"\nVerification:")
    print(f"  null={nulls}, nopg={nopg}, cite_leaks={cite_leaks}, no_info={no_info_count}")

    out_path = REPO / "data" / "private_submission_V16_SUPER_HYBRID.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False))
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
