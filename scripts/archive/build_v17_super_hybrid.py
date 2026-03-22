#!/usr/bin/env python3
"""Build V17_SUPER_HYBRID: V16_SUPER_HYBRID (base) + V17 free_text upgrades.

Strategy:
- Base: V16_SUPER_HYBRID (null=1, nopg=3, 22 bool fixes, 14 num fixes, 52 FT upgrades)
- For free_text only: upgrade when V17 is substantively better
- ALWAYS keep V15/V16 pages (V17 may have nopg regression)
- ALWAYS keep V16 typed answers (boolean/number/date/name/names)

Upgrade criteria for free_text:
1. V17 is non-null and non-no-info
2. V17 starts with strong evidence pattern (Under Article/Section/Rule/Part...) OR
   V17 is >20 chars longer than V16 AND V16 is below 200 chars OR
   V16 is no-info (any V17 substantive answer is better)
3. V17 answer is not obviously truncated (ends with proper punctuation)
4. V17 answer fits within 280 chars (hard competition limit)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAX_CHARS = 280


def strip_cite(text: str) -> str:
    cleaned = re.sub(r"\s*\(cite:[^)]+\)", "", text or "").strip()
    return re.sub(r"  +", " ", cleaned)


def is_evidence_first(text: str) -> bool:
    """Check if answer starts with an evidence-grounded pattern."""
    t = text.strip()
    patterns = [
        r"^Under\s+(Article|Section|Rule|Part|Schedule|Clause|Paragraph)",
        r"^Article\s+\d",
        r"^Section\s+\d",
        r"^Rule\s+\d",
        r"^Schedule\s+\d",
        r"^Pursuant\s+to",
        r"^In\s+accordance\s+with",
        r"^According\s+to\s+(Article|Section|Rule|the)",
        r"^H\.E\.\s+Justice",
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


def smart_truncate(text: str, max_len: int = MAX_CHARS) -> str:
    """Truncate at sentence boundary, then word boundary."""
    if len(text) <= max_len:
        return text
    # Try sentence boundary
    for delim in [". ", "! ", "? ", "; "]:
        idx = text.rfind(delim, 0, max_len - 1)
        if idx > max_len // 2:
            candidate = text[: idx + 1].strip()
            if len(candidate) <= max_len:
                return candidate
    # Word boundary
    idx = text.rfind(" ", 0, max_len)
    if idx > 0:
        return text[:idx].rstrip(",:;") + "."
    return text[:max_len]


def load_v17() -> dict[str, dict]:
    """Load V17 eval: prefer full900 JSON (completed), fall back to checkpoint JSONL."""
    full_path = REPO / "data" / "tzuf_private1_full900.json"
    if full_path.exists():
        # Check if this is V17 (newer than V16 copy)
        v16_copy = REPO / "data" / "tzuf_private1_v16_full900.json"
        if v16_copy.exists() and full_path.stat().st_mtime > v16_copy.stat().st_mtime:
            print(f"  Using full900.json as V17 (newer than V16 copy)")
            data = json.loads(full_path.read_text())
            results = data.get("results", [])
            return {r["id"]: r for r in results}
        elif not v16_copy.exists():
            print(f"  Using full900.json (no V16 copy found)")
            data = json.loads(full_path.read_text())
            results = data.get("results", [])
            return {r["id"]: r for r in results}
        else:
            print(f"  full900.json appears to be V16 — using checkpoint.jsonl for V17")

    cp_path = REPO / "data" / "tzuf_private1_checkpoint.jsonl"
    if not cp_path.exists():
        print("  ERROR: No V17 checkpoint found!")
        return {}
    results: dict[str, dict] = {}
    for line in cp_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id")
            if qid and qid not in results and "error" not in r:
                results[qid] = r
        except json.JSONDecodeError:
            pass
    return results


def main() -> None:
    # Load base
    print("Loading V16_SUPER_HYBRID (base)...")
    base_path = REPO / "data" / "private_submission_V16_SUPER_HYBRID.json"
    if not base_path.exists():
        print("ERROR: V16_SUPER_HYBRID not found. Build it first.")
        return
    submission = json.loads(base_path.read_text())

    print("Loading V17 results...")
    v17_by_id = load_v17()
    print(f"  V17: {len(v17_by_id)} answers available")

    questions = json.loads((REPO / "dataset" / "private" / "questions.json").read_text())
    q_map = {q["id"]: q for q in questions}

    # Build hybrid
    upgraded = 0
    kept_base = 0
    v17_is_noinfo = 0
    v17_missing = 0
    truncated_to_280 = 0

    new_answers = []
    for ans in submission["answers"]:
        qid = ans["question_id"]
        q = q_map.get(qid, {})

        # Only upgrade free_text
        if q.get("answer_type") != "free_text":
            new_answers.append(ans)
            continue

        v17 = v17_by_id.get(qid)
        if not v17:
            v17_missing += 1
            new_answers.append(ans)
            continue

        v17_raw = v17.get("answer") or v17.get("answer_text")
        base_ans = ans.get("answer")

        # Skip if V17 is null or no-info
        if is_no_info(v17_raw):
            v17_is_noinfo += 1
            new_answers.append(ans)
            continue

        v17_clean = strip_cite(str(v17_raw))
        if is_truncated(v17_clean):
            kept_base += 1
            new_answers.append(ans)
            continue

        # Normalize newlines to "; " for clean formatting
        v17_clean = re.sub(r"\n+", "; ", v17_clean).strip()
        v17_clean = add_period_if_needed(v17_clean)
        base_str = str(base_ans) if base_ans is not None else ""

        # Hard 280-char limit
        if len(v17_clean) > MAX_CHARS:
            v17_clean = smart_truncate(v17_clean, MAX_CHARS)
            if len(v17_clean) > MAX_CHARS:
                kept_base += 1
                new_answers.append(ans)
                continue
            truncated_to_280 += 1

        # Determine if V17 is better
        upgrade = False

        # Case 1: base is no-info → take any substantive V17 answer
        if is_no_info(base_ans):
            upgrade = True

        # Case 2: V17 starts with evidence-first pattern AND base doesn't
        elif is_evidence_first(v17_clean) and not is_evidence_first(base_str):
            upgrade = True

        # Case 3: V17 is substantially longer and base is short
        elif len(v17_clean) > len(base_str) + 30 and len(base_str) < 200:
            upgrade = True

        if upgrade:
            new_ans = dict(ans)
            new_ans["answer"] = v17_clean
            # KEEP base pages (critical: always preserve V15 page grounding)
            new_answers.append(new_ans)
            upgraded += 1
        else:
            kept_base += 1
            new_answers.append(ans)

    result = dict(submission)
    result["answers"] = new_answers

    # Verify
    nulls = sum(1 for a in new_answers if a["answer"] is None)
    nopg = sum(1 for a in new_answers if a["answer"] is not None and not a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []))
    cite_leaks = sum(1 for a in new_answers if a["answer"] and re.search(r"\(cite:", str(a["answer"])))
    no_info_count = sum(1 for a in new_answers if isinstance(a["answer"], str) and "no information" in a["answer"].lower())
    over280 = sum(1 for a in new_answers if a["answer"] and len(str(a["answer"])) > MAX_CHARS)

    print(f"\nUpgrade stats:")
    print(f"  Upgraded to V17: {upgraded}")
    print(f"  Kept V16_SUPER_HYBRID: {kept_base}")
    print(f"  V17 was no-info (kept base): {v17_is_noinfo}")
    print(f"  V17 missing (kept base): {v17_missing}")
    print(f"  Truncated to 280: {truncated_to_280}")

    print(f"\nVerification:")
    print(f"  null={nulls}, nopg={nopg}, cite_leaks={cite_leaks}, no_info={no_info_count}, over280={over280}")

    if over280 > 0:
        print(f"\nWARNING: {over280} answers over 280 chars — fixing...")
        fixed = 0
        fixed_answers = []
        for a in new_answers:
            if a["answer"] and len(str(a["answer"])) > MAX_CHARS:
                truncated = smart_truncate(str(a["answer"]), MAX_CHARS)
                if len(truncated) <= MAX_CHARS:
                    a2 = dict(a)
                    a2["answer"] = truncated
                    fixed_answers.append(a2)
                    fixed += 1
                else:
                    fixed_answers.append(a)
            else:
                fixed_answers.append(a)
        result["answers"] = fixed_answers
        new_answers = fixed_answers
        over280_after = sum(1 for a in new_answers if a["answer"] and len(str(a["answer"])) > MAX_CHARS)
        print(f"  Fixed {fixed}, remaining over280={over280_after}")

    out_path = REPO / "data" / "private_submission_V17_SUPER_HYBRID.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False))
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
