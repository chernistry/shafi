#!/usr/bin/env python3
"""Build V17_HYBRID = V16_BEST + V17 free_text improvements.

Strategy:
- Base: V16_BEST (22 bool fixes, version corrections, 6 FT expansions, V15 pages)
- Upgrade: Only where V16_BEST FT answer is short (<100c) or noinfo
           AND V17 answer is clean, longer by 20+c, non-noinfo
- Never upgrade number/boolean/date/names types (V16_BEST is more accurate there)
- Never modify page references (keep V15 enriched pages from V16_BEST)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def strip_cites(text: str) -> str:
    """Remove (cite:...) markers from answer text."""
    while "(cite:" in text:
        start = text.find("(cite:")
        end = text.find(")", start)
        if end == -1:
            break
        text = text[:start] + text[end + 1 :]
    return text.strip()


def clean_newlines(text: str) -> str:
    """Replace newlines with semicolons for list-style answers."""
    text = re.sub(r"\n\n+", "; ", text)
    text = re.sub(r"\n", "; ", text)
    text = re.sub(r";+\s*;", ";", text)
    return text.strip("; ").strip()


def truncate_280(text: str) -> str:
    """Sentence-boundary truncation at 280 chars."""
    if len(text) <= 280:
        return text
    trunc = text[:280]
    for sep in [". ", "! ", "? ", "; ", "\n"]:
        idx = trunc.rfind(sep)
        if idx > 200:
            return trunc[: idx + 1].strip()
    return trunc.rstrip(".,;").strip()


def main() -> None:
    base_path = REPO / "data" / "private_submission_V16_BEST.json"
    v17_full900 = REPO / "data" / "tzuf_private1_full900.json"
    questions_path = REPO / "dataset" / "private" / "questions.json"
    output_path = REPO / "data" / "private_submission_V17_BEST.json"

    base = json.loads(base_path.read_text())
    v17_data = json.loads(v17_full900.read_text())
    v17_items = v17_data.get("results", v17_data.get("answers", []))
    questions = json.loads(questions_path.read_text())

    v17_map = {a["id"]: a for a in v17_items}
    qt_map = {q["id"]: q for q in questions}

    upgrades = 0
    skipped_type = 0
    skipped_short = 0

    for answer in base["answers"]:
        qid = answer["question_id"]
        v17_ans = v17_map.get(qid)
        if not v17_ans:
            continue

        q = qt_map.get(qid, {})
        atype = q.get("answer_type", "free_text")

        # Only upgrade free_text answers
        if atype != "free_text":
            skipped_type += 1
            continue

        vb_text = answer.get("answer", "")
        v17_raw = v17_ans.get("answer", "")

        if not isinstance(vb_text, str) or not isinstance(v17_raw, str):
            continue

        # Clean V17 answer
        v17_clean = strip_cites(v17_raw)
        v17_clean = clean_newlines(v17_clean)
        v17_final = truncate_280(v17_clean)

        # Skip noinfo V17 answers
        if "no info" in v17_final.lower() or "no information" in v17_final.lower():
            continue

        # Only upgrade short/noinfo V16_BEST answers
        is_short = len(vb_text) < 100 or "no info" in vb_text.lower()
        is_longer = len(v17_final) > len(vb_text) + 20

        if not is_short:
            skipped_short += 1
            continue

        if not is_longer:
            continue

        # Apply upgrade
        answer["answer"] = v17_final
        upgrades += 1
        print(f"  UPGRADE {qid[:12]}: {len(vb_text)}c → {len(v17_final)}c")
        print(f"    V16: {vb_text[:80]}")
        print(f"    V17: {v17_final[:80]}")

    output_path.write_text(json.dumps(base, ensure_ascii=False, separators=(",", ":")))
    print(f"\n=== V17_HYBRID BUILD ===")
    print(f"Upgrades applied: {upgrades}")
    print(f"Skipped (wrong type): {skipped_type}")
    print(f"Skipped (V16 not short): {skipped_short}")

    # Quick stats
    answers = base["answers"]
    null_count = sum(1 for a in answers if a.get("answer") is None)
    noinfo_count = sum(
        1 for a in answers
        if isinstance(a.get("answer"), str) and "no info" in a.get("answer", "").lower()
    )
    nopg_count = sum(
        1 for a in answers
        if not a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", [])
    )
    ttft_vals = [
        a["telemetry"]["timing"]["ttft_ms"]
        for a in answers
        if a.get("telemetry", {}).get("timing", {}).get("ttft_ms")
    ]
    ttft_avg = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0

    print(f"\nV17_HYBRID stats:")
    print(f"  null={null_count}, nopg={nopg_count}, noinfo={noinfo_count}")
    print(f"  ttft_avg={ttft_avg:.0f}ms")
    print(f"\nOutput: {output_path} ({output_path.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
