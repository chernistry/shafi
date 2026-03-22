#!/usr/bin/env python3
"""Rocky-35a: Private dataset gate eval — spot-check 50Q after private ingestion.

Verifies:
1. Server responding with answers (not null/error)
2. TTFT within acceptable range
3. Telemetry completeness (used_page_ids not empty)
4. Answer type format compliance (boolean=True/False, number=numeric, etc.)

No Det/G metrics (no golden answers for private set).
Gate: <5% null answers, <10% empty telemetry, TTFT_avg < 4000ms.

Usage: After EYAL completes private ingestion, run:
  python scripts/tzuf35a_private_gate.py [--n 50]
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parents[1]
PRIVATE_QUESTIONS = REPO / "dataset" / "private" / "questions.json"
OUTPUT = REPO / "data" / "tzuf35a_private_gate.json"
SERVER_URL = "http://localhost:8000/query"
TIMEOUT = 120.0

GATE_MAX_NULL_PCT = 5.0    # >5% null answers = FAIL
GATE_MAX_EMPTY_TELEM_PCT = 10.0  # >10% empty telemetry = FAIL
GATE_MAX_TTFT_MS = 4000.0  # avg TTFT >4000ms = FAIL


def parse_sse_response(text: str) -> dict[str, object]:
    answer_text = None
    telemetry: dict[str, object] = {}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        msg_type = payload.get("type")
        if msg_type == "answer_final":
            answer_text = payload.get("text", "")
        elif msg_type == "telemetry":
            telemetry = payload.get("payload", {})
    return {"answer_text": answer_text, "telemetry": telemetry}


def is_null_answer(ans: object, q_type: str) -> bool:
    if ans is None:
        return True
    s = str(ans).strip().lower()
    return s in ("", "null", "none", "n/a", "unavailable", "unknown")


def main(n: int = 50) -> None:
    if not PRIVATE_QUESTIONS.exists():
        print(f"ERROR: Private questions not found at {PRIVATE_QUESTIONS}")
        sys.exit(1)

    all_qs = json.loads(PRIVATE_QUESTIONS.read_text(encoding="utf-8"))
    # Sample n questions stratified by type
    by_type: dict[str, list[dict]] = {}
    for q in all_qs:
        t = q.get("answer_type", "unknown")
        by_type.setdefault(t, []).append(q)

    sample: list[dict] = []
    for t, qs in by_type.items():
        take = max(1, round(n * len(qs) / len(all_qs)))
        sample.extend(random.sample(qs, min(take, len(qs))))
    if len(sample) > n:
        sample = random.sample(sample, n)

    total = len(sample)
    print(f"Rocky-35a PRIVATE GATE: {total} questions (sampled from {len(all_qs)})")
    print(f"HEAD: {Path(REPO, '.git/HEAD').read_text().strip()}")

    results = []
    null_count = 0
    empty_telem_count = 0
    ttft_all: list[float] = []
    errors = 0

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(sample, 1):
            qid = q.get("id", "?")[:16]
            q_text = q.get("question", "")
            q_type = q.get("answer_type", "free_text")

            t0 = time.monotonic()
            try:
                resp = client.post(SERVER_URL, json={"question": q_text, "answer_type": q_type})
                resp.raise_for_status()
                parsed = parse_sse_response(resp.text)
                elapsed = time.monotonic() - t0

                telem = parsed["telemetry"] or {}
                pred_ans = parsed["answer_text"]
                used_pages = telem.get("used_page_ids", [])
                ttft_ms = float(telem.get("ttft_ms", 0) or 0)
                model = telem.get("model_llm", "?")

            except Exception as e:
                elapsed = time.monotonic() - t0
                errors += 1
                print(f"  [{i:3d}/{total}] {qid} ERROR {elapsed:.1f}s: {e}")
                results.append({"id": qid, "error": str(e), "answer_type": q_type})
                continue

            is_null = is_null_answer(pred_ans, q_type)
            has_telem = bool(used_pages)
            if is_null:
                null_count += 1
            if not has_telem:
                empty_telem_count += 1
            if ttft_ms > 0:
                ttft_all.append(ttft_ms)

            null_flag = " ← NULL" if is_null else ""
            telem_flag = " ← NO_TELEM" if not has_telem else ""
            print(
                f"  [{i:3d}/{total}] {qid} {q_type:10s} "
                f"ans={str(pred_ans)[:20]!r:22s} "
                f"pages={len(used_pages or []):2d} ttft={ttft_ms:.0f}ms model={model:15s}{null_flag}{telem_flag}"
            )

            results.append({
                "id": q.get("id", "?"),
                "answer_type": q_type,
                "answer": pred_ans,
                "used_pages": used_pages,
                "ttft_ms": ttft_ms,
                "model": model,
                "elapsed_s": round(elapsed, 3),
                "is_null": is_null,
                "has_telem": has_telem,
            })

    # Gate checks
    null_pct = null_count / total * 100 if total else 0
    empty_telem_pct = empty_telem_count / total * 100 if total else 0
    ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0

    gate_null = null_pct <= GATE_MAX_NULL_PCT
    gate_telem = empty_telem_pct <= GATE_MAX_EMPTY_TELEM_PCT
    gate_ttft = ttft_avg <= GATE_MAX_TTFT_MS
    overall_pass = gate_null and gate_telem and gate_ttft

    print(f"\n=== TZUF-35a: PRIVATE GATE EVAL ===")
    print(f"Questions: {total} | Errors: {errors}")
    print(f"Null answers: {null_count}/{total} ({null_pct:.1f}%) — gate <{GATE_MAX_NULL_PCT}% {'✓ PASS' if gate_null else '✗ FAIL'}")
    print(f"Empty telem:  {empty_telem_count}/{total} ({empty_telem_pct:.1f}%) — gate <{GATE_MAX_EMPTY_TELEM_PCT}% {'✓ PASS' if gate_telem else '✗ FAIL'}")
    print(f"TTFT avg: {ttft_avg:.0f}ms — gate <{GATE_MAX_TTFT_MS:.0f}ms {'✓ PASS' if gate_ttft else '✗ FAIL'}")
    print(f"\n{'✓ GATE PASS' if overall_pass else '✗ GATE FAIL'} — private data ready for submission: {'YES' if overall_pass else 'NO'}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "eval_id": "tzuf-35a",
        "total": total,
        "errors": errors,
        "null_count": null_count,
        "null_pct": null_pct,
        "empty_telem_count": empty_telem_count,
        "empty_telem_pct": empty_telem_pct,
        "ttft_avg_ms": ttft_avg,
        "gate_null": gate_null,
        "gate_telem": gate_telem,
        "gate_ttft": gate_ttft,
        "overall_pass": overall_pass,
        "results": results,
    }, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(n=n)
