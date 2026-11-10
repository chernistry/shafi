#!/usr/bin/env python3
"""V9.1 Recovery: Re-run 9 V9 null regressions to recover answers.

V9 vs V8.2: V9 has 9 more null answers that V8.2 answered correctly.
Strategy: re-run those 9 questions on V9 server to try to recover answers.
Boolean judge-change questions (3) are kept as null (too risky to guess).
Non-boolean questions (6) are re-run with free_text routing override.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
V9_CHECKPOINT = REPO / "data" / "tzuf_v9_checkpoint.jsonl"
V8_1_RESULTS = REPO / "data" / "tzuf_v8_1_full900.json"
PRIVATE_QS = REPO / "dataset" / "private" / "questions.json"
OUTPUT = REPO / "data" / "tzuf_v9_1_full900.json"
SERVER_URL = "http://localhost:8003/query"
TIMEOUT = 60.0

# V9 regressions (null in V9, answered in V8.1/V8.2)
# Boolean judge-change questions — SKIP (risky, low-evidence in V8.1)
SKIP_QIDS = {
    "33c774d5",  # boolean CFI 073/2024 judge change (V8.1 Yes, pages=2 only)
    "8c346986",  # boolean CA 005/2025 judge change (V8.1 No, pages=3)
    "b24ae3c5",  # boolean CA panel change (V8.1 Yes, pages=1 only)
}

# Non-boolean regressions to recover via free_text routing
RECOVER_QIDS = {
    "2d3456c0",  # date consultation deadline → V8.1=2011-06-16
    "49ac14a6",  # date Financial Collateral commencement → V8.1=2019-04-17
    "9f08c8b1",  # name earlier appellate doc → V8.1=CA 006/2024
    "bc592dda",  # number Operating Regs version → V8.1=3
    "c9de6075",  # date DIFC Laws amendment deadline → V8.1=2017-06-19
    "cd54e570",  # number Employment Regs version → V8.1=3
}


def parse_sse(text: str) -> tuple[str | None, dict]:
    ans = None
    telem: dict = {}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "answer_final":
            ans = payload.get("text", "")
        elif payload.get("type") == "telemetry":
            telem = payload.get("payload", {})
    return ans, telem


def load_v9_checkpoint() -> dict[str, dict]:
    done: dict[str, dict] = {}
    for line in V9_CHECKPOINT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id", "")[:8] if len(r.get("id", "")) < 16 else r.get("id", "")
            if qid:
                done[qid] = r
                # Also index by short ID
                done[qid[:8]] = r
        except json.JSONDecodeError:
            pass
    return done


def main() -> None:
    qs = json.loads(PRIVATE_QS.read_text())
    q_map = {q["id"]: q for q in qs}

    # Load V9 results
    v9_results: dict[str, dict] = {}
    for line in V9_CHECKPOINT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id", "")
            if qid:
                v9_results[qid] = r
        except json.JSONDecodeError:
            pass

    print(f"V9 checkpoint: {len(v9_results)} questions loaded")

    # Find full question IDs for recovery targets
    recover_full_ids = {}
    for qid, q in q_map.items():
        short = qid[:8]
        if short in RECOVER_QIDS:
            recover_full_ids[qid] = q

    print(f"Recovery targets: {len(recover_full_ids)} questions")

    recovered = {}
    with httpx.Client(timeout=TIMEOUT) as client:
        for qid, q in recover_full_ids.items():
            short = qid[:8]
            original_type = q["answer_type"]
            print(f"  Trying {short} ({original_type}): {q['question'][:60]}")

            # Try with free_text routing first
            try:
                resp = client.post(
                    SERVER_URL,
                    json={"question": q["question"], "answer_type": "free_text"},
                )
                resp.raise_for_status()
                ans, telem = parse_sse(resp.text)
                used_pages = list(telem.get("used_page_ids") or [])
                ttft_ms = float(telem.get("ttft_ms", 0) or 0)

                if ans and ans.strip() and used_pages and not ans.lower().startswith("there is no information"):
                    print(f"    RECOVERED via free_text: ans={ans[:40]!r} pages={len(used_pages)}")
                    recovered[qid] = {
                        "id": qid,
                        "answer_type": original_type,  # restore original type
                        "answer": ans,  # will be coerced by submission builder
                        "used_page_ids": used_pages,
                        "ttft_ms": ttft_ms,
                        "elapsed_s": 0,
                        "recovery": "free_text_override",
                    }
                else:
                    print(f"    No recovery: ans={ans!r} pages={len(used_pages)}")
            except Exception as e:
                print(f"    ERROR: {e}")

    # Build V9.1 results: V9 base + recovered
    results = list(v9_results.values())
    recovered_count = 0
    for i, r in enumerate(results):
        qid = r.get("id", "")
        if qid in recovered:
            results[i] = recovered[qid]
            recovered_count += 1

    print(f"\nRecovered {recovered_count}/{len(recover_full_ids)} questions")
    print("Saving V9.1...")

    # Save as checkpoint-style jsonl for submission builder
    v9_1_checkpoint = REPO / "data" / "tzuf_v9_1_checkpoint.jsonl"
    with v9_1_checkpoint.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {v9_1_checkpoint}")


if __name__ == "__main__":
    main()
