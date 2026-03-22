#!/usr/bin/env python3
"""TZUF V10: Full 900Q private eval — free_text prompt fix + overlap hint + G-guard.

Code: HEAD=8882717 (EYAL predicted_outputs + NOGA G-guard + SHAI fixes)
Config: .env.local (same as V9) — RERANK_PROVIDER_MODE defaults to api = zerank-2
Server: port 8000 (current running server)

Changes vs V9.1 (9c3b503):
  - 761164e NOGA: G-guard for orchestration early-exit paths
    * Prevents non-null answers with 0 page IDs from early-exit paths
    * Expected: minimal ΔG (V9.1 already at 4 floor, all genuinely unanswerable)
  - 9c8b05a SHAI: Revert harmful 280-char removal from free_text hint
    * V9 had broken free_text prompt (no 280-char limit, "2-4 sentences")
    * Fix restores correct prompt: "280-char HARD LIMIT", "2-3 sentences, 150-250 chars"
    * Expected: ΔAsst +0.5-1pp (no mid-sentence truncation)
  - dd63bb5 SHAI: Strengthen overlap hint — role-specific matching + law firm
    * Add "law firm" to overlap pattern list
    * Require matching SPECIFIC ROLE asked about
    * Explicitly exclude registrars/clerks/courts from overlap matching
    * Expected: ΔDet +1-2 (reduce false-positive overlap answers)
  - 8882717 EYAL: Fix predicted_outputs on primary strict-type path
    * A/B test showed TTFT-neutral (-41ms = noise)

Supports incremental resume: pass --resume to skip already-answered questions.
Checkpoints every question to prevent data loss.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PRIVATE_QUESTIONS = REPO / "dataset" / "private" / "questions.json"
OUTPUT = REPO / "data" / "tzuf_v10_full900.json"
CHECKPOINT = REPO / "data" / "tzuf_v10_checkpoint.jsonl"
SERVER_URL = "http://localhost:8002/query"
TIMEOUT = 120.0

V9_1_BASELINE_G = 0.9956   # V9.1 G_proxy
V9_1_BASELINE_TTFT = 1242.0  # V9.1 avg TTFT


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


def load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT.exists():
        return {}
    done: dict[str, dict] = {}
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id")
            if qid and "error" not in r and qid not in done:
                done[qid] = r
        except json.JSONDecodeError:
            pass
    return done


def append_checkpoint(result: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def main(resume: bool = False) -> None:
    if not PRIVATE_QUESTIONS.exists():
        print(f"ERROR: Private questions not found at {PRIVATE_QUESTIONS}")
        sys.exit(1)

    qs = json.loads(PRIVATE_QUESTIONS.read_text(encoding="utf-8"))
    total = len(qs)

    done: dict[str, dict] = {}
    if resume:
        done = load_checkpoint()
        print(f"Resuming V10 eval — {len(done)} already done, {total - len(done)} remaining")
    else:
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()
        print(f"Starting V10 fresh eval — {total} questions")
        print(f"Server: {SERVER_URL}")
        print(f"Head: 8882717 (SHAI free_text fix + overlap + NOGA G-guard)")

    results: list[dict] = list(done.values())
    errors = 0
    ttft_all: list[float] = []
    by_type: dict[str, dict] = {}

    for r in results:
        t = r.get("answer_type", "unknown")
        if t not in by_type:
            by_type[t] = {"count": 0, "null": 0, "no_pages": 0, "ttft_sum": 0.0, "ttft_n": 0}
        by_type[t]["count"] += 1
        if r.get("answer") is None:
            by_type[t]["null"] += 1
        if not r.get("used_page_ids"):
            by_type[t]["no_pages"] += 1
        if r.get("ttft_ms", 0) > 0:
            ttft_all.append(float(r["ttft_ms"]))
            by_type[t]["ttft_sum"] += float(r["ttft_ms"])
            by_type[t]["ttft_n"] += 1

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(qs, 1):
            qid: str = q["id"]
            q_text: str = q["question"]
            q_type: str = q["answer_type"]

            if qid in done:
                continue

            t0 = time.monotonic()
            try:
                resp = client.post(SERVER_URL, json={"question": q_text, "answer_type": q_type})
                resp.raise_for_status()
                parsed = parse_sse_response(resp.text)
                elapsed = time.monotonic() - t0
                telem = parsed["telemetry"] or {}
                pred_ans = parsed["answer_text"]
                used_pages: list[str] = list(telem.get("used_page_ids") or [])
                ttft_ms = float(telem.get("ttft_ms", 0) or 0)
            except Exception as e:
                elapsed = time.monotonic() - t0
                errors += 1
                result = {"id": qid, "answer_type": q_type, "error": str(e), "elapsed_s": round(elapsed, 3)}
                results.append(result)
                append_checkpoint(result)
                print(f"  [{i:3d}/{total}] ERROR {elapsed:.1f}s: {e}")
                continue

            has_pages = len(used_pages) > 0
            if ttft_ms > 0:
                ttft_all.append(ttft_ms)

            if q_type not in by_type:
                by_type[q_type] = {"count": 0, "null": 0, "no_pages": 0, "ttft_sum": 0.0, "ttft_n": 0}
            by_type[q_type]["count"] += 1
            if pred_ans is None:
                by_type[q_type]["null"] += 1
            if not has_pages:
                by_type[q_type]["no_pages"] += 1
            if ttft_ms > 0:
                by_type[q_type]["ttft_sum"] += ttft_ms
                by_type[q_type]["ttft_n"] += 1

            result = {
                "id": qid,
                "answer_type": q_type,
                "answer": pred_ans,
                "used_page_ids": used_pages,
                "ttft_ms": ttft_ms,
                "elapsed_s": round(elapsed, 3),
            }
            results.append(result)
            append_checkpoint(result)

            null_marker = "✗NULL" if pred_ans is None else ""
            no_pg_marker = "✗NOPG" if not has_pages else ""
            print(
                f"  [{i:3d}/{total}] {qid[:8]} {q_type:10s} "
                f"ttft={ttft_ms:.0f}ms pg={len(used_pages):2d} "
                f"{null_marker}{no_pg_marker}"
            )

            if i % 50 == 0:
                valid = [r for r in results if "error" not in r]
                null_count = sum(1 for r in valid if r.get("answer") is None)
                no_pg_count = sum(1 for r in valid if not r.get("used_page_ids"))
                g_proxy = (len(valid) - no_pg_count) / len(valid) if valid else 0.0
                ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0
                print(
                    f"\n  === V10 PROGRESS [{i}/{total}]: "
                    f"null={null_count} no_pages={no_pg_count} G_proxy={g_proxy:.4f} "
                    f"TTFT_avg={ttft_avg:.0f}ms [V9.1={V9_1_BASELINE_TTFT:.0f}ms] ===\n"
                )

    valid = [r for r in results if "error" not in r]
    null_count = sum(1 for r in valid if r.get("answer") is None)
    no_pg_count = sum(1 for r in valid if not r.get("used_page_ids"))
    g_proxy = (len(valid) - no_pg_count) / len(valid) if valid else 0.0
    ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0

    print(f"\n{'='*60}")
    print(f"V10 EVAL COMPLETE — {len(valid)}/{total} answered (errors={errors})")
    print(f"Null:     {null_count} ({null_count / total * 100:.1f}%) [V9.1=3 (0.3%)]")
    print(f"No-pages: {no_pg_count} ({no_pg_count / total * 100:.1f}%) [V9.1=4 (0.4%)]")
    print(f"G_proxy:  {g_proxy:.4f} [V9.1={V9_1_BASELINE_G:.4f}]")
    print(f"TTFT avg: {ttft_avg:.0f}ms [V9.1={V9_1_BASELINE_TTFT:.0f}ms]")
    print("\nBy type:")
    for t in ["boolean", "date", "free_text", "name", "names", "number"]:
        d = by_type.get(t)
        if d and d.get("count", 0) > 0:
            avg_ttft = d["ttft_sum"] / d["ttft_n"] if d["ttft_n"] > 0 else 0.0
            print(
                f"  {t:12s}: n={d['count']:3d} null={d['null']:2d} no_pg={d['no_pages']:2d} "
                f"ttft={avg_ttft:.0f}ms"
            )

    output_data = {
        "eval_id": "tzuf-v10-full900",
        "code_version": "8882717",
        "config": ".env.local (zerank-2 default)",
        "total": len(valid),
        "errors": errors,
        "null_count": null_count,
        "null_pct": round(null_count / total * 100, 2),
        "no_pages_count": no_pg_count,
        "no_pages_pct": round(no_pg_count / total * 100, 2),
        "g_proxy": round(g_proxy, 4),
        "g_proxy_v9_1_baseline": V9_1_BASELINE_G,
        "g_proxy_delta": round(g_proxy - V9_1_BASELINE_G, 4),
        "ttft_avg_ms": round(ttft_avg, 1),
        "ttft_delta_ms": round(ttft_avg - V9_1_BASELINE_TTFT, 1),
        "by_type": {
            t: {
                "count": d["count"],
                "null": d["null"],
                "no_pages": d["no_pages"],
                "ttft_avg_ms": round(d["ttft_sum"] / d["ttft_n"], 1) if d["ttft_n"] > 0 else 0.0,
            }
            for t, d in by_type.items()
        },
        "results": valid,
    }
    OUTPUT.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)
