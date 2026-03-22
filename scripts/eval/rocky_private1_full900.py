#!/usr/bin/env python3
"""Rocky-private-1: Full 900Q private eval to generate submission answers.

Runs ALL 900 private questions through the pipeline and collects:
- answers (for submission)
- used_page_ids (grounding for G score)
- TTFT measurements
- model used

No Det/G metrics (no golden answers for private set).

Supports incremental resume: pass --resume to skip already-answered questions.
Saves checkpoint after every question to prevent data loss on server interruption.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parents[1]
PRIVATE_QUESTIONS = REPO / "dataset" / "private" / "questions.json"
OUTPUT = REPO / "data" / "tzuf_private1_full900.json"
CHECKPOINT = REPO / "data" / "tzuf_private1_checkpoint.jsonl"
SERVER_URL = "http://localhost:8000/query"
TIMEOUT = 120.0


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
    """Load previously answered questions from checkpoint. Returns {qid: result}."""
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
    """Append a single result to the checkpoint file."""
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


def main(resume: bool = False) -> None:
    if not PRIVATE_QUESTIONS.exists():
        print(f"ERROR: Private questions not found at {PRIVATE_QUESTIONS}")
        sys.exit(1)

    qs = json.loads(PRIVATE_QUESTIONS.read_text(encoding="utf-8"))
    total = len(qs)

    # Load checkpoint if resuming
    done_map: dict[str, dict] = {}
    if resume and CHECKPOINT.exists():
        done_map = load_checkpoint()
        print(f"Rocky-private-1: RESUME mode — {len(done_map)} questions already answered")

    print(f"Rocky-private-1: FULL {total}Q PRIVATE EVAL")
    print(f"HEAD: {Path(REPO, '.git/HEAD').read_text().strip()}")
    print(f"Server: {SERVER_URL}")

    results: list[dict] = []
    null_count = 0
    empty_telem_count = 0
    ttft_all: list[float] = []
    errors = 0
    skipped = 0
    by_type: dict[str, dict] = {}

    # Pre-populate results from checkpoint
    if done_map:
        for q in qs:
            qid = q.get("id", "?")
            if qid in done_map:
                r = done_map[qid]
                results.append(r)
                q_type = r.get("answer_type", "free_text")
                pred_ans = r.get("answer")
                used_pages = r.get("used_page_ids", [])
                ttft_ms = float(r.get("ttft_ms", 0) or 0)
                is_null = pred_ans is None or str(pred_ans).strip().lower() in ("", "null", "none")
                if is_null:
                    null_count += 1
                if not bool(used_pages):
                    empty_telem_count += 1
                if ttft_ms > 0:
                    ttft_all.append(ttft_ms)
                t = by_type.setdefault(q_type, {"n": 0, "null": 0, "no_pages": 0, "ttft": []})
                t["n"] += 1
                if is_null:
                    t["null"] += 1
                if not bool(used_pages):
                    t["no_pages"] += 1
                if ttft_ms > 0:
                    t["ttft"].append(ttft_ms)
                skipped += 1
        print(f"Pre-loaded {skipped} answers from checkpoint")

    pending_qs = [q for q in qs if q.get("id", "?") not in done_map]
    print(f"Remaining: {len(pending_qs)} questions to answer")

    with httpx.Client(timeout=TIMEOUT) as client:
        for i_pending, q in enumerate(pending_qs, 1):
            i_overall = skipped + i_pending
            qid = q.get("id", "?")
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
                print(f"  [{i_overall:4d}/{total}] ERROR {elapsed:.1f}s: {e}")
                results.append({"id": qid, "error": str(e), "answer_type": q_type})
                continue

            is_null = pred_ans is None or str(pred_ans).strip().lower() in ("", "null", "none")
            has_pages = bool(used_pages)
            if is_null:
                null_count += 1
            if not has_pages:
                empty_telem_count += 1
            if ttft_ms > 0:
                ttft_all.append(ttft_ms)

            t = by_type.setdefault(q_type, {"n": 0, "null": 0, "no_pages": 0, "ttft": []})
            t["n"] += 1
            if is_null:
                t["null"] += 1
            if not has_pages:
                t["no_pages"] += 1
            if ttft_ms > 0:
                t["ttft"].append(ttft_ms)

            status = "✗" if is_null else "✓"
            if i_overall % 50 == 0 or is_null:
                print(
                    f"  [{i_overall:4d}/{total}] {qid[:12]} {q_type:10s} {status} "
                    f"ans={str(pred_ans)[:20]!r:22s} pages={len(used_pages):2d} "
                    f"ttft={ttft_ms:.0f}ms"
                )

            result_entry = {
                "id": qid,
                "answer_type": q_type,
                "answer": pred_ans,
                "used_page_ids": used_pages,
                "ttft_ms": ttft_ms,
                "model": model,
                "elapsed_s": round(elapsed, 3),
            }
            results.append(result_entry)
            append_checkpoint(result_entry)

    ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0
    null_pct = null_count / total * 100 if total else 0
    empty_pct = empty_telem_count / total * 100 if total else 0

    print(f"\n=== TZUF-PRIVATE-1: FULL {total}Q PRIVATE EVAL ===")
    print(f"Errors: {errors} | Null: {null_count} ({null_pct:.1f}%) | No-pages: {empty_telem_count} ({empty_pct:.1f}%)")
    print(f"TTFT avg: {ttft_avg:.0f}ms")
    print(f"\nBy type:")
    for t, stats in sorted(by_type.items()):
        ttft_t = sum(stats["ttft"]) / len(stats["ttft"]) if stats["ttft"] else 0
        print(f"  {t:10s}: n={stats['n']:3d} null={stats['null']:3d} no_pages={stats['no_pages']:3d} ttft={ttft_t:.0f}ms")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "eval_id": "tzuf-private-1",
        "total": total,
        "errors": errors,
        "null_count": null_count,
        "null_pct": null_pct,
        "empty_telem_count": empty_telem_count,
        "empty_telem_pct": empty_pct,
        "ttft_avg_ms": ttft_avg,
        "by_type": {t: {k: v for k, v in s.items() if k != "ttft"} for t, s in by_type.items()},
        "results": results,
    }, indent=2))
    print(f"\nSaved to {OUTPUT}")
    print(f"Results contain answer+used_page_ids for all {total} private questions.")
    # Clean up checkpoint on clean finish
    if CHECKPOINT.exists() and errors < total * 0.1:
        CHECKPOINT.unlink()
        print("Checkpoint cleared (eval succeeded)")


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    main(resume=resume)
