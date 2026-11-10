#!/usr/bin/env python3
"""TZUF: FlashRank 30Q gate test.

Compares FlashRank reranker (port 8001) vs V7 zerank-2 baseline.

Pass conditions:
  - G_proxy_flashrank >= G_proxy_v7_on_sample - 0.02 (max 2pp regression on 30Q)
  - TTFT avg < 1500ms (vs V7 baseline ~2130ms, target saves 630ms+)

G_proxy = fraction of answers with non-empty used_page_ids.
Also measures page Jaccard overlap between FlashRank and V7 selections
as a quality signal for true G risk.

Usage:
  uv run python scripts/tzuf_flashrank_gate_30q.py
  uv run python scripts/tzuf_flashrank_gate_30q.py --port 8001 --n 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PRIVATE_QS = REPO / "dataset" / "private" / "questions.json"
V7_RESULTS = REPO / "data" / "tzuf_private1_full900.json"
OUTPUT = REPO / "data" / "tzuf_flashrank_gate_30q.json"

SEED = 42
# Stratified sample proportional to V7 type distribution
# boolean=193, date=93, free_text=270, name=95, names=90, number=159 → total=900
# 30Q target: boolean=6, date=3, free_text=10, name=3, names=3, number=5
TYPE_TARGETS: dict[str, int] = {
    "boolean": 6,
    "date": 3,
    "free_text": 10,
    "name": 3,
    "names": 3,
    "number": 5,
}

V7_G_PROXY = 0.9811  # Full-900Q V7 baseline
TTFT_BASELINE_MS = 2130.0  # V7 avg TTFT


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


def build_sample(n_total: int = 30) -> list[dict[str, object]]:
    """Build stratified sample of private questions."""
    qs = json.loads(PRIVATE_QS.read_text())
    by_type: dict[str, list[dict[str, object]]] = {}
    for q in qs:
        by_type.setdefault(q["answer_type"], []).append(q)

    rng = random.Random(SEED)
    sample: list[dict[str, object]] = []
    for t, n in TYPE_TARGETS.items():
        pool = list(by_type.get(t, []))
        rng.shuffle(pool)
        sample.extend(pool[:n])

    # Fill to n_total if needed (shouldn't be necessary with default targets)
    if len(sample) < n_total:
        remaining = [q for q in qs if q not in sample]
        rng.shuffle(remaining)
        sample.extend(remaining[: n_total - len(sample)])

    return sample[:n_total]


def load_v7_index() -> dict[str, dict[str, object]]:
    """Build {id → result} from V7 full eval."""
    v7 = json.loads(V7_RESULTS.read_text())
    return {r["id"]: r for r in v7.get("results", [])}


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    server_url = f"http://localhost:{args.port}/query"
    sample = build_sample(args.n)
    v7_index = load_v7_index()

    print(f"FlashRank Gate Test — {len(sample)}Q on port {args.port}")
    print(f"V7 baseline G_proxy (full 900Q): {V7_G_PROXY:.4f} | TTFT: {TTFT_BASELINE_MS:.0f}ms")
    print("Gate threshold: G_proxy >= [v7_on_sample] - 0.02 | TTFT < 1500ms")
    print("─" * 70)

    results: list[dict[str, object]] = []
    ttft_all: list[float] = []
    errors = 0

    with httpx.Client(timeout=args.timeout) as client:
        for i, q in enumerate(sample, 1):
            qid: str = q["id"]  # type: ignore[assignment]
            q_text: str = q["question"]  # type: ignore[assignment]
            q_type: str = q["answer_type"]  # type: ignore[assignment]

            v7r = v7_index.get(qid, {})
            v7_pages: list[str] = list(v7r.get("used_page_ids") or [])
            v7_has_pages = len(v7_pages) > 0

            t0 = time.monotonic()
            try:
                resp = client.post(server_url, json={"question": q_text, "answer_type": q_type})
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
                print(f"  [{i:2d}/{len(sample)}] ERROR {elapsed:.1f}s: {e}")
                results.append({"id": qid, "error": str(e), "answer_type": q_type})
                continue

            has_pages = len(used_pages) > 0
            jacc = jaccard(used_pages, v7_pages) if v7_has_pages or has_pages else 1.0
            change = ""
            if v7_has_pages and not has_pages:
                change = " ⚠ LOST PAGES"
            elif not v7_has_pages and has_pages:
                change = " ✓ GAINED PAGES"

            if ttft_ms > 0:
                ttft_all.append(ttft_ms)

            marker = "✓" if has_pages else "✗"
            print(
                f"  [{i:2d}/{len(sample)}] {qid[:8]} {q_type:10s} "
                f"{marker} ttft={ttft_ms:.0f}ms pg={len(used_pages):2d} "
                f"jacc={jacc:.2f} v7pg={len(v7_pages):2d}{change}"
            )

            results.append({
                "id": qid,
                "answer_type": q_type,
                "answer": pred_ans,
                "used_page_ids": used_pages,
                "v7_page_ids": v7_pages,
                "has_pages": has_pages,
                "v7_has_pages": v7_has_pages,
                "jaccard_vs_v7": round(jacc, 4),
                "ttft_ms": ttft_ms,
                "elapsed_s": round(elapsed, 3),
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    has_pages_count = sum(1 for r in valid if r["has_pages"])
    v7_has_pages_count = sum(1 for r in valid if r["v7_has_pages"])

    g_proxy_flash = has_pages_count / n_valid if n_valid else 0.0
    g_proxy_v7_sample = v7_has_pages_count / n_valid if n_valid else 0.0
    g_gate_threshold = g_proxy_v7_sample - 0.02

    ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0
    ttft_gate = 1500.0

    # Jaccard for questions where V7 had pages
    jaccard_scores = [r["jaccard_vs_v7"] for r in valid if r.get("v7_has_pages")]  # type: ignore[misc]
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    # Page coverage changes
    lost_pages = [r["id"] for r in valid if r.get("v7_has_pages") and not r.get("has_pages")]  # type: ignore[misc]
    gained_pages = [r["id"] for r in valid if not r.get("v7_has_pages") and r.get("has_pages")]  # type: ignore[misc]

    # By type breakdown
    by_type: dict[str, dict[str, int]] = {}
    for r in valid:
        t = r["answer_type"]  # type: ignore[assignment]
        if t not in by_type:
            by_type[t] = {"has": 0, "total": 0}
        by_type[t]["total"] += 1
        if r["has_pages"]:
            by_type[t]["has"] += 1

    g_pass = g_proxy_flash >= g_gate_threshold
    ttft_pass = ttft_avg <= ttft_gate

    print("\n" + "═" * 70)
    print(f"FLASHRANK GATE RESULTS — {n_valid}Q (errors={errors})")
    print("═" * 70)
    print(f"G_proxy (FlashRank):   {g_proxy_flash:.4f}  [{has_pages_count}/{n_valid} with pages]")
    print(f"G_proxy (V7 on sample):{g_proxy_v7_sample:.4f}  [{v7_has_pages_count}/{n_valid} with pages]")
    print(f"G gate threshold:      {g_gate_threshold:.4f}  (v7_sample - 0.02)")
    print(f"G_proxy ΔFlash-V7:     {g_proxy_flash - g_proxy_v7_sample:+.4f}")
    print(f"G gate: {'PASS ✓' if g_pass else 'FAIL ✗'}")
    print()
    print(f"TTFT avg (FlashRank):  {ttft_avg:.0f}ms  (target < {ttft_gate:.0f}ms, V7={TTFT_BASELINE_MS:.0f}ms)")
    print(f"TTFT gate: {'PASS ✓' if ttft_pass else 'FAIL ✗'}")
    print(f"TTFT speedup:          {TTFT_BASELINE_MS - ttft_avg:+.0f}ms ({(TTFT_BASELINE_MS - ttft_avg) / TTFT_BASELINE_MS * 100:+.1f}%)")
    print()
    print(f"Page overlap (Jaccard vs V7): {avg_jaccard:.3f}  ({len(jaccard_scores)}Q where V7 had pages)")
    print(f"Pages LOST vs V7:  {len(lost_pages)}  — {', '.join(i[:8] for i in lost_pages[:5])}")
    print(f"Pages GAINED vs V7:{len(gained_pages)}")
    print()
    print("By type:")
    for t in ["boolean", "date", "free_text", "name", "names", "number"]:
        d = by_type.get(t)
        if d:
            print(f"  {t:12s}: {d['has']}/{d['total']} ({d['has'] / d['total'] * 100:.0f}% has_pages)")
    print()

    overall_pass = g_pass and ttft_pass
    print(f"OVERALL: {'PASS ✓ — FlashRank safe for V8' if overall_pass else 'FAIL ✗ — FlashRank NOT recommended'}")

    if not g_pass:
        print(f"\n⚠ G regression too large: {g_proxy_flash:.4f} < {g_gate_threshold:.4f}")
        print("  Recommendation: DO NOT use FlashRank in V8 submission.")
        print("  FlashRank MiniLM trained on MS-MARCO (general QA) — poor fit for legal text.")
    if not ttft_pass:
        print(f"\n⚠ TTFT gate missed: {ttft_avg:.0f}ms > {ttft_gate:.0f}ms target")

    # Save
    output_data = {
        "eval_id": "tzuf-flashrank-gate-30q",
        "port": args.port,
        "n_questions": n_valid,
        "errors": errors,
        "g_proxy_flashrank": round(g_proxy_flash, 4),
        "g_proxy_v7_sample": round(g_proxy_v7_sample, 4),
        "g_gate_threshold": round(g_gate_threshold, 4),
        "g_delta": round(g_proxy_flash - g_proxy_v7_sample, 4),
        "g_pass": g_pass,
        "ttft_avg_ms": round(ttft_avg, 1),
        "ttft_pass": ttft_pass,
        "ttft_speedup_ms": round(TTFT_BASELINE_MS - ttft_avg, 1),
        "avg_jaccard_vs_v7": round(avg_jaccard, 4),
        "lost_pages_count": len(lost_pages),
        "gained_pages_count": len(gained_pages),
        "overall_pass": overall_pass,
        "by_type": by_type,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
