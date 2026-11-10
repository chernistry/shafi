#!/usr/bin/env python3
"""V9 TTFT Gate Test — 30Q stratified sample.

Compares V9 (port 8003, streaming strict) vs V8 baseline (port 8002, non-stream).

Pass conditions (both must hold):
  - G_proxy_v9 >= G_proxy_v8_sample - 0.02   (no G regression)
  - TTFT_avg_v9 < TTFT_avg_v8_sample - 100ms  (at least 100ms improvement)

Also measures per-type TTFT breakdown to understand gains by type.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PRIVATE_QS = REPO / "dataset" / "private" / "questions.json"
V8_RESULTS = REPO / "data" / "tzuf_v8_1_full900.json"  # V8.2 baseline
OUTPUT = REPO / "data" / "v9_ttft_gate_30q.json"

V8_PORT = 8002
V9_PORT = 8003
SEED = 42
TIMEOUT = 120.0

TYPE_TARGETS: dict[str, int] = {
    "boolean": 6,
    "date": 3,
    "free_text": 5,
    "name": 4,
    "names": 3,
    "number": 5,
}


def parse_sse(text: str) -> tuple[str | None, dict]:
    ans = None
    telem: dict = {}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        p = json.loads(line[6:])
        if p.get("type") == "answer_final":
            ans = p.get("text")
        elif p.get("type") == "telemetry":
            telem = p.get("payload", {})
    return ans, telem


def build_sample() -> list[dict]:
    qs = json.loads(PRIVATE_QS.read_text())
    by_type: dict[str, list[dict]] = {}
    for q in qs:
        by_type.setdefault(q["answer_type"], []).append(q)
    rng = random.Random(SEED)
    sample: list[dict] = []
    for t, n in TYPE_TARGETS.items():
        pool = list(by_type.get(t, []))
        rng.shuffle(pool)
        sample.extend(pool[:n])
    return sample


def query(client: httpx.Client, port: int, q: dict) -> dict:
    url = f"http://localhost:{port}/query"
    t0 = time.monotonic()
    try:
        resp = client.post(url, json={"question": q["question"], "answer_type": q["answer_type"]})
        resp.raise_for_status()
        elapsed = time.monotonic() - t0
        ans, telem = parse_sse(resp.text)
        pages = list(telem.get("used_page_ids") or [])
        ttft_ms = float(telem.get("ttft_ms", 0) or 0)
        return {
            "id": q["id"],
            "answer_type": q["answer_type"],
            "answer": ans,
            "used_page_ids": pages,
            "has_pages": len(pages) > 0,
            "ttft_ms": ttft_ms,
            "elapsed_s": round(elapsed, 3),
        }
    except Exception as e:
        return {"id": q["id"], "answer_type": q["answer_type"], "error": str(e)}


def summarize(results: list[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"g_proxy": 0.0, "ttft_avg": 0.0, "n": 0, "by_type": {}}
    has_pages = sum(1 for r in valid if r["has_pages"])
    g = has_pages / len(valid)
    ttfts = [r["ttft_ms"] for r in valid if r.get("ttft_ms", 0) > 0]
    ttft_avg = sum(ttfts) / len(ttfts) if ttfts else 0.0

    by_type: dict[str, dict] = {}
    for r in valid:
        t = r["answer_type"]
        if t not in by_type:
            by_type[t] = {"count": 0, "has_pages": 0, "ttft_sum": 0.0, "ttft_n": 0}
        by_type[t]["count"] += 1
        if r["has_pages"]:
            by_type[t]["has_pages"] += 1
        if r.get("ttft_ms", 0) > 0:
            by_type[t]["ttft_sum"] += r["ttft_ms"]
            by_type[t]["ttft_n"] += 1

    by_type_summary = {}
    for t, d in by_type.items():
        avg_ttft = d["ttft_sum"] / d["ttft_n"] if d["ttft_n"] > 0 else 0.0
        by_type_summary[t] = {
            "n": d["count"],
            "has_pages": d["has_pages"],
            "g": round(d["has_pages"] / d["count"], 4),
            "ttft_avg_ms": round(avg_ttft, 1),
        }
    return {"g_proxy": round(g, 4), "ttft_avg": round(ttft_avg, 1), "n": len(valid), "by_type": by_type_summary}


def main() -> None:
    sample = build_sample()
    print(f"V9 TTFT Gate Test — {len(sample)}Q (seed={SEED})")
    print(f"V8 baseline: port {V8_PORT} | V9 candidate: port {V9_PORT}")
    print("─" * 70)

    v8_results: list[dict] = []
    v9_results: list[dict] = []

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(sample, 1):
            qid = q["id"]
            qtype = q["answer_type"]

            r8 = query(client, V8_PORT, q)
            r9 = query(client, V9_PORT, q)

            ttft_delta = r9.get("ttft_ms", 0) - r8.get("ttft_ms", 0)
            pg8 = len(r8.get("used_page_ids") or [])
            pg9 = len(r9.get("used_page_ids") or [])
            v8_results.append(r8)
            v9_results.append(r9)

            marker = ""
            if r8.get("has_pages") and not r9.get("has_pages"):
                marker = " ⚠ LOST_PAGES"
            elif not r8.get("has_pages") and r9.get("has_pages"):
                marker = " ✓ GAINED"

            print(
                f"  [{i:2d}/{len(sample)}] {qid[:8]} {qtype:10s} "
                f"V8={r8.get('ttft_ms', 0):.0f}ms/{pg8}pg  "
                f"V9={r9.get('ttft_ms', 0):.0f}ms/{pg9}pg  "
                f"Δ={ttft_delta:+.0f}ms{marker}"
            )

    s8 = summarize(v8_results)
    s9 = summarize(v9_results)

    g_gate = s8["g_proxy"] - 0.02
    ttft_gate = s8["ttft_avg"] - 100.0
    g_pass = s9["g_proxy"] >= g_gate
    ttft_pass = s9["ttft_avg"] <= ttft_gate

    print("\n" + "═" * 70)
    print("V9 TTFT GATE RESULTS")
    print("═" * 70)
    print(f"G_proxy  V8={s8['g_proxy']:.4f}  V9={s9['g_proxy']:.4f}  threshold≥{g_gate:.4f}  {'PASS ✓' if g_pass else 'FAIL ✗'}")
    print(f"TTFT_avg V8={s8['ttft_avg']:.0f}ms  V9={s9['ttft_avg']:.0f}ms  target≤{ttft_gate:.0f}ms  {'PASS ✓' if ttft_pass else 'FAIL ✗'}")
    print(f"TTFT delta: {s9['ttft_avg'] - s8['ttft_avg']:+.0f}ms")
    print()
    print("By type (V8 → V9):")
    for t in ["boolean", "date", "free_text", "name", "names", "number"]:
        d8 = s8["by_type"].get(t)
        d9 = s9["by_type"].get(t)
        if d8:
            v8t = d8["ttft_avg_ms"]
            v9t = d9["ttft_avg_ms"] if d9 else 0.0
            delta = v9t - v8t
            print(f"  {t:12s}: V8={v8t:.0f}ms  V9={v9t:.0f}ms  Δ={delta:+.0f}ms  g8={d8['g']:.3f} g9={d9['g'] if d9 else 0:.3f}")

    overall = g_pass and ttft_pass
    print(f"\nOVERALL: {'PASS ✓ — V9 safe for full 900Q eval' if overall else 'FAIL ✗ — do not proceed'}")

    if not g_pass:
        lost = [r["id"][:8] for r in v9_results if r.get("has_pages") is False and
                next((r8 for r8 in v8_results if r8["id"] == r["id"]), {}).get("has_pages")]
        print(f"\n⚠ G regression: {s9['g_proxy']:.4f} < {g_gate:.4f}  lost pages: {lost}")
    if not ttft_pass:
        print(f"\n⚠ TTFT improvement insufficient: {s9['ttft_avg']:.0f}ms > target {ttft_gate:.0f}ms")

    OUTPUT.write_text(json.dumps({
        "eval_id": "v9-ttft-gate-30q",
        "v8_port": V8_PORT, "v9_port": V9_PORT,
        "n": len(sample),
        "v8": s8, "v9": s9,
        "g_gate": round(g_gate, 4),
        "ttft_gate": round(ttft_gate, 1),
        "g_pass": g_pass, "ttft_pass": ttft_pass,
        "overall_pass": overall,
        "v8_results": v8_results,
        "v9_results": v9_results,
    }, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
