#!/usr/bin/env python3
"""Quick V13 checkpoint monitor — use with: watch -n 30 python3 scripts/v13_monitor.py"""
import json, statistics
from pathlib import Path

ckpt = Path("data/tzuf_private1_checkpoint.jsonl")
if not ckpt.exists():
    print("No checkpoint found")
    exit()

v = {}
for line in ckpt.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
        qid = r.get("id")
        if not qid or "error" in r:
            continue
        v[qid] = r
    except Exception:
        pass

nulls = sum(
    1 for r in v.values()
    if not r.get("answer") or str(r.get("answer", "")).strip().lower() in ("null", "none", "")
)
nopg = sum(
    1 for r in v.values()
    if r.get("answer")
    and str(r.get("answer", "")).strip().lower() not in ("null", "none", "")
    and not r.get("used_page_ids")
)
ttfts = [
    float(r.get("ttft_ms", 0) or 0)
    for r in v.values()
    if float(r.get("ttft_ms", 0) or 0) > 0
]
f_vals = [
    1.05 if t < 1000 else 1.02 if t < 2000 else 1.00 if t < 3000
    else max(0.85, 0.99 - (t - 3000) * 0.14 / 2000) if t < 5000 else 0.85
    for t in ttfts
]

pct = len(v) * 100 // 900
over5 = sum(1 for t in ttfts if t > 5000)
print(f"V13: {len(v)}/900 ({pct}%) | null={nulls} nopg={nopg} | TTFT={statistics.mean(ttfts):.0f}ms | F={statistics.mean(f_vals):.4f} | >5s={over5}")
