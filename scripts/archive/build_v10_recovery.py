#!/usr/bin/env python3
"""V10 Recovery: Re-run nopg questions with free_text routing override.

Same strategy as V9.1 recovery, applied to V10 eval results.
Root cause is identical: strict-type hard Qdrant filter for regulation title refs.
free_text routing bypasses the filter and uses hybrid retrieval.

Strategy:
  - Load V10 full 900Q eval results
  - For each 0-page question, re-run with answer_type=free_text on V10 server (port 8000)
  - Extract correct-type answer from free_text response
  - Update eval results for recovered questions
  - Save as tzuf_v10_1_full900.json and build new submission

Expected: similar recovery to V9.1 (~12 questions recovered).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from dateutil import parser as dateutil_parser

REPO = Path(__file__).resolve().parents[1]
V10_RESULTS = REPO / "data" / "tzuf_v10_full900.json"
PRIVATE_QS = REPO / "dataset" / "private" / "questions.json"
OUTPUT = REPO / "data" / "tzuf_v10_1_full900.json"
SERVER_URL = "http://localhost:8000/query"
TIMEOUT = 60.0


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


def extract_date(text: str) -> str | None:
    """Try to extract ISO date from free text."""
    month_names = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December"
    )
    m = re.search(rf"\b(\d{{1,2}})\s+({month_names})\s+(\d{{4}})\b", text, re.IGNORECASE)
    if m:
        try:
            return dateutil_parser.parse(m.group(0)).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return m.group(0)
    return None


def extract_number(text: str) -> int | float | None:
    """Extract the most likely version/count number from text."""
    m = re.search(r"(?:No\.|Number|Version|no\.)\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return int(v) if v == int(v) else v
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    if nums:
        v = float(nums[0])
        return int(v) if v == int(v) else v
    return None


def extract_answer(raw: str, answer_type: str) -> object:
    """Extract typed answer from free_text response."""
    if not raw or raw.lower().strip() in ("null", "none", ""):
        return None
    clean = re.sub(r"\s*\(cite:[^)]+\)", "", raw).strip()
    if "no information" in clean.lower() or "not state" in clean.lower():
        return None

    if answer_type == "number":
        return extract_number(clean)
    elif answer_type == "date":
        return extract_date(clean)
    elif answer_type in ("name", "names"):
        first = clean.split(".")[0].strip()
        return first if first else clean[:200]
    return clean


def main() -> None:
    if not V10_RESULTS.exists():
        print(f"ERROR: V10 results not found at {V10_RESULTS}")
        print("Run tzuf_v10_full900.py first.")
        return

    data = json.loads(V10_RESULTS.read_text())
    qs_raw = json.loads(PRIVATE_QS.read_text())
    q_by_id = {q["id"]: q for q in qs_raw}

    results: list[dict] = data["results"]
    nopg = [r for r in results if not r.get("used_page_ids")]
    print(f"V10 nopg questions: {len(nopg)} total")

    recovered = 0
    with httpx.Client(timeout=TIMEOUT) as client:
        for r in nopg:
            qid = r["id"]
            q = q_by_id.get(qid, {})
            qtext = q.get("question", "")
            orig_type = r.get("answer_type", q.get("answer_type", "free_text"))

            # Skip genuinely unanswerable (already free_text type with text answers)
            if orig_type == "free_text" and r.get("answer") not in (None, "null"):
                print(f"  {qid[:8]} ({orig_type}): skip — already has text answer")
                continue

            print(f"  {qid[:8]} ({orig_type}): querying with free_text override...")
            try:
                resp = client.post(SERVER_URL, json={"question": qtext, "answer_type": "free_text"})
                resp.raise_for_status()
                raw_ans, telem = parse_sse(resp.text)
                pages = list(telem.get("used_page_ids") or [])
                ttft_ms = float(telem.get("ttft_ms", 0) or 0)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            if not pages:
                print(f"    still 0 pages — skip")
                continue

            extracted = extract_answer(raw_ans or "", orig_type)
            if extracted is None:
                print(f"    got {len(pages)} pages but could not extract {orig_type} answer: {repr(raw_ans or '')[:60]}")
                r["used_page_ids"] = pages
                r["answer"] = None
                r["ttft_ms"] = ttft_ms
                r["recovery_method"] = "free_text_routing_pages_only"
                recovered += 1
                print(f"    → pages recovered ({len(pages)}), answer still null")
                continue

            r["answer"] = extracted
            r["used_page_ids"] = pages
            r["ttft_ms"] = ttft_ms
            r["recovery_method"] = "free_text_routing"
            recovered += 1
            print(f"    → RECOVERED! pg={len(pages)} {orig_type}={repr(extracted)[:40]}")
            print(f"      raw: {repr(raw_ans or '')[:80]}")

            time.sleep(0.2)

    # Update summary stats
    valid = results
    null_c = sum(1 for r in valid if r.get("answer") in (None, "null"))
    nopg_c = sum(1 for r in valid if not r.get("used_page_ids"))
    g = (len(valid) - nopg_c) / len(valid) if valid else 0.0
    ttfts = [float(r["ttft_ms"]) for r in valid if r.get("ttft_ms", 0) > 0]
    ttft_avg = sum(ttfts) / len(ttfts) if ttfts else 0.0

    output = dict(data)
    output["eval_id"] = "tzuf-v10-1-recovery"
    output["null_count"] = null_c
    output["null_pct"] = round(null_c / len(valid) * 100, 2)
    output["no_pages_count"] = nopg_c
    output["no_pages_pct"] = round(nopg_c / len(valid) * 100, 2)
    output["g_proxy"] = round(g, 4)
    output["g_proxy_delta_vs_v9_1"] = round(g - data.get("g_proxy", 0.9822), 4)
    output["ttft_avg_ms"] = round(ttft_avg, 1)
    output["results"] = valid

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\n{'='*60}")
    print(f"V10.1 RECOVERY COMPLETE — {recovered} questions recovered")
    print(f"null={null_c} nopg={nopg_c} G_proxy={g:.4f}")
    print(f"TTFT_avg={ttft_avg:.0f}ms [target: <2000ms for F=1.020]")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
