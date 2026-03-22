#!/usr/bin/env python3
"""Run all warmup questions through the pipeline and score against golden labels.

Sends each question to the running server via SSE, collects answers + telemetry,
then scores using the golden_labels_v2 format.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
WARMUP_PATH = REPO / ".sdd" / "golden" / "synthetic-ai-generated" / "golden_labels_v2.json"
import os as _os
SERVER_URL = _os.environ.get("EVAL_SERVER_URL", "http://localhost:8002/query")
TIMEOUT = 120.0


def parse_sse_response(text: str) -> dict[str, object]:
    """Parse SSE stream into answer_text and telemetry."""
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


def run_eval(server_url: str = SERVER_URL) -> dict[str, object]:
    """Run all warmup questions and return raw results."""
    golden = json.loads(WARMUP_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(golden)} golden questions from {WARMUP_PATH.name}")

    results: list[dict[str, object]] = []
    errors: list[str] = []
    total = len(golden)

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(golden, 1):
            qid = q["question_id"]
            question = q["question"]
            answer_type = q["answer_type"]
            qid_short = qid[:8]

            t0 = time.monotonic()
            try:
                resp = client.post(
                    server_url,
                    json={"question": question, "answer_type": answer_type},
                )
                resp.raise_for_status()
                parsed = parse_sse_response(resp.text)
                elapsed = time.monotonic() - t0

                result = {
                    "case": {"case_id": qid},
                    "answer_text": parsed["answer_text"],
                    "telemetry": parsed["telemetry"],
                }
                results.append(result)

                used_pages = (parsed["telemetry"] or {}).get("used_page_ids", []) if isinstance(parsed["telemetry"], dict) else []
                gold_pages = q.get("golden_page_ids", [])
                overlap = len(set(used_pages or []) & set(gold_pages))

                status = "OK" if overlap > 0 else "MISS"
                print(
                    f"  [{i:3d}/{total}] {qid_short} {answer_type:10s} "
                    f"{elapsed:5.1f}s  pages={len(used_pages or []):2d}/{len(gold_pages):2d} "
                    f"overlap={overlap} {status}"
                )

            except Exception as e:
                elapsed = time.monotonic() - t0
                errors.append(f"{qid_short}: {e}")
                print(f"  [{i:3d}/{total}] {qid_short} ERROR {elapsed:.1f}s: {e}")
                results.append({
                    "case": {"case_id": qid},
                    "answer_text": None,
                    "telemetry": {},
                })

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")

    return {"results": results, "question_count": total, "error_count": len(errors)}


def main() -> int:
    """Run eval and save results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"warmup_raw_{timestamp}.json"
    score_path = out_dir / f"warmup_score_{timestamp}.json"
    score_md_path = out_dir / f"warmup_score_{timestamp}.md"

    print(f"=== Warmup Eval {timestamp} ===\n")
    eval_result = run_eval()
    results = eval_result["results"]

    raw_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nRaw results: {raw_path}")

    # Score
    from scripts.score_against_golden import score, _render_markdown

    score_result = score(raw_path, WARMUP_PATH)
    score_path.write_text(
        json.dumps(score_result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    md = _render_markdown(score_result)
    score_md_path.write_text(md, encoding="utf-8")

    print(f"\n{'='*60}")
    print(md)
    print(f"\nScore JSON: {score_path}")
    print(f"Score MD:   {score_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
