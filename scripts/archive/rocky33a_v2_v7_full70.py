#!/usr/bin/env python3
"""Rocky-33a-v2: Full 70-Q non-ft Det eval on v7_1792_enhanced standard profile.

HEAD 7677cf9 (updated from 8e6fc0d). Server: port 8000 (v7_1792_enhanced.env).
NOTE: task server runs on port 8001 — pipeline is on port 8000.
IMPORTANT: cold-restart server at HEAD 7677cf9 before running (NOGA-49a: server may be stale).
Baseline: Det=57/70 (NOGA 49a at b7210e2).
Expected improvements (14 targets): d6eb4a64+30ab0e56+6976d6d2+47cb314a+75bf397c+4ced374a+df0f24b2+61321726+bd8d0bef+bb67fc19+f0329296+d5bc7441+cd0c8f36+b249b41b → Det=67.
orev-46a new: bb67fc19, f0329296, d5bc7441, cd0c8f36. orev-46b new: b249b41b.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / ".sdd" / "golden" / "synthetic-ai-generated" / "golden_labels_v2.json"
OUTPUT = REPO / "data" / "tzuf33a_v2_v7_full70.json"
SERVER_URL = "http://localhost:8000/query"  # pipeline port (task server is 8001)
TIMEOUT = 120.0

TARGETS = [
    "d6eb4a64",  # strict_answerer: EXCEPT clause → No (CONFIRMED)
    "6976d6d2",  # strict_answerer: Art.17(b) without consent → No (1c34862)
    "bd8d0bef",  # context-free: Employment/IP same year → Yes (d4b3f91)
    "47cb314a",  # strict_answerer: Art.34(1) pre-admission liability → No (1c34862)
    "30ab0e56",  # bonus fix (b7210e2, CONFIRMED)
    "75bf397c",  # strict_answerer: Art.10 RPLAW freehold = fee simple → Yes (e1944b7)
    "4ced374a",  # strict_answerer: enactment notice no precise date → No (e1944b7)
    "df0f24b2",  # strict_answerer: ARB 034/2025 main claim not granted → No (e1944b7)
    "61321726",  # strict_answerer: Art.12 RPLAW corporation sole → Registrar (e1944b7)
    "bb67fc19",  # context-free: IP not earlier than Employment → No (8e6fc0d)
    "f0329296",  # context-free: Civil+Commercial Laws No.3 → '3' (8e6fc0d)
    "d5bc7441",  # strict_answerer: Leasing Law 2020 ≠ RPLAW Amendment 2024 → No (8e6fc0d)
    "cd0c8f36",  # strict_answerer: Art.16(1)(c) payroll type → 'gross remuneration' (8e6fc0d)
    "b249b41b",  # strict_answerer: Strata Title 2018 ≠ Financial Collateral 2019 same day → No (7677cf9)
]


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


def normalize_bool(ans: object) -> bool | None:
    if ans is None:
        return None
    s = str(ans).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


def check_det(pred_ans: object, gold_ans: object, q_type: str) -> bool | None:
    if q_type == "free_text":
        return None
    if q_type == "boolean":
        g = normalize_bool(gold_ans)
        p = normalize_bool(pred_ans)
        if g is None and p is None:
            return True
        return g == p
    if q_type in ("number", "date"):
        if gold_ans is None and pred_ans is None:
            return True
        try:
            return abs(float(str(gold_ans)) - float(str(pred_ans))) < 1e-6
        except Exception:
            return str(gold_ans).strip().lower() == str(pred_ans or "").strip().lower()
    if q_type in ("name", "names"):
        if gold_ans is None and pred_ans is None:
            return True
        return str(gold_ans or "").strip().lower() == str(pred_ans or "").strip().lower()
    return str(gold_ans or "").strip().lower() == str(pred_ans or "").strip().lower()


def main() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # Only non-free_text questions
    non_ft = [q for q in golden if q.get("answer_type", "free_text") != "free_text"]
    print(f"Loaded {len(non_ft)} non-free_text questions")

    results = []
    det_by_type: dict[str, list[bool]] = {}
    ttft_all: list[float] = []
    errors = 0
    total = len(non_ft)

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(non_ft, 1):
            qid = q["question_id"]
            q_text = q["question"]
            q_type = q.get("answer_type", "free_text")
            gold_ans = q.get("golden_answer")

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
                print(f"  [{i:3d}/{total}] {qid[:8]} ERROR {elapsed:.1f}s: {e}")
                results.append({"case_id": qid, "error": str(e), "answer_type": q_type})
                continue

            det = check_det(pred_ans, gold_ans, q_type)
            det_by_type.setdefault(q_type, []).append(det if det is not None else False)
            if ttft_ms > 0:
                ttft_all.append(ttft_ms)

            is_target = any(qid.startswith(t) for t in TARGETS)
            status = "✓" if det else "✗"
            flag = " ← TARGET" if is_target else ""
            print(
                f"  [{i:3d}/{total}] {qid[:8]} {q_type:8s} {status} "
                f"pred={str(pred_ans)[:15]!r:17s} gold={str(gold_ans)!r:10s} "
                f"model={model:15s} ttft={ttft_ms:.0f}ms{flag}"
            )

            results.append({
                "case_id": qid,
                "answer_type": q_type,
                "answer": pred_ans,
                "gold_answer": gold_ans,
                "det_correct": det,
                "used_pages": used_pages,
                "ttft_ms": ttft_ms,
                "model": model,
                "elapsed_s": round(elapsed, 3),
            })

    # Summary
    det_correct = sum(1 for r in results if r.get("det_correct") is True)
    det_total = sum(1 for r in results if r.get("det_correct") is not None and "error" not in r)
    ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0

    print(f"\n=== TZUF-33a v2: V7 STANDARD FULL 70-Q DET EVAL ===")
    print(f"HEAD: 7677cf9 | Profile: private_v7_1792_enhanced.env")
    print(f"Questions: {total} | Errors: {errors}")
    print(f"Det: {det_correct}/{det_total} (BL=57/70=81.4%, NOGA 49a at b7210e2, expected 67/70 at 7677cf9)")
    print(f"TTFT avg: {ttft_avg:.0f}ms")
    print(f"\nBy type:")
    for t in ["boolean", "name", "names", "number", "date"]:
        items = det_by_type.get(t, [])
        if items:
            correct = sum(items)
            print(f"  {t}: {correct}/{len(items)} ({correct/len(items)*100:.1f}%)")

    print(f"\nTarget questions:")
    for r in results:
        for tgt in TARGETS:
            if r.get("case_id", "").startswith(tgt):
                status = "✓" if r.get("det_correct") else "✗"
                print(f"  {tgt}: {status} answer={r.get('answer')!r} gold={r.get('gold_answer')!r} model={r.get('model','?')}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "eval_id": "tzuf-33a-v2",
        "head": "7677cf9",
        "profile": "private_v7_1792_enhanced.env",
        "det_correct": det_correct,
        "det_total": det_total,
        "ttft_avg_ms": ttft_avg,
        "det_by_type": {t: {"correct": sum(v), "total": len(v)} for t, v in det_by_type.items()},
        "results": results,
    }, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
