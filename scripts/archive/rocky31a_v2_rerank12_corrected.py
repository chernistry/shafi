#!/usr/bin/env python3
"""Rocky-31a v2: RERANK_TOP_N=12 full 100Q eval WITH correct answer_type routing.

Baseline (tzuf-33a full70 at HEAD b7210e2 + tzuf-29a ft30):
  70Q non-ft G=0.2623, Det=57/70
  30Q ft G=0.1660
  100Q combined G=0.2334
Profile: private_v9_rerank12.env (RERANK_TOP_N=12, RERANK_CANDIDATES=160)
Server: port 8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx


REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / ".sdd" / "golden" / "synthetic-ai-generated" / "golden_labels_v2.json"
OUTPUT = REPO / "data" / "tzuf31a_v2_rerank12_corrected.json"
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


def compute_g(used_pages: list, golden_pages: list) -> float:
    if not golden_pages:
        return 1.0 if not used_pages else 0.0
    used_set = set(str(p) for p in (used_pages or []))
    gold_set = set(str(p) for p in golden_pages)
    if not used_set and not gold_set:
        return 1.0
    if not used_set:
        return 0.0
    tp = len(used_set & gold_set)
    precision = tp / len(used_set)
    recall = tp / len(gold_set)
    beta = 2.5
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


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
    print(f"Loaded {len(golden)} golden questions")

    results = []
    g_scores: list[float] = []
    g_by_type: dict[str, list[float]] = {}
    ttft_all: list[float] = []
    ttft_by_type: dict[str, list[float]] = {}
    det_correct = 0
    det_total = 0
    errors = 0
    total = len(golden)

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(golden, 1):
            qid = q["question_id"]
            q_text = q["question"]
            q_type = q.get("answer_type", "free_text")
            gold_ans = q.get("golden_answer")
            golden_pages = q.get("golden_page_ids", [])

            t0 = time.monotonic()
            try:
                resp = client.post(
                    SERVER_URL,
                    json={"question": q_text, "answer_type": q_type},
                )
                resp.raise_for_status()
                parsed = parse_sse_response(resp.text)
                elapsed = time.monotonic() - t0

                telem = parsed["telemetry"] or {}
                pred_ans = parsed["answer_text"]
                used_pages = telem.get("used_page_ids", [])
                ttft_ms = float(telem.get("ttft_ms", 0) or 0)

            except Exception as e:
                elapsed = time.monotonic() - t0
                errors += 1
                print(f"  [{i:3d}/{total}] ERROR {elapsed:.1f}s: {e}")
                results.append({"case_id": qid, "error": str(e), "answer_type": q_type})
                continue

            g = compute_g(used_pages, golden_pages)
            g_scores.append(g)
            g_by_type.setdefault(q_type, []).append(g)

            if q_type != "free_text":
                det = check_det(pred_ans, gold_ans, q_type)
                if det is not None:
                    det_total += 1
                    if det:
                        det_correct += 1

            if ttft_ms > 0:
                ttft_all.append(ttft_ms)
                ttft_by_type.setdefault(q_type, []).append(ttft_ms)

            overlap = len(set(str(p) for p in (used_pages or [])) & set(str(p) for p in golden_pages))
            print(
                f"  [{i:3d}/{total}] {qid[:8]} {q_type:10s} "
                f"G={g:.3f} ttft={ttft_ms:.0f}ms "
                f"pages={len(used_pages or []):2d}/{len(golden_pages):2d} overlap={overlap}"
            )

            results.append({
                "case_id": qid,
                "answer_type": q_type,
                "answer": pred_ans,
                "gold_answer": gold_ans,
                "used_pages": used_pages,
                "golden_pages": golden_pages,
                "g": g,
                "ttft_ms": ttft_ms,
                "elapsed_s": round(elapsed, 3),
            })

    # Summary
    overall_g = sum(g_scores) / len(g_scores) if g_scores else 0.0
    det_pct = det_correct / det_total * 100 if det_total else 0.0
    ttft_avg = sum(ttft_all) / len(ttft_all) if ttft_all else 0.0

    print(f"\n=== TZUF-31a v2: RERANK12 CORRECTED EVAL ===")
    print(f"Questions: {total} | Errors: {errors}")
    print(f"Overall G: {overall_g:.4f} (BL=0.2334, Δ={overall_g - 0.2334:+.4f})")
    print(f"Det: {det_correct}/{det_total} = {det_pct:.1f}% (BL=57/70=81.4%)")
    print(f"TTFT avg: {ttft_avg:.0f}ms (BL=2150ms)")
    print(f"\nBy type:")
    for t in ["boolean", "name", "names", "number", "date", "free_text"]:
        gs = g_by_type.get(t, [])
        tts = ttft_by_type.get(t, [])
        if gs:
            g_avg = sum(gs) / len(gs)
            tt_avg = sum(tts) / len(tts) if tts else 0.0
            print(f"  {t}: G={g_avg:.4f} n={len(gs)} ttft={tt_avg:.0f}ms")

    non_ft_types = ["boolean", "name", "names", "number", "date"]
    non_ft_gs: list[float] = []
    for t in non_ft_types:
        non_ft_gs.extend(g_by_type.get(t, []))
    ft_gs = g_by_type.get("free_text", [])
    non_ft_g = sum(non_ft_gs) / len(non_ft_gs) if non_ft_gs else 0.0
    ft_g = sum(ft_gs) / len(ft_gs) if ft_gs else 0.0
    print(f"\n  non-ft G: {non_ft_g:.4f} (BL=0.2623, Δ={non_ft_g - 0.2623:+.4f})")
    print(f"  ft G:     {ft_g:.4f} (BL=0.1660, Δ={ft_g - 0.1660:+.4f})")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "eval_id": "tzuf-31a-v2",
        "config": "private_v9_rerank12.env (RERANK_TOP_N=12, RERANK_CANDIDATES=160)",
        "overall_g": overall_g,
        "baseline_g": 0.2334,
        "delta_g": overall_g - 0.2334,
        "det_correct": det_correct,
        "det_total": det_total,
        "ttft_avg_ms": ttft_avg,
        "g_by_type": {t: sum(v) / len(v) for t, v in g_by_type.items() if v},
        "ttft_by_type": {t: sum(v) / len(v) for t, v in ttft_by_type.items() if v},
        "non_ft_g": non_ft_g,
        "ft_g": ft_g,
        "results": results,
    }, indent=2))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
