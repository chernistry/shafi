#!/usr/bin/env python3
"""Rocky-61b: Warmup eval runner — regression check for current HEAD.

Runs all 100 golden_labels_v2 questions through the pipeline and scores
deterministic accuracy + grounding F-beta. No platform submission.

Usage:
    .venv/bin/python scripts/tzuf_61b_warmup_eval.py
    EVAL_SERVER_URL=http://localhost:8000/query .venv/bin/python scripts/tzuf_61b_warmup_eval.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
WARMUP_PATH = REPO / ".sdd" / "golden" / "synthetic-ai-generated" / "golden_labels_v2.json"
SERVER_URL = os.environ.get("EVAL_SERVER_URL", "http://localhost:8000/query")
TIMEOUT = 120.0
RETRY_WAIT = 5.0   # seconds to wait before retry after server error
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------

def parse_sse_response(text: str) -> dict[str, object]:
    """Parse SSE stream text into answer_text + telemetry dict."""
    answer_text: str | None = None
    telemetry: dict[str, object] = {}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        msg_type = payload.get("type")
        if msg_type == "answer_final":
            answer_text = payload.get("text", "")
        elif msg_type == "telemetry":
            telemetry = payload.get("payload", {})
    return {"answer_text": answer_text, "telemetry": telemetry}


# ---------------------------------------------------------------------------
# Scoring helpers (inlined from score_against_golden.py)
# ---------------------------------------------------------------------------

BETA = 2.5


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _extract_used_page_ids(telemetry: dict[str, Any]) -> list[str]:
    pages = _coerce_str_list(telemetry.get("used_page_ids"))
    if pages:
        return pages
    pages = _coerce_str_list(telemetry.get("cited_page_ids"))
    if pages:
        return pages
    chunk_ids = _coerce_str_list(telemetry.get("retrieved_chunk_ids"))
    seen: set[str] = set()
    out: list[str] = []
    for cid in chunk_ids:
        parts = cid.split(":")
        if len(parts) < 2:
            continue
        doc_id = parts[0].strip()
        page_raw = parts[1].strip()
        if not doc_id or not page_raw.isdigit():
            continue
        pid = f"{doc_id}_{int(page_raw) + 1}"
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _fbeta(predicted: set[str], gold: set[str], beta: float = BETA) -> tuple[float, float, float]:
    if not gold:
        return (1.0, 1.0, 1.0) if not predicted else (0.0, 1.0, 0.0)
    if not predicted:
        return (0.0, 0.0, 0.0)
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return (0.0, precision, recall)
    b2 = beta * beta
    sc = (1 + b2) * precision * recall / (b2 * precision + recall)
    return (sc, precision, recall)


def _normalize_answer(answer: Any, answer_type: str) -> Any:
    if answer is None or (isinstance(answer, str) and answer.strip().lower() in ("null", "none", "")):
        return None
    if answer_type == "boolean":
        if isinstance(answer, bool):
            return answer
        s = str(answer).strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        return None
    if answer_type == "number":
        if isinstance(answer, (int, float)) and not isinstance(answer, bool):
            return float(answer)
        s = str(answer).strip().replace(",", "").replace(" ", "")
        s = re.sub(r"[^\d.\-]", "", s)
        try:
            return float(s)
        except (ValueError, TypeError):
            return None
    if answer_type == "date":
        s = str(answer).strip()
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        return s.lower()
    if answer_type in ("name", "names"):
        s = str(answer).strip()
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s
    return str(answer).strip()


def _answers_match(our_answer: Any, golden_answer: Any, answer_type: str) -> bool | None:
    """True/False for deterministic types, None for free_text."""
    if answer_type == "free_text":
        return None
    norm_ours = _normalize_answer(our_answer, answer_type)
    norm_gold = _normalize_answer(golden_answer, answer_type)
    if norm_ours is None and norm_gold is None:
        return True
    if norm_ours is None or norm_gold is None:
        return False
    if answer_type == "number":
        if norm_ours == 0 and norm_gold == 0:
            return True
        return math.isclose(norm_ours, norm_gold, rel_tol=1e-3)
    return norm_ours == norm_gold


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval() -> tuple[list[dict[str, Any]], list[str], list[float]]:
    """Run all questions and return (results, errors, ttft_list)."""
    golden: list[dict[str, Any]] = json.loads(WARMUP_PATH.read_text(encoding="utf-8"))
    total = len(golden)
    print(f"Loaded {total} golden questions from {WARMUP_PATH.name}")
    print(f"Server: {SERVER_URL}\n")

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    ttfts: list[float] = []

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(golden, 1):
            qid: str = q["question_id"]
            question: str = q["question"]
            answer_type: str = q["answer_type"]
            qid_short = qid[:8]

            parsed: dict[str, Any] | None = None
            elapsed = 0.0
            for attempt in range(MAX_RETRIES + 1):
                if attempt > 0:
                    print(f"    Retry {attempt} after {RETRY_WAIT}s ...")
                    time.sleep(RETRY_WAIT)
                t0 = time.monotonic()
                try:
                    resp = client.post(
                        SERVER_URL,
                        json={"question": question, "answer_type": answer_type},
                    )
                    resp.raise_for_status()
                    elapsed = time.monotonic() - t0
                    parsed = parse_sse_response(resp.text)
                    break
                except Exception as e:
                    elapsed = time.monotonic() - t0
                    if attempt == MAX_RETRIES:
                        errors.append(f"{qid_short}: {e}")
                        print(f"  [{i:3d}/{total}] {qid_short} ERROR after {attempt+1} attempts {elapsed:.1f}s: {e}")

            if parsed is not None:
                tel = parsed["telemetry"]
                used_pages = _extract_used_page_ids(tel if isinstance(tel, dict) else {})
                gold_pages = q.get("golden_page_ids", [])
                overlap = len(set(used_pages) & set(gold_pages))
                ttft = tel.get("ttft_ms") if isinstance(tel, dict) else None
                if isinstance(ttft, (int, float)):
                    ttfts.append(float(ttft))

                status = "OK" if overlap > 0 else "MISS"
                null_marker = " NULL" if parsed["answer_text"] is None else ""
                print(
                    f"  [{i:3d}/{total}] {qid_short} {answer_type:10s} "
                    f"{elapsed:5.1f}s  pages={len(used_pages):2d}/{len(gold_pages):2d} "
                    f"overlap={overlap} {status}{null_marker}"
                )

                results.append({
                    "case": {"case_id": qid},
                    "answer_text": parsed["answer_text"],
                    "telemetry": tel,
                    "answer_type": answer_type,
                    "golden_answer": q.get("golden_answer"),
                    "golden_page_ids": gold_pages,
                })
            else:
                results.append({
                    "case": {"case_id": qid},
                    "answer_text": None,
                    "telemetry": {},
                    "answer_type": answer_type,
                    "golden_answer": q.get("golden_answer"),
                    "golden_page_ids": q.get("golden_page_ids", []),
                })

    return results, errors, ttfts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic accuracy and grounding F-beta per type."""
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_case = []

    null_count = sum(1 for r in results if r.get("answer_text") is None)
    no_pages_count = 0

    for r in results:
        atype = r.get("answer_type", "free_text")
        our_answer = r.get("answer_text")
        golden_answer = r.get("golden_answer")
        tel = r.get("telemetry") or {}
        our_pages = set(_extract_used_page_ids(tel if isinstance(tel, dict) else {}))
        gold_pages = set(r.get("golden_page_ids") or [])

        if not our_pages:
            no_pages_count += 1

        f_sc, prec, rec = _fbeta(our_pages, gold_pages)
        match = _answers_match(our_answer, golden_answer, atype)

        by_type[atype]["f_beta"].append(f_sc)
        if match is not None:
            by_type[atype]["exact_match"].append(1.0 if match else 0.0)

        per_case.append({
            "qid": r["case"]["case_id"][:8],
            "atype": atype,
            "match": match,
            "f_beta": round(f_sc, 4),
            "recall": round(rec, 4),
        })

    def _agg(vals: list[float]) -> dict[str, Any]:
        if not vals:
            return {"mean": 0.0, "count": 0}
        return {"mean": round(sum(vals) / len(vals), 4), "count": len(vals)}

    type_summary: dict[str, Any] = {}
    all_fbetas: list[float] = []
    all_em: list[float] = []
    for t, metrics in sorted(by_type.items()):
        type_summary[t] = {k: _agg(v) for k, v in metrics.items()}
        all_fbetas.extend(metrics["f_beta"])
        if "exact_match" in metrics:
            all_em.extend(metrics["exact_match"])

    return {
        "total": len(results),
        "null_count": null_count,
        "null_pct": round(100.0 * null_count / max(len(results), 1), 1),
        "no_pages_count": no_pages_count,
        "no_pages_pct": round(100.0 * no_pages_count / max(len(results), 1), 1),
        "overall_fbeta": round(sum(all_fbetas) / max(len(all_fbetas), 1), 4),
        "det_exact_match": round(sum(all_em) / max(len(all_em), 1), 4) if all_em else None,
        "det_count": len(all_em),
        "by_type": type_summary,
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"tzuf61b_warmup_raw_{timestamp}.json"

    print(f"=== Rocky-61b Warmup Regression Check {timestamp} ===\n")

    results, errors, ttfts = run_eval()

    raw_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    sc = score_results(results)

    ttft_avg = round(sum(ttfts) / len(ttfts)) if ttfts else None
    ttft_p95 = None
    if ttfts:
        ttfts_sorted = sorted(ttfts)
        idx = min(int(0.95 * len(ttfts_sorted)), len(ttfts_sorted) - 1)
        ttft_p95 = round(ttfts_sorted[idx])

    print(f"\n{'='*60}")
    print(f"RESULTS — Rocky-61b Warmup Regression Check {timestamp}")
    print(f"{'='*60}")
    print(f"  Total questions : {sc['total']}")
    print(f"  Errors          : {len(errors)}")
    print(f"  Null answers    : {sc['null_count']} ({sc['null_pct']}%)")
    print(f"  No-pages        : {sc['no_pages_count']} ({sc['no_pages_pct']}%)")
    print(f"  Det exact match : {sc['det_exact_match']} over {sc['det_count']} det questions")
    print(f"  Overall F-beta  : {sc['overall_fbeta']}")
    print(f"  TTFT avg (ms)   : {ttft_avg}")
    print(f"  TTFT p95 (ms)   : {ttft_p95}")
    print()
    print("  By type:")
    for t, m in sc["by_type"].items():
        em = m.get("exact_match", {})
        fb = m.get("f_beta", {})
        em_str = f"det={em['mean']:.3f}({em['count']})" if em.get("count") else "det=N/A"
        print(f"    {t:12s}  {em_str}  f_beta={fb['mean']:.3f}({fb['count']})")
    print()

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... +{len(errors)-10} more")
    print()

    # Regression check vs baseline
    BASELINE_DET = 1.0   # 70/70 perfect
    BASELINE_FBETA = None  # unknown exact
    det = sc["det_exact_match"]
    if det is not None and det < BASELINE_DET - 0.05:
        print(f"  *** REGRESSION WARNING: det={det:.3f} vs baseline={BASELINE_DET:.3f} ***")
    else:
        print(f"  PASS: det={det} (baseline ~1.0)")

    print(f"\n  Raw results saved: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
