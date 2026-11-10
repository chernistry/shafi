#!/usr/bin/env python3
"""Fast 5-question smoke test — sanity check before full eval or after private data arrival.

Sends one question of each key type (boolean, name, number, free_text, unanswerable)
to the server and verifies: non-null answers, valid format, citations present, TTFT<10s.

Runtime: ~30 seconds. Use before committing to a 30-60 min full eval.

Usage:
    python scripts/smoke_test_server.py [--server-url http://localhost:8000/query]
    python scripts/smoke_test_server.py --questions .sdd/golden/reviewed/reviewed_all_100.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = REPO / ".sdd/golden/reviewed/reviewed_all_100.json"
DEFAULT_URL = "http://localhost:8000/query"
TIMEOUT = 30.0

# Specific QIDs to use as smoke test (one per type, known stable questions)
# Falls back to first N questions of each type if not found.
SMOKE_QIDS: dict[str, str | None] = {
    "boolean": None,   # Auto-pick first boolean
    "name": None,      # Auto-pick first name
    "number": None,    # Auto-pick first number
    "free_text": None, # Auto-pick first free_text
    "unanswerable": None,  # Pick one with empty golden_page_ids
}


def _pick_smoke_questions(golden: list[dict]) -> list[dict]:
    """Pick one question per type for smoke test."""
    picked: dict[str, dict | None] = {t: None for t in SMOKE_QIDS}
    for q in golden:
        atype = q.get("answer_type", "")
        if atype in picked and picked[atype] is None:
            picked[atype] = q
        # Detect unanswerable: empty golden_page_ids or golden_answer is null
        if picked.get("unanswerable") is None:
            gold_pages = q.get("golden_page_ids") or []
            gold_ans = q.get("golden_answer")
            if not gold_pages and gold_ans in (None, "", "null", "N/A"):
                picked["unanswerable"] = q
        if all(v is not None for v in picked.values()):
            break
    return [q for q in picked.values() if q is not None]


def _parse_sse(text: str) -> tuple[str | None, dict]:
    answer_text = None
    telemetry: dict = {}
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
    return answer_text, telemetry


def run_smoke_test(
    server_url: str = DEFAULT_URL,
    questions_path: Path = DEFAULT_QUESTIONS,
) -> bool:
    """Run smoke test. Returns True if all checks pass."""
    golden = json.loads(questions_path.read_text())
    questions = _pick_smoke_questions(golden)

    if not questions:
        print("ERROR: Could not pick questions from golden labels")
        return False

    print(f"Smoke test: {len(questions)} questions → {server_url}")
    print(f"Types: {[q.get('answer_type', '?') for q in questions]}")
    print("-" * 50)

    all_pass = True
    with httpx.Client(timeout=TIMEOUT) as client:
        for q in questions:
            qid = (q.get("question_id") or "")[:12]
            atype = q.get("answer_type", "?")
            question = q.get("question", "")
            t0 = time.monotonic()

            try:
                resp = client.post(
                    server_url,
                    json={"question": question, "answer_type": atype},
                )
                resp.raise_for_status()
                elapsed = time.monotonic() - t0
                answer, tel = _parse_sse(resp.text)
                ttft = tel.get("ttft_ms", 0)

                # Checks
                checks: list[tuple[bool, str]] = [
                    (answer is not None, "answer not null"),
                    (elapsed < 30, f"TTFT < 30s (actual {elapsed:.1f}s)"),
                    (ttft < 10000 or ttft == 0, f"TTFT < 10000ms (actual {ttft}ms)"),
                ]

                # Format check — boolean must be Yes/No
                if atype == "boolean" and answer:
                    checks.append(
                        (answer.strip().lower() in ("yes", "no"),
                         f"boolean is Yes/No (actual: {answer!r})")
                    )

                # Citations check for non-unanswerable
                gold_pages = q.get("golden_page_ids") or []
                if gold_pages and answer:
                    cited = tel.get("cited_page_ids") or tel.get("used_page_ids") or []
                    checks.append((bool(cited), "citations present"))

                passed = all(ok for ok, _ in checks)
                status = "✓ PASS" if passed else "✗ FAIL"
                if not passed:
                    all_pass = False
                    failed = [msg for ok, msg in checks if not ok]
                    print(f"{status} [{atype:10}] {qid} TTFT={ttft}ms | FAILED: {', '.join(failed)}")
                    print(f"         answer: {(answer or '')[:80]!r}")
                else:
                    print(f"{status} [{atype:10}] {qid} TTFT={ttft}ms ans={str(answer)[:50]!r}")

            except httpx.ConnectError:
                print(f"✗ FAIL [{atype:10}] {qid} — CONNECTION REFUSED ({server_url})")
                all_pass = False
            except Exception as e:
                print(f"✗ FAIL [{atype:10}] {qid} — ERROR: {e}")
                all_pass = False

    print("-" * 50)
    print(f"SMOKE TEST: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    return all_pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5-question smoke test for the pipeline server")
    parser.add_argument("--server-url", default=DEFAULT_URL, help="Server URL")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Golden labels JSON")
    args = parser.parse_args(argv)
    passed = run_smoke_test(server_url=args.server_url, questions_path=args.questions)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
