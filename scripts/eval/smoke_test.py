#!/usr/bin/env python3
"""2-minute smoke test — send 5 representative questions, verify basic pipeline health.

Catches broken server, missing telemetry, citation failures, and timeout regressions
before committing 1h to a full eval. Run after any server restart.

Usage:
    uv run python scripts/smoke_test.py [--server http://localhost:8002] [--questions GOLDEN_JSON]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = REPO / ".sdd" / "golden" / "synthetic-ai-generated" / "golden_labels_v2.json"
DEFAULT_SERVER = "http://localhost:8002"
TTFT_WARN_S = 8.0

# One question per answer type for broad coverage
SMOKE_TYPES = ["boolean", "name", "number", "free_text", "date"]


def parse_sse(text: str) -> tuple[str | None, dict]:
    """Extract final answer text and telemetry from SSE stream."""
    answer_text = None
    telemetry: dict = {}
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        t = payload.get("type")
        if t == "answer_final":
            answer_text = payload.get("text")
        elif t == "telemetry":
            telemetry = payload.get("payload", {})
    return answer_text, telemetry


def pick_smoke_questions(golden_path: Path) -> list[dict]:
    """Pick one question per type from golden labels, ordered by SMOKE_TYPES."""
    golden = json.loads(golden_path.read_text())
    picked: dict[str, dict] = {}
    for q in golden:
        at = q["answer_type"]
        if at in SMOKE_TYPES and at not in picked:
            picked[at] = q
        if len(picked) == len(SMOKE_TYPES):
            break
    return [picked[t] for t in SMOKE_TYPES if t in picked]


def run_smoke(server_url: str, golden_path: Path) -> int:
    """Run smoke test. Returns 0 on pass, 1 on any failure."""
    questions = pick_smoke_questions(golden_path)
    print(f"=== SMOKE TEST — {len(questions)} questions against {server_url} ===\n")

    failures: list[str] = []
    total_ttft: list[float] = []

    with httpx.Client(timeout=30.0) as client:
        for q in questions:
            qid = q["question_id"][:8]
            at = q["answer_type"]
            t0 = time.monotonic()
            try:
                resp = client.post(
                    f"{server_url}/query",
                    json={"question": q["question"], "answer_type": at},
                )
                elapsed = time.monotonic() - t0
                resp.raise_for_status()
                answer_text, telemetry = parse_sse(resp.text)

                checks: list[str] = []
                # 1. Got an answer
                if answer_text is None:
                    checks.append("NO_ANSWER")
                # 2. Telemetry present
                if not telemetry:
                    checks.append("NO_TELEMETRY")
                # 3. Citations present (for non-null answers)
                if answer_text is not None and not telemetry.get("used_page_ids"):
                    checks.append("NO_CITATIONS")
                # 4. TTFT reasonable
                ttft = telemetry.get("ttft_ms", 0)
                if isinstance(ttft, (int, float)) and ttft > 0:
                    total_ttft.append(ttft / 1000)
                if elapsed > TTFT_WARN_S:
                    checks.append(f"SLOW({elapsed:.1f}s)")

                status = "PASS" if not checks else "FAIL:" + ",".join(checks)
                print(
                    f"  {qid} {at:10s}  elapsed={elapsed:.2f}s  ttft={ttft}ms  "
                    f"citations={len(telemetry.get('used_page_ids', []))}  [{status}]"
                )
                if checks:
                    failures.append(f"{qid} {at}: {', '.join(checks)}")
            except Exception as exc:
                elapsed = time.monotonic() - t0
                print(f"  {qid} {at:10s}  elapsed={elapsed:.2f}s  [FAIL: {exc}]")
                failures.append(f"{qid} {at}: exception {exc}")

    print()
    if total_ttft:
        avg_ttft = sum(total_ttft) / len(total_ttft)
        print(f"  Avg TTFT: {avg_ttft:.2f}s (target <3.0s for F≈1.04)")
    if failures:
        print(f"\n  SMOKE FAILED — {len(failures)} issue(s):")
        for f in failures:
            print(f"    - {f}")
        print("\n  Do NOT run full eval until smoke passes.")
        return 1
    else:
        print(f"  SMOKE PASSED — all {len(questions)} checks OK")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--questions", type=Path, default=DEFAULT_GOLDEN)
    args = parser.parse_args()
    return run_smoke(args.server, args.questions)


if __name__ == "__main__":
    raise SystemExit(main())
