#!/usr/bin/env python3
"""Prompt quality smoke test — 18 questions (3 per answer type).

Checks format compliance, not answer correctness. Run against live server:
    uv run python scripts/prompt_quality_smoke.py

Requires server at localhost:8000.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

SERVER = "http://localhost:8000/query"

QUESTIONS = [
    # boolean (3)
    {"id": "8f95ae2a", "question": "Is there any main party that appeared in both cases CA 006/2024 and CFI 084/2024?", "answer_type": "boolean"},
    {"id": "af7d4a34", "question": "Is there evidence that a final trial took place in case CFI 041/2021?", "answer_type": "boolean"},
    {"id": "102e2fcc", "question": "Were there any changes to the judges presiding over case ENF 269/2023?", "answer_type": "boolean"},
    # free_text (3)
    {"id": "1045730241a7", "question": "What is the deadline for filing an acknowledgment of service when a claim form is served out of the DIFC or Dubai?", "answer_type": "free_text"},
    {"id": "f95e3f78", "question": "In Olive v Onyx [2025] DIFC SCT 042, what did the court hold about Article 28 of the DIFC Employment Law?", "answer_type": "free_text"},
    {"id": "a58dcc68ft", "question": "What penalty applies under the Intellectual Property Law for disseminating content through computer networks?", "answer_type": "free_text"},
    # number (3)
    {"id": "a58dcc68", "question": "What is the maximum fine under the Intellectual Property Law DIFC Law No. 4 of 2019?", "answer_type": "number"},
    {"id": "1cbb2573", "question": "What is the total number of distinct claimants in case CFI 057/2025?", "answer_type": "number"},
    {"id": "c7281d77", "question": "What is the maximum fine under the Non Profit Incorporated Organisations Law?", "answer_type": "number"},
    # name (3)
    {"id": "04a51fbc", "question": "Which law has a later enactment date: Regarding The Financial Free Zones or LEASING REGULATIONS?", "answer_type": "name"},
    {"id": "4e30af68", "question": "Between TCD 001/2023 and TCD 002/2024, which case involved the larger sum claimed?", "answer_type": "name"},
    {"id": "7fbce555", "question": "Which case was issued earlier: CFI 067/2025 or CFI 069/2024?", "answer_type": "name"},
    # names (3)
    {"id": "c4c53051", "question": "Against which party was enforcement sought in case ENF 084/2023?", "answer_type": "names"},
    {"id": "25d46638", "question": "Identify the claimant in case ARB 034/2025.", "answer_type": "names"},
    {"id": "8ec07e12", "question": "List all defendants in case CFI 070/2018.", "answer_type": "names"},
    # date (3)
    {"id": "9551d920", "question": "When was the Court of Appeal document in case CA 016/2024 issued?", "answer_type": "date"},
    {"id": "fb48f644", "question": "What is the Date of Issue in SCT 365/2024?", "answer_type": "date"},
    {"id": "8f918b99", "question": "By when should public responses be submitted for Consultation Paper No. 3?", "answer_type": "date"},
]


def query_server(q: dict) -> dict:
    body = json.dumps({"question": q["question"], "question_id": q["id"], "answer_type": q["answer_type"]})
    cmd = f'''curl -s -N -X POST {SERVER} -H "Content-Type: application/json" -d '{body}' 2>/dev/null'''
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    answer = ""
    ttft = 0
    for line in out.split("\n"):
        if '"answer_final"' in line:
            try:
                answer = json.loads(line.split("data: ")[1]).get("text", "")
            except Exception:
                pass
        if '"telemetry"' in line:
            try:
                ttft = json.loads(line.split("data: ")[1])["payload"].get("ttft_ms", 0)
            except Exception:
                pass
    return {"answer": answer, "ttft": ttft}


def strip_cites(text: str) -> str:
    return re.sub(r"\s*\(cite:[^)]+\)", "", text).strip()


def check_format(q: dict, result: dict) -> tuple[bool, str]:
    """Check format compliance. Returns (passed, reason)."""
    ans = result["answer"]
    stripped = strip_cites(ans)
    atype = q["answer_type"]

    if not ans and result["ttft"] == 0:
        return False, "pipeline failure (no answer)"

    if atype == "boolean":
        if stripped not in ("Yes", "No") and stripped.lower() not in ("yes", "no", "null"):
            return False, f"bad boolean format: '{stripped[:30]}'"
        return True, "ok"

    if atype == "number":
        if stripped and not re.match(r"^-?[\d,.]+$", stripped) and stripped.lower() != "null":
            return False, f"non-numeric: '{stripped[:30]}'"
        return True, "ok"

    if atype == "date":
        if stripped and not re.match(r"^\d{4}-\d{2}-\d{2}$", stripped) and stripped.lower() != "null":
            return False, f"bad date format: '{stripped[:30]}'"
        return True, "ok"

    if atype == "free_text":
        if len(stripped) > 280:
            return False, f"over 280 chars: {len(stripped)}"
        if stripped and not stripped.endswith((".", "!", "?", '"', ")", "]")):
            return False, f"no terminal punctuation: '{stripped[-20:]}'"
        return True, "ok"

    if atype in ("name", "names"):
        if not stripped:
            return False, "empty answer"
        return True, "ok"

    return True, "ok"


def main() -> None:
    passed = 0
    failed = 0
    errors: list[str] = []

    for q in QUESTIONS:
        result = query_server(q)
        ok, reason = check_format(q, result)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append(f"{q['id']} ({q['answer_type']}): {reason}")
        stripped = strip_cites(result["answer"])
        print(f"  [{status}] {q['answer_type']:10s} {result['ttft']:5d}ms | {stripped[:60]}")

    print(f"\n=== {passed}/{passed + failed} passed ===")
    if errors:
        print("Failures:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
