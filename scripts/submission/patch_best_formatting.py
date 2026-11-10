"""Patch BEST.json formatting issues — double periods, truncation artifacts.

Run: python scripts/patch_best_formatting.py
Output: data/private_submission_BEST_patched.json

Does NOT modify the original BEST.json. Creates a new patched file.
"""

import json
import re
from pathlib import Path

SRC = Path("data/private_submission_BEST.json")
DST = Path("data/private_submission_BEST_patched.json")


def patch_answer(txt: str) -> str:
    """Apply formatting fixes to a single answer string."""
    if not isinstance(txt, str):
        return txt

    # Fix 1: Double periods -> single period (but preserve ellipsis)
    txt = re.sub(r"(?<!\.)\.\.(?!\.)", ".", txt)

    # Fix 2: Truncation artifacts — preposition + period at end of long answer
    if len(txt) > 250:
        match = re.search(r"\b(by|and|or|as|to|of|in|for|with)\.\s*$", txt)
        if match:
            last_semi = txt.rfind(";", 0, match.start())
            last_period = txt.rfind(".", 0, match.start())
            boundary = max(last_semi, last_period)
            if boundary > 100:
                txt = txt[: boundary + 1].rstrip()

    return txt


def main() -> None:
    with open(SRC) as f:
        data = json.load(f)

    patches = 0
    for entry in data["answers"]:
        original = entry.get("answer", "")
        if not isinstance(original, str):
            continue
        fixed = patch_answer(original)
        if fixed != original:
            entry["answer"] = fixed
            patches += 1
            qid = entry["question_id"][:16]
            print(f"  PATCHED {qid}: {len(original)}ch -> {len(fixed)}ch")

    with open(DST, "w") as f:
        json.dump(data, f, indent=None, ensure_ascii=False)

    size_kb = DST.stat().st_size / 1024
    print(f"\n{patches} answers patched -> {DST} ({size_kb:.1f}KB)")


if __name__ == "__main__":
    main()
