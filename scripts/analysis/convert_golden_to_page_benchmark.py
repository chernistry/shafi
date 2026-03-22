#!/usr/bin/env python3
"""Convert golden_labels_v2.json to the page-benchmark format
expected by evaluate_candidate_debug_signal.py --page-benchmark.

Usage:
    python scripts/convert_golden_to_page_benchmark.py \
        --golden .sdd/golden/synthetic-ai-generated/golden_labels_v2.json \
        --out .sdd/golden/page_benchmark_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_TRUSTED_CONFIDENCE = {"high", "medium"}


def convert(golden_path: Path, out_path: Path) -> None:
    labels = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = []
    for entry in labels:
        qid = str(entry.get("question_id") or "").strip()
        pages = entry.get("golden_page_ids") or []
        confidence = str(entry.get("confidence") or "").strip().lower()
        if not qid or not pages:
            continue
        cases.append(
            {
                "question_id": qid,
                "gold_page_ids": pages,
                "trust_tier": "trusted" if confidence in _TRUSTED_CONFIDENCE else "untrusted",
            }
        )
    payload = {"cases": cases}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trusted = sum(1 for c in cases if c["trust_tier"] == "trusted")
    print(f"Wrote {len(cases)} cases ({trusted} trusted, {len(cases) - trusted} untrusted) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert golden labels to page benchmark format")
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    convert(args.golden, args.out)


if __name__ == "__main__":
    main()
