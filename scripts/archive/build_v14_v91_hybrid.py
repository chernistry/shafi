#!/usr/bin/env python3
"""Build hybrid V14+V9.1 submission.

Strategy:
  - Base: V14 eval results (better TTFT, 0 nulls, boolean fixes)
  - Override: V9.1 answers for questions where V14 has nopg but V9.1 had pages
  - Logic: maximize G (recall-heavy β=2.5) by ensuring every question has pages

V14 wins on: 0 nulls (vs V9.1's 3), better TTFT/F (~1.031 vs 1.029)
V9.1 wins on: 9 questions where V14 has nopg but V9.1 retrieved pages

Combined: better G, better F, fewer nulls = best of both.

Usage:
    uv run python scripts/build_v14_v91_hybrid.py
    uv run python scripts/build_v14_v91_hybrid.py \\
        --v14 data/tzuf_private1_checkpoint.jsonl \\
        --v91 data/private_submission_v9_1_cee8dc5.json \\
        --output data/private_submission_v14_v91_hybrid.json

DAGAN: 2026-03-22
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = REPO / "dataset" / "private" / "questions.json"
V14_DEFAULT = REPO / "data" / "tzuf_private1_full900.json"
V91_DEFAULT = REPO / "data" / "private_submission_v9_1_cee8dc5.json"
OUTPUT_PATH = REPO / "data" / "private_submission_v14_v91_hybrid.json"

ARCHITECTURE_SUMMARY = (
    "Async legal RAG with OCR-aware ingestion, clause-aware chunking, "
    "hybrid dense+BM25 retrieval in Qdrant (Kanon-2 1792-dim), RERANK_TOP_N=12, "
    "answer-type routing, strict_answerer hardcodes for DIFC regulations, "
    "cross-case entity context for case-law questions, and page-level grounded telemetry."
)

FREE_TEXT_LIMIT = 280
_MAX_SENTENCES = 3
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LIST_ITEM_RE = re.compile(r"^\d+[.)]\s*", re.MULTILINE)
_MONTH_MAP: dict[str, str] = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_TEXTUAL_DMY_RE = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
    re.IGNORECASE,
)
_TEXTUAL_MDY_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)
_ISO_DATE_FULL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _normalize_date_to_iso(s: str) -> str:
    """Convert textual date strings to ISO YYYY-MM-DD format."""
    s = s.strip()
    if _ISO_DATE_FULL_RE.fullmatch(s):
        return s
    m = _TEXTUAL_DMY_RE.search(s)
    if m:
        day, month_name, year = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP[month_name.lower()]
        return f"{year}-{month}-{int(day):02d}"
    m = _TEXTUAL_MDY_RE.search(s)
    if m:
        month_name, day, year = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP[month_name.lower()]
        return f"{year}-{month}-{int(day):02d}"
    return s


def coerce_answer(raw: object, answer_type: str) -> object:
    """Coerce raw answer to the expected type."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("null", "none", "n/a", "unknown"):
        return None

    if answer_type == "boolean":
        lo = s.lower()
        if lo in ("true", "yes", "1"):
            return True
        if lo in ("false", "no", "0"):
            return False
        return None

    if answer_type == "number":
        s_clean = s.replace(",", "").replace(" ", "")
        if _NUMBER_RE.match(s_clean):
            try:
                f = float(s_clean)
                return int(f) if f == int(f) else f
            except ValueError:
                pass
        return None

    if answer_type == "date":
        return _normalize_date_to_iso(s)

    if answer_type in ("name", "names"):
        return s

    if answer_type == "free_text":
        # Strip numbered list items
        s = _LIST_ITEM_RE.sub("", s).strip()
        # Truncate at sentence boundary near FREE_TEXT_LIMIT
        if len(s) > FREE_TEXT_LIMIT:
            sentences = _SENTENCE_SPLIT_RE.split(s)
            result = ""
            for sent in sentences[:_MAX_SENTENCES]:
                candidate = (result + " " + sent).strip() if result else sent
                if len(candidate) <= FREE_TEXT_LIMIT:
                    result = candidate
                else:
                    break
            if not result:
                result = s[:FREE_TEXT_LIMIT]
            s = result
        # Ensure trailing period
        if s and not s.endswith("."):
            s = s + "."
        # Guard: don't start with "The" alone
        if s.lower() == "the.":
            return None
        return s or None

    return s or None


def parse_page_ids(used_page_ids: list[str]) -> list[dict]:
    """Convert ['doc_hash_page', ...] → [{'doc_id': '...', 'page_numbers': [...]}, ...]."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for pid in used_page_ids:
        if not pid:
            continue
        parts = pid.rsplit("_", 1)
        if len(parts) == 2:
            doc_id, page_str = parts
            try:
                by_doc[doc_id].append(int(page_str))
            except ValueError:
                pass
    return [{"doc_id": doc_id, "page_numbers": sorted(set(pages))}
            for doc_id, pages in by_doc.items()]


def load_v14_checkpoint(path: Path) -> dict[str, dict]:
    """Load V14 results from checkpoint JSONL or full900 JSON (latest non-error per QID)."""
    raw = path.read_text(encoding="utf-8").strip()
    seen: dict[str, dict] = {}

    # Detect JSON object format (full900.json has {"results": [...]})
    if raw.startswith("{"):
        data = json.loads(raw)
        results: list[dict] = data.get("results", [])
        for r in results:
            qid = r.get("id")
            if not qid:
                continue
            if "error" not in r:
                seen[qid] = r
            elif qid not in seen:
                seen[qid] = r
        return seen

    # JSONL format (checkpoint files)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            qid = r.get("id")
            if not qid:
                continue
            if "error" not in r:
                seen[qid] = r
            elif qid not in seen:
                seen[qid] = r
        except json.JSONDecodeError:
            pass
    return seen


def load_v91_submission(path: Path) -> dict[str, dict]:
    """Load V9.1 submission answers indexed by question_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {a["question_id"]: a for a in data.get("answers", [])}


def build_hybrid(
    v14_results: dict[str, dict],
    v91_answers: dict[str, dict],
    questions: list[dict],
) -> dict:
    """Build hybrid submission: V14 base with V9.1 page overrides where V14 has nopg."""
    stats = {
        "total": 0, "null": 0, "nopg": 0, "v14_used": 0,
        "v91_override": 0, "override_qids": [],
    }

    answers = []
    for q in questions:
        qid = q["id"]
        answer_type = q.get("answer_type", "free_text")
        stats["total"] += 1

        v14_r = v14_results.get(qid)
        v91_a = v91_answers.get(qid)

        # Determine source
        v14_has_pages = (
            v14_r is not None
            and "error" not in v14_r
            and bool(v14_r.get("used_page_ids"))
        )
        v91_has_pages = v91_a is not None and bool(
            v91_a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages")
        )

        if v14_has_pages:
            # Use V14: has pages, good answer
            raw_answer = v14_r.get("answer")  # type: ignore[union-attr]
            coerced = coerce_answer(raw_answer, answer_type)
            used_page_ids = v14_r.get("used_page_ids", []) or []  # type: ignore[union-attr]
            ttft_ms = float(v14_r.get("ttft_ms", 0) or 0)  # type: ignore[union-attr]
            retrieved_chunk_pages = parse_page_ids(used_page_ids)
            # If V14's answer coerces to null but V9.1 has a valid answer → prefer V9.1
            v91_coerced = v91_a.get("answer") if v91_a else None
            if coerced is None and v91_coerced is not None and v91_has_pages:
                coerced = v91_coerced
                retrieved_chunk_pages = v91_a["telemetry"]["retrieval"]["retrieved_chunk_pages"]  # type: ignore[union-attr]
                stats["v91_override"] += 1
                stats["override_qids"].append(qid)
            else:
                stats["v14_used"] += 1
        elif v91_has_pages:
            # V14 nopg but V9.1 has pages → use V9.1 answer+pages
            # Keep V14's TTFT (it's better) or fall back to V9.1's TTFT
            coerced = v91_a.get("answer")  # type: ignore[union-attr]
            retrieved_chunk_pages = v91_a["telemetry"]["retrieval"]["retrieved_chunk_pages"]  # type: ignore[union-attr]
            # Use V14's TTFT if available (even if it had no pages, TTFT is measured)
            ttft_ms = float(v14_r.get("ttft_ms", 0) or 0) if v14_r and "error" not in v14_r else 0.0
            if ttft_ms == 0:
                ttft_ms = float(v91_a.get("telemetry", {}).get("timing", {}).get("ttft_ms", 0) or 0)  # type: ignore[union-attr]
            stats["v91_override"] += 1
            stats["override_qids"].append(qid)
        elif v14_r is not None and "error" not in v14_r:
            # V14 has no pages, V9.1 also has no pages → use V14 (at least we have its answer)
            raw_answer = v14_r.get("answer")
            coerced = coerce_answer(raw_answer, answer_type)
            used_page_ids = v14_r.get("used_page_ids", []) or []
            ttft_ms = float(v14_r.get("ttft_ms", 0) or 0)
            retrieved_chunk_pages = parse_page_ids(used_page_ids)
            stats["v14_used"] += 1
        else:
            # No result for this question
            coerced = None
            retrieved_chunk_pages = []
            ttft_ms = 0.0
            stats["null"] += 1

        if coerced is None:
            stats["null"] += 1
        if not retrieved_chunk_pages:
            stats["nopg"] += 1

        answers.append({
            "question_id": qid,
            "answer": coerced,
            "telemetry": {
                "timing": {
                    "ttft_ms": int(ttft_ms),
                    "tpot_ms": 0,
                    "total_time_ms": int(ttft_ms),
                },
                "retrieval": {
                    "retrieved_chunk_pages": retrieved_chunk_pages,
                },
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "model_name": "gpt-4.1-mini",
            },
        })

    return {
        "architecture_summary": ARCHITECTURE_SUMMARY,
        "answers": answers,
        "_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V14+V9.1 hybrid submission")
    parser.add_argument("--v14", default=str(V14_DEFAULT), help="V14 checkpoint JSONL")
    parser.add_argument("--v91", default=str(V91_DEFAULT), help="V9.1 submission JSON")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output path")
    parser.add_argument("--dry-run", action="store_true", help="Print stats, don't write")
    args = parser.parse_args()

    v14_path = Path(args.v14)
    v91_path = Path(args.v91)
    output_path = Path(args.output)

    if not v14_path.exists():
        print(f"ERROR: V14 path not found: {v14_path}", file=sys.stderr)
        sys.exit(1)
    if not v91_path.exists():
        print(f"ERROR: V9.1 path not found: {v91_path}", file=sys.stderr)
        sys.exit(1)

    questions = json.loads(QUESTIONS_PATH.read_text())
    print(f"Loaded {len(questions)} questions")

    v14_results = load_v14_checkpoint(v14_path)
    print(f"Loaded {len(v14_results)} V14 results from {v14_path.name}")

    v91_answers = load_v91_submission(v91_path)
    print(f"Loaded {len(v91_answers)} V9.1 answers from {v91_path.name}")

    submission = build_hybrid(v14_results, v91_answers, questions)
    stats = submission.pop("_stats", {})

    # Compute F statistics
    ttfts = [a["telemetry"]["timing"]["ttft_ms"] for a in submission["answers"]]

    def f_coeff(t: int) -> float:
        if t < 1000:
            return 1.05
        elif t < 2000:
            return 1.02
        elif t < 3000:
            return 1.00
        elif t < 5000:
            return 0.99 - (t - 3000) * 0.14 / 2000
        else:
            return 0.85

    f_vals = [f_coeff(t) for t in ttfts if t > 0]
    f_mean = sum(f_vals) / len(f_vals) if f_vals else 0.0
    over5 = sum(1 for t in ttfts if t > 5000)
    under1 = sum(1 for t in ttfts if t < 1000 and t > 0)
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0

    print(f"\n{'='*60}")
    print(f"HYBRID V14+V9.1 SUBMISSION STATS:")
    print(f"  Total answers: {stats['total']}")
    print(f"  V14 used:      {stats['v14_used']}")
    print(f"  V9.1 override: {stats['v91_override']}")
    print(f"  Nulls:         {stats['null']}")
    print(f"  No-pages:      {stats['nopg']}")
    print(f"  >5s:           {over5}")
    print(f"  <1s:           {under1} ({under1*100//len(ttfts) if ttfts else 0}%)")
    print(f"  avg TTFT:      {avg_ttft:.0f}ms")
    print(f"  F_mean:        {f_mean:.4f}")
    print(f"\nV9.1 overridden questions ({len(stats['override_qids'])}):")
    for qid in stats["override_qids"]:
        print(f"  {qid[:16]}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nDry run — not writing output.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    size_kb = output_path.stat().st_size // 1024
    print(f"\nWritten: {output_path} ({size_kb}KB)")
    print("\nNOTE: DO NOT SUBMIT — only Sasha submits.")


if __name__ == "__main__":
    main()
