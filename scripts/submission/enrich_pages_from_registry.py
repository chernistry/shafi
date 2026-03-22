#!/usr/bin/env python3
"""Enrich submission pages from corpus registry.

For each answer that references a specific case/law/order, adds all pages
from matching registry documents that are NOT yet represented in the answer.
Only adds pages when the doc is completely absent (zero coverage) to avoid
hurting precision on already-good answers.

Skips noinfo answers — those correctly have empty pages.

Usage:
    python scripts/enrich_pages_from_registry.py \
        --input data/private_submission_FINAL_SUBMISSION_V2.json \
        --output data/private_submission_V2_enriched.json \
        --registry data/private_corpus_registry.json \
        --questions dataset/private/questions.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

# ---------------------------------------------------------------------------
# Case-number normalisation
# ---------------------------------------------------------------------------

# Matches: CFI 076/2024, ARB 004/2024, CA 004/2025, SCT 011/2025, ENF 022/2023,
#          TCD 001/2024, DIV 001/2023, etc.  Also dash or slash separators.
_CASE_REF_RE = re.compile(
    r"\b(CFI|SCT|CA|ARB|TCD|ENF|DIV|GAT|LIN)\s*[-/]?\s*(\d{3})\s*[-/]\s*(\d{4})\b",
    re.IGNORECASE,
)


def _normalise_case_ref(court: str, num: str, year: str) -> str:
    """Return canonical form: 'CFI 076/2024'."""
    return f"{court.upper()} {num.zfill(3)}/{year}"


def _extract_case_refs(text: str) -> list[str]:
    """Return list of canonical case refs found in text."""
    refs: list[str] = []
    for m in _CASE_REF_RE.finditer(text):
        refs.append(_normalise_case_ref(m.group(1), m.group(2), m.group(3)))
    return refs


# ---------------------------------------------------------------------------
# Registry index helpers
# ---------------------------------------------------------------------------

def _build_case_index(registry: JsonDict) -> dict[str, list[JsonDict]]:
    """Map canonical case_number -> list of registry case objects."""
    idx: dict[str, list[JsonDict]] = defaultdict(list)
    for item in registry.get("cases", {}).values():
        cn = item.get("case_number", "")
        if cn and not re.match(r"^[0-9a-f]{60,}$", cn):
            # Normalise registry case number too
            m = _CASE_REF_RE.match(cn)
            if m:
                key = _normalise_case_ref(m.group(1), m.group(2), m.group(3))
            else:
                key = cn.upper().strip()
            idx[key].append(item)
    return dict(idx)


def _build_law_order_index(registry: JsonDict) -> list[tuple[str, JsonDict]]:
    """Return list of (normalised_title, item) for laws and orders."""
    pairs: list[tuple[str, JsonDict]] = []
    for section in ("laws", "orders"):
        for item in registry.get(section, {}).values():
            title = item.get("title", "").strip()
            short = item.get("short_title", "").strip()
            if title:
                pairs.append((title.lower(), item))
            if short and short.lower() != title.lower():
                pairs.append((short.lower(), item))
    return pairs


# ---------------------------------------------------------------------------
# Page-format helpers
# ---------------------------------------------------------------------------

def _pages_to_dict(page_ids: list[str]) -> dict[str, list[int]]:
    """Convert ['hash_1', 'hash_2', ...] → {hash: [1, 2, ...]}."""
    result: dict[str, list[int]] = defaultdict(list)
    for pid in page_ids:
        if "_" not in pid:
            continue
        doc_id, _, pnum = pid.rpartition("_")
        try:
            result[doc_id].append(int(pnum))
        except ValueError:
            pass
    return dict(result)


def _current_doc_ids(retrieved_chunk_pages: list[JsonDict]) -> set[str]:
    """Return set of doc_ids already present in the answer's pages."""
    return {entry["doc_id"] for entry in retrieved_chunk_pages if "doc_id" in entry}


def _merge_pages(
    retrieved_chunk_pages: list[JsonDict],
    new_page_ids: list[str],
    doc_id: str,
) -> list[JsonDict]:
    """Add pages for doc_id to retrieved_chunk_pages (deduplicating)."""
    pages_by_doc = {e["doc_id"]: list(e["page_numbers"]) for e in retrieved_chunk_pages}

    new_pages = _pages_to_dict(new_page_ids)
    for did, pnums in new_pages.items():
        if did != doc_id:
            continue  # Only add pages for the target doc
        existing = set(pages_by_doc.get(did, []))
        combined = sorted(existing | set(pnums))
        pages_by_doc[did] = combined

    return [{"doc_id": did, "page_numbers": pnums} for did, pnums in pages_by_doc.items()]


# ---------------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------------

_NOINFO_ANSWERS = frozenset({
    "there is no information on this question",
    "there is no information on this question.",
    "null",
    "none",
    "",
})


def _is_noinfo(answer: Any) -> bool:
    if answer is None:
        return True
    if isinstance(answer, str):
        return answer.strip().lower() in _NOINFO_ANSWERS
    return False


def enrich(
    submission: JsonDict,
    registry: JsonDict,
    questions: list[JsonDict],
) -> tuple[JsonDict, dict[str, int]]:
    """Enrich submission pages; return (enriched_submission, stats)."""
    case_idx = _build_case_index(registry)
    q_by_id = {q["id"]: q for q in questions}

    stats = {"enriched": 0, "pages_added": 0, "nopg_before": 0, "nopg_after": 0}

    for answer in submission["answers"]:
        qid = answer["question_id"]
        ans_val = answer.get("answer")

        telem = answer.setdefault("telemetry", {})
        retrieval = telem.setdefault("retrieval", {})
        cur_pages: list[JsonDict] = retrieval.get("retrieved_chunk_pages", [])

        # Count nopg before
        if not cur_pages:
            stats["nopg_before"] += 1

        # Skip noinfo — these must have empty pages
        if _is_noinfo(ans_val):
            if not cur_pages:
                stats["nopg_after"] += 1
            continue

        q_text = q_by_id.get(qid, {}).get("question", "")
        already_doc_ids = _current_doc_ids(cur_pages)
        pages_before = sum(len(e.get("page_numbers", [])) for e in cur_pages)
        modified = False

        # --- Case references ---
        for ref in _extract_case_refs(q_text):
            for case_item in case_idx.get(ref, []):
                doc_id = case_item["doc_id"]
                if doc_id in already_doc_ids:
                    continue  # Already have pages from this doc — leave it alone
                page_ids = case_item.get("page_ids", [])
                if not page_ids:
                    continue
                # Add all pages for this doc
                cur_pages = _merge_pages(cur_pages, page_ids, doc_id)
                already_doc_ids.add(doc_id)
                modified = True

        # Law/order title matching intentionally omitted:
        # short titles (e.g. "Employment Law") match too many questions as
        # substrings, flooding answers with hundreds of irrelevant pages.
        # Case-number matching above is precise enough.

        if modified:
            retrieval["retrieved_chunk_pages"] = cur_pages
            pages_after = sum(len(e.get("page_numbers", [])) for e in cur_pages)
            stats["pages_added"] += pages_after - pages_before
            stats["enriched"] += 1

        if not retrieval.get("retrieved_chunk_pages"):
            stats["nopg_after"] += 1

    return submission, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/private_submission_FINAL_SUBMISSION_V2.json")
    parser.add_argument("--output", default="data/private_submission_V2_enriched.json")
    parser.add_argument("--registry", default="data/private_corpus_registry.json")
    parser.add_argument("--questions", default="dataset/private/questions.json")
    args = parser.parse_args()

    print(f"Loading submission: {args.input}")
    submission = json.loads(Path(args.input).read_text())
    print(f"Loading registry:   {args.registry}")
    registry = json.loads(Path(args.registry).read_text())
    print(f"Loading questions:  {args.questions}")
    questions = json.loads(Path(args.questions).read_text())

    print(f"Answers: {len(submission['answers'])}")

    submission, stats = enrich(submission, registry, questions)

    out = Path(args.output)
    out.write_text(json.dumps(submission, separators=(",", ":")))
    size_kb = out.stat().st_size / 1024

    print(f"\n=== Enrichment Stats ===")
    print(f"  Answers enriched:  {stats['enriched']}")
    print(f"  Pages added:       {stats['pages_added']}")
    print(f"  nopg before:       {stats['nopg_before']}")
    print(f"  nopg after:        {stats['nopg_after']}")
    print(f"\nWritten: {args.output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
