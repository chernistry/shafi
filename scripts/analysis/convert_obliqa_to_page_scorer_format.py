"""Convert ObliQA dataset into page scorer training format for augmentation.

ObliQA has (question, gold_passages) pairs from DIFC regulatory domain.
We create synthetic page-scorer training examples by mapping gold passages
to positive labels and sampling non-gold passages as negatives.

Since ObliQA uses section-level PassageIDs (not page numbers), we synthesize
page-like features from the passage metadata.  The resulting examples augment
(not replace) the reviewed page scorer training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def _passage_to_page_num(passage_id: str) -> int:
    """Heuristic: map a section-level PassageID to a synthetic page number."""
    # Extract leading section number (e.g., "7.3.4" → 7, "Schedule.Part 1" → 100)
    m = re.match(r"(\d+)", passage_id)
    if m:
        return int(m.group(1))
    if "schedule" in passage_id.lower():
        return 100
    return 1


def _build_feature_dict(
    *,
    question: str,
    passage_text: str,
    passage_id: str,
    doc_id: int,
    is_gold: bool,
    doc_passage_count: int,
    rank_in_doc: int,
) -> dict[str, object]:
    """Build a page-scorer-compatible feature dictionary from ObliQA data."""
    page_num = _passage_to_page_num(passage_id)
    snippet = passage_text[:500].casefold()
    # Simple keyword overlap for answer_in_snippet proxy.
    q_tokens = set(re.findall(r"\w+", question.lower()))
    p_tokens = set(re.findall(r"\w+", snippet))
    overlap = len(q_tokens & p_tokens)
    return {
        "scope_mode": "single_field_single_doc",
        "answer_type": "free_text",
        "doc_rank": 1,
        "page_num": page_num,
        "is_first_page": int(page_num == 1),
        "is_last_page": 0,
        "doc_selected_by_legacy": 1,
        "doc_candidate_count": 1,
        "page_candidate_count": doc_passage_count,
        "anchor_hit_count": overlap,
        "has_anchor_hit": int(overlap > 0),
        "answer_in_snippet": int(overlap > 3),
        "requires_all_docs_in_case": 0,
        "should_force_empty_grounding_on_null": 0,
        "explicit_anchor_count": 0,
        "target_page_roles_count": 0,
        "doc_ref_count": 1,
        "legacy_context_page_count": doc_passage_count,
        "sidecar_retrieved_page_count": doc_passage_count,
        "targets_title_cover": 0,
        "targets_caption": 0,
        "targets_issued_by_block": 0,
        "targets_operative_order": 0,
        "targets_costs_block": 0,
        "targets_article_clause": int("article" in passage_id.lower()),
        "targets_schedule_table": int("schedule" in passage_id.lower()),
        "legacy_context_rank": rank_in_doc,
        "sidecar_retrieved_rank": rank_in_doc,
        "from_legacy_context": 1,
        "from_legacy_used": int(is_gold),
        "from_sidecar_retrieved": 1,
        "is_strict_type": 0,
        "multi_doc_query": 0,
        "is_compare_scope": 0,
        "query_token_count": len(question.split()),
        "answer_char_count": 0,
        "from_legacy_cited": int(is_gold),
        "legacy_cited_rank": 1 if is_gold else 0,
    }


def _build_grounding_row(
    *,
    question_id: str,
    question: str,
    doc_id: int,
    gold_passage_ids: set[str],
    all_passages: list[dict[str, object]],
) -> dict[str, object]:
    """Build one GroundingMlRow-compatible JSONL row."""
    page_candidates = []
    label_page_ids = []

    for rank, passage in enumerate(all_passages, start=1):
        pid = str(passage["PassageID"])
        page_num = _passage_to_page_num(pid)
        page_id = f"obliqa_doc{doc_id}_{page_num}"
        is_gold = pid in gold_passage_ids

        features = _build_feature_dict(
            question=question,
            passage_text=str(passage.get("Passage", "")),
            passage_id=pid,
            doc_id=doc_id,
            is_gold=is_gold,
            doc_passage_count=len(all_passages),
            rank_in_doc=rank,
        )

        page_candidates.append({
            "page_id": page_id,
            "doc_id": f"obliqa_doc{doc_id}",
            "page_num": page_num,
            "candidate_sources": ["legacy_context", "sidecar_retrieved"],
            "legacy_context_rank": rank,
            "sidecar_retrieved_rank": rank,
            "anchor_hits": [],
            "snippet_excerpt": str(passage.get("Passage", ""))[:200],
        })

        if is_gold:
            label_page_ids.append(page_id)

    return {
        "question_id": question_id,
        "question": question,
        "answer_type": "free_text",
        "golden_answer": None,
        "label_page_ids": label_page_ids,
        "label_source": "soft_ai_gold",
        "label_trust_tier": "obliqa_external",
        "label_confidence": "medium",
        "label_status": "auto_mapped",
        "label_weight": 0.5,
        "label_note_present": False,
        "scope_mode": "single_field_single_doc",
        "target_page_roles": [],
        "hard_anchor_strings": [],
        "doc_ref_signatures": [],
        "holdout_doc_family_key": f"obliqa_doc{doc_id}",
        "doc_candidates": [{
            "doc_id": f"obliqa_doc{doc_id}",
            "page_candidate_count": len(all_passages),
            "candidate_sources": ["legacy_context"],
            "legacy_selected": True,
            "sidecar_selected": False,
        }],
        "page_candidates": page_candidates,
        "legacy_selected_pages": label_page_ids,
        "sidecar_selected_pages": [],
        "support_fact_features": {
            "requires_all_docs_in_case": False,
            "should_force_empty_grounding_on_null": False,
            "explicit_anchor_count": 0,
            "target_page_roles_count": 0,
            "doc_ref_count": 1,
        },
        "page_retrieval_features": {
            "legacy_retrieved_page_count": len(all_passages),
            "legacy_context_page_count": len(all_passages),
            "legacy_cited_page_count": len(label_page_ids),
            "sidecar_retrieved_page_count": len(all_passages),
            "sidecar_context_page_count": 0,
            "sidecar_cited_page_count": 0,
            "legacy_sidecar_used_overlap_count": 0,
        },
        "label_is_suspect": False,
        "source_paths": {},
    }


def main() -> int:
    """Convert ObliQA to page scorer training format."""
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "data" / "external" / "obliqa" / "raw" / "hf_snapshot" / "ObliQA" / "ObliQA_train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "derived" / "grounding_ml" / "obliqa_augmentation" / "train.jsonl",
    )
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--min-negatives", type=int, default=2)
    parser.add_argument("--seed", type=int, default=610)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} ObliQA examples")

    # Group all passages by document for negative sampling.
    doc_passages: dict[int, list[dict[str, object]]] = {}
    for ex in data:
        for p in ex["Passages"]:
            doc_id = p["DocumentID"]
            doc_passages.setdefault(doc_id, []).append(p)

    # Deduplicate passages within each document.
    for doc_id in doc_passages:
        seen: set[str] = set()
        unique: list[dict[str, object]] = []
        for p in doc_passages[doc_id]:
            pid = str(p["PassageID"])
            if pid not in seen:
                seen.add(pid)
                unique.append(p)
        doc_passages[doc_id] = unique

    print(f"Unique documents: {len(doc_passages)}")
    for doc_id, passages in sorted(doc_passages.items()):
        print(f"  Doc {doc_id}: {len(passages)} unique passages")

    # Select examples with enough negatives for meaningful training.
    rows: list[dict[str, object]] = []
    for ex in data:
        if len(rows) >= args.max_examples:
            break

        gold_pids = {str(p["PassageID"]) for p in ex["Passages"]}
        doc_id = ex["Passages"][0]["DocumentID"]
        all_doc_passages = doc_passages.get(doc_id, [])

        # Need at least min_negatives non-gold passages for training.
        neg_count = sum(1 for p in all_doc_passages if str(p["PassageID"]) not in gold_pids)
        if neg_count < args.min_negatives:
            continue

        # Cap to reasonable size: gold + up to 8 negatives.
        candidate_passages = [p for p in all_doc_passages if str(p["PassageID"]) in gold_pids]
        neg_passages = [p for p in all_doc_passages if str(p["PassageID"]) not in gold_pids]

        # Deterministic negative sampling.
        neg_passages.sort(key=lambda p: hashlib.sha256(
            f"{args.seed}:{ex['QuestionID']}:{p['PassageID']}".encode()
        ).hexdigest())
        candidate_passages.extend(neg_passages[:8])

        row = _build_grounding_row(
            question_id=ex["QuestionID"],
            question=ex["Question"],
            doc_id=doc_id,
            gold_passage_ids=gold_pids,
            all_passages=candidate_passages,
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows)} training rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
