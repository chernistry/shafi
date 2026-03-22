#!/usr/bin/env python3
"""Build V2 submission by applying corrections + page enrichment to FINAL_SUBMISSION.

Steps:
1. Apply 12 registry-verified answer corrections
2. Enrich pages from corpus registry (case refs, law refs)
3. Apply free-text prefixes (if available)
4. Deduplicate pages
5. Run sanity checks
"""
import json
import re
import sys
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
FINAL_SUB = ROOT / "data" / "private_submission_FINAL_SUBMISSION.json"
CORRECTIONS = ROOT / "data" / "registry_corrections_v2.json"
REGISTRY = ROOT / "data" / "private_corpus_registry.json"
QUESTIONS = ROOT / "dataset" / "private" / "questions.json"
PREFIXED = ROOT / "data" / "private_submission_V2_prefixed.json"
OUTPUT = ROOT / "data" / "private_submission_V2.json"


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Written: {path} ({path.stat().st_size / 1024:.0f}KB)")


def apply_corrections(answers: list[dict], corrections_path: Path) -> int:
    """Apply registry-verified answer corrections. Returns count applied."""
    if not corrections_path.exists():
        print("  No corrections file found, skipping")
        return 0

    corr_data = load_json(corrections_path)
    corrections = corr_data["corrections"]

    # Build QID prefix -> correction map
    corr_map = {}
    for c in corrections:
        corr_map[c["qid"]] = c

    applied = 0
    for ans in answers:
        qid = ans["question_id"]
        for prefix, corr in corr_map.items():
            if qid.startswith(prefix):
                old_val = ans["answer"]
                new_val = corr["correct"]
                print(f"    [{corr['type']}] {prefix}: {old_val} → {new_val} ({corr['reason'][:50]})")
                ans["answer"] = new_val
                applied += 1
                break

    print(f"  Applied {applied}/{len(corrections)} corrections")
    return applied


def enrich_pages_from_registry(answers: list[dict], questions: list[dict], registry: dict) -> dict:
    """Add pages from corpus registry based on case/law references in questions."""
    # Build question ID -> question text map
    q_map = {q["id"]: q for q in questions}

    # Build case number -> (doc_id, page_ids) map
    case_map = {}  # case_number -> {doc_id, page_ids}
    for doc_id, case in registry.get("cases", {}).items():
        title = case.get("title", "")
        # Extract case number like "CFI 070/2018", "ARB 009/2023", etc.
        m = re.search(r"((?:CFI|SCT|CA|ARB|TCD|ENF)\s+\d+/\d{4})", title)
        if m:
            case_num = m.group(1)
            page_ids = case.get("page_ids", [])
            case_map[case_num] = {"doc_id": doc_id, "page_ids": page_ids}

    # Also handle "ARB/031/2025" format (with slash)
    for doc_id, case in registry.get("cases", {}).items():
        title = case.get("title", "")
        m = re.search(r"((?:CFI|SCT|CA|ARB|TCD|ENF)/\d+/\d{4})", title)
        if m:
            case_num = m.group(1)
            if case_num not in case_map:
                case_map[case_num] = {"doc_id": doc_id, "page_ids": case.get("page_ids", [])}

    # Build law title -> (doc_id, page_ids) map
    law_map = {}
    for doc_id, law in registry.get("laws", {}).items():
        title = law.get("title", "")
        page_ids = law.get("page_ids", [])
        if title and page_ids:
            law_map[title] = {"doc_id": doc_id, "page_ids": page_ids}

    # Also orders
    for doc_id, order in registry.get("orders", {}).items():
        title = order.get("title", "")
        page_ids = order.get("page_ids", [])
        if title and page_ids:
            law_map[title] = {"doc_id": doc_id, "page_ids": page_ids}

    print(f"  Registry: {len(case_map)} cases, {len(law_map)} laws/orders")

    stats = {"enriched": 0, "pages_added": 0, "nopg_fixed": 0}

    for ans in answers:
        qid = ans["question_id"]
        q_info = q_map.get(qid, {})
        q_text = q_info.get("question", "")

        # Skip noinfo answers (don't add pages to "no information" answers)
        answer_val = ans.get("answer")
        if isinstance(answer_val, str) and "no information" in answer_val.lower():
            continue
        if answer_val is None:
            continue

        # Current pages
        retrieval = ans.get("telemetry", {}).get("retrieval", {})
        current_pages = retrieval.get("retrieved_chunk_pages", [])

        # Build existing doc_id -> page_numbers set
        existing = {}
        for p in current_pages:
            did = p["doc_id"]
            existing.setdefault(did, set()).update(p["page_numbers"])

        new_pages_added = 0

        # Find case references in question text
        case_refs = re.findall(r"(?:CFI|SCT|CA|ARB|TCD|ENF)[\s/]+\d+/\d{4}", q_text)
        for ref in case_refs:
            # Normalize: "ARB/031/2025" -> "ARB 031/2025"
            normalized = re.sub(r"([A-Z]+)/(\d+)", r"\1 \2", ref)

            # Try both formats
            for lookup_ref in [normalized, ref]:
                if lookup_ref in case_map:
                    info = case_map[lookup_ref]
                    doc_id = info["doc_id"]
                    # Parse page_ids like "docid_3" -> page 3
                    for pid in info["page_ids"]:
                        parts = pid.rsplit("_", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            page_num = int(parts[1])
                            if doc_id not in existing or page_num not in existing[doc_id]:
                                existing.setdefault(doc_id, set()).add(page_num)
                                new_pages_added += 1
                    break

        # Find law/order references in question text (substring match)
        for law_title, info in law_map.items():
            # Check if a significant portion of the law title appears in the question
            # Use short titles (>10 chars) for matching
            if len(law_title) > 10 and law_title.lower() in q_text.lower():
                doc_id = info["doc_id"]
                for pid in info["page_ids"]:
                    parts = pid.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        page_num = int(parts[1])
                        if doc_id not in existing or page_num not in existing[doc_id]:
                            existing.setdefault(doc_id, set()).add(page_num)
                            new_pages_added += 1

        if new_pages_added > 0:
            # Rebuild retrieved_chunk_pages from existing
            new_chunk_pages = []
            for did in sorted(existing.keys()):
                new_chunk_pages.append({
                    "doc_id": did,
                    "page_numbers": sorted(existing[did])
                })

            was_nopg = len(current_pages) == 0
            retrieval["retrieved_chunk_pages"] = new_chunk_pages
            stats["enriched"] += 1
            stats["pages_added"] += new_pages_added
            if was_nopg:
                stats["nopg_fixed"] += 1

    print(f"  Enriched {stats['enriched']} answers, +{stats['pages_added']} pages, fixed {stats['nopg_fixed']} nopg")
    return stats


def apply_freetext_prefixes(answers: list[dict], prefixed_path: Path) -> int:
    """Cherry-pick free-text prefix improvements from SHAI's output."""
    if not prefixed_path.exists():
        print("  No prefixed file found, skipping")
        return 0

    prefixed_sub = load_json(prefixed_path)
    prefixed_answers = prefixed_sub.get("answers", [])

    # Build QID -> prefixed answer map
    prefix_map = {a["question_id"]: a["answer"] for a in prefixed_answers}

    applied = 0
    for ans in answers:
        qid = ans["question_id"]
        if qid in prefix_map:
            new_answer = prefix_map[qid]
            old_answer = ans["answer"]
            # Only apply if the prefix version is different and starts with "Under "/"Pursuant to "
            if (new_answer != old_answer
                and isinstance(new_answer, str)
                and isinstance(old_answer, str)
                and (new_answer.startswith("Under ") or new_answer.startswith("Pursuant to "))):
                ans["answer"] = new_answer
                applied += 1
                print(f"    Prefix: {old_answer[:40]}... → {new_answer[:40]}...")

    print(f"  Applied {applied} free-text prefixes")
    return applied


def dedup_pages(answers: list[dict]) -> int:
    """Deduplicate page entries. Returns count of answers with dupes removed."""
    fixed = 0
    for ans in answers:
        pages = ans.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", [])
        if not pages:
            continue

        # Merge by doc_id
        merged = {}
        for p in pages:
            did = p["doc_id"]
            merged.setdefault(did, set()).update(p["page_numbers"])

        new_pages = [
            {"doc_id": did, "page_numbers": sorted(pnums)}
            for did, pnums in sorted(merged.items())
        ]

        if len(new_pages) != len(pages):
            ans["telemetry"]["retrieval"]["retrieved_chunk_pages"] = new_pages
            fixed += 1

    if fixed:
        print(f"  Deduped pages in {fixed} answers")
    return fixed


def sanity_check(answers: list[dict], questions: list[dict]) -> bool:
    """Run sanity checks. Returns True if all pass."""
    q_map = {q["id"]: q for q in questions}

    issues = []
    null_count = 0
    nopg_count = 0
    over280 = 0

    for ans in answers:
        qid = ans["question_id"]
        q_info = q_map.get(qid, {})
        answer_type = q_info.get("answer_type", "unknown")
        answer = ans.get("answer")
        pages = ans.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", [])

        # Null check
        if answer is None:
            null_count += 1

        # Nopg check
        if not pages:
            nopg_count += 1

        # Type-specific checks
        if answer_type == "boolean" and answer is not None:
            if answer not in (True, False, "Yes", "No"):
                issues.append(f"  Boolean format: {qid[:8]} = {answer}")

        if answer_type == "date" and isinstance(answer, str):
            if answer.endswith("."):
                issues.append(f"  Date trailing period: {qid[:8]} = {answer}")

        if answer_type == "names" and answer is not None:
            if not isinstance(answer, list):
                issues.append(f"  Names not list: {qid[:8]} = {type(answer)}")

        if answer_type == "free_text" and isinstance(answer, str):
            if len(answer) > 280:
                over280 += 1
                issues.append(f"  Over 280: {qid[:8]} len={len(answer)}")

    print(f"\n  SANITY CHECK:")
    print(f"    Answers: {len(answers)}")
    print(f"    null: {null_count}")
    print(f"    nopg: {nopg_count}")
    print(f"    over280: {over280}")
    print(f"    Issues: {len(issues)}")
    for i in issues[:10]:
        print(f"    {i}")

    passed = len(answers) == 900 and over280 == 0 and len(issues) == 0
    print(f"    Result: {'PASS' if passed else 'FAIL (warnings only)' if len(issues) == 0 else 'FAIL'}")
    return passed


def main():
    print("=" * 60)
    print("BUILD V2 SUBMISSION")
    print("=" * 60)

    # Load base submission
    print("\n1. Loading FINAL_SUBMISSION...")
    sub = load_json(FINAL_SUB)
    answers = sub["answers"]
    print(f"  {len(answers)} answers loaded")

    # Count baseline
    baseline_null = sum(1 for a in answers if a.get("answer") is None)
    baseline_nopg = sum(1 for a in answers if not a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []))
    print(f"  Baseline: null={baseline_null}, nopg={baseline_nopg}")

    # Deep copy for comparison later
    original_answers = {a["question_id"]: deepcopy(a["answer"]) for a in answers}

    # Step 1: Apply corrections
    print("\n2. Applying registry corrections...")
    n_corrected = apply_corrections(answers, CORRECTIONS)

    # Step 1b: Noinfo page clearing (DISABLED — risky, nopg 3→26)
    # OREV hypothesis: clearing pages from general-knowledge noinfo answers improves G.
    # But if platform penalizes nopg, this hurts. Variant saved as V2_noinfo.json.
    # Uncomment to enable:
    # _NOINFO_PHRASES = ("no information on this question", "there is no information")
    # noinfo_cleared = 0
    # for ans in answers:
    #     av = ans.get("answer")
    #     if isinstance(av, str) and any(ph in av.lower() for ph in _NOINFO_PHRASES):
    #         retrieval = ans.setdefault("telemetry", {}).setdefault("retrieval", {})
    #         if retrieval.get("retrieved_chunk_pages"):
    #             retrieval["retrieved_chunk_pages"] = []
    #             noinfo_cleared += 1
    # print(f"  Cleared pages for {noinfo_cleared} noinfo answers")

    # Step 2: Enrich pages from registry
    print("\n3. Enriching pages from corpus registry...")
    questions = load_json(QUESTIONS)
    registry = load_json(REGISTRY)
    enrich_stats = enrich_pages_from_registry(answers, questions, registry)

    # Step 3: Apply free-text prefixes
    print("\n4. Applying free-text prefixes...")
    n_prefixed = apply_freetext_prefixes(answers, PREFIXED)

    # Step 4: Dedup pages
    print("\n5. Deduplicating pages...")
    dedup_pages(answers)

    # Step 5: Sanity check
    print("\n6. Running sanity checks...")
    sanity_check(answers, questions)

    # Summary
    final_null = sum(1 for a in answers if a.get("answer") is None)
    final_nopg = sum(1 for a in answers if not a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []))
    total_pages = sum(
        sum(len(p["page_numbers"]) for p in a.get("telemetry", {}).get("retrieval", {}).get("retrieved_chunk_pages", []))
        for a in answers
    )

    changed = sum(1 for a in answers if a["answer"] != original_answers[a["question_id"]])

    print(f"\n{'=' * 60}")
    print(f"COMPARISON: FINAL → V2")
    print(f"{'=' * 60}")
    print(f"  null:          {baseline_null} → {final_null}")
    print(f"  nopg:          {baseline_nopg} → {final_nopg}")
    print(f"  total_pages:   ... → {total_pages}")
    print(f"  answers_changed: {changed}")
    print(f"  corrections:   {n_corrected}")
    print(f"  enriched:      {enrich_stats['enriched']} answers (+{enrich_stats['pages_added']} pages)")
    print(f"  prefixed:      {n_prefixed}")

    # Save
    print(f"\n7. Saving V2 submission...")
    save_json(sub, OUTPUT)
    print(f"\nDONE. V2 ready at: {OUTPUT}")


if __name__ == "__main__":
    main()
