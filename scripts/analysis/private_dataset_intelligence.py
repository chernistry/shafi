#!/usr/bin/env python3
"""Private dataset intelligence toolkit for rapid understanding of new competition data.

Produces four analysis artifacts:
  1. document_profile.json     -- per-document structural/entity profile
  2. query_distribution_report.json -- per-question scope/complexity profile
  3. distribution_shift_report.md   -- private vs warmup comparison
  4. synthetic_training_triples.jsonl -- weak-supervision retrieval triples

Usage:
    python scripts/private_dataset_intelligence.py \\
        --dataset-dir dataset_private/ \\
        --output-dir analysis/private/
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Project root for imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from rag_challenge.core.grounding.query_scope_classifier import (  # noqa: E402
    classify_query_scope,
)

# ---------------------------------------------------------------------------
# Entity extraction regex (mirrored from codebase anchor patterns)
# ---------------------------------------------------------------------------
_LAW_TITLE_RE = re.compile(r"\bLaw\s+No\.?\s+\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(r"\b[A-Z]{2,4}\s+\d{3}/\d{4}\b")
_ARTICLE_REF_RE = re.compile(r"\bArticle\s+\d+[A-Z]?(?:\(\w+\))?\b", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"\bSection\s+\d+[A-Z]?(?:\(\w+\))?\b", re.IGNORECASE)
_SCHEDULE_REF_RE = re.compile(r"\bSchedule\s+\d+[A-Z]?\b", re.IGNORECASE)

# Document-type heuristic patterns (applied to title strings)
_DOC_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("regulation", re.compile(r"\b(?:regulation|rules?|module)\b", re.IGNORECASE)),
    ("case_law", re.compile(r"\b(?:case|CFI|CA|ARB|claim|arbitration|court)\b", re.IGNORECASE)),
    ("contract", re.compile(r"\b(?:contract|agreement|deed|lease|MoU)\b", re.IGNORECASE)),
    ("order", re.compile(r"\b(?:order|direction|directive|notice|enactment)\b", re.IGNORECASE)),
]

# Multi-hop / complexity indicators
_MULTIHOP_INDICATORS = re.compile(
    r"\b(?:compare|both|difference|between|respectively|and\s.*\band|"
    r"across\s+all|every\s+document|multiple|total\s+number\s+of)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# TypedDicts for output schemas
# ---------------------------------------------------------------------------


class EntityCounts(TypedDict):
    """Counts of extracted legal entities."""

    law_titles: int
    case_refs: int
    article_refs: int
    section_refs: int
    schedule_refs: int


class StructuralMarkers(TypedDict):
    """Boolean flags for structural document features."""

    has_toc: bool
    has_definitions_section: bool
    has_schedule: bool


class DocumentProfile(TypedDict):
    """Per-document structural and entity profile."""

    doc_id: str
    title: str
    doc_type_raw: str
    detected_type: str
    page_count: int
    entity_counts: EntityCounts
    structural_markers: StructuralMarkers
    warmup_similarity: float


class QueryProfile(TypedDict):
    """Per-question scope and complexity profile."""

    question_id: str
    question: str
    detected_answer_type: str
    detected_scope: str
    target_page_roles: list[str]
    hard_anchor_strings: list[str]
    complexity_score: int
    most_similar_warmup_question: str
    warmup_jaccard: float


class TrainingTriple(TypedDict):
    """Weak-supervision retrieval triple."""

    query: str
    positive_doc_id: str
    negative_doc_id: str
    scope: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_entities(text: str) -> dict[str, list[str]]:
    """Extract legal entities from text using codebase regex patterns.

    Args:
        text: Input text (title, question, etc.).

    Returns:
        Dict mapping entity type to list of matched strings.
    """
    return {
        "law_titles": _LAW_TITLE_RE.findall(text),
        "case_refs": _CASE_REF_RE.findall(text),
        "article_refs": _ARTICLE_REF_RE.findall(text),
        "section_refs": _SECTION_REF_RE.findall(text),
        "schedule_refs": _SCHEDULE_REF_RE.findall(text),
    }


def _entity_set(text: str) -> set[str]:
    """Return a flat set of all entity mentions found in text.

    Args:
        text: Input text to scan.

    Returns:
        Set of lowercased entity strings.
    """
    entities: set[str] = set()
    for matches in _extract_entities(text).values():
        for m in matches:
            entities.add(m.strip().lower())
    return entities


def _jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        a: First set.
        b: Second set.

    Returns:
        Jaccard similarity in [0.0, 1.0].
    """
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two strings.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard similarity of whitespace-split token sets.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    return _jaccard(tokens_a, tokens_b)


def _detect_doc_type(title: str, raw_type: str) -> str:
    """Heuristically detect document type from title and raw type metadata.

    Args:
        title: Document title string.
        raw_type: Raw type field from document index.

    Returns:
        One of: "regulation", "case_law", "contract", "order", "law",
        "amendment_law", or "other".
    """
    combined = f"{title} {raw_type}".lower()
    # Preserve known index types directly
    if raw_type in ("law", "amendment_law", "enactment_notice"):
        return raw_type
    if "court_case" in combined or raw_type == "case":
        return "case_law"
    for label, pattern in _DOC_TYPE_PATTERNS:
        if pattern.search(combined):
            return label
    return "other"


def _structural_markers_from_title(title: str) -> StructuralMarkers:
    """Infer structural markers from document title (best-effort without PDF text).

    Args:
        title: Document title.

    Returns:
        StructuralMarkers flags.
    """
    lower = title.lower()
    return StructuralMarkers(
        has_toc="contents" in lower or "table of contents" in lower,
        has_definitions_section="definition" in lower or "interpretation" in lower,
        has_schedule=bool(re.search(r"\bschedule\b", lower)),
    )


def _complexity_score(question: str, entities: dict[str, list[str]]) -> int:
    """Compute a simple complexity score for a question.

    Score = total entity mentions + multi-hop indicator count.

    Args:
        question: Question text.
        entities: Extracted entity dict.

    Returns:
        Non-negative integer complexity score.
    """
    entity_count = sum(len(v) for v in entities.values())
    multihop_count = len(_MULTIHOP_INDICATORS.findall(question))
    return entity_count + multihop_count


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_questions(path: Path) -> list[dict[str, Any]]:
    """Load questions from a JSON file.

    Args:
        path: Path to questions.json.

    Returns:
        List of question dicts.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a JSON list.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"Expected list in {path}, got {type(data).__name__}"
        raise ValueError(msg)
    return data  # type: ignore[no-any-return]


def _load_doc_index(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Load document index, trying dataset-local then repo-root fallback.

    Searches for:
      1. dataset_dir / dataset_document_index.json
      2. repo_root / dataset_document_index.json

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        Dict mapping doc_id to metadata. Empty if not found.
    """
    candidates = [
        dataset_dir / "dataset_document_index.json",
        _REPO_ROOT / "dataset_document_index.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data  # type: ignore[no-any-return]
    return {}


def _discover_pdf_doc_ids(dataset_dir: Path) -> set[str]:
    """Discover document IDs from PDF filenames in documents/ subdirectory.

    Args:
        dataset_dir: Root dataset directory containing documents/ folder.

    Returns:
        Set of doc IDs (stem of .pdf files).
    """
    docs_dir = dataset_dir / "documents"
    if not docs_dir.is_dir():
        # Also try dataset_documents (warmup structure variant)
        docs_dir = dataset_dir / "dataset_documents"
    if not docs_dir.is_dir():
        return set()
    return {p.stem for p in docs_dir.glob("*.pdf")}


def _load_warmup_questions() -> list[dict[str, Any]]:
    """Load warmup questions from known locations.

    Returns:
        List of warmup question dicts. Empty if not found.
    """
    candidates = [
        _REPO_ROOT / "platform_runs" / "warmup" / "questions.json",
        _REPO_ROOT / "dataset" / "questions.json",
        _REPO_ROOT / "dataset" / "public_dataset.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                data = _load_questions(path)
                return data
            except (ValueError, json.JSONDecodeError):
                continue
    return []


# ---------------------------------------------------------------------------
# Analysis builders
# ---------------------------------------------------------------------------


def build_document_profiles(
    doc_index: dict[str, dict[str, Any]],
    pdf_doc_ids: set[str],
    warmup_doc_index: dict[str, dict[str, Any]],
) -> list[DocumentProfile]:
    """Build structural profiles for all discovered documents.

    Args:
        doc_index: Document index mapping doc_id to metadata.
        pdf_doc_ids: Set of doc IDs found as PDF files.
        warmup_doc_index: Warmup document index for similarity comparison.

    Returns:
        List of DocumentProfile dicts.
    """
    # Merge: all doc IDs from index + PDFs
    all_doc_ids = set(doc_index.keys()) | pdf_doc_ids

    # Pre-compute warmup entity sets per document
    warmup_entity_sets: dict[str, set[str]] = {}
    for wid, wmeta in warmup_doc_index.items():
        wtitle = wmeta.get("title", "")
        warmup_entity_sets[wid] = _entity_set(wtitle)

    # Flatten warmup entities for aggregate comparison
    warmup_all_entities: set[str] = set()
    for es in warmup_entity_sets.values():
        warmup_all_entities |= es

    profiles: list[DocumentProfile] = []
    for doc_id in sorted(all_doc_ids):
        meta = doc_index.get(doc_id, {})
        title: str = meta.get("title", "")
        raw_type: str = meta.get("type", "unknown")

        entities = _extract_entities(title)
        entity_flat = _entity_set(title)

        # Warmup similarity: best Jaccard vs any single warmup doc
        best_sim = 0.0
        if warmup_entity_sets and entity_flat:
            for ws in warmup_entity_sets.values():
                sim = _jaccard(entity_flat, ws)
                if sim > best_sim:
                    best_sim = sim

        profile = DocumentProfile(
            doc_id=doc_id,
            title=title,
            doc_type_raw=raw_type,
            detected_type=_detect_doc_type(title, raw_type),
            page_count=0,  # Cannot determine without parsing PDF
            entity_counts=EntityCounts(
                law_titles=len(entities["law_titles"]),
                case_refs=len(entities["case_refs"]),
                article_refs=len(entities["article_refs"]),
                section_refs=len(entities["section_refs"]),
                schedule_refs=len(entities["schedule_refs"]),
            ),
            structural_markers=_structural_markers_from_title(title),
            warmup_similarity=round(best_sim, 4),
        )
        profiles.append(profile)

    return profiles


def build_query_profiles(
    questions: list[dict[str, Any]],
    warmup_questions: list[dict[str, Any]],
) -> list[QueryProfile]:
    """Build scope and complexity profiles for all questions.

    Args:
        questions: List of private dataset question dicts.
        warmup_questions: List of warmup question dicts for similarity.

    Returns:
        List of QueryProfile dicts.
    """
    warmup_texts = [q.get("question", "") for q in warmup_questions]

    profiles: list[QueryProfile] = []
    for q in questions:
        qid: str = q.get("question_id", q.get("id", ""))
        question_text: str = q.get("question", "")
        answer_type: str = q.get("answer_type", "free_text")

        # Scope classification via imported classifier
        scope_pred = classify_query_scope(question_text, answer_type)

        entities = _extract_entities(question_text)
        cscore = _complexity_score(question_text, entities)

        # Find most similar warmup question
        best_warmup = ""
        best_jaccard = 0.0
        if warmup_texts:
            for wt in warmup_texts:
                j = _token_jaccard(question_text, wt)
                if j > best_jaccard:
                    best_jaccard = j
                    best_warmup = wt

        profile = QueryProfile(
            question_id=qid,
            question=question_text,
            detected_answer_type=answer_type,
            detected_scope=scope_pred.scope_mode.value,
            target_page_roles=scope_pred.target_page_roles,
            hard_anchor_strings=scope_pred.hard_anchor_strings,
            complexity_score=cscore,
            most_similar_warmup_question=best_warmup,
            warmup_jaccard=round(best_jaccard, 4),
        )
        profiles.append(profile)

    return profiles


def build_distribution_shift_report(
    query_profiles: list[QueryProfile],
    warmup_questions: list[dict[str, Any]],
    doc_profiles: list[DocumentProfile],
    warmup_doc_index: dict[str, dict[str, Any]],
) -> str:
    """Generate a markdown report comparing private vs warmup distributions.

    Args:
        query_profiles: Private dataset query profiles.
        warmup_questions: Warmup question dicts.
        doc_profiles: Private document profiles.
        warmup_doc_index: Warmup document index.

    Returns:
        Markdown report string.
    """
    lines: list[str] = []
    lines.append("# Distribution Shift Report: Private vs Warmup")
    lines.append("")

    # -- Answer type distribution --
    private_type_dist: dict[str, int] = {}
    for qp in query_profiles:
        t = qp["detected_answer_type"]
        private_type_dist[t] = private_type_dist.get(t, 0) + 1

    warmup_type_dist: dict[str, int] = {}
    for wq in warmup_questions:
        t = wq.get("answer_type", "unknown")
        warmup_type_dist[t] = warmup_type_dist.get(t, 0) + 1

    all_types = sorted(set(list(private_type_dist.keys()) + list(warmup_type_dist.keys())))

    lines.append("## Answer Type Distribution")
    lines.append("")
    lines.append("| Answer Type | Private | Warmup | Delta |")
    lines.append("|-------------|---------|--------|-------|")
    for at in all_types:
        pc = private_type_dist.get(at, 0)
        wc = warmup_type_dist.get(at, 0)
        delta = pc - wc
        sign = "+" if delta > 0 else ""
        lines.append(f"| {at} | {pc} | {wc} | {sign}{delta} |")
    lines.append(f"| **Total** | **{sum(private_type_dist.values())}** | **{sum(warmup_type_dist.values())}** | |")
    lines.append("")

    # -- Scope mode distribution --
    private_scope_dist: dict[str, int] = {}
    for qp in query_profiles:
        s = qp["detected_scope"]
        private_scope_dist[s] = private_scope_dist.get(s, 0) + 1

    warmup_scope_dist: dict[str, int] = {}
    for wq in warmup_questions:
        wq_text: str = wq.get("question", "")
        wq_type: str = wq.get("answer_type", "free_text")
        ws = classify_query_scope(wq_text, wq_type).scope_mode.value
        warmup_scope_dist[ws] = warmup_scope_dist.get(ws, 0) + 1

    all_scopes = sorted(set(list(private_scope_dist.keys()) + list(warmup_scope_dist.keys())))

    lines.append("## Scope Mode Distribution")
    lines.append("")
    lines.append("| Scope Mode | Private | Warmup | Delta |")
    lines.append("|------------|---------|--------|-------|")
    for sm in all_scopes:
        pc = private_scope_dist.get(sm, 0)
        wc = warmup_scope_dist.get(sm, 0)
        delta = pc - wc
        sign = "+" if delta > 0 else ""
        lines.append(f"| {sm} | {pc} | {wc} | {sign}{delta} |")
    lines.append("")

    # -- Complexity distribution (histogram buckets) --
    lines.append("## Complexity Distribution")
    lines.append("")
    buckets = {"0 (trivial)": 0, "1-2 (simple)": 0, "3-5 (moderate)": 0, "6+ (complex)": 0}
    for qp in query_profiles:
        c = qp["complexity_score"]
        if c == 0:
            buckets["0 (trivial)"] += 1
        elif c <= 2:
            buckets["1-2 (simple)"] += 1
        elif c <= 5:
            buckets["3-5 (moderate)"] += 1
        else:
            buckets["6+ (complex)"] += 1

    lines.append("| Complexity Bucket | Count |")
    lines.append("|-------------------|-------|")
    for bucket, count in buckets.items():
        lines.append(f"| {bucket} | {count} |")
    lines.append("")

    # -- New document families not in warmup --
    warmup_titles = {meta.get("title", "").lower() for meta in warmup_doc_index.values()}

    lines.append("## New Document Families (not seen in warmup)")
    lines.append("")
    new_docs: list[DocumentProfile] = []
    for dp in doc_profiles:
        if dp["title"].lower() not in warmup_titles and dp["warmup_similarity"] < 0.3:
            new_docs.append(dp)

    if new_docs:
        lines.append("| Doc ID (prefix) | Title | Detected Type | Warmup Similarity |")
        lines.append("|-----------------|-------|---------------|-------------------|")
        for dp in new_docs:
            short_id = dp["doc_id"][:12] + "..."
            lines.append(f"| {short_id} | {dp['title'][:60]} | {dp['detected_type']} | {dp['warmup_similarity']:.2f} |")
    else:
        lines.append("No significantly new document families detected.")
    lines.append("")

    # -- Document type comparison --
    private_doc_types: dict[str, int] = {}
    for dp in doc_profiles:
        t = dp["detected_type"]
        private_doc_types[t] = private_doc_types.get(t, 0) + 1

    warmup_doc_types: dict[str, int] = {}
    for wmeta in warmup_doc_index.values():
        t = wmeta.get("type", "other")
        warmup_doc_types[t] = warmup_doc_types.get(t, 0) + 1

    all_doc_types = sorted(set(list(private_doc_types.keys()) + list(warmup_doc_types.keys())))

    lines.append("## Document Type Comparison")
    lines.append("")
    lines.append("| Doc Type | Private | Warmup | Delta |")
    lines.append("|----------|---------|--------|-------|")
    for dt in all_doc_types:
        pc = private_doc_types.get(dt, 0)
        wc = warmup_doc_types.get(dt, 0)
        delta = pc - wc
        sign = "+" if delta > 0 else ""
        lines.append(f"| {dt} | {pc} | {wc} | {sign}{delta} |")
    lines.append("")

    # -- Predicted failure hotspots --
    lines.append("## Predicted Failure Hotspots")
    lines.append("")

    hotspots: list[str] = []

    # New answer types not in warmup
    new_types = set(private_type_dist.keys()) - set(warmup_type_dist.keys())
    if new_types:
        hotspots.append(f"- **New answer types**: {', '.join(sorted(new_types))} -- no warmup training signal")

    # New scopes not in warmup
    new_scopes = set(private_scope_dist.keys()) - set(warmup_scope_dist.keys())
    if new_scopes:
        hotspots.append(f"- **New scope modes**: {', '.join(sorted(new_scopes))} -- untested retrieval paths")

    # New doc types
    new_doc_types = set(private_doc_types.keys()) - set(warmup_doc_types.keys())
    if new_doc_types:
        hotspots.append(
            f"- **New document types**: {', '.join(sorted(new_doc_types))} -- no ingestion heuristics tuned"
        )

    # High complexity questions
    high_complexity = [qp for qp in query_profiles if qp["complexity_score"] >= 5]
    if high_complexity:
        hotspots.append(
            f"- **High-complexity questions**: {len(high_complexity)} questions with "
            f"complexity >= 5 (multi-hop, multi-entity)"
        )

    # Low warmup similarity questions
    low_sim = [qp for qp in query_profiles if qp["warmup_jaccard"] < 0.15]
    if low_sim:
        hotspots.append(
            f"- **Low warmup similarity**: {len(low_sim)} questions with token Jaccard < 0.15 vs any warmup question"
        )

    # Broad free-text with no anchors
    broad_no_anchor = [
        qp for qp in query_profiles if qp["detected_scope"] == "broad_free_text" and not qp["hard_anchor_strings"]
    ]
    if broad_no_anchor:
        hotspots.append(
            f"- **Broad free-text without anchors**: {len(broad_no_anchor)} questions -- "
            f"retrieval will rely solely on embedding similarity"
        )

    if hotspots:
        for h in hotspots:
            lines.append(h)
    else:
        lines.append("No significant failure hotspots predicted.")
    lines.append("")

    return "\n".join(lines)


def build_synthetic_triples(
    questions: list[dict[str, Any]],
    doc_index: dict[str, dict[str, Any]],
    query_profiles: list[QueryProfile],
) -> list[TrainingTriple]:
    """Generate weak-supervision retrieval triples using entity matching.

    For each question, the positive document is the one whose title entities
    best overlap with the question entities. A random other document is negative.

    Args:
        questions: Private dataset question dicts.
        doc_index: Document index mapping doc_id to metadata.
        query_profiles: Pre-computed query profiles for scope info.

    Returns:
        List of TrainingTriple dicts.
    """
    if not doc_index:
        return []

    doc_ids = list(doc_index.keys())
    doc_entity_sets: dict[str, set[str]] = {}
    for did, meta in doc_index.items():
        title: str = meta.get("title", "")
        doc_entity_sets[did] = _entity_set(title)

    # Build scope lookup
    scope_by_qid: dict[str, str] = {}
    for qp in query_profiles:
        scope_by_qid[qp["question_id"]] = qp["detected_scope"]

    rng = random.Random(42)
    triples: list[TrainingTriple] = []

    for q in questions:
        qid: str = q.get("question_id", q.get("id", ""))
        question_text: str = q.get("question", "")
        q_entities = _entity_set(question_text)

        if not q_entities:
            continue

        # Find best matching document
        best_doc_id = ""
        best_sim = -1.0
        for did, d_entities in doc_entity_sets.items():
            sim = _jaccard(q_entities, d_entities)
            if sim > best_sim:
                best_sim = sim
                best_doc_id = did

        if not best_doc_id or best_sim <= 0.0:
            continue

        # Pick a random negative (different from positive)
        negative_candidates = [d for d in doc_ids if d != best_doc_id]
        if not negative_candidates:
            continue
        neg_doc_id = rng.choice(negative_candidates)

        triple = TrainingTriple(
            query=question_text,
            positive_doc_id=best_doc_id,
            negative_doc_id=neg_doc_id,
            scope=scope_by_qid.get(qid, "unknown"),
        )
        triples.append(triple)

    return triples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run private dataset intelligence analysis.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    parser = argparse.ArgumentParser(
        description="Analyze a private dataset directory for the Legal RAG Competition.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to private dataset directory (contains documents/ and questions.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write analysis artifacts.",
    )
    parser.add_argument(
        "--warmup-dir",
        type=Path,
        default=None,
        help="Override warmup dataset directory (default: auto-detect).",
    )
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load private dataset --
    questions_path = dataset_dir / "questions.json"
    if not questions_path.is_file():
        # Fallback: try public_dataset.json (warmup structure variant)
        questions_path = dataset_dir / "public_dataset.json"
    if not questions_path.is_file():
        print(f"ERROR: no questions.json found in {dataset_dir}", file=sys.stderr)
        return 1

    print(f"Loading questions from {questions_path}")
    questions = _load_questions(questions_path)
    print(f"  Found {len(questions)} questions")

    doc_index = _load_doc_index(dataset_dir)
    pdf_doc_ids = _discover_pdf_doc_ids(dataset_dir)
    print(f"  Document index: {len(doc_index)} entries, PDFs found: {len(pdf_doc_ids)}")

    # -- Load warmup data (optional) --
    warmup_questions: list[dict[str, Any]] = []
    warmup_doc_index: dict[str, dict[str, Any]] = {}

    warmup_dir: Path | None = args.warmup_dir
    if warmup_dir is None:
        # Auto-detect warmup locations
        for candidate in [_REPO_ROOT / "dataset", _REPO_ROOT / "platform_runs" / "warmup"]:
            if candidate.is_dir():
                warmup_dir = candidate
                break

    if warmup_dir is not None and warmup_dir.is_dir():
        print(f"Loading warmup data from {warmup_dir}")
        warmup_questions = _load_warmup_questions()
        warmup_doc_index = _load_doc_index(warmup_dir)
        # Fallback: try repo-root index as warmup doc index
        if not warmup_doc_index:
            warmup_doc_index = _load_doc_index(_REPO_ROOT)
        print(f"  Warmup: {len(warmup_questions)} questions, {len(warmup_doc_index)} docs")
    else:
        print("No warmup data found -- skipping comparisons")

    # -- Build profiles --
    print("Building document profiles...")
    doc_profiles = build_document_profiles(doc_index, pdf_doc_ids, warmup_doc_index)

    print("Building query profiles...")
    query_profiles = build_query_profiles(questions, warmup_questions)

    # -- Write output 1: document_profile.json --
    doc_profile_path = output_dir / "document_profile.json"
    with open(doc_profile_path, "w", encoding="utf-8") as f:
        json.dump(doc_profiles, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {doc_profile_path} ({len(doc_profiles)} documents)")

    # -- Write output 2: query_distribution_report.json --
    query_report_path = output_dir / "query_distribution_report.json"
    with open(query_report_path, "w", encoding="utf-8") as f:
        json.dump(query_profiles, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {query_report_path} ({len(query_profiles)} questions)")

    # -- Write output 3: distribution_shift_report.md --
    print("Building distribution shift report...")
    shift_report = build_distribution_shift_report(
        query_profiles,
        warmup_questions,
        doc_profiles,
        warmup_doc_index,
    )
    shift_report_path = output_dir / "distribution_shift_report.md"
    with open(shift_report_path, "w", encoding="utf-8") as f:
        f.write(shift_report)
    print(f"  Wrote {shift_report_path}")

    # -- Write output 4: synthetic_training_triples.jsonl --
    print("Building synthetic training triples...")
    triples = build_synthetic_triples(questions, doc_index, query_profiles)
    triples_path = output_dir / "synthetic_training_triples.jsonl"
    with open(triples_path, "w", encoding="utf-8") as f:
        for triple in triples:
            f.write(json.dumps(triple, ensure_ascii=False) + "\n")
    print(f"  Wrote {triples_path} ({len(triples)} triples)")

    # -- Summary stats --
    print("\n--- Summary ---")
    print(f"Documents: {len(doc_profiles)}")
    print(f"Questions: {len(query_profiles)}")
    type_dist: dict[str, int] = {}
    for qp in query_profiles:
        t = str(qp["detected_answer_type"])
        type_dist[t] = type_dist.get(t, 0) + 1
    print(f"Answer types: {type_dist}")
    scope_dist: dict[str, int] = {}
    for qp in query_profiles:
        s = str(qp["detected_scope"])
        scope_dist[s] = scope_dist.get(s, 0) + 1
    print(f"Scope modes: {scope_dist}")
    print(f"Training triples: {len(triples)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
