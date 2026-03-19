# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
"""Deterministic dataset export helpers for grounding-sidecar ML prep."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from rag_challenge.core.grounding.query_scope_classifier import classify_query_scope

if TYPE_CHECKING:
    from pathlib import Path

LabelSource = Literal["reviewed", "soft_ai_gold", "suspect_ai_gold"]


class PageCandidateRecord(BaseModel):
    """Feature record for one candidate grounding page.

    Args:
        page_id: Competition page identifier.
        doc_id: Parent document identifier.
        page_num: 1-based page number.
        candidate_sources: Source lists that contributed this page candidate.
        legacy_retrieved_rank: Rank in legacy retrieved pages, if present.
        legacy_context_rank: Rank in legacy context pages, if present.
        legacy_cited_rank: Rank in legacy cited pages, if present.
        sidecar_retrieved_rank: Rank in sidecar retrieved pages, if present.
        sidecar_context_rank: Rank in sidecar context pages, if present.
        sidecar_cited_rank: Rank in sidecar cited pages, if present.
        anchor_hits: Query anchors detected in page snippets.
        snippet_excerpt: Short snippet union for manual review/debug.
    """

    page_id: str
    doc_id: str
    page_num: int
    candidate_sources: list[str] = Field(default_factory=list)
    legacy_retrieved_rank: int | None = None
    legacy_context_rank: int | None = None
    legacy_cited_rank: int | None = None
    sidecar_retrieved_rank: int | None = None
    sidecar_context_rank: int | None = None
    sidecar_cited_rank: int | None = None
    anchor_hits: list[str] = Field(default_factory=list)
    snippet_excerpt: str = ""


class DocCandidateRecord(BaseModel):
    """Aggregate document-level candidate record."""

    doc_id: str
    page_candidate_count: int
    candidate_sources: list[str] = Field(default_factory=list)
    legacy_selected: bool = False
    sidecar_selected: bool = False


class SupportFactFeatureRecord(BaseModel):
    """Classifier- and anchor-derived support priors for later models."""

    requires_all_docs_in_case: bool = False
    should_force_empty_grounding_on_null: bool = False
    explicit_anchor_count: int = 0
    target_page_roles_count: int = 0
    doc_ref_count: int = 0


class PageRetrievalFeatureRecord(BaseModel):
    """Summary retrieval features from legacy and sidecar telemetry."""

    legacy_retrieved_page_count: int = 0
    legacy_context_page_count: int = 0
    legacy_cited_page_count: int = 0
    sidecar_retrieved_page_count: int = 0
    sidecar_context_page_count: int = 0
    sidecar_cited_page_count: int = 0
    legacy_sidecar_used_overlap_count: int = 0


class GroundingMlRow(BaseModel):
    """One deterministic export row for grounding ML experiments."""

    question_id: str
    question: str
    answer_type: str
    golden_answer: str | bool | int | float | None = None
    label_page_ids: list[str] = Field(default_factory=list)
    label_source: LabelSource
    label_trust_tier: str = ""
    label_confidence: str = ""
    label_status: str = ""
    label_weight: float = 0.0
    label_note_present: bool = False
    scope_mode: str
    target_page_roles: list[str] = Field(default_factory=list)
    hard_anchor_strings: list[str] = Field(default_factory=list)
    doc_candidates: list[DocCandidateRecord] = Field(default_factory=list)
    page_candidates: list[PageCandidateRecord] = Field(default_factory=list)
    legacy_selected_pages: list[str] = Field(default_factory=list)
    sidecar_selected_pages: list[str] = Field(default_factory=list)
    support_fact_features: SupportFactFeatureRecord
    page_retrieval_features: PageRetrievalFeatureRecord
    label_is_suspect: bool = False
    source_paths: dict[str, str] = Field(default_factory=dict)


class ExportManifest(BaseModel):
    """Metadata manifest for one grounding ML export bundle."""

    generated_at: str
    split_seed: int
    dev_ratio: float
    source_paths: dict[str, str]
    row_count: int
    train_count: int
    dev_count: int
    label_source_counts: dict[str, int]
    reviewed_slice_counts: dict[str, int] = Field(default_factory=dict)
    label_confidence_counts: dict[str, int] = Field(default_factory=dict)


class _NormalizedRawRow(BaseModel):
    """Internal normalized view of one raw-results row."""

    question_id: str
    question: str
    answer_type: str
    answer_text: str
    source_path: str
    doc_refs: list[str] = Field(default_factory=list)
    retrieved_page_ids: list[str] = Field(default_factory=list)
    context_page_ids: list[str] = Field(default_factory=list)
    cited_page_ids: list[str] = Field(default_factory=list)
    used_page_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    chunk_snippets: dict[str, str] = Field(default_factory=dict)


class _LabelRecord(BaseModel):
    """Internal label record with provenance."""

    golden_answer: str | bool | int | float | None = None
    page_ids: list[str] = Field(default_factory=list)
    label_source: LabelSource
    trust_tier: str = ""
    confidence: str = ""
    label_status: str = ""
    label_weight: float = 0.0
    audit_note: str = ""
    current_label_problem: str = ""
    label_is_suspect: bool = False


class _ReviewedManifestMetadata(TypedDict):
    """Typed manifest metadata derived from reviewed benchmark artifacts."""

    source_paths: dict[str, str]
    slice_counts: dict[str, int]
    confidence_counts: dict[str, int]


def export_grounding_ml_dataset(
    *,
    legacy_raw_results_path: Path,
    sidecar_raw_results_path: Path,
    golden_labels_path: Path,
    page_benchmark_path: Path,
    suspect_labels_path: Path | None,
    output_dir: Path,
    split_seed: int = 601,
    dev_ratio: float = 0.2,
    reviewed_labels_path: Path | None = None,
) -> ExportManifest:
    """Export a deterministic grounding-ML dataset bundle.

    Args:
        legacy_raw_results_path: Path to legacy raw-results artifact.
        sidecar_raw_results_path: Path to sidecar raw-results artifact.
        golden_labels_path: Path to answer/page gold labels.
        page_benchmark_path: Path to page benchmark labels.
        suspect_labels_path: Optional path to matched suspect-label audit.
        output_dir: Target export directory.
        split_seed: Deterministic split seed.
        dev_ratio: Fraction of rows to place in the dev split.
        reviewed_labels_path: Optional reviewed-label file path.

    Returns:
        Export manifest describing the generated bundle.
    """
    legacy_rows = _load_raw_results(legacy_raw_results_path)
    sidecar_rows = _load_raw_results(sidecar_raw_results_path)
    labels = _load_label_records(
        golden_labels_path=golden_labels_path,
        page_benchmark_path=page_benchmark_path,
        suspect_labels_path=suspect_labels_path,
        reviewed_labels_path=reviewed_labels_path,
    )
    rows = build_grounding_ml_rows(
        legacy_rows=legacy_rows,
        sidecar_rows=sidecar_rows,
        label_records=labels,
    )
    train_rows, dev_rows = split_grounding_ml_rows(rows, split_seed=split_seed, dev_ratio=dev_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "dev.jsonl", dev_rows)

    failure_canaries = _build_failure_canaries(rows)
    (output_dir / "failure_canaries.json").write_text(
        json.dumps(failure_canaries, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "feature_inventory.md").write_text(
        _build_feature_inventory(rows=rows),
        encoding="utf-8",
    )

    reviewed_metadata = _build_reviewed_manifest_metadata(
        reviewed_labels_path=reviewed_labels_path,
        page_benchmark_path=page_benchmark_path,
        rows=rows,
    )

    manifest = ExportManifest(
        generated_at=datetime.now(UTC).isoformat(),
        split_seed=split_seed,
        dev_ratio=dev_ratio,
        source_paths={
            "legacy_raw_results": str(legacy_raw_results_path),
            "sidecar_raw_results": str(sidecar_raw_results_path),
            "golden_labels": str(golden_labels_path),
            "page_benchmark": str(page_benchmark_path),
            "suspect_labels": str(suspect_labels_path) if suspect_labels_path is not None else "",
            "reviewed_labels": str(reviewed_labels_path) if reviewed_labels_path is not None else "",
            **reviewed_metadata["source_paths"],
        },
        row_count=len(rows),
        train_count=len(train_rows),
        dev_count=len(dev_rows),
        label_source_counts=_count_label_sources(rows),
        reviewed_slice_counts=reviewed_metadata["slice_counts"],
        label_confidence_counts=reviewed_metadata["confidence_counts"],
    )
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_reviewed_manifest_metadata(
    *,
    reviewed_labels_path: Path | None,
    page_benchmark_path: Path,
    rows: list[GroundingMlRow],
) -> _ReviewedManifestMetadata:
    """Build reviewed-lineage metadata for the export manifest.

    Args:
        reviewed_labels_path: Optional reviewed-label slice used for export.
        page_benchmark_path: Page-benchmark slice used for export.
        rows: Final exported rows.

    Returns:
        Mapping containing reviewed source paths, slice counts, and confidence counts.
    """
    empty: _ReviewedManifestMetadata = {
        "source_paths": {},
        "slice_counts": {},
        "confidence_counts": {},
    }
    if reviewed_labels_path is None or not reviewed_labels_path.exists():
        return empty

    reviewed_dir = reviewed_labels_path.parent
    import_manifest_path = reviewed_dir / "import_manifest.json"
    reviewed_source_paths = {
        "reviewed_import_manifest": str(import_manifest_path) if import_manifest_path.exists() else "",
        "reviewed_all_100": str(reviewed_dir / "reviewed_all_100.json"),
        "reviewed_high_confidence_81": str(reviewed_dir / "reviewed_high_confidence_81.json"),
        "reviewed_medium_plus_high_95": str(reviewed_dir / "reviewed_medium_plus_high_95.json"),
        "reviewed_page_benchmark_all_100": str(reviewed_dir / "reviewed_page_benchmark_all_100.json"),
        "reviewed_page_benchmark_high_confidence_81": str(reviewed_dir / "reviewed_page_benchmark_high_confidence_81.json"),
        "reviewed_page_benchmark_medium_plus_high_95": str(reviewed_dir / "reviewed_page_benchmark_medium_plus_high_95.json"),
        "active_reviewed_page_benchmark": str(page_benchmark_path),
    }

    if import_manifest_path.exists():
        manifest_payload = json.loads(import_manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest_payload, dict):
            slice_counts = _coerce_str_int_dict(manifest_payload.get("slice_counts"))
            confidence_counts = _coerce_str_int_dict(manifest_payload.get("confidence_counts"))
            return {
                "source_paths": reviewed_source_paths,
                "slice_counts": slice_counts,
                "confidence_counts": confidence_counts,
            }

    confidence_counts: dict[str, int] = {}
    for row in rows:
        if row.label_source != "reviewed" or not row.label_confidence:
            continue
        confidence_counts[row.label_confidence] = confidence_counts.get(row.label_confidence, 0) + 1

    reviewed_all_path = reviewed_dir / "reviewed_all_100.json"
    reviewed_high_path = reviewed_dir / "reviewed_high_confidence_81.json"
    reviewed_medium_path = reviewed_dir / "reviewed_medium_plus_high_95.json"
    slice_counts = {
        "reviewed_all_100": _load_reviewed_slice_count(reviewed_all_path),
        "reviewed_high_confidence_81": _load_reviewed_slice_count(reviewed_high_path),
        "reviewed_medium_plus_high_95": _load_reviewed_slice_count(reviewed_medium_path),
    }
    return {
        "source_paths": reviewed_source_paths,
        "slice_counts": slice_counts,
        "confidence_counts": dict(sorted(confidence_counts.items())),
    }


def build_grounding_ml_rows(
    *,
    legacy_rows: dict[str, _NormalizedRawRow],
    sidecar_rows: dict[str, _NormalizedRawRow],
    label_records: dict[str, _LabelRecord],
) -> list[GroundingMlRow]:
    """Build deterministic export rows from raw artifacts and labels.

    Args:
        legacy_rows: Legacy raw-results keyed by question ID.
        sidecar_rows: Sidecar raw-results keyed by question ID.
        label_records: Label records keyed by question ID.

    Returns:
        Sorted grounding-ML rows.
    """
    question_ids = sorted(set(legacy_rows) | set(sidecar_rows))
    rows: list[GroundingMlRow] = []
    for question_id in question_ids:
        legacy = legacy_rows.get(question_id)
        sidecar = sidecar_rows.get(question_id)
        reference = sidecar or legacy
        if reference is None:
            continue
        scope = classify_query_scope(reference.question, reference.answer_type)
        label = label_records.get(
            question_id,
            _LabelRecord(label_source="suspect_ai_gold", label_is_suspect=True),
        )
        page_candidates = _build_page_candidates(
            legacy=legacy,
            sidecar=sidecar,
            hard_anchor_strings=scope.hard_anchor_strings,
        )
        doc_candidates = _build_doc_candidates(
            page_candidates=page_candidates,
            legacy_selected_pages=list(legacy.used_page_ids if legacy is not None else []),
            sidecar_selected_pages=list(sidecar.used_page_ids if sidecar is not None else []),
        )
        rows.append(
            GroundingMlRow(
                question_id=question_id,
                question=reference.question,
                answer_type=reference.answer_type,
                golden_answer=label.golden_answer,
                label_page_ids=list(label.page_ids),
                label_source=label.label_source,
                label_trust_tier=label.trust_tier,
                label_confidence=label.confidence,
                label_status=label.label_status,
                label_weight=label.label_weight,
                label_note_present=bool(label.audit_note or label.current_label_problem),
                scope_mode=scope.scope_mode.value,
                target_page_roles=list(scope.target_page_roles),
                hard_anchor_strings=list(scope.hard_anchor_strings),
                doc_candidates=doc_candidates,
                page_candidates=page_candidates,
                legacy_selected_pages=list(legacy.used_page_ids if legacy is not None else []),
                sidecar_selected_pages=list(sidecar.used_page_ids if sidecar is not None else []),
                support_fact_features=SupportFactFeatureRecord(
                    requires_all_docs_in_case=scope.requires_all_docs_in_case,
                    should_force_empty_grounding_on_null=scope.should_force_empty_grounding_on_null,
                    explicit_anchor_count=len(scope.hard_anchor_strings),
                    target_page_roles_count=len(scope.target_page_roles),
                    doc_ref_count=len(reference.doc_refs),
                ),
                page_retrieval_features=PageRetrievalFeatureRecord(
                    legacy_retrieved_page_count=len(legacy.retrieved_page_ids) if legacy is not None else 0,
                    legacy_context_page_count=len(legacy.context_page_ids) if legacy is not None else 0,
                    legacy_cited_page_count=len(legacy.cited_page_ids) if legacy is not None else 0,
                    sidecar_retrieved_page_count=len(sidecar.retrieved_page_ids) if sidecar is not None else 0,
                    sidecar_context_page_count=len(sidecar.context_page_ids) if sidecar is not None else 0,
                    sidecar_cited_page_count=len(sidecar.cited_page_ids) if sidecar is not None else 0,
                    legacy_sidecar_used_overlap_count=len(
                        set(legacy.used_page_ids if legacy is not None else [])
                        & set(sidecar.used_page_ids if sidecar is not None else [])
                    ),
                ),
                label_is_suspect=label.label_is_suspect,
                source_paths={
                    "legacy_raw_results": legacy.source_path if legacy is not None else "",
                    "sidecar_raw_results": sidecar.source_path if sidecar is not None else "",
                },
            )
        )
    return rows


def split_grounding_ml_rows(
    rows: list[GroundingMlRow],
    *,
    split_seed: int,
    dev_ratio: float,
) -> tuple[list[GroundingMlRow], list[GroundingMlRow]]:
    """Split rows deterministically into train/dev sets.

    Args:
        rows: Full export row list.
        split_seed: Deterministic split seed.
        dev_ratio: Fraction of rows to reserve for development.

    Returns:
        Tuple of train rows and dev rows.
    """
    if not rows:
        return [], []
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{split_seed}:{row.question_id}".encode()).hexdigest(),
            row.question_id,
        ),
    )
    dev_count = max(1, min(len(ordered) - 1, round(len(ordered) * dev_ratio))) if len(ordered) > 1 else 0
    dev_rows = ordered[:dev_count]
    train_rows = ordered[dev_count:]
    return train_rows, dev_rows


def _load_raw_results(path: Path) -> dict[str, _NormalizedRawRow]:
    """Load and normalize raw-results JSON into a question-keyed map."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else list(data.get("results") or data.get("rows") or [])
    normalized: dict[str, _NormalizedRawRow] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        case = item.get("case", {})
        telemetry = item.get("telemetry", {})
        if not isinstance(case, dict) or not isinstance(telemetry, dict):
            continue
        question_id = _coerce_str(case.get("question_id")) or _coerce_str(case.get("case_id")) or _coerce_str(
            telemetry.get("question_id")
        )
        question = _coerce_str(case.get("question"))
        answer_type = _coerce_str(case.get("answer_type")) or _coerce_str(telemetry.get("answer_type")) or "free_text"
        if not question_id or not question:
            continue
        normalized[question_id] = _NormalizedRawRow(
            question_id=question_id,
            question=question,
            answer_type=answer_type,
            answer_text=_coerce_str(item.get("answer_text")),
            source_path=str(path),
            doc_refs=_coerce_str_list(telemetry.get("doc_refs")),
            retrieved_page_ids=_coerce_str_list(telemetry.get("retrieved_page_ids")),
            context_page_ids=_coerce_str_list(telemetry.get("context_page_ids")),
            cited_page_ids=_coerce_str_list(telemetry.get("cited_page_ids")),
            used_page_ids=_coerce_str_list(telemetry.get("used_page_ids")),
            retrieved_chunk_ids=_coerce_str_list(telemetry.get("retrieved_chunk_ids")),
            context_chunk_ids=_coerce_str_list(telemetry.get("context_chunk_ids")),
            cited_chunk_ids=_coerce_str_list(telemetry.get("cited_chunk_ids")),
            chunk_snippets=_coerce_str_dict(telemetry.get("chunk_snippets")),
        )
    return normalized


def _load_label_records(
    *,
    golden_labels_path: Path,
    page_benchmark_path: Path,
    suspect_labels_path: Path | None,
    reviewed_labels_path: Path | None,
) -> dict[str, _LabelRecord]:
    """Load label provenance and page IDs for export."""
    label_rows: dict[str, _LabelRecord] = {}

    soft_answers = json.loads(golden_labels_path.read_text(encoding="utf-8"))
    if isinstance(soft_answers, list):
        for item in soft_answers:
            if not isinstance(item, dict):
                continue
            question_id = _coerce_str(item.get("question_id"))
            if not question_id:
                continue
            label_rows[question_id] = _LabelRecord(
                golden_answer=_coerce_scalar_answer(item.get("golden_answer")),
                page_ids=_coerce_str_list(item.get("golden_page_ids")),
                label_source="soft_ai_gold",
                trust_tier="",
                confidence=_coerce_str(item.get("confidence")),
                label_status=_coerce_str(item.get("label_status")),
                label_weight=_coerce_label_weight(item.get("label_weight"), fallback_confidence=_coerce_str(item.get("confidence"))),
                audit_note=_coerce_str(item.get("audit_note")),
                current_label_problem=_coerce_str(item.get("current_label_problem")),
                label_is_suspect=False,
            )

    benchmark = json.loads(page_benchmark_path.read_text(encoding="utf-8"))
    if isinstance(benchmark, dict):
        for item in benchmark.get("cases", []):
            if not isinstance(item, dict):
                continue
            question_id = _coerce_str(item.get("question_id"))
            if not question_id:
                continue
            existing = label_rows.get(question_id, _LabelRecord(label_source="soft_ai_gold"))
            label_rows[question_id] = existing.model_copy(
                update={
                    "page_ids": _coerce_str_list(item.get("gold_page_ids")) or existing.page_ids,
                    "trust_tier": _coerce_str(item.get("trust_tier")),
                }
            )

    suspect_ids: set[str] = set()
    if suspect_labels_path is not None and suspect_labels_path.exists():
        suspect_data = json.loads(suspect_labels_path.read_text(encoding="utf-8"))
        rows = suspect_data.get("rows", []) if isinstance(suspect_data, dict) else []
        for item in rows:
            if isinstance(item, dict):
                question_id = _coerce_str(item.get("question_id"))
                if question_id:
                    suspect_ids.add(question_id)

    for question_id in suspect_ids:
        existing = label_rows.get(question_id, _LabelRecord(label_source="suspect_ai_gold", label_is_suspect=True))
        label_rows[question_id] = existing.model_copy(
            update={"label_source": "suspect_ai_gold", "label_is_suspect": True}
        )

    if reviewed_labels_path is not None and reviewed_labels_path.exists():
        reviewed_data = json.loads(reviewed_labels_path.read_text(encoding="utf-8"))
        reviewed_items = reviewed_data if isinstance(reviewed_data, list) else reviewed_data.get("cases", [])
        for item in reviewed_items:
            if not isinstance(item, dict):
                continue
            question_id = _coerce_str(item.get("question_id"))
            if not question_id:
                continue
            label_rows[question_id] = _LabelRecord(
                golden_answer=_coerce_scalar_answer(item.get("golden_answer")),
                page_ids=_coerce_str_list(item.get("golden_page_ids") or item.get("gold_page_ids")),
                label_source="reviewed",
                trust_tier=_coerce_str(item.get("trust_tier")) or _coerce_str(item.get("confidence")),
                confidence=_coerce_str(item.get("confidence")),
                label_status=_coerce_str(item.get("label_status")),
                label_weight=_coerce_label_weight(item.get("label_weight"), fallback_confidence=_coerce_str(item.get("confidence"))),
                audit_note=_coerce_str(item.get("audit_note")),
                current_label_problem=_coerce_str(item.get("current_label_problem")),
                label_is_suspect=False,
            )

    return label_rows


def _build_page_candidates(
    *,
    legacy: _NormalizedRawRow | None,
    sidecar: _NormalizedRawRow | None,
    hard_anchor_strings: list[str],
) -> list[PageCandidateRecord]:
    """Build page candidate features from paired raw-results telemetry."""
    page_ids = _ordered_unique(
        [
            *(legacy.retrieved_page_ids if legacy is not None else []),
            *(legacy.context_page_ids if legacy is not None else []),
            *(legacy.cited_page_ids if legacy is not None else []),
            *(legacy.used_page_ids if legacy is not None else []),
            *(sidecar.retrieved_page_ids if sidecar is not None else []),
            *(sidecar.context_page_ids if sidecar is not None else []),
            *(sidecar.cited_page_ids if sidecar is not None else []),
            *(sidecar.used_page_ids if sidecar is not None else []),
        ]
    )
    chunk_snippets = {}
    if legacy is not None:
        chunk_snippets.update(legacy.chunk_snippets)
    if sidecar is not None:
        chunk_snippets.update(sidecar.chunk_snippets)

    rows: list[PageCandidateRecord] = []
    for page_id in page_ids:
        doc_id, page_num = _split_page_id(page_id)
        sources: list[str] = []
        snippet_excerpt = _page_snippet_excerpt(
            page_id=page_id,
            chunk_snippets=chunk_snippets,
            chunk_id_lists=[
                legacy.retrieved_chunk_ids if legacy is not None else [],
                legacy.context_chunk_ids if legacy is not None else [],
                legacy.cited_chunk_ids if legacy is not None else [],
                sidecar.retrieved_chunk_ids if sidecar is not None else [],
                sidecar.context_chunk_ids if sidecar is not None else [],
                sidecar.cited_chunk_ids if sidecar is not None else [],
            ],
        )
        anchor_hits = [
            anchor
            for anchor in hard_anchor_strings
            if anchor and anchor.casefold() in snippet_excerpt.casefold()
        ]
        legacy_retrieved_rank = _rank_in_list(legacy.retrieved_page_ids if legacy is not None else [], page_id)
        legacy_context_rank = _rank_in_list(legacy.context_page_ids if legacy is not None else [], page_id)
        legacy_cited_rank = _rank_in_list(legacy.cited_page_ids if legacy is not None else [], page_id)
        sidecar_retrieved_rank = _rank_in_list(sidecar.retrieved_page_ids if sidecar is not None else [], page_id)
        sidecar_context_rank = _rank_in_list(sidecar.context_page_ids if sidecar is not None else [], page_id)
        sidecar_cited_rank = _rank_in_list(sidecar.cited_page_ids if sidecar is not None else [], page_id)
        if legacy_retrieved_rank is not None:
            sources.append("legacy_retrieved")
        if legacy_context_rank is not None:
            sources.append("legacy_context")
        if legacy_cited_rank is not None:
            sources.append("legacy_cited")
        if legacy is not None and page_id in legacy.used_page_ids:
            sources.append("legacy_used")
        if sidecar_retrieved_rank is not None:
            sources.append("sidecar_retrieved")
        if sidecar_context_rank is not None:
            sources.append("sidecar_context")
        if sidecar_cited_rank is not None:
            sources.append("sidecar_cited")
        if sidecar is not None and page_id in sidecar.used_page_ids:
            sources.append("sidecar_used")
        rows.append(
            PageCandidateRecord(
                page_id=page_id,
                doc_id=doc_id,
                page_num=page_num,
                candidate_sources=sources,
                legacy_retrieved_rank=legacy_retrieved_rank,
                legacy_context_rank=legacy_context_rank,
                legacy_cited_rank=legacy_cited_rank,
                sidecar_retrieved_rank=sidecar_retrieved_rank,
                sidecar_context_rank=sidecar_context_rank,
                sidecar_cited_rank=sidecar_cited_rank,
                anchor_hits=anchor_hits,
                snippet_excerpt=snippet_excerpt,
            )
        )
    return rows


def _build_doc_candidates(
    *,
    page_candidates: list[PageCandidateRecord],
    legacy_selected_pages: list[str],
    sidecar_selected_pages: list[str],
) -> list[DocCandidateRecord]:
    """Aggregate page candidates into document-level candidates."""
    grouped: dict[str, list[PageCandidateRecord]] = {}
    for page in page_candidates:
        grouped.setdefault(page.doc_id, []).append(page)

    doc_rows: list[DocCandidateRecord] = []
    for doc_id in sorted(grouped):
        pages = grouped[doc_id]
        candidate_sources = sorted({source for page in pages for source in page.candidate_sources})
        doc_rows.append(
            DocCandidateRecord(
                doc_id=doc_id,
                page_candidate_count=len(pages),
                candidate_sources=candidate_sources,
                legacy_selected=any(page_id.rpartition("_")[0] == doc_id for page_id in legacy_selected_pages),
                sidecar_selected=any(page_id.rpartition("_")[0] == doc_id for page_id in sidecar_selected_pages),
            )
        )
    return doc_rows


def _build_failure_canaries(rows: list[GroundingMlRow]) -> dict[str, Any]:
    """Build a compact list of suspicious or disagreement rows for manual review."""
    canaries = []
    for row in rows:
        legacy_set = set(row.legacy_selected_pages)
        sidecar_set = set(row.sidecar_selected_pages)
        label_hit = bool(set(row.label_page_ids) & {page.page_id for page in row.page_candidates})
        if row.label_is_suspect or legacy_set != sidecar_set or not label_hit:
            canaries.append(
                {
                    "question_id": row.question_id,
                    "label_source": row.label_source,
                    "label_page_ids": row.label_page_ids,
                    "legacy_selected_pages": row.legacy_selected_pages,
                    "sidecar_selected_pages": row.sidecar_selected_pages,
                    "candidate_label_overlap": label_hit,
                }
            )
    return {"case_count": len(canaries), "rows": canaries}


def _build_feature_inventory(*, rows: list[GroundingMlRow]) -> str:
    """Render a short feature inventory markdown file."""
    label_counts = _count_label_sources(rows)
    lines = [
        "# Grounding ML Export V1",
        "",
        f"- rows: `{len(rows)}`",
        f"- label sources: `{json.dumps(label_counts, sort_keys=True)}`",
        "- soft labels are preserved with provenance; current AI-generated page gold is not treated as hard truth.",
        "",
        "## Row schema",
        "",
        "- `question_id`, `question`, `answer_type`, `golden_answer`",
        "- `scope_mode`, `target_page_roles`, `hard_anchor_strings`",
        "- `doc_candidates`, `page_candidates`",
        "- `legacy_selected_pages`, `sidecar_selected_pages`",
        "- `support_fact_features`, `page_retrieval_features`",
        "- `label_source`, `label_trust_tier`, `label_confidence`, `label_status`, `label_weight`, `label_is_suspect`",
        "",
        "## Candidate feature notes",
        "",
        "- page candidates carry source membership and legacy/sidecar retrieved/context/cited ranks",
        "- anchor hits are derived from query anchors against exported chunk snippets for that page",
        "- document candidates are aggregated from the page candidate union",
        "",
    ]
    return "\n".join(lines) + "\n"


def _count_label_sources(rows: list[GroundingMlRow]) -> dict[str, int]:
    counts: dict[str, int] = {"reviewed": 0, "soft_ai_gold": 0, "suspect_ai_gold": 0}
    for row in rows:
        counts[row.label_source] = counts.get(row.label_source, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[GroundingMlRow]) -> None:
    """Write rows as deterministic JSONL."""
    lines = [json.dumps(row.model_dump(mode="json"), ensure_ascii=True, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _page_snippet_excerpt(
    *,
    page_id: str,
    chunk_snippets: dict[str, str],
    chunk_id_lists: list[list[str]],
) -> str:
    """Build a short combined snippet excerpt for a candidate page."""
    snippets: list[str] = []
    for chunk_ids in chunk_id_lists:
        for chunk_id in chunk_ids:
            if _chunk_id_to_page_id(chunk_id) != page_id:
                continue
            snippet = chunk_snippets.get(chunk_id, "").strip()
            if snippet:
                snippets.append(snippet)
    if not snippets:
        return ""
    combined = " | ".join(_ordered_unique(snippets))
    return combined[:280]


def _chunk_id_to_page_id(chunk_id: str) -> str:
    """Convert a starter-kit chunk ID into a page ID."""
    if ":" not in chunk_id and "_" in chunk_id:
        return chunk_id
    parts = chunk_id.split(":")
    if len(parts) < 2:
        return ""
    doc_id = parts[0].strip()
    page_idx = parts[1].strip()
    if not doc_id or not page_idx.isdigit():
        return ""
    return f"{doc_id}_{int(page_idx) + 1}"


def _split_page_id(page_id: str) -> tuple[str, int]:
    """Split a page ID into document ID and page number."""
    doc_id, _, page_str = str(page_id).rpartition("_")
    if not doc_id or not page_str.isdigit():
        return str(page_id), 0
    return doc_id, int(page_str)


def _rank_in_list(items: list[str], value: str) -> int | None:
    """Return 1-based rank of a value inside an ordered list."""
    for index, item in enumerate(items, start=1):
        if item == value:
            return index
    return None


def _ordered_unique(items: list[str]) -> list[str]:
    """Return ordered unique strings."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _coerce_str(value: object) -> str:
    """Coerce a raw object to a clean string."""
    return str(value or "").strip()


def _coerce_str_list(value: object) -> list[str]:
    """Coerce a raw object to a clean string list."""
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _coerce_str(item))]


def _coerce_str_dict(value: object) -> dict[str, str]:
    """Coerce a raw object to a clean string dict."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _coerce_str(raw_key)
        item_value = _coerce_str(raw_value)
        if key and item_value:
            result[key] = item_value
    return result


def _coerce_str_int_dict(value: object) -> dict[str, int]:
    """Coerce a raw mapping into a clean string-to-int dictionary.

    Args:
        value: Raw mapping candidate.

    Returns:
        Clean string-keyed integer mapping.
    """
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = _coerce_str(raw_key)
        if not key:
            continue
        if isinstance(raw_value, int | float) and not isinstance(raw_value, bool):
            result[key] = int(raw_value)
    return dict(sorted(result.items()))


def _coerce_scalar_answer(value: object) -> str | bool | int | float | None:
    """Coerce a golden answer into a JSON-safe scalar."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return value.strip()
    return None


def _coerce_label_weight(value: object, *, fallback_confidence: str) -> float:
    """Coerce an explicit or derived label weight.

    Args:
        value: Raw weight object.
        fallback_confidence: Confidence string used when weight is omitted.

    Returns:
        Normalized non-negative label weight.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, float(value))
    confidence = fallback_confidence.strip().lower()
    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.5
    return 0.0


def _load_reviewed_slice_count(path: Path) -> int:
    """Load one reviewed slice and return its row count.

    Args:
        path: Reviewed slice JSON path.

    Returns:
        Row count, or zero when the file is missing/unreadable.
    """
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        rows = payload.get("cases")
        if isinstance(rows, list):
            return len(rows)
    return 0
