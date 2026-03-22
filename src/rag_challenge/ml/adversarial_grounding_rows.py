"""Offline adversarial and near-miss grounding corpus builders."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import defaultdict
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rag_challenge.ml.grounding_dataset import GroundingMlRow


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TAXONOMY_HINTS: dict[str, tuple[str, ...]] = {
    "CASE_NUMBER": ("cfi", "ca ", "tcd", "arb", "claim no", "case"),
    "DATE": ("date", "when", "commencement", "enacted", "issued"),
    "PROVISION": ("article", "section", "schedule", "provision", "clause", "rule"),
    "STATUTE": ("law", "regulation", "regulations", "order"),
    "COURT": ("court", "tribunal", "registrar", "judge"),
    "PETITIONER": ("claimant", "claimants", "petitioner", "appellant", "applicant"),
    "RESPONDENT": ("respondent", "respondents", "defendant"),
    "ORG": ("authority", "company", "organisation", "corporation", "registrar"),
    "JUDGE": ("judge", "justice", "registrar"),
}


class AdversarialNegativeFamily(StrEnum):
    """Adversarial negative families for offline training rows."""

    SAME_DOC_NEARBY_PAGE = "same_doc_nearby_page"
    SAME_FAMILY_WRONG_LAW = "same_family_wrong_law"
    TITLE_AUTHORITY_CONFUSER = "title_authority_confuser"
    CONTRADICTION_UNSUPPORTED = "contradiction_unsupported"
    COMPACT_SET_LEVEL = "compact_set_level"


class AdversarialLineage(BaseModel):
    """Lineage bundle for one adversarial row."""

    source_question_id: str
    source_paths: dict[str, str] = Field(default_factory=dict)
    source_positive_page_ids: list[str] = Field(default_factory=list)
    source_negative_page_ids: list[str] = Field(default_factory=list)
    source_doc_ref_signatures: list[str] = Field(default_factory=list)
    legal_ner_taxonomy_path: str = ""


class AdversarialGroundingRow(BaseModel):
    """One offline adversarial grounding row."""

    adversarial_id: str
    question_id: str
    question: str
    answer_type: str
    holdout_doc_family_key: str
    positive_page_ids: list[str] = Field(default_factory=list)
    negative_page_ids: list[str] = Field(default_factory=list)
    negative_family: AdversarialNegativeFamily
    taxonomy_labels: list[str] = Field(default_factory=list)
    doc_ref_signatures: list[str] = Field(default_factory=list)
    lineage: AdversarialLineage


class AdversarialCorpusManifest(BaseModel):
    """Manifest describing one adversarial export bundle."""

    row_count: int
    train_count: int
    dev_count: int
    negative_family_counts: dict[str, int]
    holdout_family_count: int
    lineage_complete: bool
    legal_ner_taxonomy_path: str = ""


def build_adversarial_grounding_rows(
    rows: Sequence[GroundingMlRow],
    *,
    legal_ner_taxonomy_path: Path | None = None,
) -> list[AdversarialGroundingRow]:
    """Build offline adversarial rows from grounding export rows."""

    taxonomy_labels = load_legal_ner_taxonomy_labels(legal_ner_taxonomy_path)
    by_holdout_key: dict[str, list[GroundingMlRow]] = defaultdict(list)
    for row in rows:
        by_holdout_key[row.holdout_doc_family_key].append(row)

    adversarial_rows: list[AdversarialGroundingRow] = []
    for row in sorted(rows, key=lambda item: item.question_id):
        positive_page_ids = _positive_page_ids(row)
        if not positive_page_ids and row.scope_mode != "negative_unanswerable":
            continue
        adversarial_rows.extend(
            _generate_family_rows(
                row=row,
                cohort=by_holdout_key[row.holdout_doc_family_key],
                taxonomy_labels=taxonomy_labels,
                legal_ner_taxonomy_path=legal_ner_taxonomy_path,
            )
        )
    return sorted(adversarial_rows, key=lambda item: (item.question_id, item.negative_family.value, item.adversarial_id))


def load_legal_ner_taxonomy_labels(legal_ner_taxonomy_path: Path | None) -> list[str]:
    """Load optional taxonomy labels from the Legal NER dataset."""

    if legal_ner_taxonomy_path is None:
        return []
    data_dir = legal_ner_taxonomy_path / "data"
    labels: set[str] = set()
    for path in sorted(data_dir.glob("*_class_labels.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, list):
            payload_items = cast("list[object]", payload)
            labels.update(str(item).strip().upper() for item in payload_items if str(item).strip())
    return sorted(labels)


def derive_taxonomy_labels(
    row: GroundingMlRow,
    *,
    available_labels: Sequence[str],
) -> list[str]:
    """Derive weak taxonomy labels for a grounding row."""

    if not available_labels:
        return []
    question_blob = _normalize_text(" ".join([row.question, *row.doc_ref_signatures, row.answer_type]))
    derived: list[str] = []
    for label in available_labels:
        hints = _TAXONOMY_HINTS.get(label, ())
        if any(hint in question_blob for hint in hints):
            derived.append(label)
    return derived


def write_adversarial_grounding_rows(path: Path, rows: Sequence[AdversarialGroundingRow]) -> None:
    """Write adversarial rows as deterministic JSONL."""

    lines = [json.dumps(row.model_dump(mode="json"), ensure_ascii=True, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_adversarial_manifest(
    *,
    train_rows: Sequence[AdversarialGroundingRow],
    dev_rows: Sequence[AdversarialGroundingRow],
    legal_ner_taxonomy_path: Path | None,
) -> AdversarialCorpusManifest:
    """Build manifest metadata for an adversarial export."""

    all_rows = [*train_rows, *dev_rows]
    negative_family_counts: dict[str, int] = {}
    for row in all_rows:
        negative_family_counts[row.negative_family.value] = negative_family_counts.get(row.negative_family.value, 0) + 1
    return AdversarialCorpusManifest(
        row_count=len(all_rows),
        train_count=len(train_rows),
        dev_count=len(dev_rows),
        negative_family_counts=dict(sorted(negative_family_counts.items())),
        holdout_family_count=len({row.holdout_doc_family_key for row in all_rows}),
        lineage_complete=all(bool(row.lineage.source_question_id and row.lineage.source_paths) for row in all_rows),
        legal_ner_taxonomy_path=str(legal_ner_taxonomy_path) if legal_ner_taxonomy_path is not None else "",
    )


def _generate_family_rows(
    *,
    row: GroundingMlRow,
    cohort: Sequence[GroundingMlRow],
    taxonomy_labels: Sequence[str],
    legal_ner_taxonomy_path: Path | None,
) -> list[AdversarialGroundingRow]:
    positive_page_ids = _positive_page_ids(row)
    positive_doc_ids = {page_id.rpartition("_")[0] for page_id in positive_page_ids}
    row_taxonomy = derive_taxonomy_labels(row, available_labels=taxonomy_labels)
    generated: list[AdversarialGroundingRow] = []

    same_doc_negatives = _same_doc_nearby_page_negatives(row=row, positive_page_ids=positive_page_ids)
    if same_doc_negatives:
        generated.append(
            _build_adversarial_row(
                row=row,
                negative_family=AdversarialNegativeFamily.SAME_DOC_NEARBY_PAGE,
                negative_page_ids=same_doc_negatives,
                taxonomy_labels=row_taxonomy,
                legal_ner_taxonomy_path=legal_ner_taxonomy_path,
            )
        )

    wrong_law_negatives = [
        candidate.page_id
        for candidate in row.page_candidates
        if candidate.doc_id and candidate.doc_id not in positive_doc_ids and candidate.page_id not in positive_page_ids
    ][:2]
    if wrong_law_negatives:
        generated.append(
            _build_adversarial_row(
                row=row,
                negative_family=AdversarialNegativeFamily.SAME_FAMILY_WRONG_LAW,
                negative_page_ids=wrong_law_negatives,
                taxonomy_labels=row_taxonomy,
                legal_ner_taxonomy_path=legal_ner_taxonomy_path,
            )
        )

    confuser_negatives = _title_authority_confuser_negatives(
        row=row,
        cohort=cohort,
        positive_doc_ids=positive_doc_ids,
    )
    if confuser_negatives:
        generated.append(
            _build_adversarial_row(
                row=row,
                negative_family=AdversarialNegativeFamily.TITLE_AUTHORITY_CONFUSER,
                negative_page_ids=confuser_negatives,
                taxonomy_labels=row_taxonomy,
                legal_ner_taxonomy_path=legal_ner_taxonomy_path,
            )
        )

    contradiction_negatives = _contradiction_unsupported_negatives(row=row, positive_page_ids=positive_page_ids)
    if contradiction_negatives:
        generated.append(
            _build_adversarial_row(
                row=row,
                negative_family=AdversarialNegativeFamily.CONTRADICTION_UNSUPPORTED,
                negative_page_ids=contradiction_negatives,
                taxonomy_labels=row_taxonomy,
                legal_ner_taxonomy_path=legal_ner_taxonomy_path,
            )
        )

    compact_negatives = _compact_set_level_negatives(generated)
    if compact_negatives:
        generated.append(
            _build_adversarial_row(
                row=row,
                negative_family=AdversarialNegativeFamily.COMPACT_SET_LEVEL,
                negative_page_ids=compact_negatives,
                taxonomy_labels=row_taxonomy,
                legal_ner_taxonomy_path=legal_ner_taxonomy_path,
            )
        )
    return generated


def _build_adversarial_row(
    *,
    row: GroundingMlRow,
    negative_family: AdversarialNegativeFamily,
    negative_page_ids: Sequence[str],
    taxonomy_labels: Sequence[str],
    legal_ner_taxonomy_path: Path | None,
) -> AdversarialGroundingRow:
    positive_page_ids = _positive_page_ids(row)
    negative_ids = list(dict.fromkeys(page_id for page_id in negative_page_ids if page_id and page_id not in positive_page_ids))
    identity = "|".join([row.question_id, negative_family.value, *negative_ids])
    adversarial_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return AdversarialGroundingRow(
        adversarial_id=adversarial_id,
        question_id=row.question_id,
        question=row.question,
        answer_type=row.answer_type,
        holdout_doc_family_key=row.holdout_doc_family_key,
        positive_page_ids=positive_page_ids,
        negative_page_ids=negative_ids,
        negative_family=negative_family,
        taxonomy_labels=list(taxonomy_labels),
        doc_ref_signatures=list(row.doc_ref_signatures),
        lineage=AdversarialLineage(
            source_question_id=row.question_id,
            source_paths=dict(row.source_paths),
            source_positive_page_ids=positive_page_ids,
            source_negative_page_ids=negative_ids,
            source_doc_ref_signatures=list(row.doc_ref_signatures),
            legal_ner_taxonomy_path=str(legal_ner_taxonomy_path) if legal_ner_taxonomy_path is not None else "",
        ),
    )


def _same_doc_nearby_page_negatives(
    *,
    row: GroundingMlRow,
    positive_page_ids: Sequence[str],
) -> list[str]:
    positive_pages = {_parse_page_id(page_id) for page_id in positive_page_ids}
    candidates: list[tuple[int, str]] = []
    for candidate in row.page_candidates:
        if candidate.page_id in positive_page_ids:
            continue
        for positive_doc_id, positive_page_num in positive_pages:
            if candidate.doc_id != positive_doc_id:
                continue
            candidates.append((abs(candidate.page_num - positive_page_num), candidate.page_id))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [page_id for _, page_id in candidates[:2]]


def _title_authority_confuser_negatives(
    *,
    row: GroundingMlRow,
    cohort: Sequence[GroundingMlRow],
    positive_doc_ids: set[str],
) -> list[str]:
    query_tokens = set(_normalize_text(" ".join(row.doc_ref_signatures or [row.question])).split())
    confuser_pages: list[str] = []
    for other in sorted(cohort, key=lambda item: item.question_id):
        if other.question_id == row.question_id:
            continue
        other_positive_pages = _positive_page_ids(other)
        other_doc_ids = {page_id.rpartition("_")[0] for page_id in other_positive_pages}
        if other_doc_ids & positive_doc_ids:
            continue
        other_tokens = set(_normalize_text(" ".join(other.doc_ref_signatures or [other.question])).split())
        if not (query_tokens & other_tokens):
            continue
        confuser_pages.extend(other_positive_pages[:1])
        if len(confuser_pages) >= 2:
            break
    return confuser_pages[:2]


def _contradiction_unsupported_negatives(
    *,
    row: GroundingMlRow,
    positive_page_ids: Sequence[str],
) -> list[str]:
    if row.scope_mode == "negative_unanswerable":
        return [candidate.page_id for candidate in row.page_candidates[:2]]
    if positive_page_ids:
        return []
    fallback = row.sidecar_selected_pages or row.legacy_selected_pages
    return [page_id for page_id in fallback[:2]]


def _compact_set_level_negatives(rows: Sequence[AdversarialGroundingRow]) -> list[str]:
    negative_page_ids: list[str] = []
    for row in rows:
        if row.negative_page_ids:
            negative_page_ids.append(row.negative_page_ids[0])
    return list(dict.fromkeys(negative_page_ids))


def _positive_page_ids(row: GroundingMlRow) -> list[str]:
    return list(row.label_page_ids or row.sidecar_selected_pages or row.legacy_selected_pages)


def _parse_page_id(page_id: str) -> tuple[str, int]:
    doc_id, _, page_num_text = page_id.rpartition("_")
    page_num = int(page_num_text) if page_num_text.isdigit() else 0
    return doc_id, page_num


def _normalize_text(text: str) -> str:
    return " ".join(token.lower() for token in _TOKEN_RE.findall(text))
