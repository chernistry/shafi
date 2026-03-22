"""In-memory structured registry for compare/join execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_challenge.core.legal_title_family import title_key
from rag_challenge.models.legal_objects import (
    CaseObject,
    CorpusRegistry,
    LawObject,
    OrderObject,
    PracticeDirectionObject,
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CASE_JUDGE_HINT_RE = re.compile(
    r"\b(?:justice|judge|chief justice|h\.e\.|he\.|hon\.|honourable|sir|lady|lord)\b",
    re.IGNORECASE,
)
_CASE_JUDGE_BLOCKLIST_RE = re.compile(
    r"\b(?:assistant\s+registrar|registrar|claimant|claimants|respondent|respondents|appellant|appellants|"
    r"applicant|applicants|defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\b",
    re.IGNORECASE,
)


def _normalize_value(value: str) -> str:
    """Build a stable comparison key for field values.

    Args:
        value: Raw field value.

    Returns:
        str: Lowercased comparison key.
    """

    return _NON_ALNUM_RE.sub(" ", value.casefold()).strip()


def _case_number_key(value: str) -> str:
    """Normalize case-number text into a compact comparison key.

    Args:
        value: Raw case number.

    Returns:
        str: Stable compact case-number key.
    """

    return _NON_ALNUM_RE.sub("", value.casefold())


def _normalize_judge_name(value: str) -> str:
    """Normalize a judge name for panel comparisons.

    Args:
        value: Raw judge text.

    Returns:
        str: Cleaned judge name.
    """

    normalized = re.sub(r"\s+", " ", value.strip()).strip(" ,.;:")
    normalized = re.sub(r"^(?:h\.e\.|he\.|hon\.|honourable)\s+", "", normalized, flags=re.IGNORECASE)
    return normalized.strip(" ,.;:")


def _normalize_court_title(value: str) -> str:
    """Normalize a court title for exact-field indexing.

    Args:
        value: Raw court title text.

    Returns:
        str: Cleaned court title.
    """

    normalized = re.sub(r"\s+", " ", value.strip()).strip(" ,.;:")
    if not normalized:
        return ""
    normalized = re.sub(r"^(?:the\s+)+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bof\s+(?:the\s+)?DIFC\s+Courts?\b", "", normalized, flags=re.IGNORECASE)
    return normalized.strip(" ,.;:")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate text values while preserving their first-seen order.

    Args:
        values: Raw values in observed order.

    Returns:
        list[str]: Deduplicated values.
    """

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = _normalize_value(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value.strip())
    return deduped


def _selector_ids_for_case(case: CaseObject) -> list[str]:
    """Build doc-selector IDs for a compiled case.

    Args:
        case: Compiled case object.

    Returns:
        list[str]: Canonical-like selector IDs for the case.
    """

    selectors = [f"case:{case.doc_id}"]
    if case.case_number.strip():
        selectors.append(f"case_number:{_case_number_key(case.case_number)}")
    return selectors


def _selector_ids_for_law(law: LawObject) -> list[str]:
    """Build doc-selector IDs for a compiled law.

    Args:
        law: Compiled law object.

    Returns:
        list[str]: Canonical-like selector IDs for the law.
    """

    selectors = [f"law:{law.doc_id}"]
    title_source = law.short_title or law.title or law.doc_id
    selectors.append(f"law_title:{title_key(title_source)}")
    return selectors


@dataclass(frozen=True, slots=True)
class PanelRegistry:
    """Structured inverted indexes over the compiled corpus registry."""

    document_entities: dict[str, tuple[str, ...]]
    entity_documents: dict[str, tuple[str, ...]]
    document_fields: dict[str, dict[str, tuple[str, ...]]]
    document_field_pages: dict[str, dict[str, tuple[str, ...]]]
    document_labels: dict[str, str]
    selector_documents: dict[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: str | Path) -> PanelRegistry:
        """Load a panel registry from a persisted corpus registry JSON file.

        Args:
            path: Path to the persisted corpus registry.

        Returns:
            PanelRegistry: In-memory compare/join registry.
        """

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.build_from_corpus(CorpusRegistry.model_validate(payload))

    @classmethod
    def build_from_corpus(cls, registry: CorpusRegistry) -> PanelRegistry:
        """Build panel indexes from a compiled corpus registry.

        Args:
            registry: Compiled corpus registry artifact.

        Returns:
            PanelRegistry: In-memory structured indexes.
        """

        document_entities: dict[str, tuple[str, ...]] = {}
        entity_documents: dict[str, set[str]] = {}
        document_fields: dict[str, dict[str, tuple[str, ...]]] = {}
        document_field_pages: dict[str, dict[str, tuple[str, ...]]] = {}
        document_labels: dict[str, str] = {}
        selector_documents: dict[str, set[str]] = {}

        for doc_id, law in registry.laws.items():
            document_fields[doc_id] = cls._law_fields(law)
            document_field_pages[doc_id] = cls._field_pages(law.field_page_ids)
            document_labels[doc_id] = law.short_title or law.title or doc_id
            for selector in _selector_ids_for_law(law):
                selector_documents.setdefault(selector, set()).add(doc_id)

        for doc_id, case in registry.cases.items():
            document_fields[doc_id] = cls._case_fields(case)
            document_field_pages[doc_id] = cls._field_pages(case.field_page_ids)
            document_labels[doc_id] = case.case_number or case.title or doc_id
            for selector in _selector_ids_for_case(case):
                selector_documents.setdefault(selector, set()).add(doc_id)

        for doc_id, order in registry.orders.items():
            document_fields[doc_id] = cls._order_fields(order)
            document_field_pages[doc_id] = cls._field_pages(order.field_page_ids)
            document_labels[doc_id] = order.title or doc_id
            selector_documents.setdefault(f"order:{doc_id}", set()).add(doc_id)

        for doc_id, practice in registry.practice_directions.items():
            document_fields[doc_id] = cls._practice_direction_fields(practice)
            document_field_pages[doc_id] = cls._field_pages(practice.field_page_ids)
            document_labels[doc_id] = practice.title or doc_id
            selector_documents.setdefault(f"practice_direction:{doc_id}", set()).add(doc_id)

        entity_sets: dict[str, set[str]] = {doc_id: set() for doc_id in document_fields}
        for entity in registry.entities.values():
            for doc_id in entity.source_doc_ids:
                if not doc_id:
                    continue
                entity_sets.setdefault(doc_id, set()).add(entity.entity_id)
                entity_documents.setdefault(entity.entity_id, set()).add(doc_id)
        document_entities = {
            doc_id: tuple(sorted(entity_ids))
            for doc_id, entity_ids in sorted(entity_sets.items())
            if entity_ids
        }

        return cls(
            document_entities=document_entities,
            entity_documents={
                entity_id: tuple(sorted(doc_ids)) for entity_id, doc_ids in sorted(entity_documents.items())
            },
            document_fields=document_fields,
            document_field_pages=document_field_pages,
            document_labels=document_labels,
            selector_documents={
                selector: tuple(sorted(doc_ids)) for selector, doc_ids in sorted(selector_documents.items())
            },
        )

    @staticmethod
    def _field_pages(raw_field_pages: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
        """Normalize field-page provenance mappings.

        Args:
            raw_field_pages: Raw field-to-page-id mapping.

        Returns:
            dict[str, tuple[str, ...]]: Immutable provenance mapping.
        """

        return {
            field_name: tuple(page_id for page_id in page_ids if page_id)
            for field_name, page_ids in raw_field_pages.items()
            if page_ids
        }

    @staticmethod
    def _law_fields(law: LawObject) -> dict[str, tuple[str, ...]]:
        """Extract comparable law fields.

        Args:
            law: Compiled law object.

        Returns:
            dict[str, tuple[str, ...]]: Comparable law fields.
        """

        return {
            "title": tuple(_dedupe_preserve_order([law.title, law.short_title])),
            "authority": tuple(_dedupe_preserve_order([law.issuing_authority])),
            "issued_by": tuple(_dedupe_preserve_order([law.issuing_authority])),
            "date": tuple(_dedupe_preserve_order([law.commencement_date])),
            "law_number": tuple(_dedupe_preserve_order([law.law_number])),
        }

    @staticmethod
    def _case_fields(case: CaseObject) -> dict[str, tuple[str, ...]]:
        """Extract comparable case fields.

        Args:
            case: Compiled case object.

        Returns:
            dict[str, tuple[str, ...]]: Comparable case fields.
        """

        claimants = [party.name for party in case.parties if party.role.casefold() == "claimant"]
        respondents = [party.name for party in case.parties if party.role.casefold() == "respondent"]
        parties = [party.name for party in case.parties]
        court = _normalize_court_title(case.court)
        judges = [
            _normalize_judge_name(judge)
            for judge in case.judges
            if _normalize_judge_name(judge)
            and not _CASE_JUDGE_BLOCKLIST_RE.search(judge)
            and _CASE_JUDGE_HINT_RE.search(judge)
        ]
        return {
            "title": tuple(_dedupe_preserve_order([case.title])),
            "case_number": tuple(_dedupe_preserve_order([case.case_number])),
            "date": tuple(_dedupe_preserve_order([case.date])),
            "judge": tuple(_dedupe_preserve_order(judges)),
            "claimant": tuple(_dedupe_preserve_order(claimants)),
            "respondent": tuple(_dedupe_preserve_order(respondents)),
            "party": tuple(_dedupe_preserve_order(parties)),
            "outcome": tuple(_dedupe_preserve_order([case.outcome_summary])),
            "court": tuple(_dedupe_preserve_order([court] if court else [])),
            "authority": tuple(_dedupe_preserve_order([court] if court else [])),
        }

    @staticmethod
    def _order_fields(order: OrderObject) -> dict[str, tuple[str, ...]]:
        """Extract comparable order fields.

        Args:
            order: Compiled order object.

        Returns:
            dict[str, tuple[str, ...]]: Comparable order fields.
        """

        return {
            "title": tuple(_dedupe_preserve_order([order.title])),
            "authority": tuple(_dedupe_preserve_order([order.issued_by])),
            "issued_by": tuple(_dedupe_preserve_order([order.issued_by])),
            "date": tuple(_dedupe_preserve_order([order.effective_date])),
        }

    @staticmethod
    def _practice_direction_fields(practice: PracticeDirectionObject) -> dict[str, tuple[str, ...]]:
        """Extract comparable practice-direction fields.

        Args:
            practice: Compiled practice direction object.

        Returns:
            dict[str, tuple[str, ...]]: Comparable practice-direction fields.
        """

        return {
            "title": tuple(_dedupe_preserve_order([practice.title])),
            "authority": tuple(_dedupe_preserve_order([practice.issued_by])),
            "issued_by": tuple(_dedupe_preserve_order([practice.issued_by])),
            "date": tuple(_dedupe_preserve_order([practice.effective_date])),
            "law_number": tuple(_dedupe_preserve_order([practice.number])),
        }

    def resolve_document_ids(self, selector_ids: list[str], *, source_doc_ids: list[str] | None = None) -> list[str]:
        """Resolve canonical-like selector IDs into concrete document IDs.

        Args:
            selector_ids: Contract canonical IDs.
            source_doc_ids: Optional source-doc hints from resolved entities.

        Returns:
            list[str]: Unique document IDs in deterministic order.
        """

        resolved: list[str] = []
        seen: set[str] = set()
        for doc_id in source_doc_ids or []:
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                resolved.append(doc_id)
        for selector_id in selector_ids:
            for doc_id in self.selector_documents.get(selector_id, ()):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                resolved.append(doc_id)
        return resolved

    def get_entities_by_document(self, doc_id: str) -> list[str]:
        """Return canonical entities linked to one document.

        Args:
            doc_id: Target document ID.

        Returns:
            list[str]: Canonical entity IDs associated with the document.
        """

        return list(self.document_entities.get(doc_id, ()))

    def get_documents_by_entity(self, entity_id: str) -> list[str]:
        """Return documents linked to one canonical entity.

        Args:
            entity_id: Canonical entity ID.

        Returns:
            list[str]: Matching document IDs.
        """

        return list(self.entity_documents.get(entity_id, ()))

    def get_documents_by_field(self, field: str, value: str) -> list[str]:
        """Return documents whose structured field matches a value.

        Args:
            field: Structured field name.
            value: Expected field value.

        Returns:
            list[str]: Matching document IDs.
        """

        normalized = _normalize_value(value)
        if not normalized:
            return []
        matching: list[str] = []
        for doc_id, field_map in self.document_fields.items():
            if any(_normalize_value(candidate) == normalized for candidate in field_map.get(field, ())):
                matching.append(doc_id)
        return matching

    def intersect_entities(self, doc_ids: list[str]) -> list[str]:
        """Return canonical entities shared by all referenced documents.

        Args:
            doc_ids: Document IDs to intersect.

        Returns:
            list[str]: Shared canonical entity IDs.
        """

        if not doc_ids:
            return []
        sets = [set(self.document_entities.get(doc_id, ())) for doc_id in doc_ids]
        if not sets or any(not entity_set for entity_set in sets):
            return []
        head, *tail = sets
        shared: set[str] = set(head)
        for entity_set in tail:
            shared &= entity_set
        return sorted(shared)

    def intersect_attributes(self, doc_ids: list[str], attribute: str) -> list[str]:
        """Return field values shared by all referenced documents.

        Args:
            doc_ids: Document IDs to intersect.
            attribute: Structured field name.

        Returns:
            list[str]: Shared field values in original display form.
        """

        if not doc_ids:
            return []
        value_sets: list[set[str]] = []
        display_values: dict[str, str] = {}
        for doc_id in doc_ids:
            values = self.document_fields.get(doc_id, {}).get(attribute, ())
            normalized_values = {_normalize_value(value) for value in values if _normalize_value(value)}
            if not normalized_values:
                return []
            value_sets.append(normalized_values)
            for value in values:
                normalized = _normalize_value(value)
                if normalized and normalized not in display_values:
                    display_values[normalized] = value
        head, *tail = value_sets
        shared: set[str] = set(head)
        for value_set in tail:
            shared &= value_set
        return [display_values[key] for key in sorted(shared)]

    def compare_attributes(self, doc_ids: list[str], attribute: str) -> dict[str, list[str]]:
        """Return field values for each referenced document.

        Args:
            doc_ids: Target document IDs.
            attribute: Structured field name.

        Returns:
            dict[str, list[str]]: Field values keyed by document ID.
        """

        return {
            doc_id: list(self.document_fields.get(doc_id, {}).get(attribute, ()))
            for doc_id in doc_ids
        }

    def get_field_pages(self, doc_id: str, field: str) -> list[str]:
        """Return provenance pages for one document field.

        Args:
            doc_id: Target document ID.
            field: Structured field name.

        Returns:
            list[str]: Page IDs carrying the structured field evidence.
        """

        return list(self.document_field_pages.get(doc_id, {}).get(field, ()))

    def get_document_label(self, doc_id: str) -> str:
        """Return the preferred display label for a document.

        Args:
            doc_id: Target document ID.

        Returns:
            str: Human-readable label.
        """

        return self.document_labels.get(doc_id, doc_id)
