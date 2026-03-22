"""Structured compare/join execution over the compiled legal corpus."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag_challenge.core.panel_registry import PanelRegistry
from rag_challenge.core.query_contract import PredicateType, QueryContract

if TYPE_CHECKING:
    from pathlib import Path


def _string_list_factory() -> list[str]:
    """Build a typed empty string-list default.

    Returns:
        list[str]: Empty list.
    """

    return []


def _dict_list_factory() -> dict[str, list[str]]:
    """Build a typed empty mapping of string lists.

    Returns:
        dict[str, list[str]]: Empty mapping.
    """

    return {}


class CompareType(StrEnum):
    """Supported structured comparison families."""

    COMMON_PARTY = "common_party"
    COMMON_JUDGE = "common_judge"
    COMMON_ENTITY = "common_entity"
    COMMON_ATTRIBUTE = "common_attribute"
    ATTRIBUTE_COMPARE = "attribute_compare"
    ENUMERATE_BY_FILTER = "enumerate_by_filter"


class CompareResult(BaseModel):
    """Structured compare-engine result with provenance."""

    model_config = ConfigDict(frozen=True)

    result_type: CompareType
    entities: list[str] = Field(default_factory=_string_list_factory)
    attributes: dict[str, list[str]] = Field(default_factory=_dict_list_factory)
    provenance: dict[str, list[str]] = Field(default_factory=_dict_list_factory)
    source_doc_ids: list[str] = Field(default_factory=_string_list_factory)
    source_page_ids: list[str] = Field(default_factory=_string_list_factory)
    formatted_answer: str = ""


class CompareEngine:
    """Execute structured compare questions over compiled corpus artifacts."""

    def __init__(self, *, panel_registry: PanelRegistry) -> None:
        """Initialize the compare engine.

        Args:
            panel_registry: In-memory structured registry for joins.
        """

        self._panel_registry = panel_registry

    @classmethod
    def from_path(cls, registry_path: str | Path) -> CompareEngine:
        """Load the compare engine from a persisted corpus registry.

        Args:
            registry_path: Path to the persisted corpus-registry JSON file.

        Returns:
            CompareEngine: Runtime-ready compare engine.
        """

        registry = PanelRegistry.load(registry_path)
        return cls(panel_registry=registry)

    def execute(self, contract: QueryContract) -> CompareResult | None:
        """Attempt structured execution for a compare contract.

        Args:
            contract: Compiled query contract.

        Returns:
            CompareResult | None: Structured answer when safely handled.
        """

        if self.should_fallback(contract):
            return None
        doc_ids = self._resolve_doc_ids(contract)
        if len(doc_ids) < 2:
            return None
        compare_type = self.detect_compare_type(contract)
        if compare_type is CompareType.COMMON_JUDGE:
            return self.execute_common_judge(contract, doc_ids)
        if compare_type is CompareType.COMMON_PARTY:
            return self.execute_common_party(contract, doc_ids)
        if compare_type is CompareType.COMMON_ENTITY:
            return self.execute_common_entity(contract, doc_ids)
        if compare_type is CompareType.ATTRIBUTE_COMPARE:
            return self.execute_attribute_compare(contract, doc_ids)
        if compare_type is CompareType.COMMON_ATTRIBUTE:
            return self.execute_common_attribute(contract, doc_ids)
        return None

    def should_fallback(self, contract: QueryContract) -> bool:
        """Decide whether structured compare execution is unsafe.

        Args:
            contract: Compiled query contract.

        Returns:
            bool: True when standard RAG should handle the question.
        """

        if contract.predicate is not PredicateType.COMPARE:
            return True
        if len(contract.primary_entities) < 2:
            return True
        if contract.polarity.value == "negative":
            return True
        return not contract.comparison_axes and contract.answer_type.strip().lower() == "free_text"

    def detect_compare_type(self, contract: QueryContract) -> CompareType:
        """Infer the compare family from the compiled contract.

        Args:
            contract: Compiled query contract.

        Returns:
            CompareType: Structured compare family.
        """

        axes = {axis.strip().lower() for axis in contract.comparison_axes}
        query = contract.query_text.casefold()
        if "judge" in axes:
            return CompareType.COMMON_JUDGE
        if "party" in axes:
            return CompareType.COMMON_PARTY
        if any(axis in axes for axis in {"date", "title", "authority", "outcome"}):
            if any(marker in query for marker in (" earlier ", " later ", " first", " second", " before ", " after ")):
                return CompareType.ATTRIBUTE_COMPARE
            return CompareType.COMMON_ATTRIBUTE
        return CompareType.COMMON_ENTITY

    def execute_common_party(self, contract: QueryContract, doc_ids: list[str]) -> CompareResult | None:
        """Execute a shared-party join across compiled cases.

        Args:
            contract: Compiled compare contract.
            doc_ids: Target document IDs.

        Returns:
            CompareResult | None: Structured answer when supported.
        """

        return self._execute_common_field(contract, doc_ids, CompareType.COMMON_PARTY, "party")

    def execute_common_judge(self, contract: QueryContract, doc_ids: list[str]) -> CompareResult | None:
        """Execute a shared-judge join across compiled cases.

        Args:
            contract: Compiled compare contract.
            doc_ids: Target document IDs.

        Returns:
            CompareResult | None: Structured answer when supported.
        """

        return self._execute_common_field(contract, doc_ids, CompareType.COMMON_JUDGE, "judge")

    def execute_common_entity(self, contract: QueryContract, doc_ids: list[str]) -> CompareResult | None:
        """Execute a generic shared-entity join across documents.

        Args:
            contract: Compiled compare contract.
            doc_ids: Target document IDs.

        Returns:
            CompareResult | None: Structured answer when supported.
        """

        shared_entities = self._panel_registry.intersect_entities(doc_ids)
        if not shared_entities:
            return None if contract.answer_type.strip().lower() == "free_text" else CompareResult(
                result_type=CompareType.COMMON_ENTITY,
                source_doc_ids=doc_ids,
                source_page_ids=[],
                formatted_answer="No",
            )
        return CompareResult(
            result_type=CompareType.COMMON_ENTITY,
            entities=shared_entities,
            source_doc_ids=doc_ids,
            source_page_ids=[],
            formatted_answer=self.format_result(
                result_type=CompareType.COMMON_ENTITY,
                values=shared_entities,
                answer_type=contract.answer_type,
                contract=contract,
                doc_ids=doc_ids,
            ),
        )

    def execute_common_attribute(self, contract: QueryContract, doc_ids: list[str]) -> CompareResult | None:
        """Execute a shared-attribute join across documents.

        Args:
            contract: Compiled compare contract.
            doc_ids: Target document IDs.

        Returns:
            CompareResult | None: Structured answer when supported.
        """

        axis = self._primary_attribute_axis(contract)
        if not axis:
            return None
        return self._execute_common_field(contract, doc_ids, CompareType.COMMON_ATTRIBUTE, axis)

    def execute_attribute_compare(self, contract: QueryContract, doc_ids: list[str]) -> CompareResult | None:
        """Compare one structured attribute across referenced documents.

        Args:
            contract: Compiled compare contract.
            doc_ids: Target document IDs.

        Returns:
            CompareResult | None: Structured answer when supported.
        """

        axis = self._primary_attribute_axis(contract)
        if axis != "date":
            return None
        attribute_values = self._panel_registry.compare_attributes(doc_ids, axis)
        if any(not values for values in attribute_values.values()):
            return None
        ordered = self._order_docs_by_date(attribute_values)
        if not ordered:
            return None
        source_page_ids: list[str] = []
        provenance: dict[str, list[str]] = {}
        for doc_id in doc_ids:
            pages = self._panel_registry.get_field_pages(doc_id, axis)
            source_page_ids.extend(pages)
            provenance[doc_id] = pages
        ordered_doc_ids = [doc_id for doc_id, _date in ordered]
        answer = self._format_attribute_compare_answer(contract, ordered_doc_ids)
        return CompareResult(
            result_type=CompareType.ATTRIBUTE_COMPARE,
            attributes={doc_id: values for doc_id, values in attribute_values.items()},
            provenance=provenance,
            source_doc_ids=doc_ids,
            source_page_ids=self._dedupe_pages(source_page_ids),
            formatted_answer=answer,
        )

    def format_result(
        self,
        *,
        result_type: CompareType,
        values: list[str],
        answer_type: str,
        contract: QueryContract,
        doc_ids: list[str],
    ) -> str:
        """Format a structured compare result for pipeline emission.

        Args:
            result_type: Structured compare family.
            values: Result values.
            answer_type: Requested answer type.
            contract: Compiled query contract.
            doc_ids: Documents involved in the compare.

        Returns:
            str: Final answer string.
        """

        normalized_answer_type = answer_type.strip().lower()
        if normalized_answer_type == "boolean":
            return "Yes" if values else "No"
        if not values:
            return ""
        if result_type in {CompareType.COMMON_JUDGE, CompareType.COMMON_PARTY, CompareType.COMMON_ATTRIBUTE}:
            if normalized_answer_type in {"name", "names"}:
                return ", ".join(values)
            if normalized_answer_type == "free_text":
                return ", ".join(values)
        if result_type is CompareType.COMMON_ENTITY:
            return ", ".join(values)
        return self._format_attribute_compare_answer(contract, doc_ids)

    def _execute_common_field(
        self,
        contract: QueryContract,
        doc_ids: list[str],
        result_type: CompareType,
        field_name: str,
    ) -> CompareResult | None:
        """Execute a shared-field join across referenced documents.

        Args:
            contract: Compiled compare contract.
            doc_ids: Target document IDs.
            result_type: Structured compare family.
            field_name: Structured field to intersect.

        Returns:
            CompareResult | None: Structured answer when supported.
        """

        values = self._panel_registry.intersect_attributes(doc_ids, field_name)
        answer = self.format_result(
            result_type=result_type,
            values=values,
            answer_type=contract.answer_type,
            contract=contract,
            doc_ids=doc_ids,
        )
        if not answer.strip() and contract.answer_type.strip().lower() != "boolean":
            return None
        provenance: dict[str, list[str]] = {}
        source_page_ids: list[str] = []
        for doc_id in doc_ids:
            pages = self._panel_registry.get_field_pages(doc_id, field_name)
            provenance[doc_id] = pages
            source_page_ids.extend(pages)
        return CompareResult(
            result_type=result_type,
            entities=values,
            provenance=provenance,
            source_doc_ids=doc_ids,
            source_page_ids=self._dedupe_pages(source_page_ids),
            formatted_answer=answer,
        )

    def _resolve_doc_ids(self, contract: QueryContract) -> list[str]:
        """Resolve contract primary entities into document IDs.

        Args:
            contract: Compiled compare contract.

        Returns:
            list[str]: Resolved document IDs.
        """

        selector_ids = [entity.canonical_id for entity in contract.primary_entities]
        source_doc_ids = [
            doc_id
            for entity in contract.primary_entities
            for doc_id in entity.source_doc_ids
        ]
        return self._panel_registry.resolve_document_ids(selector_ids, source_doc_ids=source_doc_ids)

    @staticmethod
    def _primary_attribute_axis(contract: QueryContract) -> str:
        """Choose the main structured attribute axis for a compare query.

        Args:
            contract: Compiled compare contract.

        Returns:
            str: Chosen field name, or an empty string.
        """

        for candidate in ("date", "authority", "title", "outcome"):
            if candidate in contract.comparison_axes:
                return candidate
        return ""

    @staticmethod
    def _parse_legal_date(raw_value: str) -> datetime | None:
        """Parse a legal date string conservatively.

        Args:
            raw_value: Raw structured date value.

        Returns:
            datetime | None: Parsed date, or ``None`` when unsupported.
        """

        cleaned = raw_value.strip()
        if not cleaned:
            return None
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%Y-%m-%d", "%Y"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    def _order_docs_by_date(self, attribute_values: dict[str, list[str]]) -> list[tuple[str, datetime]]:
        """Sort documents by their structured date field.

        Args:
            attribute_values: Document-date mapping.

        Returns:
            list[tuple[str, datetime]]: Ordered document/date pairs.
        """

        parsed: list[tuple[str, datetime]] = []
        for doc_id, values in attribute_values.items():
            if not values:
                return []
            parsed_date = self._parse_legal_date(values[0])
            if parsed_date is None:
                return []
            parsed.append((doc_id, parsed_date))
        return sorted(parsed, key=lambda item: item[1])

    def _format_attribute_compare_answer(self, contract: QueryContract, ordered_doc_ids: list[str]) -> str:
        """Format earlier/later attribute comparisons.

        Args:
            contract: Compiled compare contract.
            ordered_doc_ids: Documents ordered by the compared attribute.

        Returns:
            str: Final answer string.
        """

        if not ordered_doc_ids:
            return ""
        query = contract.query_text.casefold()
        entity_by_doc_id = {
            doc_id: entity
            for entity in contract.primary_entities
            for doc_id in entity.source_doc_ids
        }
        if "later" in query or "after" in query or "second" in query:
            target_doc_id = ordered_doc_ids[-1]
        else:
            target_doc_id = ordered_doc_ids[0]
        entity = entity_by_doc_id.get(target_doc_id)
        if entity is not None and entity.canonical_form.strip():
            return entity.canonical_form.strip()
        return self._panel_registry.get_document_label(target_doc_id)

    @staticmethod
    def _dedupe_pages(page_ids: list[str]) -> list[str]:
        """Deduplicate provenance page IDs while preserving order.

        Args:
            page_ids: Raw page IDs.

        Returns:
            list[str]: Deduplicated page IDs.
        """

        seen: set[str] = set()
        ordered: list[str] = []
        for page_id in page_ids:
            if not page_id or page_id in seen:
                continue
            seen.add(page_id)
            ordered.append(page_id)
        return ordered
