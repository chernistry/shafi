"""Structured field lookup tables built from the compiled corpus registry."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag_challenge.core.law_notice_support import (
    extract_enactment_authority,
    extract_enactment_date,
    extract_law_number_year,
    is_law_like_order_title,
    normalize_law_like_title,
)

if TYPE_CHECKING:
    from rag_challenge.models.legal_objects import (
        CaseObject,
        CorpusRegistry,
        LawObject,
        OrderObject,
        PracticeDirectionObject,
    )


def _field_answer_table_factory() -> dict[str, dict[str, FieldAnswer]]:
    """Build a typed empty field-answer table."""

    return {}


class FieldType(StrEnum):
    """Supported structured field lookups."""

    TITLE = "title"
    ISSUED_BY = "issued_by"
    AUTHORITY = "authority"
    DATE = "date"
    COMMENCEMENT_DATE = "commencement_date"
    LAW_NUMBER = "law_number"
    CLAIMANT = "claimant"
    RESPONDENT = "respondent"
    APPELLANT = "appellant"
    APPELLEE = "appellee"
    PARTY = "party"
    JUDGE = "judge"
    CASE_NUMBER = "case_number"
    OUTCOME = "outcome"


class FieldAnswer(BaseModel):
    """Deterministic field answer with source provenance."""

    model_config = ConfigDict(frozen=True)

    value: str
    field_type: FieldType
    source_doc_id: str
    source_page_ids: list[str] = Field(default_factory=list)
    canonical_entity_id: str
    confidence: float = 0.0


class FieldLookupTable(BaseModel):
    """In-memory lookup table for structured field answers."""

    model_config = ConfigDict(frozen=True)

    table: dict[str, dict[str, FieldAnswer]] = Field(default_factory=_field_answer_table_factory)

    @classmethod
    def build_from_registry(cls, corpus_registry: CorpusRegistry) -> FieldLookupTable:
        """Build a lookup table from a compiled corpus registry.

        Args:
            corpus_registry: Compiled corpus registry with structured objects.

        Returns:
            FieldLookupTable: Lookup table keyed by canonical entity ID.
        """

        table: dict[str, dict[str, FieldAnswer]] = {}
        for doc_id, law in corpus_registry.laws.items():
            table[f"law:{doc_id}"] = cls._law_fields(f"law:{doc_id}", law)
        for doc_id, case in corpus_registry.cases.items():
            table[f"case:{doc_id}"] = cls._case_fields(f"case:{doc_id}", case)
        for doc_id, order in corpus_registry.orders.items():
            table[f"order:{doc_id}"] = cls._order_fields(f"order:{doc_id}", order)
        for doc_id, practice in corpus_registry.practice_directions.items():
            table[f"practice_direction:{doc_id}"] = cls._practice_direction_fields(
                f"practice_direction:{doc_id}",
                practice,
            )
        return cls(table=table)

    @staticmethod
    def _law_fields(canonical_id: str, law: LawObject) -> dict[str, FieldAnswer]:
        """Build lookup rows for a compiled law."""

        rows: dict[str, FieldAnswer] = {}
        derived_authority = extract_enactment_authority(source_text=law.source_text, fallback=law.issuing_authority)
        derived_date = extract_enactment_date(source_text=law.source_text, fallback=law.commencement_date)
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=law.doc_id,
            field_type=FieldType.TITLE,
            value=law.title,
            page_ids=law.field_page_ids.get(FieldType.TITLE.value, law.page_ids[:1]),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=law.doc_id,
            field_type=FieldType.ISSUED_BY,
            value=derived_authority,
            page_ids=law.field_page_ids.get(FieldType.ISSUED_BY.value) or law.field_page_ids.get(FieldType.AUTHORITY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=law.doc_id,
            field_type=FieldType.AUTHORITY,
            value=derived_authority,
            page_ids=law.field_page_ids.get(FieldType.AUTHORITY.value) or law.field_page_ids.get(FieldType.ISSUED_BY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=law.doc_id,
            field_type=FieldType.DATE,
            value=derived_date,
            page_ids=law.field_page_ids.get(FieldType.DATE.value) or law.field_page_ids.get(FieldType.COMMENCEMENT_DATE.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=law.doc_id,
            field_type=FieldType.COMMENCEMENT_DATE,
            value=derived_date,
            page_ids=law.field_page_ids.get(FieldType.COMMENCEMENT_DATE.value) or law.field_page_ids.get(FieldType.DATE.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=law.doc_id,
            field_type=FieldType.LAW_NUMBER,
            value=law.law_number,
            page_ids=law.field_page_ids.get(FieldType.LAW_NUMBER.value, law.page_ids[:1]),
        )
        return rows

    @staticmethod
    def _case_fields(canonical_id: str, case: CaseObject) -> dict[str, FieldAnswer]:
        """Build lookup rows for a compiled case."""

        rows: dict[str, FieldAnswer] = {}
        claimants = [party.name for party in case.parties if party.role.casefold() == "claimant"]
        respondents = [party.name for party in case.parties if party.role.casefold() == "respondent"]
        appellants = [party.name for party in case.parties if party.role.casefold() == "appellant"]
        appellees = [party.name for party in case.parties if party.role.casefold() == "appellee"]
        parties = [party.name for party in case.parties]
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.TITLE,
            value=case.title,
            page_ids=case.field_page_ids.get(FieldType.TITLE.value, case.page_ids[:1]),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.CASE_NUMBER,
            value=case.case_number,
            page_ids=case.field_page_ids.get(FieldType.CASE_NUMBER.value, case.page_ids[:1]),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.DATE,
            value=case.date,
            page_ids=case.field_page_ids.get(FieldType.DATE.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.JUDGE,
            value=", ".join(case.judges),
            page_ids=case.field_page_ids.get(FieldType.JUDGE.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.CLAIMANT,
            value=", ".join(claimants),
            page_ids=case.field_page_ids.get(FieldType.CLAIMANT.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.RESPONDENT,
            value=", ".join(respondents),
            page_ids=case.field_page_ids.get(FieldType.RESPONDENT.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.APPELLANT,
            value=", ".join(appellants),
            page_ids=case.field_page_ids.get(FieldType.APPELLANT.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.APPELLEE,
            value=", ".join(appellees),
            page_ids=case.field_page_ids.get(FieldType.APPELLEE.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.PARTY,
            value=", ".join(parties),
            page_ids=case.field_page_ids.get(FieldType.PARTY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=case.doc_id,
            field_type=FieldType.OUTCOME,
            value=case.outcome_summary,
            page_ids=case.field_page_ids.get(FieldType.OUTCOME.value, []),
        )
        return rows

    @staticmethod
    def _order_fields(canonical_id: str, order: OrderObject) -> dict[str, FieldAnswer]:
        """Build lookup rows for a compiled order."""

        rows: dict[str, FieldAnswer] = {}
        derived_title = (
            normalize_law_like_title(title=order.title, source_text=order.source_text)
            if is_law_like_order_title(title=order.title, source_text=order.source_text)
            else order.title
        )
        derived_law_number, _derived_year = extract_law_number_year(title=order.title, source_text=order.source_text)
        derived_authority = extract_enactment_authority(source_text=order.source_text, fallback=order.issued_by)
        derived_date = extract_enactment_date(source_text=order.source_text, fallback=order.effective_date)
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=order.doc_id,
            field_type=FieldType.TITLE,
            value=derived_title,
            page_ids=order.field_page_ids.get(FieldType.TITLE.value, order.page_ids[:1]),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=order.doc_id,
            field_type=FieldType.ISSUED_BY,
            value=derived_authority,
            page_ids=order.field_page_ids.get(FieldType.ISSUED_BY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=order.doc_id,
            field_type=FieldType.AUTHORITY,
            value=derived_authority,
            page_ids=order.field_page_ids.get(FieldType.AUTHORITY.value) or order.field_page_ids.get(FieldType.ISSUED_BY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=order.doc_id,
            field_type=FieldType.DATE,
            value=derived_date,
            page_ids=order.field_page_ids.get(FieldType.DATE.value) or order.field_page_ids.get("effective_date", []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=order.doc_id,
            field_type=FieldType.LAW_NUMBER,
            value=derived_law_number,
            page_ids=order.field_page_ids.get(FieldType.TITLE.value, order.page_ids[:1]),
        )
        return rows

    @staticmethod
    def _practice_direction_fields(
        canonical_id: str,
        practice: PracticeDirectionObject,
    ) -> dict[str, FieldAnswer]:
        """Build lookup rows for a compiled practice direction."""

        rows: dict[str, FieldAnswer] = {}
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=practice.doc_id,
            field_type=FieldType.TITLE,
            value=practice.title,
            page_ids=practice.field_page_ids.get(FieldType.TITLE.value, practice.page_ids[:1]),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=practice.doc_id,
            field_type=FieldType.ISSUED_BY,
            value=practice.issued_by,
            page_ids=practice.field_page_ids.get(FieldType.ISSUED_BY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=practice.doc_id,
            field_type=FieldType.AUTHORITY,
            value=practice.issued_by,
            page_ids=practice.field_page_ids.get(FieldType.AUTHORITY.value) or practice.field_page_ids.get(FieldType.ISSUED_BY.value, []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=practice.doc_id,
            field_type=FieldType.DATE,
            value=practice.effective_date,
            page_ids=practice.field_page_ids.get(FieldType.DATE.value) or practice.field_page_ids.get("effective_date", []),
        )
        FieldLookupTable._maybe_add(
            rows,
            canonical_id=canonical_id,
            doc_id=practice.doc_id,
            field_type=FieldType.LAW_NUMBER,
            value=practice.number,
            page_ids=practice.field_page_ids.get(FieldType.LAW_NUMBER.value, practice.page_ids[:1]),
        )
        return rows

    @staticmethod
    def _maybe_add(
        rows: dict[str, FieldAnswer],
        *,
        canonical_id: str,
        doc_id: str,
        field_type: FieldType,
        value: str,
        page_ids: list[str],
    ) -> None:
        """Conditionally add a populated field answer to a row mapping."""

        cleaned = value.strip()
        if not cleaned:
            return
        confidence = 0.98 if page_ids else 0.82
        rows[field_type.value] = FieldAnswer(
            value=cleaned,
            field_type=field_type,
            source_doc_id=doc_id,
            source_page_ids=list(page_ids),
            canonical_entity_id=canonical_id,
            confidence=confidence,
        )

    def lookup(self, canonical_id: str, field_type: FieldType | str) -> FieldAnswer | None:
        """Resolve one field for a canonical entity.

        Args:
            canonical_id: Canonical entity identifier.
            field_type: Desired field type.

        Returns:
            FieldAnswer | None: Matched field answer, if present.
        """

        normalized = field_type.value if isinstance(field_type, FieldType) else str(field_type).strip()
        return self.table.get(canonical_id, {}).get(normalized)

    def get_all_fields(self, canonical_id: str) -> dict[FieldType, str]:
        """Return all populated fields for a canonical entity.

        Args:
            canonical_id: Canonical entity identifier.

        Returns:
            dict[FieldType, str]: Available field values.
        """

        result: dict[FieldType, str] = {}
        for raw_field, answer in self.table.get(canonical_id, {}).items():
            result[FieldType(raw_field)] = answer.value
        return result

    def search_by_field(self, field_type: FieldType | str, value: str) -> list[str]:
        """Reverse-search canonical IDs by normalized field value.

        Args:
            field_type: Field to search.
            value: Target field value.

        Returns:
            list[str]: Matching canonical IDs in deterministic order.
        """

        normalized_field = field_type.value if isinstance(field_type, FieldType) else str(field_type).strip()
        normalized_value = value.strip().casefold()
        if not normalized_field or not normalized_value:
            return []
        matches: list[str] = []
        for canonical_id, rows in self.table.items():
            answer = rows.get(normalized_field)
            if answer is None:
                continue
            if answer.value.casefold() == normalized_value:
                matches.append(canonical_id)
        return matches

    def export(self, path: str | Path) -> Path:
        """Persist the lookup table to JSON.

        Args:
            path: Output JSON path.

        Returns:
            Path: Resolved output path.
        """

        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return resolved_path

    @classmethod
    def load(cls, path: str | Path) -> FieldLookupTable:
        """Load a persisted lookup table from disk.

        Args:
            path: JSON path previously created by :meth:`export`.

        Returns:
            FieldLookupTable: Loaded lookup table, or an empty one if missing.
        """

        resolved_path = Path(path)
        if not resolved_path.exists():
            return cls()
        return cls.model_validate_json(resolved_path.read_text(encoding="utf-8"))
