"""Deterministic manual metadata overrides for high-EV domain documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from rag_challenge.models.legal_objects import CaseObject, CaseParty, LawObject, OrderObject, PracticeDirectionObject

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_challenge.ingestion.corpus_compiler import CompiledLegalObject


def _tuple_factory() -> tuple[str, ...]:
    """Build a typed immutable empty tuple for dataclass defaults."""

    return ()


@dataclass(frozen=True, slots=True)
class ManualDomainOverride:
    """Typed manual override for a compiled legal object.

    Args:
        title: Canonical full title override.
        short_title: Canonical short title override for law-like documents.
        aliases: Extra deterministic aliases for title/case matching.
        issued_by: Issuing authority override for law/order-like documents.
        enactment_date: Generic enactment/effective date hint.
        commencement_date: Commencement date override for law-like documents.
        effective_date: Effective date override for order/practice-direction documents.
        law_number: Law or direction number override.
        year: Law year override.
        case_number: Case-number override.
        claimant: Replacement claimant party names.
        respondent: Replacement respondent party names.
        judges: Replacement judge/panel names.
        note: Human-readable audit note for why the override exists.
    """

    title: str = ""
    short_title: str = ""
    aliases: tuple[str, ...] = field(default_factory=_tuple_factory)
    issued_by: str = ""
    enactment_date: str = ""
    commencement_date: str = ""
    effective_date: str = ""
    law_number: str = ""
    year: str = ""
    case_number: str = ""
    claimant: tuple[str, ...] = field(default_factory=_tuple_factory)
    respondent: tuple[str, ...] = field(default_factory=_tuple_factory)
    judges: tuple[str, ...] = field(default_factory=_tuple_factory)
    note: str = ""


@lru_cache(maxsize=1)
def _all_manual_domain_overrides() -> dict[str, ManualDomainOverride]:
    """Load the merged manual override table once per process."""

    from rag_challenge.ingestion.manual_case_overrides import MANUAL_CASE_OVERRIDES
    from rag_challenge.ingestion.manual_law_overrides import MANUAL_LAW_OVERRIDES

    merged: dict[str, ManualDomainOverride] = {}
    merged.update(MANUAL_LAW_OVERRIDES)
    merged.update(MANUAL_CASE_OVERRIDES)
    return merged


def get_manual_domain_override(doc_id: str) -> ManualDomainOverride | None:
    """Look up a manual override by stable parsed-document ID.

    Args:
        doc_id: Parser-generated stable document ID.

    Returns:
        ManualDomainOverride | None: Override payload when available.
    """

    return _all_manual_domain_overrides().get(str(doc_id or "").strip())


def apply_manual_domain_override(
    compiled: CompiledLegalObject,
    override: ManualDomainOverride,
) -> CompiledLegalObject:
    """Apply a typed manual override to a compiled legal object.

    Args:
        compiled: Compiled legal object to patch.
        override: Manual override payload for the document.

    Returns:
        CompiledLegalObject: Patched compiled object.
    """

    merged_aliases = _merge_aliases(getattr(compiled, "aliases", ()), override.aliases)
    if isinstance(compiled, LawObject):
        commencement_date = override.commencement_date or override.enactment_date or compiled.commencement_date
        return compiled.model_copy(
            update={
                "title": override.title or compiled.title,
                "short_title": override.short_title or compiled.short_title,
                "aliases": merged_aliases,
                "issuing_authority": override.issued_by or compiled.issuing_authority,
                "commencement_date": commencement_date,
                "law_number": override.law_number or compiled.law_number,
                "year": override.year or compiled.year,
            }
        )
    if isinstance(compiled, CaseObject):
        return compiled.model_copy(
            update={
                "title": override.title or compiled.title,
                "aliases": merged_aliases,
                "case_number": override.case_number or compiled.case_number,
                "judges": _apply_string_override(compiled.judges, override.judges),
                "parties": _apply_case_party_override(compiled.parties, override),
            }
        )
    if isinstance(compiled, OrderObject):
        effective_date = override.effective_date or override.enactment_date or compiled.effective_date
        return compiled.model_copy(
            update={
                "title": override.title or compiled.title,
                "aliases": merged_aliases,
                "issued_by": override.issued_by or compiled.issued_by,
                "effective_date": effective_date,
                "order_number": override.case_number or compiled.order_number,
            }
        )
    if isinstance(compiled, PracticeDirectionObject):
        effective_date = override.effective_date or override.enactment_date or compiled.effective_date
        return compiled.model_copy(
            update={
                "title": override.title or compiled.title,
                "aliases": merged_aliases,
                "issued_by": override.issued_by or compiled.issued_by,
                "effective_date": effective_date,
                "number": override.law_number or compiled.number,
            }
        )
    return compiled.model_copy(update={"title": override.title or compiled.title, "aliases": merged_aliases})


def _merge_aliases(existing: Sequence[str], updates: Sequence[str]) -> list[str]:
    """Merge alias lists while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()
    for raw in [*existing, *updates]:
        cleaned = str(raw or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def _apply_string_override(existing: Sequence[str], updates: Sequence[str]) -> list[str]:
    """Replace a string list only when the override provides values."""

    if not updates:
        return [str(value) for value in existing]
    return [str(value).strip() for value in updates if str(value).strip()]


def _apply_case_party_override(
    existing: Sequence[CaseParty],
    override: ManualDomainOverride,
) -> list[CaseParty]:
    """Replace claimant/respondent parties while preserving unrelated roles."""

    preserved: list[CaseParty] = []
    if not override.claimant:
        preserved.extend(
            CaseParty(name=party.name, role=party.role, canonical_entity_id=party.canonical_entity_id)
            for party in existing
            if party.role.casefold() == "claimant"
        )
    else:
        preserved.extend(CaseParty(name=name, role="claimant") for name in override.claimant if name.strip())

    if not override.respondent:
        preserved.extend(
            CaseParty(name=party.name, role=party.role, canonical_entity_id=party.canonical_entity_id)
            for party in existing
            if party.role.casefold() == "respondent"
        )
    else:
        preserved.extend(CaseParty(name=name, role="respondent") for name in override.respondent if name.strip())

    preserved.extend(
        CaseParty(name=party.name, role=party.role, canonical_entity_id=party.canonical_entity_id)
        for party in existing
        if party.role.casefold() not in {"claimant", "respondent"}
    )

    unique: list[CaseParty] = []
    seen: set[tuple[str, str]] = set()
    for party in preserved:
        key = (party.name.casefold(), party.role.casefold())
        if key in seen or not party.name.strip():
            continue
        seen.add(key)
        unique.append(party)
    return unique
