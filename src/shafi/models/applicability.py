"""Applicability-graph models for amendment and temporal relationships."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field


def _edge_list_factory() -> list[ApplicabilityEdge]:
    """Build a typed empty edge list for Pydantic defaults."""

    return []


def _commencement_list_factory() -> list[CommencementRecord]:
    """Build a typed empty commencement list for Pydantic defaults."""

    return []


def _warning_list_factory() -> list[GraphWarning]:
    """Build a typed empty warning list for Pydantic defaults."""

    return []


class ApplicabilityEdgeType(StrEnum):
    """Supported applicability-edge types."""

    AMENDS = "amends"
    COMMENCES = "commences"
    SUPERSEDES = "supersedes"
    REPLACES = "replaces"
    REPEALS = "repeals"
    EXTENDS = "extends"
    INSERTS = "inserts"
    SUBSTITUTES = "substitutes"
    REVOKES = "revokes"


class ApplicabilityEdge(BaseModel):
    """Directed applicability edge between two legal objects."""

    model_config = ConfigDict(frozen=True)

    source_doc_id: str
    target_doc_id: str
    edge_type: ApplicabilityEdgeType
    effective_date: str | None = None
    scope: str = ""
    evidence_text: str = ""
    evidence_page_id: str = ""


class CommencementRecord(BaseModel):
    """Commencement date record for one law."""

    model_config = ConfigDict(frozen=True)

    law_id: str
    commencement_date: str
    commencement_notice_id: str = ""
    evidence_page_id: str = ""


class GraphWarning(BaseModel):
    """Validation warning emitted by graph checks."""

    model_config = ConfigDict(frozen=True)

    warning_type: str
    message: str
    doc_ids: list[str] = Field(default_factory=list)


class ApplicabilityGraph(BaseModel):
    """Temporal and amendment graph across compiled legal objects."""

    nodes: list[str] = Field(default_factory=list)
    edges: list[ApplicabilityEdge] = Field(default_factory=_edge_list_factory)
    commencements: list[CommencementRecord] = Field(default_factory=_commencement_list_factory)
    laws: dict[str, object] = Field(default_factory=dict)
    warnings: list[GraphWarning] = Field(default_factory=_warning_list_factory)

    def get_amendments(self, law_id: str) -> list[ApplicabilityEdge]:
        """Return amendment edges that target the given law."""

        return [
            edge
            for edge in self.edges
            if edge.target_doc_id == law_id and edge.edge_type is ApplicabilityEdgeType.AMENDS
        ]

    def get_commencement(self, law_id: str) -> CommencementRecord | None:
        """Return the first commencement record for the given law."""

        for record in self.commencements:
            if record.law_id == law_id:
                return record
        return None

    def get_current_version(self, law_id: str) -> LawObject | None:
        """Follow supersession/replacement edges to the newest known law object."""

        current_id = law_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            replacement = next(
                (
                    edge.source_doc_id
                    for edge in self.edges
                    if edge.target_doc_id == current_id
                    and edge.edge_type in {ApplicabilityEdgeType.SUPERSEDES, ApplicabilityEdgeType.REPLACES}
                ),
                None,
            )
            if replacement is None:
                break
            current_id = replacement
        return cast("LawObject | None", self.laws.get(current_id))

    def get_amendment_history(self, law_id: str) -> list[ApplicabilityEdge]:
        """Return amendment edges for the given law sorted by effective date."""

        return sorted(
            self.get_amendments(law_id),
            key=lambda edge: ((edge.effective_date or "9999-99-99"), edge.source_doc_id, edge.scope),
        )


if TYPE_CHECKING:
    from shafi.models.legal_objects import LawObject
