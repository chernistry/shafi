"""Structured temporal reasoning over the compiled applicability graph."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict, Field

from rag_challenge.core.query_contract import PredicateType, QueryContract, TimeScopeType
from rag_challenge.ingestion.applicability_graph import build_applicability_graph
from rag_challenge.models.applicability import ApplicabilityEdge, ApplicabilityEdgeType, ApplicabilityGraph
from rag_challenge.models.legal_objects import CorpusRegistry, LawObject

if TYPE_CHECKING:
    from collections.abc import Iterable


def _edge_list_factory() -> list[ApplicabilityEdge]:
    """Build a typed empty applicability-edge list.

    Returns:
        list[ApplicabilityEdge]: Empty list.
    """

    return []


def _string_list_factory() -> list[str]:
    """Build a typed empty string-list default.

    Returns:
        list[str]: Empty list.
    """

    return []


class TemporalQueryType(StrEnum):
    """Supported temporal query families."""

    COMMENCEMENT = "commencement"
    AMENDMENT_DATE = "amendment_date"
    AMENDMENT_LIST = "amendment_list"
    SUPERSESSION = "supersession"
    EFFECTIVE_DATE = "effective_date"
    TEMPORAL_STATUS = "temporal_status"


class TemporalResult(BaseModel):
    """Structured temporal answer with graph provenance."""

    model_config = ConfigDict(frozen=True)

    query_type: TemporalQueryType
    answer_value: str
    answer_formatted: str
    provenance_edges: list[ApplicabilityEdge] = Field(default_factory=_edge_list_factory)
    provenance_page_ids: list[str] = Field(default_factory=_string_list_factory)
    confidence: float = 0.0


class ApplicabilityEngine:
    """Answer temporal questions from the compiled applicability graph."""

    def __init__(
        self,
        *,
        applicability_graph: ApplicabilityGraph,
        document_titles: dict[str, str],
    ) -> None:
        """Initialize the temporal engine.

        Args:
            applicability_graph: Structured applicability graph.
            document_titles: Display titles keyed by document ID.
        """

        self._graph = applicability_graph
        self._document_titles = document_titles

    @classmethod
    def from_registry_path(cls, registry_path: str | Path) -> ApplicabilityEngine:
        """Load the temporal engine from a persisted corpus registry.

        Args:
            registry_path: Path to the persisted corpus-registry JSON file.

        Returns:
            ApplicabilityEngine: Runtime-ready temporal engine.
        """

        payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        registry = CorpusRegistry.model_validate(payload)
        graph = build_applicability_graph(registry)
        document_titles = {
            **{doc_id: law.title for doc_id, law in registry.laws.items()},
            **{doc_id: order.title for doc_id, order in registry.orders.items()},
            **{doc_id: practice.title for doc_id, practice in registry.practice_directions.items()},
        }
        return cls(applicability_graph=graph, document_titles=document_titles)

    def answer(self, contract: QueryContract) -> TemporalResult | None:
        """Attempt a structured temporal answer from the applicability graph.

        Args:
            contract: Compiled temporal query contract.

        Returns:
            TemporalResult | None: Structured temporal answer when safely handled.
        """

        if self.should_fallback(contract):
            return None
        law_id = self._resolve_law_id(contract)
        if not law_id:
            return None
        query_type = self.detect_temporal_type(contract)
        if query_type is TemporalQueryType.COMMENCEMENT:
            return self.answer_commencement(law_id)
        if query_type is TemporalQueryType.AMENDMENT_DATE:
            return self.answer_amendment_date(law_id)
        if query_type is TemporalQueryType.AMENDMENT_LIST:
            return self.answer_amendment_list(law_id)
        if query_type is TemporalQueryType.SUPERSESSION:
            return self.answer_supersession(law_id)
        if query_type is TemporalQueryType.EFFECTIVE_DATE:
            return self.answer_effective_date(law_id)
        if query_type is TemporalQueryType.TEMPORAL_STATUS:
            return self.answer_temporal_status(law_id, contract)
        return None

    def detect_temporal_type(self, contract: QueryContract) -> TemporalQueryType:
        """Infer the temporal question family from the compiled contract.

        Args:
            contract: Compiled temporal query contract.

        Returns:
            TemporalQueryType: Structured temporal family.
        """

        query = contract.query_text.casefold()
        if "current version" in query or "replaced" in query or "supersed" in query or "repealed" in query:
            return TemporalQueryType.SUPERSESSION
        if "what amendments" in query or "which amendments" in query or "amendment list" in query:
            return TemporalQueryType.AMENDMENT_LIST
        if "when was" in query and "amend" in query:
            return TemporalQueryType.AMENDMENT_DATE
        if "take effect" in query or "effective date" in query:
            return TemporalQueryType.EFFECTIVE_DATE
        if (
            contract.time_scope.scope_type is TimeScopeType.SPECIFIC_DATE
            or "currently in force" in query
            or "in force" in query
        ):
            return TemporalQueryType.TEMPORAL_STATUS
        return TemporalQueryType.COMMENCEMENT

    def should_fallback(self, contract: QueryContract) -> bool:
        """Decide whether standard RAG should handle the temporal query.

        Args:
            contract: Compiled temporal query contract.

        Returns:
            bool: True when structured temporal handling is unsafe.
        """

        if contract.predicate is not PredicateType.TEMPORAL:
            return True
        if not contract.primary_entities:
            return True
        query = contract.query_text.casefold()
        return any(marker in query for marker in ("article ", "section ", "schedule "))

    def answer_commencement(self, law_id: str) -> TemporalResult | None:
        """Answer a commencement-date question.

        Args:
            law_id: Target law document ID.

        Returns:
            TemporalResult | None: Structured commencement answer.
        """

        record = self._graph.get_commencement(law_id)
        if record is None or not record.commencement_date.strip():
            return None
        return TemporalResult(
            query_type=TemporalQueryType.COMMENCEMENT,
            answer_value=record.commencement_date,
            answer_formatted=record.commencement_date,
            provenance_page_ids=[record.evidence_page_id] if record.evidence_page_id else [],
            confidence=0.95,
        )

    def answer_amendment_date(self, law_id: str) -> TemporalResult | None:
        """Answer a latest-amendment-date question.

        Args:
            law_id: Target law document ID.

        Returns:
            TemporalResult | None: Structured amendment-date answer.
        """

        history = [edge for edge in self._graph.get_amendment_history(law_id) if edge.effective_date]
        if not history:
            return None
        latest = history[-1]
        return TemporalResult(
            query_type=TemporalQueryType.AMENDMENT_DATE,
            answer_value=str(latest.effective_date or ""),
            answer_formatted=str(latest.effective_date or ""),
            provenance_edges=[latest],
            provenance_page_ids=[latest.evidence_page_id] if latest.evidence_page_id else [],
            confidence=0.85,
        )

    def answer_amendment_list(self, law_id: str) -> TemporalResult | None:
        """Answer an amendment-list question.

        Args:
            law_id: Target law document ID.

        Returns:
            TemporalResult | None: Structured amendment-list answer.
        """

        history = self._graph.get_amendment_history(law_id)
        if not history:
            return None
        amendment_titles = [self._document_label(edge.source_doc_id) for edge in history]
        return TemporalResult(
            query_type=TemporalQueryType.AMENDMENT_LIST,
            answer_value=", ".join(amendment_titles),
            answer_formatted=", ".join(amendment_titles),
            provenance_edges=history,
            provenance_page_ids=self._edge_page_ids(history),
            confidence=0.82,
        )

    def answer_supersession(self, law_id: str) -> TemporalResult | None:
        """Answer a supersession/current-version question.

        Args:
            law_id: Target law document ID.

        Returns:
            TemporalResult | None: Structured supersession answer.
        """

        chain = self._supersession_chain(law_id)
        current = self._graph.get_current_version(law_id)
        if current is None:
            return None
        return TemporalResult(
            query_type=TemporalQueryType.SUPERSESSION,
            answer_value=current.title,
            answer_formatted=current.title,
            provenance_edges=chain,
            provenance_page_ids=self._edge_page_ids(chain),
            confidence=0.87 if chain else 0.75,
        )

    def answer_effective_date(self, law_id: str) -> TemporalResult | None:
        """Answer an effective-date question.

        Args:
            law_id: Target law document ID.

        Returns:
            TemporalResult | None: Structured effective-date answer.
        """

        commencement = self.answer_commencement(law_id)
        if commencement is not None:
            return commencement.model_copy(update={"query_type": TemporalQueryType.EFFECTIVE_DATE})
        return self.answer_amendment_date(law_id)

    def answer_temporal_status(self, law_id: str, contract: QueryContract) -> TemporalResult | None:
        """Answer a boolean/current-status temporal question.

        Args:
            law_id: Target law document ID.
            contract: Compiled temporal query contract.

        Returns:
            TemporalResult | None: Structured temporal-status answer.
        """

        commencement = self._graph.get_commencement(law_id)
        if commencement is None:
            return None
        reference_date = self._reference_date(contract)
        commencement_date = self._parse_date(commencement.commencement_date)
        if reference_date is None or commencement_date is None:
            return None
        end_edge = self._first_ending_edge(law_id)
        end_date = self._parse_date(str(end_edge.effective_date)) if end_edge is not None and end_edge.effective_date else None
        in_force = reference_date >= commencement_date and (end_date is None or reference_date < end_date)
        provenance_edges = [edge for edge in [end_edge] if edge is not None]
        provenance_page_ids = self._edge_page_ids(provenance_edges)
        if commencement.evidence_page_id:
            provenance_page_ids.append(commencement.evidence_page_id)
        return TemporalResult(
            query_type=TemporalQueryType.TEMPORAL_STATUS,
            answer_value="yes" if in_force else "no",
            answer_formatted=self.format_temporal_answer("yes" if in_force else "no", contract.answer_type),
            provenance_edges=provenance_edges,
            provenance_page_ids=self._dedupe_pages(provenance_page_ids),
            confidence=0.8,
        )

    @staticmethod
    def format_temporal_answer(answer_value: str, answer_type: str) -> str:
        """Format a temporal answer for pipeline emission.

        Args:
            answer_value: Structured answer value.
            answer_type: Requested answer type.

        Returns:
            str: Final answer string.
        """

        if answer_type.strip().lower() == "boolean":
            return "Yes" if answer_value.casefold() == "yes" else "No"
        return answer_value.strip()

    def _resolve_law_id(self, contract: QueryContract) -> str:
        """Resolve the primary law ID for a temporal contract.

        Args:
            contract: Compiled temporal query contract.

        Returns:
            str: Target law document ID, or an empty string.
        """

        for entity in contract.primary_entities:
            for doc_id in entity.source_doc_ids:
                if doc_id in self._graph.nodes:
                    return doc_id
        return ""

    def _supersession_chain(self, law_id: str) -> list[ApplicabilityEdge]:
        """Follow supersession edges from one law to its latest replacement.

        Args:
            law_id: Starting law ID.

        Returns:
            list[ApplicabilityEdge]: Traversed supersession chain.
        """

        chain: list[ApplicabilityEdge] = []
        current_id = law_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            next_edge = next(
                (
                    edge
                    for edge in self._graph.edges
                    if edge.target_doc_id == current_id
                    and edge.edge_type in {
                        ApplicabilityEdgeType.SUPERSEDES,
                        ApplicabilityEdgeType.REPLACES,
                        ApplicabilityEdgeType.REPEALS,
                        ApplicabilityEdgeType.REVOKES,
                    }
                ),
                None,
            )
            if next_edge is None:
                break
            chain.append(next_edge)
            current_id = next_edge.source_doc_id
        return chain

    def _first_ending_edge(self, law_id: str) -> ApplicabilityEdge | None:
        """Return the earliest known edge that ends or replaces a law.

        Args:
            law_id: Target law document ID.

        Returns:
            ApplicabilityEdge | None: Earliest terminating edge.
        """

        candidates = [
            edge
            for edge in self._graph.edges
            if edge.target_doc_id == law_id
            and edge.edge_type in {
                ApplicabilityEdgeType.SUPERSEDES,
                ApplicabilityEdgeType.REPLACES,
                ApplicabilityEdgeType.REPEALS,
                ApplicabilityEdgeType.REVOKES,
            }
            and edge.effective_date
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda edge: (self._parse_date(str(edge.effective_date)) or datetime.max.replace(tzinfo=UTC)),
        )[0]

    @staticmethod
    def _parse_date(raw_value: str) -> datetime | None:
        """Parse a legal date string conservatively.

        Args:
            raw_value: Raw date text.

        Returns:
            datetime | None: Parsed date, or ``None`` when unsupported.
        """

        cleaned = raw_value.strip()
        if not cleaned or cleaned.casefold() == "date specified in the enactment notice":
            return None
        for fmt in ("%d %B %Y", "%B %Y", "%Y-%m-%d", "%Y"):
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

    def _reference_date(self, contract: QueryContract) -> datetime | None:
        """Resolve the temporal reference date for a status query.

        Args:
            contract: Compiled temporal query contract.

        Returns:
            datetime | None: Reference date for in-force evaluation.
        """

        if contract.time_scope.scope_type is TimeScopeType.SPECIFIC_DATE:
            return self._parse_date(contract.time_scope.reference_date)
        return datetime.now(tz=UTC)

    @staticmethod
    def _edge_page_ids(edges: Iterable[ApplicabilityEdge]) -> list[str]:
        """Collect deduplicated page IDs from temporal provenance edges.

        Args:
            edges: Applicability edges with evidence pages.

        Returns:
            list[str]: Deduplicated provenance page IDs.
        """

        ordered: list[str] = []
        seen: set[str] = set()
        for edge in edges:
            page_id = edge.evidence_page_id.strip()
            if not page_id or page_id in seen:
                continue
            seen.add(page_id)
            ordered.append(page_id)
        return ordered

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

    def _document_label(self, doc_id: str) -> str:
        """Return the preferred display label for a document.

        Args:
            doc_id: Target document ID.

        Returns:
            str: Human-readable title.
        """

        if doc_id in self._document_titles:
            return self._document_titles[doc_id]
        law = cast("LawObject | None", self._graph.laws.get(doc_id))
        if law is not None and law.title.strip():
            return law.title
        return doc_id
