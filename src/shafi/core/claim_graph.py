from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from shafi.core.span_grounder import EvidenceSpan, SpanGrounder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from shafi.models import RankedChunk


_CITE_RE = re.compile(r"\(cite:[^)]+\)", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)-]|[-*])\s*")
_WHITESPACE_RE = re.compile(r"\s+")


class SupportType(StrEnum):
    """Support classes for claim-to-span provenance."""

    DIRECTLY_STATED = "directly_stated"
    INFERRED = "inferred"
    COMPUTED = "computed"
    UNSUPPORTED = "unsupported"


class Claim(BaseModel):
    """Atomic claim produced by claim decomposition."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_text: str
    support_type: SupportType = SupportType.UNSUPPORTED
    evidence_spans: list[EvidenceSpan] = []
    confidence: float = 0.0
    depends_on: list[str] = []


class ClaimGraph(BaseModel):
    """Answer-level claim graph with provenance links."""

    model_config = ConfigDict(frozen=True)

    claims: list[Claim] = []
    answer_text: str = ""
    query_contract: object | None = None
    unsupported_claims: list[str] = []
    support_coverage: float = 0.0
    dependency_edges: list[tuple[str, str]] = []


class ClaimGraphBuilder:
    """Build a claim graph by grounding answer sentences against context."""

    def __init__(
        self,
        *,
        claim_extractor: Callable[[str], list[str]] | None = None,
        grounder: SpanGrounder | None = None,
    ) -> None:
        """Initialize the claim-graph builder.

        Args:
            claim_extractor: Optional custom claim extractor for LLM-backed tests.
            grounder: Optional span grounder dependency.
        """

        self._claim_extractor = claim_extractor
        self._grounder = grounder or SpanGrounder()

    def build(
        self,
        answer_text: str,
        retrieved_context: Sequence[RankedChunk],
        segments: Sequence[object] | None,
        query_contract: object | None,
    ) -> ClaimGraph:
        """Build a grounded claim graph from a final answer and context.

        Args:
            answer_text: Final answer text to decompose.
            retrieved_context: Ranked chunks used to support the answer.
            segments: Optional compiled segments for segment-level upgrades.
            query_contract: Compiled query contract for routing context.

        Returns:
            ClaimGraph: Structured claim graph with support metadata.
        """

        claims_text = self.decompose_claims(answer_text)
        claims: list[Claim] = []
        for index, claim_text in enumerate(claims_text):
            evidence_spans = self.find_evidence(claim_text, retrieved_context, segments or [])
            support_type = self.classify_support(claim_text, evidence_spans)
            claims.append(
                Claim(
                    claim_id=f"claim_{index + 1}",
                    claim_text=claim_text,
                    support_type=support_type,
                    evidence_spans=evidence_spans,
                    confidence=self._confidence_for_claim(support_type, evidence_spans),
                    depends_on=[],
                )
            )

        dependency_edges = self.build_dependency_edges(claims)
        claims_by_id = {claim.claim_id: claim for claim in claims}
        for upstream_id, downstream_id in dependency_edges:
            downstream = claims_by_id.get(downstream_id)
            if downstream is None:
                continue
            depends_on = [*downstream.depends_on, upstream_id]
            claims_by_id[downstream_id] = downstream.model_copy(update={"depends_on": self._dedupe_ids(depends_on)})
        claims = [claims_by_id[claim.claim_id] for claim in claims]
        unsupported_claims = [claim.claim_id for claim in claims if claim.support_type is SupportType.UNSUPPORTED]
        coverage = self.compute_coverage(claims)

        return ClaimGraph(
            claims=claims,
            answer_text=answer_text,
            query_contract=query_contract,
            unsupported_claims=unsupported_claims,
            support_coverage=coverage,
            dependency_edges=dependency_edges,
        )

    def decompose_claims(self, answer_text: str) -> list[str]:
        """Decompose answer text into atomic claim strings.

        Args:
            answer_text: Free-text or strict answer candidate.

        Returns:
            list[str]: Decomposed claim texts.
        """

        if self._claim_extractor is not None:
            extracted = [self._clean_claim_text(text) for text in self._claim_extractor(answer_text)]
            return [text for text in extracted if text]
        text = self._strip_citations(str(answer_text or ""))
        text = text.replace("\r", "\n")
        chunks = re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
        claims: list[str] = []
        for chunk in chunks:
            candidate = self._clean_claim_text(chunk)
            if not candidate:
                continue
            if len(candidate) < 8 and claims:
                claims[-1] = self._clean_claim_text(f"{claims[-1]} {candidate}")
            else:
                claims.append(candidate)
        if not claims and self._clean_claim_text(text):
            claims.append(self._clean_claim_text(text))
        return claims

    def find_evidence(
        self,
        claim_text: str,
        context_pages: Sequence[RankedChunk],
        segments: Sequence[object],
    ) -> list[EvidenceSpan]:
        """Find evidence spans for a claim using page- and segment-level context.

        Args:
            claim_text: Atomic claim text.
            context_pages: Ranked chunks used as supporting context.
            segments: Optional compiled segments for segment upgrades.

        Returns:
            list[EvidenceSpan]: Ordered evidence spans.
        """

        return self._grounder.ground_claim_to_spans(claim_text, context_pages, segments)

    def classify_support(self, claim_text: str, evidence_spans: Sequence[EvidenceSpan]) -> SupportType:
        """Classify the provenance support class for one claim.

        Args:
            claim_text: Atomic claim text.
            evidence_spans: Candidate evidence spans.

        Returns:
            SupportType: Support class.
        """

        if not evidence_spans:
            return SupportType.UNSUPPORTED
        normalized = self._clean_claim_text(claim_text).casefold()
        if any(self._clean_claim_text(span.span_text).casefold().find(normalized) >= 0 for span in evidence_spans):
            return SupportType.DIRECTLY_STATED
        if self._looks_computed(claim_text, evidence_spans):
            return SupportType.COMPUTED
        if any(span.match_method == "exact" for span in evidence_spans):
            return SupportType.DIRECTLY_STATED
        return SupportType.INFERRED

    def build_dependency_edges(self, claims: Sequence[Claim]) -> list[tuple[str, str]]:
        """Build a lightweight dependency graph over sequential claims.

        Args:
            claims: Ordered claims.

        Returns:
            list[tuple[str, str]]: Directed dependency edges.
        """

        edges: list[tuple[str, str]] = []
        previous: Claim | None = None
        for claim in claims:
            if previous is not None:
                edges.append((previous.claim_id, claim.claim_id))
            previous = claim
        return edges

    @staticmethod
    def compute_coverage(claims: Sequence[Claim]) -> float:
        """Compute the fraction of claims with supporting evidence.

        Args:
            claims: Ordered claims.

        Returns:
            float: Coverage in `[0.0, 1.0]`.
        """

        if not claims:
            return 0.0
        supported = sum(1 for claim in claims if claim.support_type is not SupportType.UNSUPPORTED)
        return supported / len(claims)

    @staticmethod
    def _clean_claim_text(text: str) -> str:
        """Normalize a claim text chunk for downstream matching.

        Args:
            text: Raw claim text.

        Returns:
            str: Cleaned claim text.
        """

        cleaned = _CITE_RE.sub(" ", str(text or ""))
        cleaned = _LIST_PREFIX_RE.sub("", cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned.rstrip(".").strip()

    @staticmethod
    def _strip_citations(text: str) -> str:
        """Remove inline citation markers from an answer string.

        Args:
            text: Answer text with inline citations.

        Returns:
            str: Text without inline citation markers.
        """

        return _CITE_RE.sub(" ", str(text or ""))

    @staticmethod
    def _dedupe_ids(ids: Sequence[str]) -> list[str]:
        """Deduplicate identifiers while preserving order.

        Args:
            ids: Candidate identifiers.

        Returns:
            list[str]: Deduplicated identifiers.
        """

        seen: set[str] = set()
        out: list[str] = []
        for raw in ids:
            identifier = str(raw).strip()
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            out.append(identifier)
        return out

    @staticmethod
    def _confidence_for_claim(support_type: SupportType, evidence_spans: Sequence[EvidenceSpan]) -> float:
        """Derive a claim confidence score from support type and evidence count.

        Args:
            support_type: Classified support type.
            evidence_spans: Attached evidence spans.

        Returns:
            float: Confidence score in ``[0.0, 0.99]``.
        """

        if support_type is SupportType.UNSUPPORTED:
            return 0.1
        base = {
            SupportType.DIRECTLY_STATED: 0.9,
            SupportType.COMPUTED: 0.85,
            SupportType.INFERRED: 0.75,
        }.get(support_type, 0.5)
        return min(0.99, base + min(0.05, 0.01 * len(evidence_spans)))

    @staticmethod
    def _looks_computed(claim_text: str, evidence_spans: Sequence[EvidenceSpan]) -> bool:
        """Detect claims that are likely computed from multiple evidence spans.

        Args:
            claim_text: Claim text under inspection.
            evidence_spans: Candidate evidence spans.

        Returns:
            bool: True when the claim looks computed.
        """

        lowered = claim_text.casefold()
        numeric_terms = ("total", "sum", "difference", "greater", "less", "earlier", "later", "before", "after")
        if not any(term in lowered for term in numeric_terms):
            return False
        if len(evidence_spans) >= 2:
            return True
        return bool(re.search(r"\b\d+(?:[.,]\d+)?\b", claim_text))
