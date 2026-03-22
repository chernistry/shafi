from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from shafi.core.claim_graph import Claim, ClaimGraph, SupportType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.core.query_contract import QueryContract


_WHITESPACE_RE = re.compile(r"\s+")


class ProofAnswer(BaseModel):
    """A proof-carrying answer and its provenance chain."""

    model_config = ConfigDict(frozen=True)

    answer_text: str
    verified_claims: list[Claim] = []
    dropped_claims: list[Claim] = []
    provenance_chain: dict[int, list[str]] = {}
    support_coverage: float = 0.0
    is_fully_supported: bool = False
    fallback_reason: str = ""


@dataclass(frozen=True)
class ProofCompilerConfig:
    """Configuration for proof-carried answer compilation."""

    min_support_coverage: float = 0.5
    allow_partial_answers: bool = True
    fluency_pass_enabled: bool = True


class ProofCarryingCompiler:
    """Compile answers from verified claims and attached evidence only."""

    def __init__(self, config: ProofCompilerConfig | None = None) -> None:
        """Initialize the proof compiler.

        Args:
            config: Optional compiler configuration.
        """

        self._config = config or ProofCompilerConfig()

    def compile(self, claim_graph: ClaimGraph | None, query_contract: QueryContract | None) -> ProofAnswer:
        """Compile a proof-carrying answer from a claim graph.

        Args:
            claim_graph: Source claim graph.
            query_contract: Typed query contract for routing.

        Returns:
            ProofAnswer: Compiled proof answer or a fallback sentinel.
        """

        if claim_graph is None or not claim_graph.claims:
            return ProofAnswer(
                answer_text="",
                fallback_reason="claim_graph_unavailable",
            )

        verified_claims, dropped_claims = self.filter_verified_claims(claim_graph.claims)
        if not verified_claims:
            return ProofAnswer(
                answer_text="",
                dropped_claims=dropped_claims,
                support_coverage=0.0,
                is_fully_supported=False,
                fallback_reason="no_verified_claims",
            )

        ordered_verified = self.order_claims(verified_claims, query_contract)
        answer_text = self._compile_answer_text(ordered_verified, query_contract, claim_graph.answer_text)
        answer_text = self.ensure_fluency(answer_text)
        provenance = self.attach_provenance(answer_text, ordered_verified)
        coverage = claim_graph.support_coverage
        fully_supported = coverage >= self._config.min_support_coverage and not dropped_claims

        if coverage < self._config.min_support_coverage and not self._config.allow_partial_answers:
            return ProofAnswer(
                answer_text="",
                verified_claims=ordered_verified,
                dropped_claims=dropped_claims,
                provenance_chain=provenance,
                support_coverage=coverage,
                is_fully_supported=False,
                fallback_reason="coverage_below_threshold",
            )

        return ProofAnswer(
            answer_text=answer_text,
            verified_claims=ordered_verified,
            dropped_claims=dropped_claims,
            provenance_chain=provenance,
            support_coverage=coverage,
            is_fully_supported=fully_supported,
        )

    def filter_verified_claims(self, claims: Sequence[Claim]) -> tuple[list[Claim], list[Claim]]:
        """Separate verified claims from unsupported claims.

        Args:
            claims: Candidate claims from the claim graph.

        Returns:
            tuple[list[Claim], list[Claim]]: Verified and dropped claims.
        """

        verified: list[Claim] = []
        dropped: list[Claim] = []
        for claim in claims:
            if (
                claim.support_type in {SupportType.DIRECTLY_STATED, SupportType.INFERRED, SupportType.COMPUTED}
                and claim.evidence_spans
            ):
                verified.append(claim)
            else:
                dropped.append(claim)
        return verified, dropped

    def order_claims(self, verified: Sequence[Claim], contract: QueryContract | None) -> list[Claim]:
        """Order claims by support strength and contract relevance.

        Args:
            verified: Verified claims.
            contract: Typed query contract.

        Returns:
            list[Claim]: Ordered claims for compilation.
        """

        field_name = str(getattr(contract, "field_name", "") or "").strip().casefold()
        answer_type = str(getattr(contract, "answer_type", "") or "").strip().casefold()

        def _score(claim: Claim) -> tuple[int, int, int]:
            support_rank = {
                SupportType.DIRECTLY_STATED: 3,
                SupportType.COMPUTED: 2,
                SupportType.INFERRED: 1,
            }.get(claim.support_type, 0)
            relevance = 0
            claim_lower = claim.claim_text.casefold()
            if field_name and field_name in claim_lower:
                relevance += 3
            if answer_type in {"boolean", "name", "names", "date", "number"} and claim_lower.startswith(("yes", "no")):
                relevance += 2
            if any(marker in claim_lower for marker in ("because", "therefore", "thus", "hence")):
                relevance += 1
            return (support_rank, relevance, len(claim.evidence_spans))

        return sorted(list(verified), key=_score, reverse=True)

    def compile_boolean(self, verified: Sequence[Claim], contract: QueryContract | None) -> ProofAnswer:
        """Compile a proof answer for boolean queries."""

        if not verified:
            return ProofAnswer(answer_text="", fallback_reason="no_verified_claims")
        original = str(getattr(contract, "query_text", "") or "").casefold()
        yes_like = any(
            claim.claim_text.casefold().startswith(("yes", "true", "it applies", "applies")) for claim in verified
        )
        no_like = any(
            any(
                term in claim.claim_text.casefold()
                for term in ("no ", "not ", "does not", "did not", "without", "doesn't")
            )
            for claim in verified
        )
        if no_like and not yes_like:
            return self._compile_from_sentences(["No."], verified)
        if yes_like or " not " in original:
            return self._compile_from_sentences(["Yes."], verified)
        return self._compile_from_sentences([verified[0].claim_text], verified)

    def compile_field(self, verified: Sequence[Claim], contract: QueryContract | None) -> ProofAnswer:
        """Compile a proof answer for strict field lookups."""

        if not verified:
            return ProofAnswer(answer_text="", fallback_reason="no_verified_claims")
        ordered = self.order_claims(verified, contract)
        best = ordered[0]
        answer = self._strip_leading_prefix(best.claim_text)
        return self._compile_from_sentences([answer], ordered)

    def compile_free_text(self, verified: Sequence[Claim], contract: QueryContract | None) -> ProofAnswer:
        """Compile a proof answer for free-text questions."""

        if not verified:
            return ProofAnswer(answer_text="Insufficient information.", fallback_reason="no_verified_claims")
        ordered = self.order_claims(verified, contract)
        sentences: list[str] = []
        for claim in ordered:
            sentence = self._strip_leading_prefix(claim.claim_text)
            citations = self._claim_citations(claim)
            if citations:
                sentence = f"{sentence} (cite: {', '.join(citations)})"
            sentences.append(sentence)
        return self._compile_from_sentences(sentences, ordered)

    def attach_provenance(self, answer_text: str, claims: Sequence[Claim]) -> dict[int, list[str]]:
        """Attach a sentence-to-claim provenance chain to an answer."""

        sentences = [
            segment for segment in re.split(r"(?<=[.!?])\s+", str(answer_text or "").strip()) if segment.strip()
        ]
        provenance: dict[int, list[str]] = {}
        if not sentences:
            return provenance
        if len(sentences) == 1:
            provenance[0] = [claim.claim_id for claim in claims]
            return provenance
        for index, claim in enumerate(claims):
            sentence_index = min(index, len(sentences) - 1)
            provenance.setdefault(sentence_index, []).append(claim.claim_id)
        return provenance

    def handle_insufficient_evidence(self, dropped: Sequence[Claim], contract: QueryContract | None) -> ProofAnswer:
        """Build the fallback proof answer when evidence is insufficient."""

        del contract
        return ProofAnswer(
            answer_text="",
            dropped_claims=list(dropped),
            support_coverage=0.0,
            is_fully_supported=False,
            fallback_reason="insufficient_evidence",
        )

    def ensure_fluency(self, compiled_text: str) -> str:
        """Lightly normalize a compiled answer without adding content."""

        text = str(compiled_text or "").strip()
        if not text:
            return ""
        text = _WHITESPACE_RE.sub(" ", text)
        text = re.sub(r"\(\s*cite:\s*", "(cite: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+\)", ")", text)
        return text

    def _compile_answer_text(
        self,
        ordered_verified: Sequence[Claim],
        contract: QueryContract | None,
        original_answer_text: str,
    ) -> str:
        """Compile the final answer text from verified claims.

        Args:
            ordered_verified: Claims ordered for compilation.
            contract: Typed query contract.
            original_answer_text: Fallback original answer text.

        Returns:
            str: Compiled answer text.
        """

        answer_type = str(getattr(contract, "answer_type", "") or "").strip().casefold()
        if answer_type == "boolean":
            compiled = self.compile_boolean(ordered_verified, contract)
            if compiled.answer_text:
                return compiled.answer_text
        if answer_type in {"name", "names", "date", "number"}:
            compiled = self.compile_field(ordered_verified, contract)
            if compiled.answer_text:
                return compiled.answer_text
        compiled = self.compile_free_text(ordered_verified, contract)
        if compiled.answer_text and compiled.answer_text != "Insufficient information.":
            return compiled.answer_text
        return str(original_answer_text or "").strip()

    def _compile_from_sentences(self, sentences: Sequence[str], claims: Sequence[Claim]) -> ProofAnswer:
        """Compile a proof answer from a sequence of sentence fragments.

        Args:
            sentences: Sentence fragments to compile.
            claims: Claims backing the fragments.

        Returns:
            ProofAnswer: Compiled proof answer.
        """

        normalized_sentences = [self.ensure_fluency(sentence) for sentence in sentences if str(sentence).strip()]
        answer_text = " ".join(normalized_sentences).strip()
        if not answer_text:
            return ProofAnswer(answer_text="", fallback_reason="empty_compilation")
        provenance = self.attach_provenance(answer_text, claims)
        return ProofAnswer(
            answer_text=answer_text,
            verified_claims=list(claims),
            dropped_claims=[],
            provenance_chain=provenance,
            support_coverage=1.0 if claims else 0.0,
            is_fully_supported=bool(claims),
        )

    @staticmethod
    def _strip_leading_prefix(text: str) -> str:
        """Remove obvious answer-style prefixes from a claim sentence.

        Args:
            text: Candidate sentence.

        Returns:
            str: Sentence without answer-style prefixes.
        """

        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^\s*(?:answer|the answer|result|conclusion)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:yes|no)\s*[,;:-]\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _claim_citations(claim: Claim) -> list[str]:
        """Collect source chunk identifiers from a claim.

        Args:
            claim: Verified claim.

        Returns:
            list[str]: Unique chunk identifiers in order.
        """

        chunk_ids: list[str] = []
        seen: set[str] = set()
        for span in claim.evidence_spans:
            chunk_id = str(getattr(span, "chunk_id", "") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
        return chunk_ids
