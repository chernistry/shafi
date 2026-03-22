"""Bounded page/fact relevance verification for grounding-sidecar arbitration."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from shafi.prompts import load_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shafi.llm.provider import LLMProvider
    from shafi.models.schemas import RetrievedPage

logger = logging.getLogger(__name__)

_VERIFY_SYSTEM_PROMPT = load_prompt("relevance_verifier/system")
_VERIFY_USER_PROMPT = load_prompt("relevance_verifier/user")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

EvidenceRole = Literal["primary", "secondary", "reference", "insufficient"]
SelectionMode = Literal["single", "pair", "empty"]


def _coerce_bounded_float(value: object, default: float = 0.0) -> float:
    """Convert a JSON scalar into a bounded float-compatible value.

    Args:
        value: Parsed JSON value.
        default: Fallback when the value is not float-compatible.

    Returns:
        float: Parsed float or the fallback default.
    """

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """Structured assessment for one candidate page.

    Args:
        page_id: Candidate page ID.
        evidence_role: Evidence role assigned by the verifier.
        covered_slots: Typed support slots covered by the page.
        reasons: Short machine-readable reasons.
    """

    page_id: str
    evidence_role: EvidenceRole
    covered_slots: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelevanceVerificationResult:
    """Outcome of bounded page/fact relevance verification.

    Args:
        used: Whether the verifier produced a valid decision.
        selected_page_ids: Ordered winning page IDs.
        selection_mode: Single, pair, or empty selection mode.
        confidence: Verifier confidence in ``[0.0, 1.0]``.
        candidate_assessments: Structured assessments for candidates.
        reasons: Ordered short decision reasons.
        fallback_reason: Fail-closed reason when verification was rejected.
    """

    used: bool
    selected_page_ids: tuple[str, ...]
    selection_mode: SelectionMode
    confidence: float
    candidate_assessments: tuple[CandidateAssessment, ...]
    reasons: tuple[str, ...]
    fallback_reason: str = ""


class BoundedPageRelevanceVerifier:
    """Run one bounded LLM verification pass over top page candidates."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        model: str,
        max_tokens: int,
        temperature: float,
        min_confidence: float,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._min_confidence = min_confidence

    async def verify(
        self,
        *,
        query: str,
        answer_type: str,
        required_slots: Sequence[str],
        candidate_pages: Sequence[RetrievedPage],
        max_selected_pages: int,
    ) -> RelevanceVerificationResult:
        """Run bounded verification and parse the structured result.

        Args:
            query: Raw user query.
            answer_type: Normalized answer type.
            required_slots: Typed support slots requested by the query family.
            candidate_pages: Top candidate pages under consideration.
            max_selected_pages: Maximum number of pages allowed in the result.

        Returns:
            RelevanceVerificationResult: Validated verifier decision or a
            fail-closed fallback result.
        """

        if not candidate_pages:
            return RelevanceVerificationResult(
                used=False,
                selected_page_ids=(),
                selection_mode="empty",
                confidence=0.0,
                candidate_assessments=(),
                reasons=(),
                fallback_reason="no_candidate_pages",
            )

        user_prompt = _VERIFY_USER_PROMPT.format(
            question=query,
            answer_type=answer_type,
            required_slots=", ".join(required_slots) if required_slots else "none",
            candidates=self._format_candidates(candidate_pages),
        )

        try:
            result = await self._llm.generate(
                system_prompt=_VERIFY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as exc:  # pragma: no cover - exercised in integration
            logger.warning("Grounding relevance verifier failed", exc_info=True)
            return RelevanceVerificationResult(
                used=False,
                selected_page_ids=(),
                selection_mode="empty",
                confidence=0.0,
                candidate_assessments=(),
                reasons=(),
                fallback_reason=f"provider_error:{type(exc).__name__}",
            )

        parsed = self._parse_verification_json(result.text)
        if parsed is None:
            return RelevanceVerificationResult(
                used=False,
                selected_page_ids=(),
                selection_mode="empty",
                confidence=0.0,
                candidate_assessments=(),
                reasons=(),
                fallback_reason="invalid_json",
            )
        validation_error = self._validate_result(
            parsed=parsed,
            candidate_pages=candidate_pages,
            max_selected_pages=max_selected_pages,
        )
        if validation_error:
            return RelevanceVerificationResult(
                used=False,
                selected_page_ids=(),
                selection_mode="empty",
                confidence=0.0,
                candidate_assessments=(),
                reasons=parsed.reasons,
                fallback_reason=validation_error,
            )
        if parsed.confidence < self._min_confidence:
            return RelevanceVerificationResult(
                used=False,
                selected_page_ids=(),
                selection_mode="empty",
                confidence=parsed.confidence,
                candidate_assessments=parsed.candidate_assessments,
                reasons=parsed.reasons,
                fallback_reason="low_confidence",
            )
        return parsed

    @staticmethod
    def _format_candidates(candidate_pages: Sequence[RetrievedPage]) -> str:
        """Render candidate pages into compact verifier context.

        Args:
            candidate_pages: Top candidate pages.

        Returns:
            str: Compact structured candidate description.
        """

        sections: list[str] = []
        for page in candidate_pages:
            parts = [
                f"page_id: {page.page_id}",
                f"doc_id: {page.doc_id}",
                f"page_num: {page.page_num}",
                f"doc_title: {page.doc_title}",
                f"page_template_family: {page.page_template_family}",
                f"document_template_family: {page.document_template_family}",
                f"officialness_score: {page.officialness_score:.3f}",
                f"source_vs_reference_prior: {page.source_vs_reference_prior:.3f}",
                f"heading_lines: {' | '.join(page.heading_lines[:4])}",
                f"top_lines: {' | '.join(page.top_lines[:6])}",
                f"field_labels_present: {' | '.join(page.field_labels_present)}",
                f"article_refs: {' | '.join(page.article_refs[:4])}",
                f"law_titles: {' | '.join(page.law_titles[:3])}",
                f"text_excerpt: {page.page_text[:1200]}",
            ]
            sections.append("\n".join(parts))
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _parse_verification_json(text: str) -> RelevanceVerificationResult | None:
        """Parse verifier JSON into a structured result.

        Args:
            text: Raw model output.

        Returns:
            RelevanceVerificationResult | None: Parsed result or ``None``.
        """

        raw = text.strip()
        if not raw:
            return None
        candidates: list[str] = [raw]
        if raw.startswith("```"):
            stripped = re.sub(r"^```[\w-]*\s*", "", raw)
            stripped = re.sub(r"\s*```$", "", stripped)
            candidates.append(stripped.strip())
        match = _JSON_BLOCK_RE.search(raw)
        if match is not None:
            candidates.append(match.group(0).strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            parsed = BoundedPageRelevanceVerifier._coerce_payload(payload)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _coerce_payload(payload: object) -> RelevanceVerificationResult | None:
        """Coerce parsed JSON payload into a verification result.

        Args:
            payload: Parsed JSON payload.

        Returns:
            RelevanceVerificationResult | None: Structured result or ``None``.
        """

        if not isinstance(payload, dict):
            return None
        obj = cast("dict[str, object]", payload)
        selected_page_ids_raw = obj.get("selected_page_ids", [])
        if not isinstance(selected_page_ids_raw, list):
            selected_page_ids_raw = []
        selected_page_ids = tuple(
            str(page_id).strip() for page_id in cast("list[object]", selected_page_ids_raw) if str(page_id).strip()
        )

        selection_mode_raw = str(obj.get("selection_mode", "single")).strip().lower()
        selection_mode: SelectionMode = "single"
        if selection_mode_raw in {"single", "pair", "empty"}:
            selection_mode = cast("SelectionMode", selection_mode_raw)

        confidence = _coerce_bounded_float(obj.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        assessments_raw = obj.get("candidate_assessments", [])
        assessments: list[CandidateAssessment] = []
        if isinstance(assessments_raw, list):
            for item in cast("list[object]", assessments_raw):
                assessment = BoundedPageRelevanceVerifier._coerce_assessment(item)
                if assessment is not None:
                    assessments.append(assessment)

        reasons_raw = obj.get("reasons", [])
        reasons = tuple(
            str(reason).strip()
            for reason in cast("list[object]", reasons_raw if isinstance(reasons_raw, list) else [])
            if str(reason).strip()
        )
        return RelevanceVerificationResult(
            used=True,
            selected_page_ids=selected_page_ids,
            selection_mode=selection_mode,
            confidence=confidence,
            candidate_assessments=tuple(assessments),
            reasons=reasons,
        )

    @staticmethod
    def _coerce_assessment(payload: object) -> CandidateAssessment | None:
        """Coerce one candidate assessment from verifier JSON.

        Args:
            payload: Candidate payload.

        Returns:
            CandidateAssessment | None: Structured assessment or ``None``.
        """

        if not isinstance(payload, dict):
            return None
        obj = cast("dict[str, object]", payload)
        page_id = str(obj.get("page_id", "")).strip()
        role_raw = str(obj.get("evidence_role", "insufficient")).strip().lower()
        if role_raw not in {"primary", "secondary", "reference", "insufficient"}:
            role_raw = "insufficient"
        slots_raw = obj.get("covered_slots", [])
        reasons_raw = obj.get("reasons", [])
        slots = tuple(
            str(slot).strip()
            for slot in cast("list[object]", slots_raw if isinstance(slots_raw, list) else [])
            if str(slot).strip()
        )
        reasons = tuple(
            str(reason).strip()
            for reason in cast("list[object]", reasons_raw if isinstance(reasons_raw, list) else [])
            if str(reason).strip()
        )
        if not page_id:
            return None
        return CandidateAssessment(
            page_id=page_id,
            evidence_role=cast("EvidenceRole", role_raw),
            covered_slots=slots,
            reasons=reasons,
        )

    @staticmethod
    def _validate_result(
        *,
        parsed: RelevanceVerificationResult,
        candidate_pages: Sequence[RetrievedPage],
        max_selected_pages: int,
    ) -> str:
        """Validate the verifier result against hard bounded constraints.

        Args:
            parsed: Parsed verifier result.
            candidate_pages: Candidate pages presented to the model.
            max_selected_pages: Maximum allowed selection size.

        Returns:
            str: Empty string when valid, otherwise a fallback reason.
        """

        candidate_ids = {page.page_id for page in candidate_pages if page.page_id}
        if len(parsed.selected_page_ids) > max_selected_pages:
            return "selected_too_many_pages"
        if any(page_id not in candidate_ids for page_id in parsed.selected_page_ids):
            return "selected_unknown_page"
        if parsed.selection_mode == "single" and len(parsed.selected_page_ids) not in {0, 1}:
            return "single_mode_size_mismatch"
        if parsed.selection_mode == "pair" and len(parsed.selected_page_ids) != 2:
            return "pair_mode_size_mismatch"
        if parsed.selection_mode == "empty" and parsed.selected_page_ids:
            return "empty_mode_size_mismatch"
        return ""
