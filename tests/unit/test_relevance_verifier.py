"""Tests for bounded page relevance verifier parsing and validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from shafi.core.grounding.relevance_verifier import BoundedPageRelevanceVerifier
from shafi.models.schemas import RetrievedPage


class _FakeLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text=self._text)


def _make_page(page_id: str) -> RetrievedPage:
    doc_id, _, page_num = page_id.rpartition("_")
    return RetrievedPage(
        page_id=page_id,
        doc_id=doc_id,
        page_num=int(page_num),
        doc_title="Employment Law",
        doc_type="statute",
        page_text="Article 16 requires annual returns.",
        score=0.8,
    )


def test_parse_verification_json_accepts_valid_payload() -> None:
    parsed = BoundedPageRelevanceVerifier._parse_verification_json(
        """
        {
          "selected_page_ids": ["law_2"],
          "selection_mode": "single",
          "confidence": 0.88,
          "candidate_assessments": [
            {
              "page_id": "law_2",
              "evidence_role": "primary",
              "covered_slots": ["article_anchor"],
              "reasons": ["exact_article"]
            }
          ],
          "reasons": ["exact_article_page"]
        }
        """
    )

    assert parsed is not None
    assert parsed.used is True
    assert parsed.selected_page_ids == ("law_2",)
    assert parsed.selection_mode == "single"
    assert parsed.candidate_assessments[0].evidence_role == "primary"


@pytest.mark.asyncio
async def test_verify_rejects_selection_that_exceeds_max_selected_pages() -> None:
    verifier = BoundedPageRelevanceVerifier(
        llm=_FakeLLM(
            """
            {
              "selected_page_ids": ["law_1", "law_2"],
              "selection_mode": "pair",
              "confidence": 0.92,
              "candidate_assessments": [],
              "reasons": ["pair_required"]
            }
            """
        ),
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        min_confidence=0.7,
    )

    result = await verifier.verify(
        query="Who was the claimant?",
        answer_type="name",
        required_slots=("party_title",),
        candidate_pages=[_make_page("law_1"), _make_page("law_2")],
        max_selected_pages=1,
    )

    assert result.used is False
    assert result.fallback_reason == "selected_too_many_pages"
