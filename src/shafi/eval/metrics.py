# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from deepeval.metrics import BaseMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepeval.test_case import LLMTestCase

_CHUNK_ID_PATTERN = re.compile(r"chunk_id\s*=\s*([A-Za-z0-9._:\-]+)")
_CITE_PATTERN = re.compile(r"\(cite:\s*([^)]+)\)", re.IGNORECASE)
_DATE_LIKE_PATTERN = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})$"
)
_NUMBER_LIKE_PATTERN = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")


@dataclass
class GoldChunkRecallAtK(BaseMetric):
    """Fraction of gold chunk IDs found in retrieved chunk IDs within top-k."""

    k: int = 10
    threshold: float = 1.0
    name_override: str = ""

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        del args, kwargs
        retrieval_context_obj: object = (
            getattr(test_case, "retrieval_context", None) or getattr(test_case, "context", None) or []
        )
        retrieval_context = self.coerce_context_items(retrieval_context_obj)
        retrieved_ids = self.extract_context_chunk_ids(retrieval_context)[: max(0, int(self.k))]
        gold_ids = self._parse_gold_ids(str(getattr(test_case, "expected_output", "") or ""))

        if not gold_ids:
            self.score = 0.0
            self.success = False
            self.reason = "No gold chunk IDs found in expected_output"
            return 0.0

        hits = len(gold_ids.intersection(set(retrieved_ids)))
        score = hits / len(gold_ids)
        self.score = score
        self.success = score >= float(self.threshold)
        self.reason = f"hits={hits}/{len(gold_ids)} within top-{self.k}"
        return score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def name(self) -> str:
        return self.name_override or f"gold_chunk_recall@{self.k}"

    @staticmethod
    def extract_context_chunk_ids(context_items: Sequence[str]) -> list[str]:
        ids: list[str] = []
        for item in context_items:
            match = _CHUNK_ID_PATTERN.search(item)
            if match is not None:
                ids.append(match.group(1))
        return ids

    @staticmethod
    def coerce_context_items(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items = cast("list[object]", value)
        return [text for item in items if (text := str(item).strip())]

    @staticmethod
    def _parse_gold_ids(expected: str) -> set[str]:
        raw = expected.strip()
        if not raw:
            return set()
        if "gold_chunk_ids=" in raw:
            raw = raw.split("gold_chunk_ids=", maxsplit=1)[1]
        return {part.strip() for part in raw.split(",") if part.strip()}


@dataclass
class CitationCoverage(BaseMetric):
    """Fraction of cited chunk IDs that are present in retrieval context."""

    threshold: float = 1.0

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        del args, kwargs
        answer = str(getattr(test_case, "actual_output", "") or "")
        cited_ids = self.extract_cited_chunk_ids(answer)
        if not cited_ids:
            self.score = 0.0
            self.success = False
            self.reason = "No citations found in answer"
            return 0.0

        retrieval_context_obj: object = getattr(test_case, "retrieval_context", None) or []
        retrieval_context = GoldChunkRecallAtK.coerce_context_items(retrieval_context_obj)
        context_ids = set(GoldChunkRecallAtK.extract_context_chunk_ids(retrieval_context))
        if not context_ids:
            self.score = 0.0
            self.success = False
            self.reason = "No retrieval_context chunk IDs found"
            return 0.0

        valid = cited_ids.intersection(context_ids)
        score = len(valid) / len(cited_ids)
        self.score = score
        self.success = score >= float(self.threshold)
        self.reason = f"valid_citations={len(valid)}/{len(cited_ids)}"
        return score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def name(self) -> str:
        return "citation_coverage"

    @staticmethod
    def extract_cited_chunk_ids(answer: str) -> set[str]:
        ids: set[str] = set()
        for match in _CITE_PATTERN.finditer(answer):
            for raw_id in re.split(r"[,;]", match.group(1)):
                chunk_id = raw_id.strip()
                if chunk_id:
                    ids.add(chunk_id)
        return ids


@dataclass
class AnswerTypeFormatCompliance(BaseMetric):
    """Checks whether answer formatting matches dataset `answer_type`."""

    threshold: float = 1.0

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        del args, kwargs
        answer = str(getattr(test_case, "actual_output", "") or "").strip()
        answer_type = self._extract_answer_type(test_case)

        if not answer_type:
            self.score = 0.0
            self.success = False
            self.reason = "Missing answer_type"
            return 0.0

        ok = self.is_answer_format_compliant(answer, answer_type)
        self.score = 1.0 if ok else 0.0
        self.success = self.score >= float(self.threshold)
        self.reason = f"answer_type={answer_type}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def name(self) -> str:
        return "answer_type_format_compliance"

    @classmethod
    def is_answer_format_compliant(cls, answer: str, answer_type: str) -> bool:
        text = cls._normalize_for_format_check(answer)
        if not text:
            return False

        # "null" is the correct organizer-aligned answer for unanswerable strict-type questions.
        if text.lower() == "null":
            kind_check = answer_type.strip().lower()
            return kind_check in ("boolean", "number", "date", "name", "names")

        kind = answer_type.strip().lower()
        if kind == "boolean":
            lowered = text.lower()
            return lowered.startswith("yes") or lowered.startswith("no")
        if kind == "number":
            return _NUMBER_LIKE_PATTERN.fullmatch(text) is not None
        if kind == "date":
            return _DATE_LIKE_PATTERN.fullmatch(text) is not None
        if kind == "names":
            # Allow a single-name list (some questions ask for plural but have only one item).
            return cls._looks_like_name(text) or ("," in text) or (" and " in text.lower()) or ("\n" in text)
        if kind == "name":
            return cls._looks_like_name(text)
        if kind == "free_text":
            return bool(text)

        return bool(text)

    @staticmethod
    def _extract_answer_type(test_case: LLMTestCase) -> str:
        metadata_obj: object = getattr(test_case, "additional_metadata", None) or {}
        if isinstance(metadata_obj, dict):
            metadata = cast("dict[str, object]", metadata_obj)
            answer_type_obj = metadata.get("answer_type")
            if isinstance(answer_type_obj, str) and answer_type_obj.strip():
                return answer_type_obj.strip()

        expected = str(getattr(test_case, "expected_output", "") or "")
        for part in expected.split(";"):
            segment = part.strip()
            if segment.startswith("answer_type="):
                return segment.split("=", maxsplit=1)[1].strip()
        return ""

    @staticmethod
    def _normalize_for_format_check(answer: str) -> str:
        # Remove citations so `(cite: ...)` does not break number/date/boolean checks.
        text = _CITE_PATTERN.sub("", answer).strip()
        text = re.sub(r"\s+", " ", text)
        return text.rstrip(" .;")

    @staticmethod
    def _looks_like_name(text: str) -> bool:
        if len(text.split()) > 12:
            return False
        # Reject ! and ? (signals free text), but allow periods which are
        # common in entity names: "L.L.C", "Inc.", "Dr.", "Art. 5".
        return "!" not in text and "?" not in text
