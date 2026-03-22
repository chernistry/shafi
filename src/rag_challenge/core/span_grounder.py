from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rag_challenge.models import RankedChunk


_CITE_RE = re.compile(r"\(cite:[^)]+\)", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


class EvidenceSpan(BaseModel):
    """A grounded evidence span linked to a page or segment."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = ""
    page_id: str = ""
    segment_id: str = ""
    char_start: int = 0
    char_end: int = 0
    span_text: str = ""
    match_method: str = ""


class SpanGrounder:
    """Deterministic claim-to-span matcher over ranked chunks and segments."""

    def ground_claim_to_spans(
        self,
        claim_text: str,
        page_texts: Sequence[RankedChunk],
        segments: Sequence[object] | None = None,
    ) -> list[EvidenceSpan]:
        """Ground a claim against the provided pages and segments.

        Args:
            claim_text: Atomic claim text to ground.
            page_texts: Ranked chunks used as page-level evidence.
            segments: Optional compiled legal segments for segment-id upgrades.

        Returns:
            list[EvidenceSpan]: Ordered evidence spans, deduplicated by chunk/page.
        """

        normalized_claim = self._normalize_claim(claim_text)
        if not normalized_claim:
            return []

        spans: list[EvidenceSpan] = []
        for page in page_texts:
            page_text = str(getattr(page, "text", "") or "")
            if not page_text.strip():
                continue
            page_id = self._page_id_from_chunk(getattr(page, "chunk_id", ""))
            chunk_id = str(getattr(page, "chunk_id", "") or "").strip()
            if not page_id and not chunk_id:
                continue

            exact_spans = self.exact_match(claim_text, page_text, chunk_id=chunk_id, page_id=page_id)
            fuzzy_spans = self.fuzzy_match(claim_text, page_text, chunk_id=chunk_id, page_id=page_id)
            semantic_spans = self.semantic_match(claim_text, page_text, chunk_id=chunk_id, page_id=page_id)

            if exact_spans:
                spans.extend(exact_spans)
                continue
            if fuzzy_spans:
                spans.extend(fuzzy_spans)
                continue
            if semantic_spans:
                spans.extend(semantic_spans)

        matched = self.match_to_segments(spans, segments or [])
        return self._dedupe_spans(matched)

    def exact_match(
        self,
        claim_text: str,
        page_text: str,
        *,
        chunk_id: str = "",
        page_id: str = "",
    ) -> list[EvidenceSpan]:
        """Find exact claim matches in a page text.

        Args:
            claim_text: Atomic claim text.
            page_text: Candidate page text.
            chunk_id: Source chunk identifier.
            page_id: Normalized page identifier.

        Returns:
            list[EvidenceSpan]: Exact-match spans.
        """

        claim = self._normalize_claim(claim_text)
        if not claim:
            return []
        pattern = self._exact_pattern(claim)
        spans: list[EvidenceSpan] = []
        for match in pattern.finditer(page_text):
            spans.append(
                EvidenceSpan(
                    chunk_id=chunk_id,
                    page_id=page_id,
                    char_start=match.start(),
                    char_end=match.end(),
                    span_text=page_text[match.start() : match.end()].strip(),
                    match_method="exact",
                )
            )
            if len(spans) >= 3:
                break
        return spans

    def fuzzy_match(
        self,
        claim_text: str,
        page_text: str,
        *,
        chunk_id: str = "",
        page_id: str = "",
        threshold: float = 0.72,
    ) -> list[EvidenceSpan]:
        """Find fuzzy claim matches in a page text.

        Args:
            claim_text: Atomic claim text.
            page_text: Candidate page text.
            chunk_id: Source chunk identifier.
            page_id: Normalized page identifier.
            threshold: Minimum similarity for a fuzzy match.

        Returns:
            list[EvidenceSpan]: Fuzzy-match spans.
        """

        claim = self._normalize_claim(claim_text)
        if not claim:
            return []
        best = self._best_sentence_match(claim, page_text)
        if best is None or best[0] < threshold:
            return []
        start, end, sentence = best[1], best[2], best[3]
        return [
            EvidenceSpan(
                chunk_id=chunk_id,
                page_id=page_id,
                char_start=start,
                char_end=end,
                span_text=sentence.strip(),
                match_method="fuzzy",
            )
        ]

    def semantic_match(
        self,
        claim_text: str,
        page_text: str,
        *,
        chunk_id: str = "",
        page_id: str = "",
        threshold: float = 0.62,
    ) -> list[EvidenceSpan]:
        """Find semantic claim matches in a page text.

        Args:
            claim_text: Atomic claim text.
            page_text: Candidate page text.
            chunk_id: Source chunk identifier.
            page_id: Normalized page identifier.
            threshold: Minimum semantic overlap for a match.

        Returns:
            list[EvidenceSpan]: Semantic-match spans.
        """

        claim = self._normalize_claim(claim_text)
        if not claim:
            return []
        score, start, end, sentence = self._best_token_overlap_match(claim, page_text)
        if score < threshold:
            return []
        return [
            EvidenceSpan(
                chunk_id=chunk_id,
                page_id=page_id,
                char_start=start,
                char_end=end,
                span_text=sentence.strip(),
                match_method="semantic",
            )
        ]

    def match_to_segments(
        self,
        spans: Sequence[EvidenceSpan],
        segments: Sequence[object],
    ) -> list[EvidenceSpan]:
        """Upgrade page-level spans to segment-level spans when possible.

        Args:
            spans: Page-level evidence spans.
            segments: Compiled segment-like objects.

        Returns:
            list[EvidenceSpan]: Spans with segment ids attached where matched.
        """

        if not spans or not segments:
            return list(spans)

        upgraded: list[EvidenceSpan] = []
        for span in spans:
            segment_id = str(span.segment_id or "").strip()
            if segment_id:
                upgraded.append(span)
                continue
            matched_segment_id = ""
            span_page_id = str(span.page_id or "").strip()
            for segment in segments:
                candidate_segment_id = str(getattr(segment, "segment_id", "") or "").strip()
                page_ids = [str(page_id).strip() for page_id in getattr(segment, "page_ids", []) if str(page_id).strip()]
                if candidate_segment_id and candidate_segment_id == span.chunk_id:
                    matched_segment_id = candidate_segment_id
                    break
                if span_page_id and span_page_id in page_ids:
                    matched_segment_id = candidate_segment_id
                    break
            upgraded.append(span.model_copy(update={"segment_id": matched_segment_id}))
        return upgraded

    @staticmethod
    def _page_id_from_chunk(chunk_id: object) -> str:
        """Derive a platform-style page identifier from a chunk identifier.

        Args:
            chunk_id: Raw chunk identifier.

        Returns:
            str: Normalized page identifier or an empty string.
        """

        chunk = str(chunk_id or "").strip()
        if not chunk:
            return ""
        if ":" not in chunk and "_" in chunk:
            return chunk
        parts = chunk.split(":")
        if len(parts) < 2:
            return ""
        doc_id = parts[0].strip()
        page_idx_raw = parts[1].strip()
        if not doc_id or not page_idx_raw.isdigit():
            return ""
        return f"{doc_id}_{int(page_idx_raw) + 1}"

    @staticmethod
    def _normalize_claim(text: str) -> str:
        """Normalize claim text for deterministic matching.

        Args:
            text: Raw claim text.

        Returns:
            str: Normalized text suitable for span matching.
        """

        cleaned = _CITE_RE.sub(" ", str(text or ""))
        cleaned = cleaned.replace("•", " ")
        cleaned = re.sub(r"^\s*(?:\d+[.)-]|[-*])\s*", "", cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        return cleaned

    @staticmethod
    def _exact_pattern(claim: str) -> re.Pattern[str]:
        """Build a tolerant exact-match pattern from claim text.

        Args:
            claim: Normalized claim text.

        Returns:
            re.Pattern[str]: Compiled regex pattern.
        """

        tokens = [re.escape(token) for token in claim.split() if token.strip()]
        if not tokens:
            return re.compile(r"(?!x)x")
        pattern = r"\b" + r"\s+".join(tokens) + r"\b"
        return re.compile(pattern, re.IGNORECASE)

    @staticmethod
    def _sentence_spans(page_text: str) -> list[tuple[int, int, str]]:
        """Split page text into sentence-like spans with offsets.

        Args:
            page_text: Raw page text.

        Returns:
            list[tuple[int, int, str]]: Sentence spans with offsets.
        """

        text = str(page_text or "")
        if not text.strip():
            return []
        spans: list[tuple[int, int, str]] = []
        start = 0
        for match in _SENTENCE_SPLIT_RE.finditer(text):
            end = match.start()
            sentence = text[start:end].strip()
            if sentence:
                spans.append((start, end, sentence))
            start = match.end()
        tail = text[start:].strip()
        if tail:
            spans.append((start, len(text), tail))
        if not spans:
            spans.append((0, len(text), text.strip()))
        return spans

    def _best_sentence_match(self, claim: str, page_text: str) -> tuple[float, int, int, str] | None:
        """Find the best fuzzy sentence match for a claim.

        Args:
            claim: Normalized claim text.
            page_text: Candidate page text.

        Returns:
            tuple[float, int, int, str] | None: Best fuzzy match or ``None``.
        """

        best: tuple[float, int, int, str] | None = None
        claim_tokens = self._token_set(claim)
        for start, end, sentence in self._sentence_spans(page_text):
            candidate = self._normalize_claim(sentence)
            if not candidate:
                continue
            candidate_tokens = self._token_set(candidate)
            if len(claim_tokens & candidate_tokens) < 2:
                continue
            score = SequenceMatcher(None, claim.casefold(), candidate.casefold()).ratio()
            if best is None or score > best[0]:
                best = (score, start, end, sentence)
        return best

    def _best_token_overlap_match(self, claim: str, page_text: str) -> tuple[float, int, int, str]:
        """Find the best overlap-based semantic match for a claim.

        Args:
            claim: Normalized claim text.
            page_text: Candidate page text.

        Returns:
            tuple[float, int, int, str]: Best overlap match tuple.
        """

        claim_tokens = self._token_set(claim)
        best = (0.0, 0, 0, "")
        if not claim_tokens:
            return best
        for start, end, sentence in self._sentence_spans(page_text):
            candidate_tokens = self._token_set(sentence)
            if not candidate_tokens:
                continue
            overlap_count = len(claim_tokens & candidate_tokens)
            if overlap_count < 2:
                continue
            overlap = overlap_count / max(1, len(claim_tokens | candidate_tokens))
            if overlap > best[0]:
                best = (overlap, start, end, sentence)
        return best

    @staticmethod
    def _token_set(text: str) -> set[str]:
        """Convert text into a normalized token set for overlap scoring.

        Args:
            text: Raw text input.

        Returns:
            set[str]: Normalized token set.
        """

        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9./-]*", str(text or ""))
            if len(token) > 1
        }

    @staticmethod
    def _dedupe_spans(spans: Iterable[EvidenceSpan]) -> list[EvidenceSpan]:
        """Deduplicate spans by chunk/page/match position.

        Args:
            spans: Candidate evidence spans.

        Returns:
            list[EvidenceSpan]: Deduplicated spans in order.
        """

        seen: set[tuple[str, str, int, int, str]] = set()
        deduped: list[EvidenceSpan] = []
        for span in spans:
            key = (span.chunk_id, span.page_id, span.char_start, span.char_end, span.match_method)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(span)
        return deduped
