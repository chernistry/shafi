# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false, reportAttributeAccessIssue=false, reportUnknownArgumentType=false
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from rag_challenge.core.classifier import QueryClassifier
from rag_challenge.models import RankedChunk, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


from .constants import (
    _AMENDMENT_TITLE_RE,
    _COMMON_ELEMENTS_TITLE_STOPWORDS,
    _COMMON_ELEMENTS_TOKEN_RE,
    _DIFC_CASE_ID_RE,
    _GENERIC_SELF_ADMIN_RE,
    _ISO_DATE_RE,
    _LAW_NO_REF_RE,
    _REGISTRAR_SELF_ADMIN_RE,
    _SLASH_DATE_RE,
    _SUPPORT_STOPWORDS,
    _SUPPORT_TOKEN_RE,
    _TEXTUAL_DATE_RE,
    _TITLE_CONTEXT_BAD_LEAD_RE,
    _TITLE_GENERIC_QUESTION_LEAD_RE,
    _TITLE_LAW_NO_SUFFIX_RE,
    _TITLE_LEADING_CONNECTOR_RE,
    _TITLE_PREPOSITION_BAD_LEAD_RE,
    _TITLE_QUERY_BAD_LEAD_RE,
    _TITLE_REF_BAD_LEAD_RE,
    _TITLE_REF_RE,
)
from .query_rules import (
    _extract_question_title_refs,
    _is_account_effective_dates_query,
    _is_broad_enumeration_query,
    _is_case_issue_date_name_compare_query,
    _is_common_elements_query,
    _is_common_judge_compare_query,
    _is_interpretation_sections_common_elements_query,
    _is_named_amendment_query,
    _is_named_commencement_query,
    _is_named_multi_title_lookup_query,
)

logger = logging.getLogger(__name__)
RAGPipelineBuilder: Any = None

class RetrievalLogicMixin:
    @staticmethod
    def _augment_query_for_sparse_retrieval(query: str) -> str:
        """Add BM25-friendly variants for Article references (PDFs often render as '11. (1)' not 'Article 11(1)')."""
        raw = (query or "").strip()
        if not raw:
            return ""
        out = raw
        for match in re.finditer(r"\bArticle\s+(\d+)\s*\(\s*([^)]+?)\s*\)", raw, flags=re.IGNORECASE):
            num = match.group(1).strip()
            sub = match.group(2).strip()
            # Common PDF renderings.
            out += f" {num}({sub}) {num} ({sub}) {num}. ({sub})"
        return re.sub(r"\s+", " ", out).strip()

    @staticmethod
    def _extract_provision_refs(query: str) -> list[str]:
        raw = (query or "").strip()
        if not raw:
            return []
        refs: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(
            r"\b(?:Article|Section|Schedule|Part|Chapter)\s+\d+(?:\s*\(\s*[^)]+\s*\))?",
            re.IGNORECASE,
        )
        for match in pattern.finditer(raw):
            normalized = re.sub(r"\s+", " ", match.group(0)).strip()
            normalized = re.sub(
                r"\b(article|section|schedule|part|chapter)\b",
                lambda m: m.group(1).title(),
                normalized,
                count=1,
            )
            normalized = re.sub(r"\s*\(\s*", "(", normalized)
            normalized = re.sub(r"\s*\)\s*", ")", normalized)
            key = normalized.casefold()
            if not normalized or key in seen:
                continue
            seen.add(key)
            refs.append(normalized)
        return refs

    @classmethod
    def _targeted_provision_ref_query(
        cls,
        *,
        query: str,
        ref: str,
        refs: Sequence[str],
    ) -> str:
        base_query = query or ""
        for other_ref in refs:
            other_clean = str(other_ref).strip()
            if not other_clean or other_clean.casefold() == ref.casefold():
                continue
            base_query = re.sub(re.escape(other_clean), " ", base_query, flags=re.IGNORECASE)
        base_query = re.sub(r"\s+", " ", base_query).strip()

        provision_terms: list[str] = []
        for provision_ref in cls._extract_provision_refs(query)[:3]:
            provision_terms.append(provision_ref)
            if provision_ref.lower().startswith("article "):
                short = provision_ref[8:].strip()
                if short:
                    provision_terms.append(short)
                    provision_terms.append(re.sub(r"\(\s*", " (", short))

        targeted = " ".join([ref, *provision_terms, base_query]).strip()
        return re.sub(r"\s+", " ", targeted).strip()

    @staticmethod
    def _seed_terms_for_query(query: str) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []
        q_lower = q.lower()
        terms: list[str] = []

        if "enact" in q_lower:
            terms += ["enactment notice", "hereby enact", "ruler of dubai", "enacted"]
        if "come into force" in q_lower or "commencement" in q_lower:
            terms += ["come into force", "commencement", "commence"]
        if "administ" in q_lower:
            terms += ["administer", "administered", "administration", "commissioner", "relevant authority"]
        if "claim value" in q_lower or "claim amount" in q_lower or "amount claimed" in q_lower:
            terms += ["claim value", "claim amount", "amount claimed", "value of the claim"]
        if "financial services" in q_lower:
            terms += ["financial services", "undertake", "shall not", "may not", "prohibit", "prohibited"]
        if "liable" in q_lower or "liability" in q_lower:
            terms += ["can be held liable", "cannot be held liable", "liable", "liability", "bad faith", "does not apply"]
        if "delegate" in q_lower or "delegat" in q_lower:
            terms += ["delegate", "delegat", "approval"]
        if "restriction" in q_lower and "transfer" in q_lower:
            terms += ["restriction", "ineffective", "actual knowledge", "uncertificated", "notified"]

        # Article references: add both "article 11(1)" and "11 (1)".
        for match in re.finditer(r"\bArticle\s+\d+(?:\([^)]+\))?", q, flags=re.IGNORECASE):
            key = re.sub(r"\s+", " ", match.group(0)).strip().lower()
            if not key:
                continue
            terms.append(key)
            short = key.replace("article ", "").strip()
            if short:
                terms.append(short)
                terms.append(re.sub(r"\(\s*", " (", short))

        # Dedupe, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for term in terms:
            t = term.strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    @staticmethod
    def _dedupe_chunk_ids(chunk_ids: Sequence[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in chunk_ids:
            chunk_id = str(raw).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            out.append(chunk_id)
        return out

    @classmethod
    def _merge_retrieved_preserving_chunk_ids(
        cls,
        *,
        retrieved: Sequence[RetrievedChunk],
        extra: Sequence[RetrievedChunk],
        must_keep_chunk_ids: Sequence[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        for chunk in [*retrieved, *extra]:
            existing = merged.get(chunk.chunk_id)
            if existing is None or float(chunk.score) > float(existing.score):
                merged[chunk.chunk_id] = chunk

        ranked = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)
        keep_ids = cls._dedupe_chunk_ids(must_keep_chunk_ids)
        keep_set = set(keep_ids)

        ordered: list[RetrievedChunk] = []
        for chunk_id in keep_ids:
            chunk = merged.get(chunk_id)
            if chunk is not None:
                ordered.append(chunk)
        for chunk in ranked:
            if chunk.chunk_id in keep_set:
                continue
            ordered.append(chunk)
        return ordered[: max(0, int(limit))]

    @staticmethod
    def _section_page_num(section_path: str) -> int:
        m = re.search(r"page:(\d+)", section_path or "", flags=re.IGNORECASE)
        if m is None:
            return 10_000
        try:
            return int(m.group(1))
        except ValueError:
            return 10_000

    def _select_seed_chunk_id(self, chunks: list[RetrievedChunk], seed_terms: list[str]) -> str | None:
        if not chunks or not seed_terms:
            return None

        best: tuple[int, int, float, str] | None = None  # (score, -page, retrieval_score, chunk_id)
        for chunk in chunks[: max(1, min(12, len(chunks)))]:
            text = (chunk.text or "").lower()
            if not text:
                continue
            score = sum(1 for term in seed_terms if term and term in text)
            if score <= 0:
                continue
            page = self._section_page_num(getattr(chunk, "section_path", "") or "")
            candidate = (score, -page, float(chunk.score), chunk.chunk_id)
            if best is None or candidate > best:
                best = candidate
        return best[3] if best is not None else None

    @classmethod
    def _boolean_year_seed_chunk_score(cls, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        score = cls._boolean_year_compare_chunk_score(ref=ref, chunk=chunk)
        if score <= 0:
            return 0

        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        score += cls._ref_doc_family_consistency_adjustment(ref=ref, chunk=chunk)
        if "may be cited as" in text:
            score += 220
        if "repeal" in text or "replaced" in text or "replaces" in text or "as amended by" in text:
            score -= 260
        if "consolidated version" in text or "last updated" in text:
            score -= 120
        return score

    @classmethod
    def _boolean_admin_seed_chunk_score(cls, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        base = cls._named_commencement_title_match_score(ref, chunk)
        if base <= 0:
            return 0

        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        score = base + cls._ref_doc_family_consistency_adjustment(ref=ref, chunk=chunk)
        if cls._page_num(str(getattr(chunk, "section_path", "") or "")) <= 4:
            score += 200
        if "administ" in text or "administration" in text:
            score += 220
        if any(marker in text for marker in ("relevant authority", "registrar", "difca", "difc authority")):
            score += 80
        if "may be cited as" in text or "title" in text:
            score += 40
        return score

    @classmethod
    def _case_judge_seed_chunk_score(cls, *, chunk: RetrievedChunk | RankedChunk) -> int:
        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        if not text:
            return 0

        score = 0
        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num == 1:
            score += 320
        elif page_num == 2:
            score += 160
        elif page_num > 2:
            score -= min(180, (page_num - 2) * 28)
        if (
            "order with reasons" in text
            or "judgment of" in text
            or "reasons of" in text
            or "hearing held before" in text
            or "before h.e." in text
            or "judgment of the court of appeal" in text
        ):
            score += 260
        if any(marker in text for marker in ("chief justice", "justice ", "assistant registrar", "registrar", "sct judge")):
            score += 260
        if "claim no." in text or "case no:" in text:
            score += 40
        if any(marker in text for marker in ("issued by:", "introduction", "background", "discussion and determination")):
            score -= 40
        return score

    @classmethod
    def _case_ref_identity_score(cls, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        normalized_ref = cls._normalize_support_text(ref).casefold()
        if not normalized_ref:
            return 0

        haystack = cls._normalize_support_text(
            " ".join(
                part
                for part in (
                    str(getattr(chunk, "doc_title", "") or ""),
                    str(getattr(chunk, "doc_summary", "") or ""),
                    str(getattr(chunk, "text", "") or "")[:900],
                )
                if part
            )
        ).casefold()
        if not haystack:
            return 0

        if normalized_ref in haystack:
            return 1000 - min(haystack.find(normalized_ref), 600)

        ordered_ref_tokens = [
            token.casefold()
            for token in _SUPPORT_TOKEN_RE.findall(normalized_ref)
            if token.casefold() not in _SUPPORT_STOPWORDS and len(token) > 2
        ]
        if not ordered_ref_tokens:
            return 0

        overlap = 0
        cursor = 0
        for token in ordered_ref_tokens:
            idx = haystack.find(token, cursor)
            if idx >= 0:
                overlap += 1
                cursor = idx + len(token)
            elif token in haystack:
                overlap += 1
        if overlap < min(2, len(ordered_ref_tokens)):
            return 0
        return overlap * 120

    def _select_case_judge_seed_chunk_id(self, chunks: Sequence[RetrievedChunk]) -> str | None:
        best_chunk_id = ""
        best_key: tuple[int, int, float] | None = None
        for chunk in chunks:
            score = self._case_judge_seed_chunk_score(chunk=chunk)
            if score <= 0:
                continue
            page_num = self._page_num(str(getattr(chunk, "section_path", "") or ""))
            candidate = (score, -max(page_num, 0), float(chunk.score))
            if best_key is None or candidate > best_key:
                best_key = candidate
                best_chunk_id = chunk.chunk_id
        return best_chunk_id or None

    @classmethod
    def _case_issue_date_seed_chunk_score(cls, *, chunk: RetrievedChunk | RankedChunk) -> int:
        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        if not text:
            return 0

        score = 0
        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num <= 2:
            score += 220
        if "date of issue" in text:
            score += 320
        if "issued by" in text or "at:" in text:
            score += 60
        if "decision date" in text or "judgment" in text or "judgement" in text:
            score -= 80
        if "claim no." in text:
            score += 20
        return score

    def _select_case_issue_date_seed_chunk_id(self, chunks: Sequence[RetrievedChunk]) -> str | None:
        best_chunk_id = ""
        best_score = 0
        for chunk in chunks:
            score = self._case_issue_date_seed_chunk_score(chunk=chunk)
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
        return best_chunk_id or None

    @classmethod
    def _case_outcome_seed_chunk_score(cls, *, chunk: RetrievedChunk | RankedChunk) -> int:
        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        if not text:
            return 0

        score = 0
        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num == 1:
            score += 260
        elif page_num == 2:
            score += 120
        if "it is hereby ordered that" in text:
            score += 320
        if "order with reasons" in text:
            score += 180
        if "application is refused" in text or "application was dismissed" in text:
            score += 220
        if "no order as to costs" in text or "costs" in text:
            score += 40
        return score

    def _select_case_outcome_seed_chunk_id(self, chunks: Sequence[RetrievedChunk]) -> str | None:
        best_chunk_id = ""
        best_score = 0
        for chunk in chunks:
            score = self._case_outcome_seed_chunk_score(chunk=chunk)
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
        return best_chunk_id or None

    @classmethod
    def _ref_doc_family_consistency_adjustment(cls, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        law_ref_match = _LAW_NO_REF_RE.search(ref)
        if law_ref_match is None:
            return 0

        target_pair = (int(law_ref_match.group(1)), law_ref_match.group(2))
        identity_blob = cls._normalize_support_text(
            " ".join(
                part
                for part in (
                    str(getattr(chunk, "doc_title", "") or ""),
                    str(getattr(chunk, "doc_summary", "") or ""),
                )
                if part
            )
        ).casefold()
        if not identity_blob:
            return 0

        score = 0
        law_pairs = {
            (int(match.group(1)), match.group(2))
            for match in _LAW_NO_REF_RE.finditer(identity_blob)
        }
        if target_pair in law_pairs:
            score += 140

        foreign_pairs = {
            pair for pair in law_pairs
            if pair != target_pair and pair[1] != target_pair[1]
        }
        if foreign_pairs:
            score -= min(260, len(foreign_pairs) * 90)

        if any(marker in identity_blob for marker in ("consolidated version", "amendments up to", "as amended by")):
            if foreign_pairs:
                score -= 120
            else:
                score -= 40

        return score

    @classmethod
    def _is_notice_focus_query(cls, query: str) -> bool:
        normalized = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        if not normalized:
            return False
        return (
            "enactment notice" in normalized
            or "enacted law" in normalized
            or ("come into force" in normalized and "precise calendar date" in normalized)
        )

    @classmethod
    def _notice_doc_score(cls, *, query: str, raw: RetrievedChunk) -> int:
        normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "")).strip().casefold()
        if not normalized:
            return 0

        doc_title = re.sub(r"\s+", " ", str(getattr(raw, "doc_title", "") or "")).strip().casefold()
        explicit_notice_doc = "enactment notice" in doc_title or normalized.startswith("enactment notice")
        if not explicit_notice_doc and "hereby enact" not in normalized:
            return 0

        score = 0
        if explicit_notice_doc:
            score += 320
        if "hereby enact" in normalized:
            score += 220
        if "shall come into force" in normalized or "comes into force" in normalized:
            score += 140
        if re.search(r"\b(?:on\s+this\s+)?\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[a-z]+\s+\d{4}\b", normalized):
            score += 200
        if re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", normalized):
            score += 80
        if "date specified in the enactment notice" in normalized and not explicit_notice_doc:
            score -= 260

        query_lower = cls._normalize_support_text(query).casefold()
        if "full title" in query_lower and "in the form now attached" in normalized:
            score += 120
        if "precise calendar date" in query_lower and re.search(r"\b\d{4}\b", normalized):
            score += 60
        return score

    @classmethod
    def _is_consolidated_or_amended_family_chunk(cls, *, chunk: RetrievedChunk | RankedChunk) -> bool:
        family_blob = cls._normalize_support_text(
            " ".join(
                part
                for part in (
                    str(getattr(chunk, "doc_title", "") or ""),
                    str(getattr(chunk, "doc_summary", "") or ""),
                )
                if part
            )
        ).casefold()
        return any(marker in family_blob for marker in ("consolidated version", "amendments up to", "as amended by"))

    @classmethod
    def _is_canonical_ref_family_chunk(
        cls,
        *,
        ref: str,
        chunk: RetrievedChunk | RankedChunk,
    ) -> bool:
        law_ref_match = _LAW_NO_REF_RE.search(ref)
        if law_ref_match is None:
            return False
        if cls._is_consolidated_or_amended_family_chunk(chunk=chunk):
            return False

        target_key = f"law no. {int(law_ref_match.group(1))} of {law_ref_match.group(2)}"
        combined_blob = cls._normalize_support_text(
            " ".join(
                part
                for part in (
                    str(getattr(chunk, "doc_title", "") or ""),
                    str(getattr(chunk, "doc_summary", "") or ""),
                    str(getattr(chunk, "text", "") or ""),
                )
                if part
            )
        ).casefold()
        if target_key not in combined_blob:
            return False
        return cls._named_commencement_title_match_score(ref, chunk) > 0

    @classmethod
    def _best_named_administration_chunk(
        cls,
        *,
        ref: str,
        chunks: Sequence[RetrievedChunk],
        excluded_doc_ids: Sequence[str] = (),
    ) -> RetrievedChunk | None:
        excluded = {str(doc_id).strip() for doc_id in excluded_doc_ids if str(doc_id).strip()}
        best_canonical_clause_chunk: RetrievedChunk | None = None
        best_canonical_clause_tuple: tuple[int, int, float] | None = None
        best_clause_chunk: RetrievedChunk | None = None
        best_clause_tuple: tuple[int, int, float] | None = None
        best_anchor_chunk: RetrievedChunk | None = None
        best_anchor_tuple: tuple[int, int, float] | None = None

        for chunk in chunks:
            doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
            if doc_id and doc_id in excluded:
                continue

            anchor_score = cls._boolean_admin_seed_chunk_score(ref=ref, chunk=chunk)
            clause_score = cls._named_administration_clause_score(ref=ref, text=str(getattr(chunk, "text", "") or ""))
            if anchor_score <= 0 and clause_score <= 0:
                continue

            page_bonus = 1_000 - min(cls._page_num(str(getattr(chunk, "section_path", "") or "")), 999)
            retrieval_score = float(getattr(chunk, "score", getattr(chunk, "rerank_score", 0.0)) or 0.0)
            anchor_tuple = (anchor_score, page_bonus, retrieval_score)
            if best_anchor_tuple is None or anchor_tuple > best_anchor_tuple:
                best_anchor_tuple = anchor_tuple
                best_anchor_chunk = chunk

            if clause_score > 0:
                family_adjustment = cls._ref_doc_family_consistency_adjustment(ref=ref, chunk=chunk)
                clause_tuple = (clause_score, family_adjustment + anchor_score + page_bonus, retrieval_score)
                if cls._is_canonical_ref_family_chunk(ref=ref, chunk=chunk) and (
                    best_canonical_clause_tuple is None or clause_tuple > best_canonical_clause_tuple
                ):
                    best_canonical_clause_tuple = clause_tuple
                    best_canonical_clause_chunk = chunk
                if best_clause_tuple is None or clause_tuple > best_clause_tuple:
                    best_clause_tuple = clause_tuple
                    best_clause_chunk = chunk

        return best_canonical_clause_chunk or best_clause_chunk or best_anchor_chunk

    def _select_targeted_title_seed_chunk_id(
        self,
        *,
        query: str,
        answer_type: str,
        ref: str,
        chunks: Sequence[RetrievedChunk],
        seed_terms: Sequence[str],
    ) -> str | None:
        normalized_query = re.sub(r"\s+", " ", query).strip().casefold()
        if not chunks:
            return None

        scorer: Callable[[RetrievedChunk], int] | None = None
        if answer_type == "boolean" and "same year" in normalized_query:
            def _score_year_seed(chunk: RetrievedChunk) -> int:
                return self._boolean_year_seed_chunk_score(ref=ref, chunk=chunk)

            scorer = _score_year_seed
        elif answer_type == "boolean" and "administ" in normalized_query:
            def _score_admin_seed(chunk: RetrievedChunk) -> int:
                return self._boolean_admin_seed_chunk_score(ref=ref, chunk=chunk)

            scorer = _score_admin_seed

        if scorer is not None:
            best_chunk_id = ""
            best_score = 0
            for chunk in chunks:
                score = scorer(chunk)
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id
            if best_chunk_id:
                return best_chunk_id

        return self._select_seed_chunk_id(list(chunks), list(seed_terms))

    @staticmethod
    def _extract_title_refs_from_query(query: str) -> list[str]:
        raw = (query or "").strip()
        if not raw:
            return []
        found: list[str] = []
        for match in _AMENDMENT_TITLE_RE.finditer(raw):
            ref = re.sub(r"\s+", " ", match.group(1).strip())
            ref = _TITLE_REF_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_QUERY_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", ref)
            ref = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_LEADING_CONNECTOR_RE.sub("", ref).strip(" ,.;:")
            if ref:
                found.append(ref)
        for match in _TITLE_REF_RE.finditer(raw):
            title = re.sub(r"\s+", " ", match.group(1).strip())
            title = _TITLE_REF_BAD_LEAD_RE.sub("", title).strip()
            title = _TITLE_CONTEXT_BAD_LEAD_RE.sub("", title).strip()
            title = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", title).strip()
            title = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", title).strip()
            title = _TITLE_LEADING_CONNECTOR_RE.sub("", title).strip(" ,.;:")
            year = match.group(2).strip() if match.lastindex and match.lastindex >= 2 and match.group(2) else ""
            if not title:
                continue
            # Normalize suffix casing and pluralization for matching ingestion citations.
            words = title.split(" ")
            if words:
                last = words[-1].lower()
                if last == "law":
                    words[-1] = "Law"
                elif last in {"regulation", "regulations"}:
                    words[-1] = "Regulations"
            normalized = " ".join(words).strip()
            if year:
                normalized = f"{normalized} {year}"
            found.append(normalized)

        seen: set[str] = set()
        out: list[str] = []
        for item in found:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        pruned: list[str] = []
        lowered_out = [item.casefold() for item in out]
        for idx, item in enumerate(out):
            lowered = lowered_out[idx]
            if any(
                idx != other_idx
                and lowered != other_lowered
                and re.search(rf"\b{re.escape(lowered)}\b", other_lowered)
                for other_idx, other_lowered in enumerate(lowered_out)
            ):
                continue
            pruned.append(item)
        return pruned

    @staticmethod
    def _extract_title_ref_from_chunk_text(chunk: RetrievedChunk) -> str:
        text = str(getattr(chunk, "text", "") or "")
        for match in _TITLE_REF_RE.finditer(text):
            title = re.sub(r"\s+", " ", match.group(1).strip())
            year = match.group(2).strip() if match.lastindex and match.lastindex >= 2 and match.group(2) else ""
            if title:
                return f"{title} {year}".strip()
        return re.sub(r"\s+", " ", str(getattr(chunk, "doc_title", "") or "").strip())

    @staticmethod
    def _detect_coverage_gaps(
        query: str,
        context_chunks: list[object],
        doc_refs: list[str] | None = None,
    ) -> str:
        """Detect entities mentioned in query but absent from context chunks.

        Returns a prompt hint string warning the LLM about missing entities,
        or empty string if all entities are covered.
        """
        # Gather all entity references from the query.
        title_refs = RAGPipelineBuilder._extract_title_refs_from_query(query)
        all_refs = list(doc_refs or []) + title_refs
        if len(all_refs) < 2:
            # Only flag gaps for multi-entity queries.
            return ""

        # Build a searchable text from all context chunk titles and texts.
        chunks_text_parts: list[str] = []
        for chunk in context_chunks:
            doc_title = getattr(chunk, "doc_title", "") or ""
            text = getattr(chunk, "text", "") or ""
            chunks_text_parts.append(f"{doc_title} {text}".lower())
        context_blob = " ".join(chunks_text_parts)

        # Check each entity for presence in context.
        missing: list[str] = []
        seen: set[str] = set()
        for ref in all_refs:
            ref_clean = ref.strip()
            if not ref_clean:
                continue
            key = ref_clean.lower()
            if key in seen:
                continue
            seen.add(key)
            # Check if ANY significant words from the ref appear together in context.
            ref_words = [w for w in key.split() if len(w) > 2 and w not in {"the", "of", "and", "in", "for", "law", "no."}]
            if not ref_words:
                continue
            # Require at least 60% of distinctive words to appear.
            found_count = sum(1 for w in ref_words if w in context_blob)
            if len(ref_words) > 0 and found_count / len(ref_words) < 0.6:
                missing.append(ref_clean)

        if not missing:
            return ""

        missing_list = ", ".join(missing)
        return (
            f"IMPORTANT: The retrieved sources do NOT contain information about: {missing_list}. "
            f"Do NOT guess or fabricate information about these items. "
            f'For any part of the question about these items, state that information is not available for [item name].'
        )

    @staticmethod
    def _build_entity_scope(context_chunks: Sequence[object]) -> str:
        """Build an entity scope constraint from context chunk doc_titles.

        Returns a prompt hint listing the exact documents available in context,
        preventing the LLM from referencing laws/entities from parametric memory.
        """
        doc_titles: set[str] = set()
        for chunk in context_chunks:
            title = (getattr(chunk, "doc_title", "") or "").strip()
            if title:
                doc_titles.add(title)

        if len(doc_titles) < 2:
            return ""  # Not useful for single-doc queries.

        titles_str = "; ".join(sorted(doc_titles))
        return (
            f"ENTITY SCOPE: Your retrieved sources cover ONLY these documents: [{titles_str}]. "
            f"When listing specific laws or documents in your answer, reference ONLY those named above "
            f"or entities EXPLICITLY mentioned by exact name within the source text you were given. "
            f"Do NOT add any laws, regulations, or documents from your own knowledge."
        )

    @staticmethod
    def _ensure_must_include_context(
        *,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        must_include_chunk_ids: list[str],
        top_n: int,
    ) -> list[RankedChunk]:
        if not must_include_chunk_ids or top_n <= 0:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved}

        selected: list[RankedChunk] = []
        seen: set[str] = set()

        for chunk_id in must_include_chunk_ids:
            if chunk_id in seen:
                continue
            chunk = reranked_by_id.get(chunk_id)
            if chunk is None:
                raw = retrieved_by_id.get(chunk_id)
                if raw is None:
                    continue
                chunk = RankedChunk(
                    chunk_id=raw.chunk_id,
                    doc_id=raw.doc_id,
                    doc_title=raw.doc_title,
                    doc_type=raw.doc_type,
                    section_path=raw.section_path,
                    text=raw.text,
                    retrieval_score=float(raw.score),
                    # These injected chunks didn't go through the reranker; use retrieval score as a stable proxy.
                    rerank_score=float(raw.score),
                    doc_summary=raw.doc_summary,
                    page_family=getattr(raw, "page_family", ""),
                    doc_family=getattr(raw, "doc_family", ""),
                    chunk_type=getattr(raw, "chunk_type", ""),
                    amount_roles=list(getattr(raw, "amount_roles", []) or []),
                )
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen:
                continue
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n]

    @staticmethod
    def _ensure_page_one_context(
        *,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved:
            return reranked[: max(0, int(top_n))]

        page_one_by_doc: dict[str, RetrievedChunk] = {}
        for chunk in retrieved:
            doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
            section_path = str(getattr(chunk, "section_path", "") or "")
            if not doc_id or RAGPipelineBuilder._page_num(section_path) != 1:
                continue
            current = page_one_by_doc.get(doc_id)
            if current is None or float(chunk.score) > float(current.score):
                page_one_by_doc[doc_id] = chunk

        selected: list[RankedChunk] = []
        seen: set[str] = set()

        for chunk in reranked:
            page_one = page_one_by_doc.get(chunk.doc_id)
            if page_one is not None and page_one.chunk_id not in seen:
                selected.append(
                    RankedChunk(
                        chunk_id=page_one.chunk_id,
                        doc_id=page_one.doc_id,
                        doc_title=page_one.doc_title,
                        doc_type=page_one.doc_type,
                        section_path=page_one.section_path,
                        text=page_one.text,
                        retrieval_score=float(page_one.score),
                        rerank_score=float(page_one.score),
                        doc_summary=page_one.doc_summary,
                        page_family=getattr(page_one, "page_family", ""),
                        doc_family=getattr(page_one, "doc_family", ""),
                        chunk_type=getattr(page_one, "chunk_type", ""),
                        amount_roles=list(getattr(page_one, "amount_roles", []) or []),
                    )
                )
                seen.add(page_one.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]
            if chunk.chunk_id in seen:
                continue
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n]

    @classmethod
    def _doc_family_collapse_candidate_score(cls, *, query: str, chunk: RetrievedChunk | RankedChunk) -> tuple[int, int, float]:
        normalized_query_refs = [ref.casefold() for ref in cls._support_question_refs(query)[:4]]
        haystack = re.sub(
            r"\s+",
            " ",
            " ".join(
                part
                for part in (
                    str(getattr(chunk, "doc_title", "") or ""),
                    str(getattr(chunk, "text", "") or "")[:500],
                )
                if part
            ),
        ).strip().casefold()
        ref_bonus = 0
        if haystack and any(ref in haystack for ref in normalized_query_refs):
            ref_bonus = 2
        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num <= 2:
            page_bonus = 2
        elif page_num <= 4:
            page_bonus = 1
        else:
            page_bonus = 0
        retrieval_score = float(
            getattr(
                chunk,
                "score",
                getattr(chunk, "retrieval_score", 0.0),
            )
        )
        return ref_bonus, page_bonus, retrieval_score

    @classmethod
    def _collapse_doc_family_crowding_context(
        cls,
        *,
        query: str,
        answer_type: str,
        doc_ref_count: int,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        must_include_chunk_ids: Sequence[str],
        top_n: int,
    ) -> list[RankedChunk]:
        bounded = reranked[: max(0, int(top_n))]
        if top_n <= 1 or len(bounded) <= 1 or doc_ref_count < 2 or not retrieved:
            return bounded
        if QueryClassifier.extract_explicit_page_reference(query) is not None:
            return bounded
        if _is_broad_enumeration_query(query):
            return bounded

        q_lower = re.sub(r"\s+", " ", query).strip().lower()
        normalized_answer_type = answer_type.strip().lower()
        compare_like = normalized_answer_type in {"boolean", "name", "names", "date", "number"} and (
            len(_DIFC_CASE_ID_RE.findall(query or "")) >= 2
            or len(cls._support_question_refs(query)) >= 2
            or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
            or _is_common_judge_compare_query(query)
            or "same year" in q_lower
            or "same party" in q_lower
            or "appeared in both" in q_lower
            or "administ" in q_lower
        )
        metadata_like = (
            cls._is_named_metadata_support_query(query)
            or _is_named_multi_title_lookup_query(query)
            or _is_named_commencement_query(query)
        )
        if not (compare_like or metadata_like):
            return bounded

        reranked_doc_ids = [
            str(getattr(chunk, "doc_id", "") or "").strip()
            for chunk in bounded
            if str(getattr(chunk, "doc_id", "") or "").strip()
        ]
        if not reranked_doc_ids:
            return bounded
        distinct_reranked_doc_ids = list(dict.fromkeys(reranked_doc_ids))
        target_doc_count = min(2, int(top_n))
        if len(distinct_reranked_doc_ids) >= target_doc_count:
            return bounded

        dominant_doc_id = distinct_reranked_doc_ids[0]
        alternative_by_doc: dict[str, RetrievedChunk] = {}
        for raw in retrieved:
            doc_id = str(getattr(raw, "doc_id", "") or "").strip()
            if not doc_id or doc_id == dominant_doc_id:
                continue
            current = alternative_by_doc.get(doc_id)
            if current is None or cls._doc_family_collapse_candidate_score(query=query, chunk=raw) > cls._doc_family_collapse_candidate_score(
                query=query,
                chunk=current,
            ):
                alternative_by_doc[doc_id] = raw
        if not alternative_by_doc:
            return bounded

        replacement_index: int | None = None
        must_include_set = {str(chunk_id).strip() for chunk_id in must_include_chunk_ids if str(chunk_id).strip()}
        for idx in range(len(bounded) - 1, 0, -1):
            chunk = bounded[idx]
            if chunk.chunk_id in must_include_set:
                continue
            if str(getattr(chunk, "doc_id", "") or "").strip() == dominant_doc_id:
                replacement_index = idx
                break
        if replacement_index is None:
            return bounded

        alternative = max(
            alternative_by_doc.values(),
            key=lambda raw: cls._doc_family_collapse_candidate_score(query=query, chunk=raw),
        )
        alternative_ranked = cls._raw_to_ranked(alternative)
        if any(chunk.chunk_id == alternative_ranked.chunk_id for chunk in bounded):
            return bounded

        collapsed = list(bounded)
        collapsed[replacement_index] = alternative_ranked
        return collapsed[: max(0, int(top_n))]

    @staticmethod
    def _raw_to_ranked(chunk: RetrievedChunk) -> RankedChunk:
        return RankedChunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            doc_type=chunk.doc_type,
            section_path=chunk.section_path,
            text=chunk.text,
            retrieval_score=float(chunk.score),
            rerank_score=float(chunk.score),
            doc_summary=chunk.doc_summary,
            page_family=getattr(chunk, "page_family", ""),
            doc_family=getattr(chunk, "doc_family", ""),
            chunk_type=getattr(chunk, "chunk_type", ""),
            amount_roles=list(getattr(chunk, "amount_roles", []) or []),
        )

    @classmethod
    def _augment_strict_context_chunks(
        cls,
        *,
        query: str,
        answer_type: str,
        context_chunks: Sequence[RankedChunk],
        retrieved: Sequence[RetrievedChunk],
    ) -> tuple[list[RankedChunk], bool]:
        if answer_type.strip().lower() != "name":
            return list(context_chunks), False

        explicit_ref = QueryClassifier.extract_explicit_page_reference(query)
        if explicit_ref is None or explicit_ref.requested_page is None or explicit_ref.requested_page <= 0:
            return list(context_chunks), False

        query_lower = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        if "claim number" not in query_lower and "claim no" not in query_lower:
            return list(context_chunks), False
        if not any(token in query_lower for token in ("originate", "originated", "arose", "arisen")):
            return list(context_chunks), False

        requested_page = explicit_ref.requested_page

        def _rescue_score(raw: RetrievedChunk) -> tuple[int, float]:
            text = re.sub(r"\s+", " ", str(raw.text or "")).strip().casefold()
            score = 0
            if cls._page_num(str(raw.section_path or "")) == requested_page:
                score += 500
            if "claim no." in text or "claim no " in text:
                score += 160
            if "appeal against" in text or "urgent application" in text:
                score += 140
            if "origin" in text:
                score += 40
            if "/2" in text:
                score += 80
            return score, float(raw.score)

        rescue_candidates = [
            raw for raw in retrieved if cls._page_num(str(raw.section_path or "")) == requested_page
        ]
        if not rescue_candidates and requested_page == 2:
            rescue_candidates = [
                raw for raw in retrieved if cls._page_num(str(raw.section_path or "")) in {1, 2}
            ]
        if not rescue_candidates:
            return list(context_chunks), False

        augmented = list(context_chunks)
        seen_chunk_ids = {chunk.chunk_id for chunk in augmented}
        added = False
        for raw in sorted(rescue_candidates, key=_rescue_score, reverse=True)[:2]:
            if raw.chunk_id in seen_chunk_ids:
                continue
            augmented.append(cls._raw_to_ranked(raw))
            seen_chunk_ids.add(raw.chunk_id)
            added = True
        return augmented, added

    @classmethod
    def _named_commencement_title_match_score(cls, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        normalized_ref = re.sub(r"\s+", " ", ref).strip().casefold()
        if not normalized_ref:
            return 0

        haystack = " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "text", "") or "")[:1600],
            )
            if part
        )
        normalized_haystack = re.sub(r"\s+", " ", haystack).strip().casefold()
        if not normalized_haystack:
            return 0

        position = normalized_haystack.find(normalized_ref)
        if position >= 0:
            return 1200 - min(position, 600)

        law_ref_match = _LAW_NO_REF_RE.search(ref)
        if law_ref_match is not None:
            law_no_key = f"law no. {int(law_ref_match.group(1))} of {law_ref_match.group(2)}"
            position = normalized_haystack.find(law_no_key)
            if position >= 0:
                return 1000 - min(position, 600)
            return 0

        ref_tokens = [
            token
            for token in _COMMON_ELEMENTS_TOKEN_RE.findall(normalized_ref)
            if token and token not in _COMMON_ELEMENTS_TITLE_STOPWORDS and len(token) > 2
        ]
        if not ref_tokens:
            return 0
        haystack_tokens = set(_COMMON_ELEMENTS_TOKEN_RE.findall(normalized_haystack))
        overlap = sum(1 for token in ref_tokens if token in haystack_tokens)
        if overlap == len(ref_tokens):
            return 400 + overlap
        if overlap >= max(1, len(ref_tokens) - 1):
            return 240 + overlap
        if overlap >= max(1, (len(ref_tokens) + 1) // 2):
            return 120 + overlap
        return 0

    @staticmethod
    def _named_commencement_clause_score(text: str) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return 0

        score = 0
        if "commencement" in normalized:
            score += 6
        if "comes into force" in normalized or "shall come into force" in normalized:
            score += 8
        if "enactment notice" in normalized:
            score += 4
        if "90" in normalized and "days following" in normalized:
            score += 4
        if re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", normalized):
            score += 3
        return score

    @classmethod
    def _named_multi_title_clause_score(cls, *, query: str, text: str) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return 0

        query_lower = re.sub(r"\s+", " ", (query or "").strip()).lower()
        score = 0
        if "citation title" in query_lower or "title of" in query_lower or "titles of" in query_lower:
            if "may be cited as" in normalized:
                score += 20
            if "the title is" in normalized or "citation title" in normalized:
                score += 8
        if "administ" in query_lower:
            if cls._chunk_has_self_registrar_clause(text=text):
                score += 24
            elif "registrar" in normalized:
                score += 6
        if _is_named_commencement_query(query):
            score += cls._named_commencement_clause_score(text)
        if "updated" in query_lower:
            if "updated" in normalized or "amended" in normalized or "effective from" in normalized:
                score += 10
            if _ISO_DATE_RE.search(normalized) or _SLASH_DATE_RE.search(normalized) or _TEXTUAL_DATE_RE.search(normalized):
                score += 6
        return score

    @classmethod
    def _named_amendment_clause_score(cls, *, query: str, ref: str, text: str) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return 0

        score = 0
        if "amended by" in normalized:
            score += 18
        if "as amended by" in normalized:
            score += 12
        if "enacted on" in normalized or "hereby enact" in normalized:
            score += 8

        ref_terms = {
            token
            for token in cls._support_terms(ref)
            if token not in _SUPPORT_STOPWORDS and len(token) > 2
        }
        if ref_terms:
            score += len(ref_terms.intersection(cls._support_terms(normalized))) * 8

        query_terms = {
            token
            for token in cls._support_terms(query)
            if token not in _SUPPORT_STOPWORDS and token not in ref_terms and len(token) > 2
        }
        if query_terms:
            score += len(query_terms.intersection(cls._support_terms(normalized))) * 2

        return score

    @classmethod
    def _named_penalty_clause_score(cls, *, query: str, ref: str, text: str) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return 0

        amount_match = re.search(
            r"\b(?:usd|us\\$)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})(?:\.\d+)?\b",
            normalized,
        )
        if amount_match is None:
            return 0

        score = 0
        if "penalt" in normalized:
            score += 10
        if "offence" in normalized or "offense" in normalized:
            score += 6
        if "illegal" in normalized:
            score += 12
        score += 10

        ref_terms = {
            token
            for token in cls._support_terms(ref)
            if token not in _SUPPORT_STOPWORDS and len(token) > 2 and token not in {"law", "regulations", "regulation"}
        }
        if ref_terms:
            score += len(ref_terms.intersection(cls._support_terms(normalized))) * 8

        query_terms = {
            token
            for token in cls._support_terms(query)
            if token not in _SUPPORT_STOPWORDS and token not in ref_terms and len(token) > 2
        }
        if query_terms:
            score += len(query_terms.intersection(cls._support_terms(normalized))) * 3

        return score

    @classmethod
    def _chunk_has_named_administration_clause(cls, *, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return False
        return _GENERIC_SELF_ADMIN_RE.search(normalized) is not None

    @classmethod
    def _named_administration_clause_score(cls, *, ref: str, text: str) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized or not cls._chunk_has_named_administration_clause(text=text):
            return 0

        score = 18
        if "administered by" in normalized or "shall administer this law" in normalized:
            score += 8
        if "difca" in normalized or "registrar" in normalized:
            score += 4

        ref_terms = {
            token
            for token in cls._support_terms(ref)
            if token not in _SUPPORT_STOPWORDS and len(token) > 2 and token not in {"law", "regulations", "regulation"}
        }
        if ref_terms:
            score += len(ref_terms.intersection(cls._support_terms(normalized))) * 8
        return score

    @classmethod
    def _chunk_has_self_registrar_clause(cls, *, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return False
        return _REGISTRAR_SELF_ADMIN_RE.search(normalized) is not None

    @classmethod
    def _ensure_self_registrar_context(
        cls,
        *,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved:
            return reranked[: max(0, int(top_n))]

        evidence_by_doc: dict[str, RetrievedChunk] = {}
        page_one_by_doc: dict[str, RetrievedChunk] = {}
        best_by_doc: dict[str, RetrievedChunk] = {}
        for chunk in retrieved:
            doc_id = str(getattr(chunk, "doc_id", "") or "").strip()
            if not doc_id:
                continue
            current_best = best_by_doc.get(doc_id)
            if current_best is None or float(chunk.score) > float(current_best.score):
                best_by_doc[doc_id] = chunk
            section_path = str(getattr(chunk, "section_path", "") or "").lower()
            if "page:1" in section_path:
                current_page_one = page_one_by_doc.get(doc_id)
                if current_page_one is None or float(chunk.score) > float(current_page_one.score):
                    page_one_by_doc[doc_id] = chunk
            if not cls._chunk_has_self_registrar_clause(text=str(getattr(chunk, "text", "") or "")):
                continue
            current_evidence = evidence_by_doc.get(doc_id)
            if current_evidence is None or float(chunk.score) > float(current_evidence.score):
                evidence_by_doc[doc_id] = chunk

        if not evidence_by_doc:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen: set[str] = set()

        matched_doc_ids: list[str] = []
        for chunk in reranked:
            if chunk.doc_id in evidence_by_doc and chunk.doc_id not in matched_doc_ids:
                matched_doc_ids.append(chunk.doc_id)
        for doc_id in evidence_by_doc:
            if doc_id not in matched_doc_ids:
                matched_doc_ids.append(doc_id)

        for doc_id in matched_doc_ids:
            preferred = [page_one_by_doc.get(doc_id) or best_by_doc.get(doc_id), evidence_by_doc.get(doc_id)]
            for raw in preferred:
                if raw is None or raw.chunk_id in seen:
                    continue
                ranked = reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw)
                selected.append(ranked)
                seen.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        for chunk in reranked:
            if chunk.doc_id not in evidence_by_doc or chunk.chunk_id in seen:
                continue
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _common_elements_ref_tokens(cls, text: str) -> tuple[str, ...]:
        normalized = _TITLE_LAW_NO_SUFFIX_RE.sub("", text or "")
        normalized = re.sub(r"\b(19|20)\d{2}\b", " ", normalized)
        tokens = [
            token
            for token in _COMMON_ELEMENTS_TOKEN_RE.findall(normalized.lower())
            if token and token not in _COMMON_ELEMENTS_TITLE_STOPWORDS and len(token) > 2
        ]
        return tuple(dict.fromkeys(tokens))

    @classmethod
    def _common_elements_title_match_score(cls, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        ref_tokens = cls._common_elements_ref_tokens(ref)
        if not ref_tokens:
            return 0

        haystack = " ".join(
            part
            for part in (
                str(getattr(chunk, "doc_title", "") or ""),
                str(getattr(chunk, "text", "") or "")[:1200],
            )
            if part
        )
        haystack_tokens = set(_COMMON_ELEMENTS_TOKEN_RE.findall(haystack.lower()))
        overlap = sum(1 for token in ref_tokens if token in haystack_tokens)
        if overlap <= 0:
            return 0
        if overlap == len(ref_tokens):
            return 100 + overlap
        if overlap >= max(1, len(ref_tokens) - 1):
            return 60 + overlap
        if overlap >= max(1, (len(ref_tokens) + 1) // 2):
            return 20 + overlap
        return 0

    @staticmethod
    def _common_elements_evidence_score(text: str, *, interpretation_sections: bool = False) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return 0

        score = 0
        if interpretation_sections:
            if "rules of interpretation" in normalized:
                score += 18
            if "a statutory provision includes a reference" in normalized:
                score += 22
            if "reference to a person includes" in normalized:
                score += 20
            if "interpretation" in normalized:
                score += 6
            if "schedule 1" in normalized:
                score += 2
            if "interpretative provisions" in normalized:
                score += 1
            if (
                "defined terms" in normalized
                and "a statutory provision includes a reference" not in normalized
                and "reference to a person includes" not in normalized
            ):
                score -= 8
            return score

        if "schedule 1" in normalized:
            score += 5
        if "interpretation" in normalized:
            score += 4
        if "rules of interpretation" in normalized:
            score += 7
        if "interpretative provisions" in normalized:
            score += 4
        if "defined terms" in normalized:
            score += 2
        if "a statutory provision includes a reference" in normalized:
            score += 2
        return score

    @classmethod
    def _ensure_named_commencement_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved or not _is_named_commencement_query(query):
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or _extract_question_title_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_anchor: RetrievedChunk | None = None
            best_anchor_score = 0
            for raw in retrieved:
                score = cls._named_commencement_title_match_score(ref, raw)
                if score > best_anchor_score:
                    best_anchor = raw
                    best_anchor_score = score
            if best_anchor is None or best_anchor.doc_id in matched_doc_ids:
                continue

            matched_doc_ids.add(best_anchor.doc_id)
            preferred_raw: list[RetrievedChunk] = [best_anchor]
            best_clause: RetrievedChunk | None = None
            best_clause_score = 0
            for raw in retrieved:
                if raw.doc_id != best_anchor.doc_id:
                    continue
                score = cls._named_commencement_clause_score(str(getattr(raw, "text", "") or ""))
                if score > best_clause_score:
                    best_clause = raw
                    best_clause_score = score
            if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
                preferred_raw.append(best_clause)

            for raw in preferred_raw:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_named_penalty_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved:
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_anchor: RetrievedChunk | None = None
            best_anchor_score = 0
            for raw in retrieved:
                score = cls._named_commencement_title_match_score(ref, raw)
                if score > best_anchor_score:
                    best_anchor = raw
                    best_anchor_score = score
            if best_anchor is None or best_anchor.doc_id in matched_doc_ids:
                continue

            matched_doc_ids.add(best_anchor.doc_id)
            preferred_raw: list[RetrievedChunk] = [best_anchor]
            best_clause: RetrievedChunk | None = None
            best_clause_score = 0
            for raw in retrieved:
                if raw.doc_id != best_anchor.doc_id:
                    continue
                score = cls._named_penalty_clause_score(
                    query=query,
                    ref=ref,
                    text=str(getattr(raw, "text", "") or ""),
                )
                if score > best_clause_score:
                    best_clause = raw
                    best_clause_score = score
            if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
                preferred_raw.append(best_clause)

            for raw in preferred_raw:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_boolean_year_compare_context(
        cls,
        *,
        query: str,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not retrieved:
            return reranked[: max(0, int(top_n))]

        refs = cls._paired_support_question_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_raw: RetrievedChunk | None = None
            best_score = 0
            for raw in retrieved:
                if raw.doc_id in matched_doc_ids:
                    continue
                score = cls._boolean_year_seed_chunk_score(ref=ref, chunk=raw)
                if score > best_score:
                    best_raw = raw
                    best_score = score
            if best_raw is None:
                continue
            matched_doc_ids.add(best_raw.doc_id)
            if best_raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(best_raw.chunk_id) or cls._raw_to_ranked(best_raw))
            seen_chunk_ids.add(best_raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_boolean_admin_compare_context(
        cls,
        *,
        query: str,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not retrieved:
            return reranked[: max(0, int(top_n))]

        refs = cls._paired_support_question_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_raw = cls._best_named_administration_chunk(
                ref=ref,
                chunks=retrieved,
                excluded_doc_ids=tuple(matched_doc_ids),
            )
            if best_raw is None:
                continue
            doc_id = str(best_raw.doc_id or "").strip()
            if doc_id:
                matched_doc_ids.add(doc_id)
            if best_raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(best_raw.chunk_id) or cls._raw_to_ranked(best_raw))
            seen_chunk_ids.add(best_raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_boolean_judge_compare_context(
        cls,
        *,
        query: str,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not retrieved:
            return reranked[: max(0, int(top_n))]

        refs = cls._paired_support_question_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_raw: RetrievedChunk | None = None
            best_score = 0
            for raw in retrieved:
                doc_id = str(raw.doc_id or "").strip()
                if doc_id in matched_doc_ids:
                    continue
                identity_score = cls._case_ref_identity_score(ref=ref, chunk=raw)
                if identity_score <= 0:
                    continue
                score = identity_score + cls._case_judge_seed_chunk_score(chunk=raw)
                if score > best_score:
                    best_raw = raw
                    best_score = score
            if best_raw is None:
                continue
            doc_id = str(best_raw.doc_id or "").strip()
            if doc_id:
                matched_doc_ids.add(doc_id)
            if best_raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(best_raw.chunk_id) or cls._raw_to_ranked(best_raw))
            seen_chunk_ids.add(best_raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_notice_doc_context(
        cls,
        *,
        query: str,
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not retrieved or not cls._is_notice_focus_query(query):
            return reranked[: max(0, int(top_n))]

        desired_docs = 2 if "precise calendar date" in cls._normalize_support_text(query).casefold() else 1
        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        seen_doc_ids: set[str] = set()

        scored: list[tuple[int, float, int, RetrievedChunk]] = []
        for raw in retrieved:
            score = cls._notice_doc_score(query=query, raw=raw)
            if score <= 0:
                continue
            page_num = cls._page_num(str(getattr(raw, "section_path", "") or ""))
            scored.append((score, float(raw.score), -page_num, raw))

        scored.sort(reverse=True)
        for _score, _retrieval_score, _page_rank, raw in scored:
            doc_id = str(raw.doc_id or "").strip()
            if doc_id and doc_id in seen_doc_ids:
                continue
            selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
            seen_chunk_ids.add(raw.chunk_id)
            if doc_id:
                seen_doc_ids.add(doc_id)
            if len(seen_doc_ids) >= desired_docs or len(selected) >= top_n:
                break

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_account_effective_dates_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not retrieved or not _is_account_effective_dates_query(query):
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if not refs:
            return reranked[: max(0, int(top_n))]

        ref = refs[0]
        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()

        best_anchor: RetrievedChunk | None = None
        best_anchor_score = 0
        for raw in retrieved:
            score = cls._named_commencement_title_match_score(ref, raw)
            if score > best_anchor_score:
                best_anchor = raw
                best_anchor_score = score

        if best_anchor is not None:
            best_effective: RetrievedChunk | None = None
            best_effective_score = 0
            for raw in retrieved:
                if raw.doc_id != best_anchor.doc_id:
                    continue
                score = cls._account_effective_clause_score(text=str(getattr(raw, "text", "") or ""))
                if score > best_effective_score:
                    best_effective = raw
                    best_effective_score = score
            for raw in [best_anchor, best_effective] if best_effective is not None else [best_anchor]:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        best_enactment: RetrievedChunk | None = None
        best_enactment_score = 0
        for raw in retrieved:
            score = cls._account_enactment_clause_score(ref=ref, raw=raw)
            if score > best_enactment_score:
                best_enactment = raw
                best_enactment_score = score
        if best_enactment is not None and best_enactment.chunk_id not in seen_chunk_ids:
            selected.append(reranked_by_id.get(best_enactment.chunk_id) or cls._raw_to_ranked(best_enactment))
            seen_chunk_ids.add(best_enactment.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _account_effective_support_family_seed_chunk_ids(
        cls,
        *,
        ref: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> list[str]:
        best_effective: RetrievedChunk | None = None
        best_effective_score = 0
        best_enactment: RetrievedChunk | None = None
        best_enactment_score = 0

        for raw in retrieved:
            effective_score = cls._account_effective_clause_score(text=str(getattr(raw, "text", "") or ""))
            if effective_score > best_effective_score:
                best_effective = raw
                best_effective_score = effective_score

            enactment_score = cls._account_enactment_clause_score(ref=ref, raw=raw)
            if enactment_score > best_enactment_score:
                best_enactment = raw
                best_enactment_score = enactment_score

        seeds: list[str] = []
        if best_effective is not None and best_effective_score > 0:
            seeds.append(best_effective.chunk_id)
        if best_enactment is not None and best_enactment_score > 0:
            seeds.append(best_enactment.chunk_id)
        return cls._dedupe_chunk_ids(seeds)

    _SKIP_ADMIN_ARTICLE_RE = re.compile(
        r"(?:under|in|of|per|pursuant to)\s+article\s+\d+",
        re.IGNORECASE,
    )

    @classmethod
    def _prune_boolean_context_for_single_doc_article(
        cls,
        *,
        query: str,
        answer_type: str,
        doc_refs: list[str] | tuple[str, ...] | None,
        context_chunks: list[RankedChunk],
    ) -> list[RankedChunk]:
        """For boolean + single doc_ref + explicit article queries, restrict context
        to chunks from the referenced document to prevent cross-doc contamination."""
        if (answer_type or "").strip().lower() != "boolean":
            return list(context_chunks)
        refs = [str(r).strip() for r in (doc_refs or []) if str(r).strip()]
        if len(refs) != 1:
            return list(context_chunks)
        if not cls._SKIP_ADMIN_ARTICLE_RE.search(query or ""):
            return list(context_chunks)

        ref_lower = refs[0].casefold()
        ref_tokens = set(ref_lower.split())
        matching_doc_ids: set[str] = set()
        for chunk in context_chunks:
            title_lower = (chunk.doc_title or "").casefold()
            if ref_lower in title_lower or title_lower in ref_lower:
                matching_doc_ids.add(chunk.doc_id)
                continue
            title_tokens = set(title_lower.split())
            if ref_tokens and len(ref_tokens & title_tokens) >= max(1, len(ref_tokens) - 1):
                matching_doc_ids.add(chunk.doc_id)

        if not matching_doc_ids:
            return list(context_chunks)

        pruned = [c for c in context_chunks if c.doc_id in matching_doc_ids]
        if len(pruned) >= 2:
            return pruned
        return list(context_chunks)

    @classmethod
    def _administration_support_family_seed_chunk_ids(
        cls,
        *,
        ref: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> list[str]:
        best_chunk = cls._best_named_administration_chunk(ref=ref, chunks=retrieved)
        chunk_id = str(getattr(best_chunk, "chunk_id", "") or "").strip() if best_chunk is not None else ""
        return [chunk_id] if chunk_id else []

    @classmethod
    def _remuneration_recordkeeping_clause_score(cls, raw: RetrievedChunk) -> int:
        normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "").strip()).casefold()
        if not normalized:
            return 0
        score = 0
        if "article 16" in normalized or "16. payroll records" in normalized or "payroll records" in normalized:
            score += 80
        if "remuneration" in normalized:
            score += 80
        if "pay period" in normalized:
            score += 80
        if "gross and net" in normalized:
            score += 40
        return score

    @classmethod
    def _ensure_named_amendment_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved or not _is_named_amendment_query(query):
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if not refs:
            return reranked[: max(0, int(top_n))]

        amendment_ref = refs[0]
        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()

        best_anchor: RetrievedChunk | None = None
        best_anchor_score = 0
        for raw in retrieved:
            score = cls._named_commencement_title_match_score(amendment_ref, raw) + cls._named_amendment_clause_score(
                query=query,
                ref=amendment_ref,
                text=str(getattr(raw, "text", "") or ""),
            )
            normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "").strip()).casefold()
            if "hereby enact" in normalized or "enacted on" in normalized:
                score += 40
            if score > best_anchor_score:
                best_anchor = raw
                best_anchor_score = score

        amender_doc_id = str(best_anchor.doc_id or "").strip() if best_anchor is not None else ""
        if best_anchor is not None:
            preferred_amender: list[RetrievedChunk] = [best_anchor]
            best_clause: RetrievedChunk | None = None
            best_clause_score = 0
            for raw in retrieved:
                if str(raw.doc_id or "").strip() != amender_doc_id:
                    continue
                score = cls._named_amendment_clause_score(
                    query=query,
                    ref=amendment_ref,
                    text=str(getattr(raw, "text", "") or ""),
                )
                if score > best_clause_score:
                    best_clause = raw
                    best_clause_score = score
            if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
                preferred_amender.append(best_clause)
            for raw in preferred_amender:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        doc_best_clause: dict[str, RetrievedChunk] = {}
        doc_best_score: dict[str, int] = {}
        for raw in retrieved:
            doc_id = str(raw.doc_id or "").strip()
            if not doc_id or doc_id == amender_doc_id:
                continue
            score = cls._named_amendment_clause_score(
                query=query,
                ref=amendment_ref,
                text=str(getattr(raw, "text", "") or ""),
            )
            if score <= 0:
                continue
            if score > doc_best_score.get(doc_id, 0):
                doc_best_score[doc_id] = score
                doc_best_clause[doc_id] = raw

        for raw in sorted(
            doc_best_clause.values(),
            key=lambda chunk: (doc_best_score.get(str(chunk.doc_id or "").strip(), 0), float(chunk.score)),
            reverse=True,
        ):
            if raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
            seen_chunk_ids.add(raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_named_administration_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved:
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_anchor = cls._best_named_administration_chunk(
                ref=ref,
                chunks=retrieved,
                excluded_doc_ids=tuple(matched_doc_ids),
            )
            best_doc_id = str(best_anchor.doc_id or "").strip() if best_anchor is not None else ""
            if best_anchor is None or best_doc_id in matched_doc_ids:
                continue

            matched_doc_ids.add(best_doc_id)
            preferred_raw: list[RetrievedChunk] = [best_anchor]
            best_clause: RetrievedChunk | None = None
            best_clause_score = 0
            for raw in retrieved:
                if str(raw.doc_id or "").strip() != best_doc_id:
                    continue
                score = cls._named_administration_clause_score(ref=ref, text=str(getattr(raw, "text", "") or ""))
                if score > best_clause_score:
                    best_clause = raw
                    best_clause_score = score
            if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
                preferred_raw.append(best_clause)

            for raw in preferred_raw:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_named_multi_title_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved or not _is_named_multi_title_lookup_query(query):
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs:
            best_anchor: RetrievedChunk | None = None
            best_anchor_score = 0
            for raw in retrieved:
                score = cls._named_commencement_title_match_score(ref, raw)
                if score > best_anchor_score:
                    best_anchor = raw
                    best_anchor_score = score
            if best_anchor is None or best_anchor.doc_id in matched_doc_ids:
                continue

            matched_doc_ids.add(best_anchor.doc_id)
            preferred_raw: list[RetrievedChunk] = [best_anchor]
            best_clause: RetrievedChunk | None = None
            best_clause_score = 0
            for raw in retrieved:
                if raw.doc_id != best_anchor.doc_id:
                    continue
                score = cls._named_multi_title_clause_score(
                    query=query,
                    text=str(getattr(raw, "text", "") or ""),
                )
                if score > best_clause_score:
                    best_clause = raw
                    best_clause_score = score
            if best_clause is not None and best_clause.chunk_id != best_anchor.chunk_id:
                preferred_raw.append(best_clause)

            for raw in preferred_raw:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]

    @classmethod
    def _ensure_common_elements_context(
        cls,
        *,
        query: str,
        doc_refs: list[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved or not _is_common_elements_query(query):
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._extract_title_refs_from_query(query)
        if len(refs) < 2:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        interpretation_sections_query = _is_interpretation_sections_common_elements_query(query)

        for ref in refs:
            best_anchor: RetrievedChunk | None = None
            best_anchor_key: tuple[int, int, float, int] | None = None
            for raw in retrieved:
                title_match = cls._common_elements_title_match_score(ref, raw)
                if title_match <= 0:
                    continue
                evidence_score = cls._common_elements_evidence_score(
                    str(getattr(raw, "text", "") or ""),
                    interpretation_sections=interpretation_sections_query,
                )
                page_num = cls._section_page_num(str(getattr(raw, "section_path", "") or ""))
                candidate = (evidence_score, title_match, float(raw.score), page_num)
                if best_anchor_key is None or candidate > best_anchor_key:
                    best_anchor_key = candidate
                    best_anchor = raw

            if best_anchor is None:
                continue

            preferred_raw: list[RetrievedChunk] = [best_anchor]
            if interpretation_sections_query:
                best_clause: RetrievedChunk | None = None
                best_clause_key: tuple[int, float, int] | None = None
                for raw in retrieved:
                    if raw.doc_id != best_anchor.doc_id:
                        continue
                    evidence_score = cls._common_elements_evidence_score(
                        str(getattr(raw, "text", "") or ""),
                        interpretation_sections=True,
                    )
                    if evidence_score <= 0:
                        continue
                    page_num = cls._section_page_num(str(getattr(raw, "section_path", "") or ""))
                    candidate = (evidence_score, float(raw.score), page_num)
                    if best_clause_key is None or candidate > best_clause_key:
                        best_clause_key = candidate
                        best_clause = raw
                if best_clause is not None:
                    preferred_raw = [best_clause]

            for raw in preferred_raw:
                if raw.chunk_id in seen_chunk_ids:
                    continue
                selected.append(reranked_by_id.get(raw.chunk_id) or cls._raw_to_ranked(raw))
                seen_chunk_ids.add(raw.chunk_id)
                if len(selected) >= top_n:
                    return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            if len(selected) >= top_n:
                break

        return selected[:top_n] if selected else reranked[: max(0, int(top_n))]
