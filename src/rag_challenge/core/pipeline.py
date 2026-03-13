# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false
from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, TypedDict, cast

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from rag_challenge.config import get_settings
from rag_challenge.core.conflict_detector import ConflictDetector
from rag_challenge.core.decomposer import QueryDecomposer
from rag_challenge.core.premise_guard import check_query_premise
from rag_challenge.core.strict_answerer import StrictAnswerer
from rag_challenge.models import Citation, QueryComplexity, RankedChunk, RetrievedChunk, TelemetryPayload
from rag_challenge.prompts.loader import load_prompt
from rag_challenge.telemetry import TelemetryCollector  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag_challenge.core.classifier import QueryClassifier
    from rag_challenge.core.reranker import RerankerClient
    from rag_challenge.core.retriever import HybridRetriever
    from rag_challenge.core.verifier import AnswerVerifier
    from rag_challenge.llm.generator import RAGGenerator
logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SLASH_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_TEXTUAL_DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MONEY_VALUE_RE = re.compile(r"\b(?:aed|usd|gbp|eur)\s*[\d,]+(?:\.\d+)?\b", re.IGNORECASE)
_CITE_RE = re.compile(r"\(cite:\s*([^)]+)\)")
_CASE_REF_PREFIX_RE = re.compile(
    r"^(?:case\s+)?(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s+\d{1,4}/\d{4}\s*[:\-,.]?\s*",
    re.IGNORECASE,
)
_DIFC_CASE_ID_RE = re.compile(r"\b(CFI|CA|SCT|ENF|DEC|TCD|ARB)\s*0*(\d{1,4})\s*[/-]\s*(\d{4})\b", re.IGNORECASE)
_TITLE_REF_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9]*(?:\s+(?:of|the|in|on|and|for|to|by|Non|Incorporated|Limited|General|Data|Protection|Application|Civil|Commercial|Strata|Title|Trust|Contract|Liability|Partnership|Profit|Organisations?|Operating|Companies|Insolvency|Foundations?|Employment|Arbitration|Securities|Investment|Personal|Property|Obligations|Netting|Courts|Court|Common|Reporting|Standard|Dematerialised|Investments?|Implied|Terms|Unfair|Amendment|DIFC|DFSA))*\s+(?:Law|Regulations?)))\b(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_AMENDMENT_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Laws?\s+Amendment\s+Law(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
_SUPPORT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NUMBERED_ITEM_RE = re.compile(r"(?:^|\n)\s*\d+\.\s+(.+?)(?=(?:\n\s*\d+\.\s+)|$)", re.DOTALL)
_BULLET_ITEM_RE = re.compile(r"(?m)^\s*[-*]\s+(.+?)\s*$")
_TITLE_FIELD_RE = re.compile(
    r"\btitle:\s*(.+?)(?=\s+-\s+(?:last\s+updated|the\s+provided\s+sources)|$)",
    re.IGNORECASE,
)
_LAST_UPDATED_FIELD_RE = re.compile(
    r"\blast\s+updated(?:\s+\(consolidated\s+version\))?:\s*([^()]+?)(?=\s+\(cite:|$)",
    re.IGNORECASE,
)
_ENACTED_ON_FIELD_RE = re.compile(
    r"\b(?:was\s+enacted\s+on|date\s+of\s+enactment\s+is)\s+([^()]+?)(?=\s+\(cite:|$)",
    re.IGNORECASE,
)
_COMMENCEMENT_FIELD_RE = re.compile(
    r"\b(?:come(?:s)?\s+into\s+force|effective\s+date\s+is)\s+([^()]+?)(?=\s+\(cite:|$)",
    re.IGNORECASE,
)
_SUPPORT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "to",
    "by",
    "with",
    "what",
    "which",
    "when",
    "where",
    "who",
    "whose",
    "their",
    "there",
    "this",
    "that",
    "these",
    "those",
    "law",
    "laws",
    "regulation",
    "regulations",
    "document",
    "documents",
    "case",
    "cases",
    "title",
    "titles",
    "date",
    "dates",
}
_MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_NUMBER_TO_NAME = {value: key for key, value in _MONTH_NAME_TO_NUMBER.items()}
_STRICT_REPAIR_HINT_TEMPLATE = load_prompt("llm/strict_repair_hint")
_UNANSWERABLE_FREE_TEXT = "There is no information on this question."
_UNANSWERABLE_STRICT = "null"


def _needs_long_free_text_answer(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    # Common enumeration/list patterns that often exceed small max_tokens caps.
    if any(
        phrase in q
        for phrase in (
            "which laws were amended",
            "which laws were made",
            "which laws",
            "list all",
            "identify all",
            "name all",
            "which articles",
            "which sections",
            "common elements",
            "elements in common",
            "in common",
        )
    ):
        return True
    return q.startswith("which ") and any(word in q for word in ("laws", "cases", "regulations", "articles"))

_ENUMERATION_RE = re.compile(
    r"(?:which|what|list|name|identify)\s+(?:specific\s+)?(?:laws?|regulations?|rules?|documents?|statutes?)"
    r"|(?:how\s+(?:do|does|did))\s+.*\s+and\s+.*\s+(?:define|address|regulate|handle)",
    re.IGNORECASE,
)

_MULTI_CRITERIA_ENUM_HINTS = (
    " and were ",
    " and was ",
    " and their ",
    " enacted in ",
    " commencement date ",
    " administered by ",
    " made by ",
    " amended by ",
    " mention ",
    " mentions ",
    " contain ",
    " contains ",
    " include ",
    " includes ",
)
_REGISTRAR_SELF_ADMIN_RE = re.compile(
    r"\b(?:the\s+registrar\s+shall\s+administer\s+this\s+law"
    r"|this\s+law\s+shall\s+be\s+administered\s+by\s+the\s+registrar"
    r"|this\s+law\s+is\s+administered\s+by\s+the\s+registrar"
    r"|administration\s+of\s+this\s+law\b[^.]{0,160}\bregistrar\b)\b",
    re.IGNORECASE | re.DOTALL,
)
_GENERIC_SELF_ADMIN_RE = re.compile(
    r"\b(?:this\s+law(?:\s+and\s+any\s+(?:legislation\s+made\s+for\s+the\s+purposes?\s+of\s+this\s+law|"
    r"regulations?\s+made\s+under\s+it))?\s+(?:is|are|shall\s+be)\s+administered\s+by\b"
    r"|the\s+[^.]+?\s+shall\s+administer\s+this\s+law)\b",
    re.IGNORECASE | re.DOTALL,
)
_COMMON_ELEMENTS_TITLE_STOPWORDS = {
    "the",
    "of",
    "and",
    "in",
    "on",
    "for",
    "to",
    "by",
    "law",
    "laws",
    "regulation",
    "regulations",
    "difc",
}
_COMMON_ELEMENTS_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TITLE_LAW_NO_SUFFIX_RE = re.compile(r"\s*,?\s*DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_TITLE_LEADING_CONNECTOR_RE = re.compile(r"^(?:(?:of|and|the)\s+)+", re.IGNORECASE)
_TITLE_REF_BAD_LEAD_RE = re.compile(
    r"^(?:(?:which|what|how|mention|mentions|reference|references|their|these|those|do|does|did|"
    r"administer|administers|administered|administering)\s+)+",
    re.IGNORECASE,
)
_TITLE_QUERY_BAD_LEAD_RE = re.compile(r"^(?:(?:is|are)\s+)?(?:the\s+)?titles?\s+of\s+", re.IGNORECASE)
_TITLE_GENERIC_QUESTION_LEAD_RE = re.compile(
    r"^(?:(?:on\s+what\s+date|in\s+what\s+year|what|which|when|where|who|how|was|were|is|are|did|does|do)\s+)+"
    r"(?:(?:the|its)\s+)?(?:(?:citation\s+)?titles?\s+of\s+)?",
    re.IGNORECASE,
)
_TITLE_CONTEXT_BAD_LEAD_RE = re.compile(
    r"^(?:(?:interpretation\s+sections?|sections?|section\s+\d+|schedule\s+\d+)\s+of\s+)+",
    re.IGNORECASE,
)
_TITLE_PREPOSITION_BAD_LEAD_RE = re.compile(
    r"^(?:(?:under|for|to|about|regarding|concerning|within|as|than)\s+)+(?:the\s+)?",
    re.IGNORECASE,
)
_LAW_NO_REF_RE = re.compile(r"\blaw\s+no\.?\s*(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)

def _is_enumeration_query(query: str) -> bool:
    """Check if query is asking for enumeration across multiple entities."""
    return bool(_ENUMERATION_RE.search(query))


def _is_multi_criteria_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or not _is_enumeration_query(q):
        return False
    criteria_hits = sum(1 for hint in _MULTI_CRITERIA_ENUM_HINTS if hint in q)
    if criteria_hits >= 2:
        return True
    return " and " in q and criteria_hits >= 1


def _is_broad_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or not _is_enumeration_query(q):
        return False
    return q.startswith(("which laws", "which regulations", "which documents", "which cases", "list all", "identify all", "name all"))


def _is_registrar_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return _is_enumeration_query(q) and "administered by the registrar" in q


def _is_ruler_enactment_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return _is_enumeration_query(q) and ("made by the ruler" in q or "ruler of dubai" in q) and "enactment notice" in q


def _is_common_elements_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return "common elements" in q or "elements in common" in q or " in common" in q


def _is_common_judge_compare_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    return "judge" in q and (
        "in common" in q
        or "same judge" in q
        or "judges in common" in q
        or "judge who presided over both" in q
        or "presided over both" in q
        or ("did any judge" in q and "both" in q)
        or "judge involved in both" in q
        or "judge participated in both" in q
        or "judge who participated in both" in q
    )


def _is_case_outcome_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    if "result of the application" in q or "outcome of the specific order or application" in q:
        return True
    if "it is hereby ordered that" in q or "it is hereby ordered" in q:
        return True
    if "final ruling" in q or "court of appeal rule" in q:
        return True
    return ("outcome" in q or "result" in q) and (
        "application" in q or "appeal" in q or "order" in q
    )


def _is_case_issue_date_name_compare_query(query: str, *, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if answer_type.strip().lower() != "name":
        return False
    case_ref_count = len(_DIFC_CASE_ID_RE.findall(query or ""))
    if case_ref_count != 2:
        return False
    return (
        "date of issue" in q
        or "issue date" in q
        or "issued first" in q
        or "issued earlier" in q
        or ("issued" in q and "earlier" in q)
    )


def _is_case_monetary_claim_compare_query(query: str, *, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if answer_type.strip().lower() != "name":
        return False
    case_ref_count = len(_DIFC_CASE_ID_RE.findall(query or ""))
    if case_ref_count != 2:
        return False
    if "claim" not in q:
        return False
    return any(
        phrase in q
        for phrase in (
            "higher monetary claim",
            "lower monetary claim",
            "greater monetary claim",
            "largest monetary claim",
            "smallest monetary claim",
            "higher claim",
            "lower claim",
        )
    )


def _is_case_party_overlap_compare_query(query: str, *, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if answer_type.strip().lower() not in {"boolean", "name"}:
        return False
    if len(_DIFC_CASE_ID_RE.findall(query or "")) < 2:
        return False
    if any(
        phrase in q
        for phrase in (
            "same legal",
            "same parties",
            "same party",
            "same entities",
            "main party common to both",
            "main party to both",
            "appeared in both",
            "appears in both",
            "appears as a main party in both",
            "named as a main party in both",
            "as parties",
        )
    ):
        return True
    has_party_subject = any(
        token in q for token in ("party", "parties", "claimant", "defendant", "entity", "entities", "individual")
    )
    has_overlap_signal = any(token in q for token in ("common", "same", "appeared", "appears", "named", "both"))
    return has_party_subject and has_overlap_signal


def _is_case_party_role_name_query(query: str, *, answer_type: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if answer_type.strip().lower() not in {"name", "names"}:
        return False
    if len(_DIFC_CASE_ID_RE.findall(query or "")) < 1:
        return False
    has_party_subject = any(
        token in q
        for token in (
            "claimant",
            "claimants",
            "defendant",
            "defendants",
            "appellant",
            "appellants",
            "respondent",
            "respondents",
            "applicant",
            "applicants",
            "party",
            "parties",
        )
    )
    if not has_party_subject:
        return False
    return any(
        phrase in q
        for phrase in (
            "who were",
            "who are",
            "who is",
            "listed as",
            "listed in",
            "listed on",
            "named in",
        )
    )


def _is_interpretation_sections_common_elements_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    return _is_common_elements_query(query) and "interpretation section" in q


def _is_named_reference_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return _is_broad_enumeration_query(query) and any(term in q for term in ("mention", "mentions", "reference", "references"))


def _is_company_structure_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not _is_broad_enumeration_query(query):
        return False
    return (
        "company structures" in q
        or ("schedule 2" in q and "arbitration law" in q)
        or "application of the arbitration law" in q
    )


def _is_named_commencement_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    return any(term in q for term in ("commencement", "come into force", "effective date", "enactment notice"))


def _is_named_multi_title_lookup_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or len(_extract_question_title_refs(query)) + len(_LAW_NO_REF_RE.findall(query or "")) < 2:
        return False
    if _is_broad_enumeration_query(query) or _is_common_elements_query(query):
        return False
    return any(
        term in q
        for term in (
            "commencement",
            "come into force",
            "effective date",
            "enactment",
            "administration",
            "administered",
            "citation title",
            "citation titles",
            "title of",
            "titles of",
            "last updated",
            "updated",
        )
    )


def _is_named_amendment_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q or _is_broad_enumeration_query(query):
        return False
    ref_count = len(_extract_question_title_refs(query)) + len(_LAW_NO_REF_RE.findall(query or ""))
    return ref_count >= 1 and "enact" in q and (
        "what law did it amend" in q or "what laws did it amend" in q
    )


def _is_account_effective_dates_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    return bool(q) and "pre-existing" in q and "new accounts" in q and "effective date" in q and "enact" in q


def _is_remuneration_recordkeeping_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    return "article 16(1)(c)" in q and "keep records" in q and "remuneration" in q


def _is_restriction_effectiveness_query(query: str) -> bool:
    q = re.sub(r"\s+", " ", (query or "").strip()).lower()
    if not q:
        return False
    return (
        "article" in q
        and "restriction" in q
        and "transfer" in q
        and "actual knowledge" in q
        and ("effective" in q or "ineffective" in q)
    )


def _is_citation_title_query(query: str) -> bool:
    q = (query or "").strip().lower()
    return "citation title" in q or "citation titles" in q


def _is_recall_sensitive_broad_enumeration_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q or not _is_broad_enumeration_query(q):
        return False
    if "interpretative provisions" in q:
        return True
    if "difc law no. 2 of 2022" in q and "amended by" in q:
        return True
    return "enactment notice" in q and ("made by the ruler" in q or "ruler of dubai" in q)


def _extract_question_title_refs(query: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    for match in _AMENDMENT_TITLE_RE.finditer(query or ""):
        ref = re.sub(r"\s+", " ", match.group(1).strip())
        ref = _TITLE_REF_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_QUERY_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", ref)
        ref = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", ref)
        ref = _TITLE_LEADING_CONNECTOR_RE.sub("", ref).strip(" ,.;:")
        if not ref:
            continue
        key = ref.casefold()
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)

    for title, year in _TITLE_REF_RE.findall(query or ""):
        normalized_title = _TITLE_REF_BAD_LEAD_RE.sub("", title.strip())
        normalized_title = _TITLE_QUERY_BAD_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_CONTEXT_BAD_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", normalized_title)
        normalized_title = _TITLE_LEADING_CONNECTOR_RE.sub("", normalized_title).strip(" ,.;:")
        parts = [normalized_title, year.strip()] if year else [normalized_title]
        ref = re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip(" ,.;:")
        if ref.casefold() in {"law", "difc law"}:
            continue
        if not ref:
            continue
        key = ref.casefold()
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    pruned: list[str] = []
    lowered_refs = [ref.casefold() for ref in refs]
    for idx, ref in enumerate(refs):
        lowered = lowered_refs[idx]
        if any(
            idx != other_idx
            and lowered != other_lowered
            and re.search(rf"\b{re.escape(lowered)}\b", other_lowered)
            for other_idx, other_lowered in enumerate(lowered_refs)
        ):
            continue
        pruned.append(ref)
    return pruned


class RAGState(TypedDict, total=False):
    # Input
    query: str
    request_id: str
    question_id: str
    answer_type: str
    doc_refs: list[str]

    # Routing
    complexity: QueryComplexity
    model: str
    max_tokens: int
    sub_queries: list[str]

    # Retrieval/rerank
    retrieved: list[RetrievedChunk]
    must_include_chunk_ids: list[str]
    reranked: list[RankedChunk]
    context_chunks: list[RankedChunk]
    max_rerank_score: float
    conflict_prompt_context: str

    # Generation
    answer: str
    citations: list[Citation]
    cited_chunk_ids: list[str]
    streamed: bool

    # Retry
    retried: bool

    # Telemetry
    collector: TelemetryCollector
    telemetry: TelemetryPayload


class RAGPipelineBuilder:
    """Build a minimal LangGraph-based RAG pipeline with one conditional retry edge."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        reranker: RerankerClient,
        generator: RAGGenerator,
        classifier: QueryClassifier,
        verifier: AnswerVerifier | None = None,
        decomposer: QueryDecomposer | None = None,
        conflict_detector: ConflictDetector | None = None,
        strict_answerer: StrictAnswerer | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._classifier = classifier
        self._verifier = verifier
        self._decomposer = decomposer or QueryDecomposer()
        self._conflict_detector = conflict_detector or ConflictDetector()
        self._strict_answerer = strict_answerer or StrictAnswerer()
        self._settings = get_settings()

    def build(self) -> Any:
        graph = cast("Any", StateGraph(RAGState))

        graph.add_node("classify", self._classify)
        graph.add_node("decompose", self._decompose)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rerank", self._rerank)
        graph.add_node("detect_conflicts", self._detect_conflicts)
        graph.add_node("confidence_check", self._confidence_check)
        graph.add_node("retry_retrieve", self._retry_retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.add_node("emit", self._emit)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "detect_conflicts")
        graph.add_edge("detect_conflicts", "confidence_check")
        graph.add_conditional_edges(
            "confidence_check",
            self._route_after_confidence,
            {
                "retry_retrieve": "retry_retrieve",
                "generate": "generate",
            },
        )
        graph.add_edge("retry_retrieve", "generate")
        graph.add_edge("generate", "verify")
        graph.add_edge("verify", "emit")
        graph.add_edge("emit", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def compile(self) -> Any:
        return self.build().compile()

    async def _classify(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        question_id = str(state.get("question_id") or state["request_id"])
        answer_type = str(state.get("answer_type") or "free_text").strip() or "free_text"
        with collector.timed("classify"):
            normalized_query = self._classifier.normalize_query(state["query"])
            complexity = self._classifier.classify(normalized_query)
            model = self._classifier.select_model(complexity)
            max_tokens = self._classifier.select_max_tokens(complexity)
            answer_type_lower = answer_type.strip().lower()
            if answer_type_lower in {"boolean", "number", "date", "name", "names"}:
                llm_settings = getattr(self._settings, "llm", None)
                strict_max = int(getattr(llm_settings, "strict_max_tokens", 150)) if llm_settings is not None else 150
                max_tokens = min(int(max_tokens), strict_max)
                if answer_type_lower == "boolean":
                    boolean_max = int(getattr(self._settings.pipeline, "boolean_max_tokens", 96))
                    max_tokens = min(max_tokens, max(1, boolean_max))
                if bool(getattr(self._settings.pipeline, "strict_types_force_simple_model", True)):
                    complexity = QueryComplexity.SIMPLE
                    if llm_settings is not None:
                        model = str(
                            getattr(llm_settings, "strict_model", getattr(llm_settings, "simple_model", model))
                        )
            elif answer_type_lower == "free_text":
                # Harder free_text questions benefit from the complex model when they
                # either reference multiple named titles or combine several filters.
                if complexity == QueryComplexity.SIMPLE and (
                    len(_TITLE_REF_RE.findall(normalized_query)) >= 2
                    or _is_multi_criteria_enumeration_query(normalized_query)
                    or _is_registrar_enumeration_query(normalized_query)
                    or _is_ruler_enactment_query(normalized_query)
                    or _is_broad_enumeration_query(normalized_query)
                ):
                    complexity = QueryComplexity.COMPLEX
                    model = self._classifier.select_model(complexity)
                    max_tokens = self._classifier.select_max_tokens(complexity)
                # Prevent truncation on list/enumeration questions that still classify as SIMPLE.
                if _needs_long_free_text_answer(normalized_query):
                    llm_settings = getattr(self._settings, "llm", None)
                    complex_cap = int(getattr(llm_settings, "complex_max_tokens", max_tokens)) if llm_settings is not None else int(max_tokens)
                    max_tokens = max(int(max_tokens), max(600, complex_cap))
                if _is_common_elements_query(normalized_query):
                    common_elements_cap = int(getattr(self._settings.pipeline, "common_elements_max_tokens", 360))
                    max_tokens = min(int(max_tokens), max(200, common_elements_cap))
            doc_refs = []
            # For retrieval filtering, free_text benefits from broader ref extraction (e.g., "Trust Law 2018"),
            # while strict types keep a narrower set (law numbers + case IDs).
            extractor_name = "extract_doc_refs"
            if answer_type_lower == "free_text":
                extractor_name = "extract_query_refs"
            extract_obj: object = getattr(self._classifier, extractor_name, None)
            if not callable(extract_obj) and extractor_name != "extract_doc_refs":
                extract_obj = getattr(self._classifier, "extract_doc_refs", None)
            if callable(extract_obj):
                try:
                    doc_refs_obj: object = extract_obj(normalized_query)
                    if isinstance(doc_refs_obj, list):
                        items = cast("list[object]", doc_refs_obj)
                        doc_refs = [text for item in items if (text := str(item).strip())]
                except Exception:
                    logger.warning("Failed extracting doc refs from query", exc_info=True)

            # Strict-types benefit from title refs too (e.g., "Personal Property Law 2005") to avoid
            # falling back to dense retrieval and to improve grounding for Article-based questions.
            if not doc_refs and answer_type_lower in {"boolean", "number", "date", "name", "names"}:
                extract_any: object = getattr(self._classifier, "extract_query_refs", None)
                if callable(extract_any):
                    try:
                        refs_obj: object = extract_any(normalized_query)
                        if isinstance(refs_obj, list):
                            items = cast("list[object]", refs_obj)
                            extra_refs = [text for item in items if (text := str(item).strip())]
                            extra_refs = [
                                ref
                                for ref in extra_refs
                                if (" Law" in ref or ref.endswith("Law") or "Regulations" in ref)
                            ]
                            # Bound to keep sparse filters stable and avoid over-filtering.
                            doc_refs = extra_refs[:3]
                    except Exception:
                        logger.warning("Failed extracting title refs for strict types", exc_info=True)

            collector.set_request_metadata(question_id=question_id, answer_type=answer_type, doc_refs=doc_refs)

            llm_settings = getattr(self._settings, "llm", None)
            upgrade_model = getattr(llm_settings, "upgrade_model", "")
            if (
                answer_type_lower == "free_text"
                and complexity == QueryComplexity.COMPLEX
                and upgrade_model
            ):
                triggers: list[str] = []
                if len(doc_refs) >= 3:
                    triggers.append("multi_entity")
                if _is_enumeration_query(normalized_query):
                    triggers.append("enumeration")

                if triggers:
                    model = upgrade_model
                    max_tokens = int(getattr(llm_settings, "upgrade_max_tokens", 1800))
                    collector.set_model_upgraded(True)
                    logger.info(
                        "Selective upgrade triggered on %s due to %s",
                        question_id,
                        triggers,
                        extra={"request_id": state.get("request_id"), "triggers": triggers}
                    )

        logger.info(
            "Classified query as %s -> model=%s",
            complexity.value,
            model,
            extra={
                "request_id": state.get("request_id"),
                "question_id": question_id,
                "answer_type": answer_type,
                "doc_refs": doc_refs,
            },
        )
        return {
            "query": normalized_query,
            "complexity": complexity,
            "model": model,
            "max_tokens": max_tokens,
            "question_id": question_id,
            "answer_type": answer_type,
            "doc_refs": doc_refs,
        }

    async def _decompose(self, state: RAGState) -> dict[str, object]:
        if not bool(getattr(self._settings.pipeline, "enable_multi_hop", False)):
            return {"sub_queries": []}
        complexity = state.get("complexity", QueryComplexity.SIMPLE)
        query = state["query"]
        if not self._decomposer.should_decompose(query, complexity):
            return {"sub_queries": []}

        max_subqueries = int(getattr(self._settings.pipeline, "multi_hop_max_subqueries", 3))
        sub_queries = self._decomposer.decompose(query, max_subqueries=max_subqueries)
        logger.info(
            "Decomposed query into %d sub-queries",
            len(sub_queries),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {"sub_queries": sub_queries}

    async def _retrieve(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        sub_queries = [text for text in state.get("sub_queries", []) if str(text).strip()]
        doc_refs = [text for text in state.get("doc_refs", []) if str(text).strip()]
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        is_boolean = answer_type == "boolean"
        is_strict = answer_type in {"boolean", "number", "date", "name", "names"}
        seed_terms = self._seed_terms_for_query(state.get("query", ""))
        must_include_chunk_ids: list[str] = []

        if sub_queries and bool(getattr(self._settings.pipeline, "enable_multi_hop", False)):
            search_queries = [state["query"], *sub_queries]
            # Dedupe while preserving order.
            seen: set[str] = set()
            deduped_queries: list[str] = []
            for item in search_queries:
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                deduped_queries.append(normalized)

            with collector.timed("embed"):
                vectors = await asyncio.gather(*[self._retriever.embed_query(query) for query in deduped_queries])

            with collector.timed("qdrant"):
                results = await asyncio.gather(
                    *[
                        self._retriever.retrieve(
                            query,
                            query_vector=vector,
                            doc_refs=state.get("doc_refs"),
                        )
                        for query, vector in zip(deduped_queries, vectors, strict=True)
                    ]
                )
            merged: dict[str, RetrievedChunk] = {}
            for row in results:
                for chunk in row:
                    existing = merged.get(chunk.chunk_id)
                    if existing is None or chunk.score > existing.score:
                        merged[chunk.chunk_id] = chunk
            limit = int(self._settings.reranker.rerank_candidates)
            retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]
        else:
            retrieved: list[RetrievedChunk] = []
            use_doc_ref_sparse_only = bool(getattr(self._settings.pipeline, "doc_ref_sparse_only", True))
            if doc_refs and use_doc_ref_sparse_only:
                with collector.timed("qdrant"):
                    if len(doc_refs) >= 2 and bool(getattr(self._settings.pipeline, "doc_ref_multi_retrieve", True)):
                        # Multi-ref questions are common in legal Q&A (e.g., "Did cases A and B share a judge?").
                        # A single query with MatchAny can return chunks from only one document, hurting grounding.
                        # We solve this by running one sparse-only retrieval per ref and forcing at least one
                        # chunk per ref into the candidate set.
                        default_per_ref_top_k = int(
                            getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30)
                        )
                        if is_boolean:
                            per_ref_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                        elif is_strict:
                            per_ref_top_k = int(
                                getattr(self._settings.pipeline, "strict_multi_ref_top_k_per_ref", 12)
                            )
                        else:
                            per_ref_top_k = default_per_ref_top_k
                        # Use per-ref query strings that remove the *other* refs to avoid BM25 "query dilution"
                        # (e.g., when scoring chunks for ref A, terms from ref B are absent and can push down
                        # relevant pages like judges/claim values).
                        base_query = state["query"]
                        for other_ref in doc_refs:
                            base_query = re.sub(re.escape(other_ref), " ", base_query, flags=re.IGNORECASE)
                        base_query = re.sub(r"\s+", " ", base_query).strip()
                        results = await asyncio.gather(
                            *[
                                self._retriever.retrieve(
                                    self._augment_query_for_sparse_retrieval(
                                        f"{ref} {base_query}".strip() if base_query else ref
                                    ),
                                    query_vector=None,
                                    doc_refs=[ref],
                                    sparse_only=True,
                                    top_k=per_ref_top_k,
                                )
                                for ref in doc_refs
                            ]
                        )
                        if _is_common_judge_compare_query(state["query"]):
                            judge_results = await asyncio.gather(
                                *[
                                    self._retriever.retrieve(
                                        self._augment_query_for_sparse_retrieval(
                                            f"{ref} chief justice justice judge order with reasons before h.e."
                                        ),
                                        query_vector=None,
                                        doc_refs=[ref],
                                        sparse_only=True,
                                        top_k=min(per_ref_top_k, 8),
                                    )
                                    for ref in doc_refs
                                ]
                            )
                            merged_rows: list[list[RetrievedChunk]] = []
                            for base_row, judge_row in zip(results, judge_results, strict=True):
                                merged_rows.append([*base_row, *judge_row])
                            results = merged_rows
                        merged: dict[str, RetrievedChunk] = {}
                        for row in results:
                            if row:
                                if _is_common_judge_compare_query(state["query"]):
                                    seed = (
                                        self._select_case_judge_seed_chunk_id(row)
                                        or self._select_seed_chunk_id(row, seed_terms)
                                        or row[0].chunk_id
                                    )
                                elif _is_case_issue_date_name_compare_query(
                                    state["query"], answer_type=state["answer_type"]
                                ):
                                    seed = (
                                        self._select_case_issue_date_seed_chunk_id(row)
                                        or (
                                            self._select_case_monetary_claim_seed_chunk_id(row)
                                            if _is_case_monetary_claim_compare_query(
                                                state["query"], answer_type=state["answer_type"]
                                            )
                                            else None
                                        )
                                        or self._select_seed_chunk_id(row, seed_terms)
                                        or row[0].chunk_id
                                    )
                                elif _is_case_monetary_claim_compare_query(
                                    state["query"], answer_type=state["answer_type"]
                                ):
                                    seed = (
                                        self._select_case_monetary_claim_seed_chunk_id(row)
                                        or self._select_seed_chunk_id(row, seed_terms)
                                        or row[0].chunk_id
                                    )
                                else:
                                    seed = self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                                must_include_chunk_ids.append(seed)
                            for chunk in row:
                                existing = merged.get(chunk.chunk_id)
                                if existing is None or chunk.score > existing.score:
                                    merged[chunk.chunk_id] = chunk

                        # Dedupe seeds while preserving order.
                        seen_seed: set[str] = set()
                        seeds: list[str] = []
                        for chunk_id in must_include_chunk_ids:
                            if chunk_id in seen_seed:
                                continue
                            seen_seed.add(chunk_id)
                            seeds.append(chunk_id)
                        must_include_chunk_ids = seeds

                        ranked = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)
                        seed_set = set(must_include_chunk_ids)
                        ordered: list[RetrievedChunk] = []
                        for chunk_id in must_include_chunk_ids:
                            seed_chunk = merged.get(chunk_id)
                            if seed_chunk is not None:
                                ordered.append(seed_chunk)
                        for chunk in ranked:
                            if chunk.chunk_id in seed_set:
                                continue
                            ordered.append(chunk)

                        limit = int(self._settings.reranker.rerank_candidates)
                        retrieved = ordered[:limit]
                    else:
                        retrieval_query = self._augment_query_for_sparse_retrieval(state["query"])
                        retrieved = await self._retriever.retrieve(
                            retrieval_query,
                            query_vector=None,
                            doc_refs=doc_refs,
                            sparse_only=True,
                            top_k=(
                                int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                                if is_boolean
                                else (
                                    int(getattr(self._settings.pipeline, "strict_doc_ref_top_k", 16))
                                    if is_strict
                                    else None
                                )
                            ),
                        )
                        if doc_refs and _is_case_outcome_query(state["query"]):
                            outcome_results = await asyncio.gather(
                                *[
                                    self._retriever.retrieve(
                                        self._augment_query_for_sparse_retrieval(
                                            f"{ref} order with reasons it is hereby ordered that application appeal costs"
                                        ),
                                        query_vector=None,
                                        doc_refs=[ref],
                                        sparse_only=True,
                                        top_k=8,
                                    )
                                    for ref in doc_refs
                                ]
                            )
                            merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                            for row in outcome_results:
                                if row:
                                    seed = self._select_case_outcome_seed_chunk_id(row) or row[0].chunk_id
                                    if seed not in must_include_chunk_ids:
                                        must_include_chunk_ids.append(seed)
                                for chunk in row:
                                    existing = merged.get(chunk.chunk_id)
                                    if existing is None or chunk.score > existing.score:
                                        merged[chunk.chunk_id] = chunk
                            ranked = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)
                            seed_set = set(must_include_chunk_ids)
                            ordered: list[RetrievedChunk] = []
                            for chunk_id in must_include_chunk_ids:
                                seed_chunk = merged.get(chunk_id)
                                if seed_chunk is not None:
                                    ordered.append(seed_chunk)
                            for chunk in ranked:
                                if chunk.chunk_id in seed_set:
                                    continue
                                ordered.append(chunk)
                            limit = int(self._settings.reranker.rerank_candidates)
                            retrieved = ordered[:limit]
                        if retrieved and seed_terms:
                            seed = self._select_seed_chunk_id(retrieved, seed_terms)
                            if seed:
                                must_include_chunk_ids.append(seed)

            if not retrieved:
                with collector.timed("embed"):
                    query_vector = await self._retriever.embed_query(state["query"])

                with collector.timed("qdrant"):
                    prefetch_dense_override = None
                    prefetch_sparse_override = None
                    retrieve_top_k_override = None
                    if is_boolean and not doc_refs:
                        prefetch_dense_override = int(getattr(self._settings.pipeline, "boolean_prefetch_dense", 40))
                        prefetch_sparse_override = int(getattr(self._settings.pipeline, "boolean_prefetch_sparse", 40))
                        retrieve_top_k_override = int(
                            getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12)
                        )
                    elif is_strict and not doc_refs:
                        prefetch_dense_override = int(getattr(self._settings.pipeline, "strict_prefetch_dense", 24))
                        prefetch_sparse_override = int(getattr(self._settings.pipeline, "strict_prefetch_sparse", 24))
                        retrieve_top_k_override = int(
                            getattr(self._settings.pipeline, "rerank_max_candidates_strict_types", 20)
                        )
                    retrieved = await self._retriever.retrieve(
                        state["query"],
                        query_vector=query_vector,
                        prefetch_dense=prefetch_dense_override,
                        prefetch_sparse=prefetch_sparse_override,
                        top_k=retrieve_top_k_override,
                        doc_refs=doc_refs,
                    )

        # Title-based multi-retrieve: ensure we pull at least one chunk per *title* when the query mentions
        # multiple laws/regulations. This runs even when doc_refs are present (mixed refs like "Law No..." + "X Regulations").
        title_refs = self._extract_title_refs_from_query(state["query"])
        if title_refs and bool(getattr(self._settings.pipeline, "doc_ref_multi_retrieve", True)):
            default_per_ref_top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
            if is_boolean:
                per_ref_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
            elif is_strict:
                per_ref_top_k = int(getattr(self._settings.pipeline, "strict_multi_ref_top_k_per_ref", 12))
            elif answer_type == "free_text" and (
                _is_common_elements_query(state["query"])
                or _is_named_reference_enumeration_query(state["query"])
                or _is_company_structure_enumeration_query(state["query"])
            ):
                per_ref_top_k = int(getattr(self._settings.pipeline, "free_text_targeted_multi_ref_top_k", 12))
            else:
                per_ref_top_k = default_per_ref_top_k
            cap_titles = 3

            # Avoid re-retrieving titles that were already used as retrieval filters.
            doc_ref_lower = {ref.lower() for ref in doc_refs}
            title_refs = [title for title in title_refs if title.lower() not in doc_ref_lower]

            should_title_retrieve = (not doc_refs and len(title_refs) >= 2) or (bool(doc_refs) and bool(title_refs))
            if (
                not should_title_retrieve
                and not doc_refs
                and len(title_refs) == 1
                and (
                    self._should_apply_doc_shortlist_gating(
                        query=state["query"],
                        answer_type=answer_type,
                        doc_refs=[],
                    )
                    or _is_named_amendment_query(state["query"])
                    or _is_account_effective_dates_query(state["query"])
                )
            ):
                should_title_retrieve = True
            if should_title_retrieve and title_refs:
                base_query = state["query"]
                for other_ref in list(doc_refs) + title_refs[:cap_titles]:
                    base_query = re.sub(re.escape(other_ref), " ", base_query, flags=re.IGNORECASE)
                base_query = re.sub(r"\s+", " ", base_query).strip()
                use_targeted_title_query = (
                    is_strict or _is_named_amendment_query(state["query"]) or _is_account_effective_dates_query(state["query"])
                )

                with collector.timed("qdrant"):
                    title_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                self._augment_query_for_sparse_retrieval(
                                    self._targeted_named_ref_query(
                                        query=state["query"],
                                        ref=title,
                                        refs=doc_refs or title_refs,
                                    )
                                    if use_targeted_title_query
                                    else (f"{title} {base_query}".strip() if base_query else title)
                                ),
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_ref_top_k,
                            )
                            for title in title_refs[:cap_titles]
                        ]
                    )

                title_merged: dict[str, RetrievedChunk] = {}
                seeds: list[str] = []
                for title, row in zip(title_refs[:cap_titles], title_results, strict=False):
                    if row:
                        seed = self._select_targeted_title_seed_chunk_id(
                            query=state["query"],
                            answer_type=answer_type,
                            ref=title,
                            chunks=row,
                            seed_terms=seed_terms,
                        ) or row[0].chunk_id
                        seeds.append(seed)
                    for chunk in row:
                        existing = title_merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            title_merged[chunk.chunk_id] = chunk

                if seeds:
                    # Preserve order, dedupe.
                    seen_seed: set[str] = set()
                    for chunk_id in seeds:
                        if chunk_id in seen_seed:
                            continue
                        seen_seed.add(chunk_id)
                        must_include_chunk_ids.append(chunk_id)

                if title_merged:
                    # Merge title-based candidates into the main retrieved set (bounded).
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=list(title_merged.values()),
                        must_keep_chunk_ids=seeds,
                        limit=limit,
                    )

        if (
            answer_type == "free_text"
            and (
                _is_named_multi_title_lookup_query(state["query"])
                or _is_named_amendment_query(state["query"])
                or _is_common_elements_query(state["query"])
                or (
                    "administ" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
                    and len([ref for ref in state.get("doc_refs", []) if str(ref).strip()]) >= 2
                    and not _is_broad_enumeration_query(state["query"])
                )
                or (
                    "penalt" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
                    and len([ref for ref in state.get("doc_refs", []) if str(ref).strip()]) >= 2
                    and not _is_broad_enumeration_query(state["query"])
                )
            )
        ):
            named_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            missing_refs = self._missing_named_ref_targets(
                query=state["query"],
                doc_refs=named_refs,
                retrieved=retrieved,
            )
            if missing_refs:
                per_ref_top_k = int(getattr(self._settings.pipeline, "free_text_targeted_multi_ref_top_k", 12))
                refs_for_query = named_refs or self._support_question_refs(state["query"])
                with collector.timed("qdrant"):
                    targeted_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                self._augment_query_for_sparse_retrieval(
                                    self._targeted_named_ref_query(
                                        query=state["query"],
                                        ref=ref,
                                        refs=refs_for_query,
                                    )
                                ),
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_ref_top_k,
                            )
                            for ref in missing_refs[:3]
                        ]
                    )

                merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                extra_seeds: list[str] = []
                query_lower = re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
                for ref, row in zip(missing_refs[:3], targeted_results, strict=False):
                    if row:
                        family_seeds: list[str] = []
                        if "administ" in query_lower:
                            family_seeds = self._administration_support_family_seed_chunk_ids(ref=ref, retrieved=row)
                        if family_seeds:
                            extra_seeds.extend(family_seeds)
                        else:
                            seed = self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                            extra_seeds.append(seed)
                    for chunk in row:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk

                seen_seed: set[str] = set()
                for chunk_id in extra_seeds:
                    if chunk_id in seen_seed:
                        continue
                    seen_seed.add(chunk_id)
                    must_include_chunk_ids.append(chunk_id)

                if merged:
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

        if (
            answer_type == "boolean"
            and "administ" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
            and not _is_broad_enumeration_query(state["query"])
        ):
            admin_refs = self._extract_title_refs_from_query(state["query"]) or self._support_question_refs(state["query"])
            missing_refs = self._missing_named_ref_targets(
                query=state["query"],
                doc_refs=admin_refs,
                retrieved=retrieved,
            )
            if missing_refs:
                per_ref_top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                with collector.timed("qdrant"):
                    targeted_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                self._augment_query_for_sparse_retrieval(
                                    self._targeted_named_ref_query(
                                        query=state["query"],
                                        ref=ref,
                                        refs=admin_refs,
                                    )
                                ),
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_ref_top_k,
                            )
                            for ref in missing_refs[:3]
                        ]
                    )

                merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                extra_seeds: list[str] = []
                for ref, row in zip(missing_refs[:3], targeted_results, strict=False):
                    if row:
                        family_seeds = self._administration_support_family_seed_chunk_ids(ref=ref, retrieved=row)
                        if family_seeds:
                            extra_seeds.extend(family_seeds)
                        else:
                            seed = self._select_targeted_title_seed_chunk_id(
                                query=state["query"],
                                answer_type=answer_type,
                                ref=ref,
                                chunks=row,
                                seed_terms=seed_terms,
                            ) or self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                            extra_seeds.append(seed)
                    for chunk in row:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk

                if extra_seeds:
                    must_include_chunk_ids.extend(extra_seeds)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=list(merged.values()),
                        must_keep_chunk_ids=extra_seeds,
                        limit=limit,
                    )

        if answer_type == "free_text" and _is_named_amendment_query(state["query"]):
            amendment_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            amendment_ref = (amendment_refs or self._support_question_refs(state["query"]))[:1]
            if amendment_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    f'{amendment_ref[0]} "as amended by" "amended by" enacted enactment'
                )
                top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    amendment_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if amendment_results:
                    merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                    seed = self._select_seed_chunk_id(amendment_results, seed_terms) or amendment_results[0].chunk_id
                    must_include_chunk_ids.append(seed)
                    for chunk in amendment_results:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

        if answer_type == "free_text" and _is_account_effective_dates_query(state["query"]):
            account_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            account_ref = (account_refs or self._support_question_refs(state["query"]))[:1]
            if account_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    self._targeted_named_ref_query(
                        query=state["query"],
                        ref=account_ref[0],
                        refs=account_refs or account_ref,
                    )
                )
                top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    account_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if account_results:
                    family_seeds = self._account_effective_support_family_seed_chunk_ids(
                        ref=account_ref[0],
                        retrieved=account_results,
                    )
                    if not family_seeds:
                        family_seeds = [self._select_seed_chunk_id(account_results, seed_terms) or account_results[0].chunk_id]
                    must_include_chunk_ids.extend(family_seeds)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=account_results,
                        must_keep_chunk_ids=family_seeds,
                        limit=limit,
                    )

        if answer_type == "free_text" and _is_remuneration_recordkeeping_query(state["query"]):
            remuneration_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            remuneration_ref = (
                remuneration_refs
                or self._support_question_refs(state["query"])
                or self._extract_title_refs_from_query(state["query"])
            )[:1]
            if remuneration_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    f"{remuneration_ref[0]} article 16 payroll records remuneration gross and net pay period"
                )
                top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    remuneration_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if remuneration_results:
                    best_chunk: RetrievedChunk | None = None
                    best_score = 0
                    for chunk in remuneration_results:
                        score = self._remuneration_recordkeeping_clause_score(chunk)
                        if score > best_score:
                            best_chunk = chunk
                            best_score = score
                    must_keep = [best_chunk.chunk_id] if best_chunk is not None and best_score > 0 else []
                    if must_keep:
                        must_include_chunk_ids.extend(must_keep)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=remuneration_results,
                        must_keep_chunk_ids=must_keep,
                        limit=limit,
                    )

        if answer_type == "boolean" and _is_restriction_effectiveness_query(state["query"]):
            restriction_refs = [str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()]
            restriction_ref = (restriction_refs or self._support_question_refs(state["query"]))[:1]
            if restriction_ref:
                targeted_query = self._augment_query_for_sparse_retrieval(
                    self._targeted_named_ref_query(
                        query=state["query"],
                        ref=restriction_ref[0],
                        refs=restriction_refs or restriction_ref,
                    )
                )
                top_k = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                with collector.timed("qdrant"):
                    restriction_results = await self._retriever.retrieve(
                        targeted_query,
                        query_vector=None,
                        doc_refs=None,
                        sparse_only=True,
                        top_k=top_k,
                    )
                if restriction_results:
                    best_chunk: RetrievedChunk | None = None
                    best_score = 0
                    best_rank_score = 0.0
                    for chunk in restriction_results:
                        clause_score = self._restriction_effectiveness_clause_score(
                            ref=restriction_ref[0],
                            chunk=chunk,
                        )
                        rank_score = float(getattr(chunk, "score", 0.0) or 0.0)
                        if clause_score > best_score or (clause_score == best_score and rank_score > best_rank_score):
                            best_chunk = chunk
                            best_score = clause_score
                            best_rank_score = rank_score
                    if best_chunk is not None and best_score > 0:
                        must_include_chunk_ids.append(best_chunk.chunk_id)
                    limit = int(self._settings.reranker.rerank_candidates)
                    retrieved = self._merge_retrieved_preserving_chunk_ids(
                        retrieved=retrieved,
                        extra=restriction_results,
                        must_keep_chunk_ids=[best_chunk.chunk_id] if best_chunk is not None and best_score > 0 else [],
                        limit=limit,
                    )

        if (
            is_strict
            and doc_refs
            and self._extract_provision_refs(state["query"])
            and not _is_restriction_effectiveness_query(state["query"])
        ):
            per_ref_top_k = (
                int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                if is_boolean
                else int(getattr(self._settings.pipeline, "strict_doc_ref_top_k", 16))
            )
            with collector.timed("qdrant"):
                targeted_results = await asyncio.gather(
                    *[
                        self._retriever.retrieve(
                            self._augment_query_for_sparse_retrieval(
                                self._targeted_provision_ref_query(
                                    query=state["query"],
                                    ref=ref,
                                    refs=doc_refs,
                                )
                            ),
                            query_vector=None,
                            doc_refs=None,
                            sparse_only=True,
                            top_k=per_ref_top_k,
                        )
                        for ref in doc_refs[:3]
                    ]
                )

            merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
            extra_seeds: list[str] = []
            for row in targeted_results:
                if row:
                    seed = self._select_seed_chunk_id(row, seed_terms) or row[0].chunk_id
                    extra_seeds.append(seed)
                for chunk in row:
                    existing = merged.get(chunk.chunk_id)
                    if existing is None or chunk.score > existing.score:
                        merged[chunk.chunk_id] = chunk

            if extra_seeds:
                must_include_chunk_ids.extend(extra_seeds)
                limit = int(self._settings.reranker.rerank_candidates)
                retrieved = self._merge_retrieved_preserving_chunk_ids(
                    retrieved=retrieved,
                    extra=list(merged.values()),
                    must_keep_chunk_ids=extra_seeds,
                    limit=limit,
                )

        if _is_registrar_enumeration_query(state["query"]):
            candidate_titles: list[str] = []
            seen_titles: set[str] = set()
            for chunk in retrieved[:12]:
                section_path = str(getattr(chunk, "section_path", "") or "").lower()
                if "page:1" not in section_path and "page:4" not in section_path:
                    continue
                title_ref = self._extract_title_ref_from_chunk_text(chunk)
                normalized = re.sub(r"\s+", " ", title_ref).strip(" ,.;:")
                if not normalized or "law" not in normalized.lower():
                    continue
                key = normalized.casefold()
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                candidate_titles.append(normalized)
                if len(candidate_titles) >= 4:
                    break

            if candidate_titles:
                per_title_top_k = int(getattr(self._settings.pipeline, "doc_ref_multi_top_k_per_ref", 30))
                with collector.timed("qdrant"):
                    registrar_results = await asyncio.gather(
                        *[
                            self._retriever.retrieve(
                                f"{title} administration of this law registrar",
                                query_vector=None,
                                doc_refs=None,
                                sparse_only=True,
                                top_k=per_title_top_k,
                            )
                            for title in candidate_titles
                        ]
                    )
                merged: dict[str, RetrievedChunk] = {chunk.chunk_id: chunk for chunk in retrieved}
                for row in registrar_results:
                    for chunk in row:
                        existing = merged.get(chunk.chunk_id)
                        if existing is None or chunk.score > existing.score:
                            merged[chunk.chunk_id] = chunk
                limit = max(int(self._settings.reranker.rerank_candidates), len(retrieved))
                retrieved = sorted(merged.values(), key=lambda chunk: chunk.score, reverse=True)[:limit]

        if self._should_apply_doc_shortlist_gating(
            query=state["query"],
            answer_type=answer_type,
            doc_refs=doc_refs,
        ):
            shortlisted = self._apply_doc_shortlist_gating(
                query=state["query"],
                doc_refs=doc_refs,
                retrieved=retrieved,
                must_keep_chunk_ids=must_include_chunk_ids,
            )
            if shortlisted:
                retrieved = shortlisted
                if must_include_chunk_ids:
                    retrieved_ids = {chunk.chunk_id for chunk in retrieved}
                    must_include_chunk_ids = [chunk_id for chunk_id in must_include_chunk_ids if chunk_id in retrieved_ids]

        anchor_must_include = self._select_anchor_must_include_chunk_ids(
            query=state["query"],
            answer_type=answer_type,
            doc_refs=doc_refs,
            retrieved=retrieved,
        )
        if anchor_must_include:
            must_include_chunk_ids.extend(anchor_must_include)
            must_include_chunk_ids = self._dedupe_chunk_ids(must_include_chunk_ids)

        collector.set_retrieved_ids([chunk.chunk_id for chunk in retrieved])
        collector.set_models(embed=self._settings.embedding.model)
        logger.info(
            "Retrieved %d chunks",
            len(retrieved),
            extra={
                "request_id": state.get("request_id"),
                "question_id": state.get("question_id"),
                "doc_refs": state.get("doc_refs"),
            },
        )
        payload: dict[str, object] = {"retrieved": retrieved}
        if must_include_chunk_ids:
            payload["must_include_chunk_ids"] = must_include_chunk_ids
        return payload

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

    def _select_anchor_must_include_chunk_ids(
        self,
        *,
        query: str,
        answer_type: str,
        doc_refs: Sequence[str],
        retrieved: Sequence[RetrievedChunk],
    ) -> list[str]:
        if not retrieved:
            return []

        q_lower = re.sub(r"\s+", " ", query).strip().lower()
        needs_page_two = "page 2" in q_lower or "second page" in q_lower
        needs_explicit_title_or_caption = any(
            term in q_lower
            for term in ("title page", "cover page", "first page", "header", "caption")
        )
        needs_case_compare_anchor = (
            _is_common_judge_compare_query(query)
            or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
            or _is_case_monetary_claim_compare_query(query, answer_type=answer_type)
            or _is_case_party_overlap_compare_query(query, answer_type=answer_type)
        )
        needs_outcome_anchor = (
            answer_type == "free_text" and _is_case_outcome_query(query)
        ) or any(term in q_lower for term in ("it is hereby ordered", "order", "costs"))
        if not needs_page_two and not needs_explicit_title_or_caption and not needs_case_compare_anchor and not needs_outcome_anchor:
            return []

        multi_doc = (
            len([ref for ref in doc_refs if str(ref).strip()]) >= 2
            or needs_case_compare_anchor
        )
        max_docs = 3 if multi_doc else 1

        best_by_doc: dict[str, tuple[int, float, str]] = {}
        best_global: tuple[int, float, str] | None = None

        for chunk in retrieved[: min(len(retrieved), 24)]:
            score = self._anchor_candidate_score(
                chunk=chunk,
                needs_page_two=needs_page_two,
                needs_title_or_caption=needs_explicit_title_or_caption or needs_case_compare_anchor,
                needs_outcome_anchor=needs_outcome_anchor,
            )
            if score <= 0:
                continue
            retrieval_score = float(getattr(chunk, "score", 0.0) or 0.0)
            doc_id = str(getattr(chunk, "doc_id", "") or "").strip() or chunk.chunk_id
            current = best_by_doc.get(doc_id)
            candidate = (score, retrieval_score, chunk.chunk_id)
            if current is None or candidate > current:
                best_by_doc[doc_id] = candidate
            if best_global is None or candidate > best_global:
                best_global = candidate

        if multi_doc:
            ordered = sorted(best_by_doc.values(), reverse=True)
            return [chunk_id for _score, _retrieval_score, chunk_id in ordered[:max_docs]]
        if best_global is not None:
            return [best_global[2]]
        return []

    @classmethod
    def _anchor_candidate_score(
        cls,
        *,
        chunk: RetrievedChunk,
        needs_page_two: bool,
        needs_title_or_caption: bool,
        needs_outcome_anchor: bool,
    ) -> int:
        page_type = str(getattr(chunk, "page_type", "") or "").strip().lower()
        page_num = getattr(chunk, "page_number", None)
        if page_num is None:
            page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))

        score = 0
        if needs_page_two and page_num == 2:
            score += 600
            if page_type == "page2_anchor":
                score += 180

        if needs_title_or_caption and page_num == 1:
            score += 140
            if page_type == "title_anchor":
                score += 220
            if page_type == "caption_anchor":
                score += 260
            if bool(getattr(chunk, "has_caption_terms", False)):
                score += 80

        if needs_outcome_anchor:
            if bool(getattr(chunk, "has_order_terms", False)):
                score += 240
            if page_type == "heading_window":
                score += 180

        if score <= 0:
            return 0
        if page_num in {1, 2}:
            score += 40
        return score

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
    def _case_monetary_claim_seed_chunk_score(cls, *, chunk: RetrievedChunk | RankedChunk) -> int:
        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        if not text:
            return 0

        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        page_type = str(getattr(chunk, "page_type", "") or "").strip().casefold()
        has_money = bool(_MONEY_VALUE_RE.search(text)) or any(
            marker in text for marker in ("financial limit", "claim form", "claim amount", "amount of aed", "aed ")
        )
        if not has_money:
            return 0

        score = 180
        if page_num > 1:
            score += 120
        elif page_num == 1:
            score -= 40

        if page_type == "heading_window":
            score += 60
        if page_type in {"title_anchor", "caption_anchor"}:
            score -= 120

        if any(
            marker in text
            for marker in (
                "claim amount",
                "claim form",
                "financial limit",
                "claimant filed a claim",
                "claimant seeks payment",
                "seeking payment",
                "judgment sum",
                "amount of aed",
            )
        ):
            score += 220
        if "claim" in text:
            score += 60
        if "penalty" in text and "claim" not in text:
            score -= 160
        if "costs" in text and "claim" not in text:
            score -= 80
        if "order" in text and "claim" not in text:
            score -= 40
        return max(score, 0)

    def _select_case_monetary_claim_seed_chunk_id(self, chunks: Sequence[RetrievedChunk]) -> str | None:
        best_chunk_id = ""
        best_key: tuple[int, int, float] | None = None
        for chunk in chunks:
            score = self._case_monetary_claim_seed_chunk_score(chunk=chunk)
            if score <= 0:
                continue
            page_num = self._page_num(str(getattr(chunk, "section_path", "") or ""))
            candidate = (score, -max(page_num, 0), float(chunk.score))
            if best_key is None or candidate > best_key:
                best_key = candidate
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

    async def _rerank(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        answer_type_raw = str(state.get("answer_type") or "free_text").strip().lower()
        is_strict = answer_type_raw in {"boolean", "number", "date", "name", "names"}
        is_boolean = answer_type_raw == "boolean"
        doc_ref_count = len([ref for ref in state.get("doc_refs", []) if str(ref).strip()])

        # Reranker gating: strict types need fewer chunks — reduce latency
        if is_boolean:
            top_n = int(getattr(self._settings.pipeline, "boolean_context_top_n", 2))
        elif is_strict:
            top_n = int(getattr(self._settings.pipeline, "strict_types_context_top_n", 3))
        else:
            top_n = self._settings.reranker.top_n
        query_text = str(state.get("query") or "")
        if not is_strict and _is_broad_enumeration_query(query_text):
            target_top_n = 12 if _is_recall_sensitive_broad_enumeration_query(query_text) else 8
            top_n = max(int(top_n), target_top_n)
        elif not is_strict and _is_enumeration_query(query_text):
            top_n = max(int(top_n), 8)
        # Multi-ref questions need broader context to preserve grounding against multiple identifiers.
        if doc_ref_count >= 2:
            if is_boolean:
                top_n = max(int(top_n), int(getattr(self._settings.pipeline, "boolean_multi_ref_top_n", 3)))
            else:
                top_n = max(int(top_n), min(int(self._settings.reranker.top_n), doc_ref_count * 2))
        retrieved_all = list(state.get("retrieved", []))
        retrieved = retrieved_all
        if is_strict:
            if is_boolean:
                strict_cap = int(getattr(self._settings.pipeline, "boolean_rerank_candidates_cap", 12))
                if "same year" in query_text.casefold():
                    strict_cap = max(strict_cap, 24)
            else:
                strict_cap = int(getattr(self._settings.pipeline, "rerank_max_candidates_strict_types", 20))
            if len(retrieved) > strict_cap:
                retrieved = retrieved[:strict_cap]

        rerank_enabled_for_strict = bool(getattr(self._settings.pipeline, "rerank_enabled_strict_types", True))
        rerank_enabled_for_boolean = bool(getattr(self._settings.pipeline, "rerank_enabled_boolean", False))
        rerank_enabled = rerank_enabled_for_boolean if is_boolean else (rerank_enabled_for_strict if is_strict else True)
        if doc_ref_count == 1 and bool(getattr(self._settings.pipeline, "rerank_skip_on_single_doc_ref", True)):
            q_lower = str(state.get("query") or "").lower()
            amount_question = answer_type_raw == "number" and any(
                key in q_lower for key in ("claim value", "claim amount", "fine", "amount")
            )
            # Skip rerank only when the candidate set comes from a single document. Some doc refs (e.g., case IDs)
            # can map to multiple PDFs/points; reranking is needed to pick the correct page/chunk (e.g., claim value).
            if not amount_question:
                doc_ids = {chunk.doc_id for chunk in retrieved if getattr(chunk, "doc_id", "").strip()}
                if len(doc_ids) <= 1:
                    rerank_enabled = False
        if not rerank_enabled:
            reranked = self._raw_ranked(retrieved, top_n=top_n)
        else:
            prefer_fast = bool(getattr(self._settings.pipeline, "use_fast_reranker_for_simple", False)) and (
                state.get("complexity", QueryComplexity.SIMPLE) == QueryComplexity.SIMPLE
            )
            with collector.timed("rerank"):
                reranked = await self._reranker.rerank(
                    state["query"],
                    retrieved,
                    top_n=top_n,
                    prefer_fast=prefer_fast,
                )

        must_include = [str(v).strip() for v in state.get("must_include_chunk_ids", []) if str(v).strip()]
        if must_include:
            reranked = self._ensure_must_include_context(
                reranked=reranked,
                retrieved=retrieved,
                must_include_chunk_ids=must_include,
                top_n=top_n,
            )
        if not is_strict:
            reranked = self._ensure_doc_family_page_localizer_context(
                query=str(state.get("query") or ""),
                answer_type=answer_type_raw,
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved_all,
                top_n=top_n,
            )
        if (
            not is_strict
            and _is_broad_enumeration_query(str(state.get("query") or ""))
            and _is_multi_criteria_enumeration_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_page_one_context(
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and _is_registrar_enumeration_query(str(state.get("query") or "")):
            reranked = self._ensure_self_registrar_context(
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if (
            not is_strict
            and _is_named_multi_title_lookup_query(str(state.get("query") or ""))
            and not _is_named_commencement_query(str(state.get("query") or ""))
            and not _is_common_elements_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_named_multi_title_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and _is_named_amendment_query(str(state.get("query") or "")):
            reranked = self._ensure_named_amendment_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if (
            not is_strict
            and "administ" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
            and not _is_broad_enumeration_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_named_administration_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if (
            not is_strict
            and "penalt" in re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
            and not _is_broad_enumeration_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_named_penalty_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and doc_ref_count >= 2 and _is_named_commencement_query(str(state.get("query") or "")):
            reranked = self._ensure_named_commencement_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        if not is_strict and _is_common_elements_query(str(state.get("query") or "")):
            reranked = self._ensure_common_elements_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=top_n,
            )
        normalized_query = re.sub(r"\s+", " ", str(state.get("query") or "")).strip().lower()
        if is_boolean and "same year" in normalized_query:
            refs_for_year_compare = self._support_question_refs(str(state.get("query") or ""))
            if len(refs_for_year_compare) >= 2:
                reranked = self._ensure_boolean_year_compare_context(
                    query=str(state.get("query") or ""),
                    reranked=reranked,
                    retrieved=retrieved,
                    top_n=max(int(top_n), min(4, len(refs_for_year_compare) * 2)),
                )
                reranked = self._ensure_page_one_context(
                    reranked=reranked,
                    retrieved=retrieved,
                    top_n=max(int(top_n), min(4, len(refs_for_year_compare))),
                )
        if is_boolean and _is_common_judge_compare_query(normalized_query):
            refs_for_judge_compare = self._support_question_refs(str(state.get("query") or ""))
            if len(refs_for_judge_compare) >= 2:
                reranked = self._ensure_boolean_judge_compare_context(
                    query=str(state.get("query") or ""),
                    reranked=reranked,
                    retrieved=retrieved_all,
                    top_n=max(int(top_n), min(4, len(refs_for_judge_compare) * 2)),
                )
        if is_boolean and "administ" in normalized_query:
            refs_for_admin_compare = self._support_question_refs(str(state.get("query") or ""))
            if len(refs_for_admin_compare) >= 2:
                reranked = self._ensure_boolean_admin_compare_context(
                    query=str(state.get("query") or ""),
                    reranked=reranked,
                    retrieved=retrieved_all,
                    top_n=max(int(top_n), min(4, len(refs_for_admin_compare) * 2)),
                )
        if is_strict and self._is_notice_focus_query(str(state.get("query") or "")):
            reranked = self._ensure_notice_doc_context(
                query=str(state.get("query") or ""),
                reranked=reranked,
                retrieved=retrieved,
                top_n=max(int(top_n), 2),
            )
        if str(state.get("answer_type") or "").strip().lower() == "free_text" and _is_account_effective_dates_query(
            str(state.get("query") or "")
        ):
            reranked = self._ensure_account_effective_dates_context(
                query=str(state.get("query") or ""),
                doc_refs=[str(ref).strip() for ref in state.get("doc_refs", []) if str(ref).strip()],
                reranked=reranked,
                retrieved=retrieved,
                top_n=max(int(top_n), 4),
            )
        if (
            str(state.get("answer_type") or "").strip().lower() == "free_text"
            and _is_case_outcome_query(str(state.get("query") or ""))
        ):
            reranked = self._ensure_page_one_context(
                reranked=reranked,
                retrieved=retrieved,
                top_n=max(int(top_n), 3),
            )

        collector.set_context_ids([chunk.chunk_id for chunk in reranked])
        rerank_model = "raw_retrieval_fallback" if not rerank_enabled else self._settings.reranker.primary_model
        if rerank_enabled:
            get_last_model = getattr(self._reranker, "get_last_used_model", None)
            if callable(get_last_model):
                model_obj = get_last_model()
                if isinstance(model_obj, str) and model_obj.strip():
                    rerank_model = model_obj
        collector.set_models(rerank=rerank_model)
        max_score = reranked[0].rerank_score if reranked else 0.0
        logger.info(
            "Reranked %d chunks (max score %.3f)",
            len(reranked),
            max_score,
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {
            "reranked": reranked,
            "context_chunks": reranked,
            "max_rerank_score": max_score,
        }

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
                    page_number=raw.page_number,
                    page_type=raw.page_type,
                    heading_text=raw.heading_text,
                    doc_refs=list(raw.doc_refs),
                    law_no=raw.law_no,
                    law_year=raw.law_year,
                    article_refs=list(raw.article_refs),
                    has_caption_terms=raw.has_caption_terms,
                    has_order_terms=raw.has_order_terms,
                    doc_summary=raw.doc_summary,
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
                        page_number=page_one.page_number,
                        page_type=page_one.page_type,
                        heading_text=page_one.heading_text,
                        doc_refs=list(page_one.doc_refs),
                        law_no=page_one.law_no,
                        law_year=page_one.law_year,
                        article_refs=list(page_one.article_refs),
                        has_caption_terms=page_one.has_caption_terms,
                        has_order_terms=page_one.has_order_terms,
                        doc_summary=page_one.doc_summary,
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
            page_number=chunk.page_number,
            page_type=chunk.page_type,
            heading_text=chunk.heading_text,
            doc_refs=list(chunk.doc_refs),
            law_no=chunk.law_no,
            law_year=chunk.law_year,
            article_refs=list(chunk.article_refs),
            has_caption_terms=chunk.has_caption_terms,
            has_order_terms=chunk.has_order_terms,
            doc_summary=chunk.doc_summary,
        )

    @classmethod
    def _should_apply_doc_family_page_localizer(
        cls,
        *,
        query: str,
        answer_type: str,
        doc_refs: Sequence[str],
    ) -> bool:
        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if not refs:
            return False
        q_lower = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        explicit_page_anchor = "page 2" in q_lower or "second page" in q_lower or any(
            term in q_lower for term in ("title page", "cover page", "first page", "header", "caption")
        )
        return (
            explicit_page_anchor
            or _is_common_judge_compare_query(query)
            or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
            or _is_case_monetary_claim_compare_query(query, answer_type=answer_type)
            or _is_case_party_overlap_compare_query(query, answer_type=answer_type)
            or _is_case_party_role_name_query(query, answer_type=answer_type)
            or _is_case_outcome_query(query)
        )

    @classmethod
    def _case_doc_family_page_localizer_score(
        cls,
        *,
        query: str,
        answer_type: str,
        ref: str,
        chunk: RetrievedChunk | RankedChunk,
    ) -> int:
        identity_score = cls._case_ref_identity_score(ref=ref, chunk=chunk)
        if identity_score <= 0:
            return 0

        q_lower = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        score = identity_score
        score += cls._anchor_candidate_score(
            chunk=cast("RetrievedChunk", chunk),
            needs_page_two="page 2" in q_lower or "second page" in q_lower,
            needs_title_or_caption=any(
                term in q_lower for term in ("title page", "cover page", "first page", "header", "caption")
            ),
            needs_outcome_anchor=_is_case_outcome_query(query),
        )
        if _is_case_party_overlap_compare_query(query, answer_type=answer_type) or _is_case_party_role_name_query(
            query,
            answer_type=answer_type,
        ):
            score += cls._case_party_anchor_chunk_score(
                query=query,
                chunk=chunk,
                ref=ref,
            )
        if _is_common_judge_compare_query(query):
            score += cls._case_judge_seed_chunk_score(chunk=chunk)
        if _is_case_issue_date_name_compare_query(query, answer_type=answer_type):
            score += cls._case_issue_date_seed_chunk_score(chunk=chunk)
        if _is_case_monetary_claim_compare_query(query, answer_type=answer_type):
            score += cls._case_monetary_claim_seed_chunk_score(chunk=chunk)
        if _is_case_outcome_query(query):
            score += cls._case_outcome_seed_chunk_score(chunk=chunk)
        return score

    @classmethod
    def _case_doc_family_identity_key(
        cls,
        *,
        ref: str,
        doc_chunks: Sequence[RetrievedChunk],
    ) -> tuple[int, int, float, int] | None:
        best: tuple[int, int, float, int] | None = None
        for raw in doc_chunks:
            identity_score = cls._case_ref_identity_score(ref=ref, chunk=raw)
            if identity_score <= 0:
                continue
            family_adjustment = cls._ref_doc_family_consistency_adjustment(ref=ref, chunk=raw)
            title_match = cls._named_commencement_title_match_score(ref, raw)
            page_num = cls._page_num(str(getattr(raw, "section_path", "") or ""))
            candidate = (
                identity_score + family_adjustment,
                title_match,
                float(raw.score),
                -max(page_num, 0),
            )
            if best is None or candidate > best:
                best = candidate
        return best

    @classmethod
    def _case_doc_family_page_selection_key(
        cls,
        *,
        query: str,
        answer_type: str,
        ref: str,
        chunk: RetrievedChunk,
    ) -> tuple[int, int, float] | None:
        score = cls._case_doc_family_page_localizer_score(
            query=query,
            answer_type=answer_type,
            ref=ref,
            chunk=chunk,
        )
        if score <= 0:
            return None
        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        return (score, -max(page_num, 0), float(chunk.score))

    @classmethod
    def _ensure_doc_family_page_localizer_context(
        cls,
        *,
        query: str,
        answer_type: str,
        doc_refs: Sequence[str],
        reranked: list[RankedChunk],
        retrieved: list[RetrievedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0 or not reranked or not retrieved:
            return reranked[: max(0, int(top_n))]
        if not cls._should_apply_doc_family_page_localizer(
            query=query,
            answer_type=answer_type,
            doc_refs=doc_refs,
        ):
            return reranked[: max(0, int(top_n))]

        refs = [ref for ref in doc_refs if str(ref).strip()]
        if not refs:
            refs = cls._paired_support_question_refs(query)
        if not refs:
            refs = cls._support_question_refs(query)
        if not refs:
            return reranked[: max(0, int(top_n))]

        reranked_by_id = {chunk.chunk_id: chunk for chunk in reranked}
        retrieved_by_doc: dict[str, list[RetrievedChunk]] = {}
        for raw in retrieved:
            doc_key = str(raw.doc_id or "").strip() or raw.chunk_id
            retrieved_by_doc.setdefault(doc_key, []).append(raw)

        selected: list[RankedChunk] = []
        seen_chunk_ids: set[str] = set()
        matched_doc_ids: set[str] = set()

        for ref in refs[:4]:
            best_raw: RetrievedChunk | None = None
            best_key: tuple[int, int, int, float, int, float] | None = None
            best_doc_key = ""
            for doc_key, doc_chunks in retrieved_by_doc.items():
                if doc_key in matched_doc_ids:
                    continue
                family_identity_key = cls._case_doc_family_identity_key(
                    ref=ref,
                    doc_chunks=doc_chunks,
                )
                if family_identity_key is None:
                    continue
                best_doc_raw: RetrievedChunk | None = None
                best_doc_page_key: tuple[int, int, float] | None = None
                for raw in doc_chunks:
                    page_key = cls._case_doc_family_page_selection_key(
                        query=query,
                        answer_type=answer_type,
                        ref=ref,
                        chunk=raw,
                    )
                    if page_key is None:
                        continue
                    if best_doc_page_key is None or page_key > best_doc_page_key:
                        best_doc_page_key = page_key
                        best_doc_raw = raw
                if best_doc_raw is None or best_doc_page_key is None:
                    continue
                candidate = (
                    family_identity_key[0],
                    family_identity_key[1],
                    best_doc_page_key[0],
                    family_identity_key[2],
                    best_doc_page_key[1],
                    best_doc_page_key[2],
                )
                if best_key is None or candidate > best_key:
                    best_key = candidate
                    best_raw = best_doc_raw
                    best_doc_key = doc_key
            if best_raw is None:
                continue
            if best_doc_key:
                matched_doc_ids.add(best_doc_key)
            if best_raw.chunk_id in seen_chunk_ids:
                continue
            selected.append(reranked_by_id.get(best_raw.chunk_id) or cls._raw_to_ranked(best_raw))
            seen_chunk_ids.add(best_raw.chunk_id)
            if len(selected) >= top_n:
                return selected[:top_n]

        for chunk in reranked:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            doc_key = str(getattr(chunk, "doc_id", "") or "").strip()
            if not doc_key or doc_key in matched_doc_ids:
                continue
            selected.append(chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            matched_doc_ids.add(doc_key)
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

    async def _detect_conflicts(self, state: RAGState) -> dict[str, object]:
        if not bool(getattr(self._settings.pipeline, "enable_conflict_detection", False)):
            return {}
        if state.get("complexity", QueryComplexity.SIMPLE) != QueryComplexity.COMPLEX:
            return {}

        max_chunks = int(getattr(self._settings.pipeline, "conflict_max_chunks", 8))
        report = self._conflict_detector.detect(state.get("context_chunks", [])[:max_chunks])
        prompt_context = report.to_prompt_context()
        if not prompt_context:
            return {}
        logger.info(
            "Conflict detector found %d potential conflicts",
            len(report.conflicts),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {"conflict_prompt_context": prompt_context}

    async def _confidence_check(self, state: RAGState) -> dict[str, object]:
        del state
        return {}

    def _route_after_confidence(self, state: RAGState) -> str:
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        if answer_type in {"boolean", "number", "date", "name", "names"}:
            return "generate"
        threshold = float(self._settings.pipeline.confidence_threshold)
        max_score = float(state.get("max_rerank_score", 0.0))
        already_retried = bool(state.get("retried", False))
        if max_score < threshold and not already_retried:
            logger.info(
                "Low confidence %.3f < %.3f; retrying retrieval",
                max_score,
                threshold,
                extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
            )
            return "retry_retrieve"
        return "generate"

    async def _retry_retrieve(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        collector.set_retried(True)

        expanded_query = self._expand_retry_query(state)
        with collector.timed("qdrant"):
            retrieved = await self._retriever.retrieve_with_retry(
                state["query"],
                expanded_query=expanded_query,
                doc_refs=state.get("doc_refs"),
            )
        collector.set_retrieved_ids([chunk.chunk_id for chunk in retrieved])
        reranked_state = await self._rerank({**state, "retrieved": retrieved, "collector": collector})
        reranked_obj = reranked_state.get("reranked", [])
        reranked = cast("list[RankedChunk]", reranked_obj if isinstance(reranked_obj, list) else [])
        max_score_raw = reranked_state.get("max_rerank_score", 0.0)
        max_score = float(max_score_raw) if isinstance(max_score_raw, (int, float, str)) else 0.0

        logger.info(
            "Retry retrieval produced %d chunks; reranked to %d",
            len(retrieved),
            len(reranked),
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )

        result_state: dict[str, object] = {
            "retrieved": retrieved,
            **reranked_state,
            "retried": True,
        }

        # If confidence is still low after retry, upgrade the model if configured
        threshold = float(getattr(self._settings.pipeline, "confidence_threshold", 0.5))
        llm_settings = getattr(self._settings, "llm", None)
        upgrade_model = getattr(llm_settings, "upgrade_model", "")
        if max_score < threshold and upgrade_model:
            collector.set_model_upgraded(True)
            result_state["model"] = upgrade_model
            result_state["max_tokens"] = int(getattr(llm_settings, "upgrade_max_tokens", 1800))
            logger.info(
                "Selective upgrade triggered on low confidence (%f < %f)",
                max_score,
                threshold,
                extra={"request_id": state.get("request_id")}
            )

        return result_state

    async def _generate(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        writer = self._get_stream_writer_or_noop()
        collector.set_models(llm=state["model"])

        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        strict_types = {"boolean", "number", "date", "name", "names"}
        strict_non_stream_types = {"boolean", "number", "date", "name", "names"}
        prompt_hint = str(state.get("conflict_prompt_context") or "").strip()
        strict_repair_enabled = bool(getattr(self._settings.pipeline, "strict_repair_enabled", True))

        # Pre-generation anti-hallucination guardrails for free_text.
        if answer_type == "free_text":
            context_for_guard = state.get("context_chunks", [])
            entity_scope_hint = self._build_entity_scope(context_for_guard)
            if entity_scope_hint:
                prompt_hint = f"{prompt_hint}\n\n{entity_scope_hint}".strip() if prompt_hint else entity_scope_hint
            query_text = str(state["query"] or "")
            query_lower = query_text.lower()
            common_elements_query = _is_common_elements_query(query_text)
            if common_elements_query:
                common_elements_hint = (
                    "For common-elements questions, answer concisely without IRAC or Issue/Rule/Application/"
                    "Conclusion headings. Every claimed common element must be supported by at least one citation "
                    "from every referenced document in the question; each numbered common element must cite that "
                    "support in the same item; if you cannot cite every referenced document for an element, omit "
                    "that element. Output ONLY a numbered list of common elements; do not add an explanatory "
                    "preamble, postscript, or cross-document caveat outside the numbered list. Each numbered "
                    "item must end with one parenthetical citation that includes at least one chunk ID from "
                    "every referenced document, for example (cite: id_a, id_b, id_c). Keep the list compact "
                    "and merge closely related interpretative rules into one item when they belong to the same "
                    "clause family. A structural "
                    "overlap counts as a valid common element "
                    "when each referenced document explicitly states the same structure, such as that Schedule 1 "
                    "contains interpretative provisions or a list of defined terms. List a more specific sub-item "
                    "only if that same sub-item is explicitly shown in every referenced document. If one "
                    "referenced document only shows the Schedule 1 structure but does not quote the substantive "
                    "interpretative rules, do not infer those sub-items as common from the other documents. "
                    "Do not end with "
                    "\"There is no information on this question.\" if you have already identified one or more "
                    "supported common elements. If no explicit common element remains after this check, say "
                    "exactly: \"There is no information on this question.\""
                )
                prompt_hint = f"{prompt_hint}\n\n{common_elements_hint}".strip() if prompt_hint else common_elements_hint
                named_docs = [ref for ref in (state.get("doc_refs") or []) if str(ref).strip()]
                if len(named_docs) >= 2:
                    docs_hint = (
                        "The referenced documents for this common-elements question are: "
                        f"{'; '.join(named_docs)}. If a claimed element is not explicitly supported in each of "
                        "those referenced documents, omit it."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{docs_hint}".strip() if prompt_hint else docs_hint
            if _is_broad_enumeration_query(query_text):
                source_block_count = len([chunk for chunk in context_for_guard if getattr(chunk, "chunk_id", "").strip()])
                enumeration_hint = (
                    "For this broad enumeration question, answer as a compact numbered list. Each item should "
                    "start with the exact matching law or document title supported by the sources. Use the "
                    "minimum number of citations needed to support each item. If separate source blocks are "
                    "needed to prove all criteria for the same law or to supply the exact citation title "
                    "requested, cite each needed block in that same item. Inspect every source block provided "
                    f"({source_block_count} total) before stopping. Output ONLY the numbered list items; do "
                    "not add caveats, comparative commentary, or summary text after the list."
                )
                prompt_hint = f"{prompt_hint}\n\n{enumeration_hint}".strip() if prompt_hint else enumeration_hint
                if (
                    query_lower.startswith("which laws")
                    and not _is_citation_title_query(query_text)
                    and not _is_registrar_enumeration_query(query_text)
                ):
                    titles_only_hint = (
                        "This question asks only which laws match. Each numbered item should mainly give the "
                        "law title itself. Do not add article-level, schedule-level, or explanatory detail "
                        "unless the question explicitly asks for that detail."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{titles_only_hint}".strip() if prompt_hint else titles_only_hint
                if _is_multi_criteria_enumeration_query(query_text):
                    multi_criteria_hint = (
                        "Review every source block before stopping. When multiple source blocks clearly refer to "
                        "the same law title, you may combine those blocks for that law only. Deduplicate by law "
                        "title, but do not stop after the first few matches if later blocks also satisfy all criteria. "
                        "List a law only if the source block or combined source blocks for that same law explicitly "
                        "support every criterion in the question; if one criterion is missing for that law, exclude it. "
                        "Each listed item's citations must collectively support every criterion for that same law."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{multi_criteria_hint}".strip() if prompt_hint else multi_criteria_hint
                if _is_multi_criteria_enumeration_query(query_text) and re.search(r"\b(19|20)\d{2}\b", query_text):
                    year_hint = (
                        "Use the exact year shown in each source block, such as in a law title like "
                        "\"DIFC Law No. 4 of 2018\" or \"Limited Partnership Law 2006\". Do not infer the "
                        "year from consolidated-version dates or amendment dates, and check every source block "
                        "before you finish the list. If the year/title evidence and the substantive matching "
                        "criterion appear in different source blocks for the same law, cite both blocks in that "
                        "same item."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{year_hint}".strip() if prompt_hint else year_hint
                if "administered by the registrar" in query_lower:
                    registrar_hint = (
                        "For this question, include a law only if the source block or combined source blocks for "
                        "that law explicitly state that the law is administered by the Registrar. A title, "
                        "enactment reference, or year alone is not enough. A definitional passage in another "
                        "document about 'legislation administered by the Registrar' or 'Prescribed Laws' does "
                        "NOT make that current document itself a match unless that same document explicitly "
                        "says this law is administered by the Registrar. Use one separate numbered item per "
                        "matching law, and do not merge two different laws into the same numbered item even if "
                        "their supporting clauses are similar."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{registrar_hint}".strip() if prompt_hint else registrar_hint
                if _is_citation_title_query(query_text):
                    citation_title_hint = (
                        "For each listed law, state the exact citation title as written in the source, usually "
                        "from a clause like \"This Law may be cited as ...\". Do not paraphrase, shorten, or "
                        "replace it with generic uppercase document headers. If the title clause and another "
                        "matching criterion appear in different source blocks for the same law, cite both blocks "
                        "in that same item."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{citation_title_hint}".strip() if prompt_hint else citation_title_hint
                question_title_refs = _extract_question_title_refs(query_text)
                if len(question_title_refs) >= 2 and any(term in query_lower for term in ("mention", "mentions", "reference", "references")):
                    named_refs_hint = (
                        "The named law references in this question are: "
                        f"{'; '.join(question_title_refs)}. List a document only if that same document explicitly "
                        "mentions every named law reference above. If a document mentions only one of them, "
                        "exclude it silently. If multiple documents in the sources satisfy all named references, "
                        "list all of them before stopping."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{named_refs_hint}".strip() if prompt_hint else named_refs_hint
                if _is_ruler_enactment_query(query_text):
                    enactment_hint = (
                        "For this question, a law matches if the Enactment Notice itself states both that the "
                        "Ruler of Dubai enacted the law and when it comes into force. A commencement rule written "
                        "as a relative period, such as 'on the 5th business day after enactment' or '90 days after "
                        "enactment', still counts as the commencement being specified in the Enactment Notice. "
                        "Do not exclude a law only because the notice gives a relative commencement period instead "
                        "of a calendar date."
                    )
                    prompt_hint = f"{prompt_hint}\n\n{enactment_hint}".strip() if prompt_hint else enactment_hint

        streamed = False
        streamed_raw = ""
        answer = ""
        extracted = False
        strict_cited_ids: list[str] = []
        context_chunks = state.get("context_chunks", [])
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        get_context_debug_stats = getattr(self._generator, "get_context_debug_stats", None)
        if callable(get_context_debug_stats):
            context_stats_obj = get_context_debug_stats(
                question=state["query"],
                chunks=context_chunks,
                complexity=state.get("complexity", QueryComplexity.SIMPLE),
                answer_type=answer_type,
            )
            if isinstance(context_stats_obj, tuple):
                context_stats_items = cast("tuple[object, ...]", context_stats_obj)
                if len(context_stats_items) != 2:
                    context_stats_items = ()
            else:
                context_stats_items = ()
            if len(context_stats_items) == 2:
                chunk_count_obj, budget_obj = context_stats_items
                if isinstance(chunk_count_obj, int) and isinstance(budget_obj, int):
                    collector.set_context_stats(chunk_count=chunk_count_obj, budget_tokens=budget_obj)
        if answer_type == "free_text" and not context_chunks:
            answer = self._insufficient_sources_answer(())
            collector.set_generation_mode("single_shot")
            collector.set_models(llm="insufficient-sources")
            logger.info(
                "free_text_no_context_fallback",
                extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
            )
        if (
            answer_type == "free_text"
            and bool(getattr(self._settings.pipeline, "premise_guard_enabled", True))
        ):
            terms_obj = getattr(self._settings.pipeline, "premise_guard_terms", [])
            disallowed_terms: list[str] = []
            if isinstance(terms_obj, list):
                terms = cast("list[object]", terms_obj)
                disallowed_terms = [text for term in terms if (text := str(term).strip())]
            guard = check_query_premise(
                query=state["query"],
                context_chunks=context_chunks,
                disallowed_terms=disallowed_terms,
            )
            if guard.triggered:
                answer = self._insufficient_sources_answer(tuple(context_chunk_ids[:1]))
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="premise-guard")
                logger.warning(
                    "premise_guard_triggered",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "term": guard.term,
                        "answer_type": answer_type,
                    },
                )
        if answer_type in strict_types and bool(getattr(self._settings.pipeline, "strict_types_extraction_enabled", True)):
            strict_result = self._strict_answerer.answer(
                answer_type=answer_type,
                query=state["query"],
                context_chunks=context_chunks,
                max_chunks=int(getattr(self._settings.pipeline, "strict_types_extraction_max_chunks", 4)),
            )
            if strict_result is not None and strict_result.confident:
                answer = strict_result.answer.strip()
                extracted = True
                strict_cited_ids = list(strict_result.cited_chunk_ids)
                collector.set_generation_mode("single_shot")
                collector.set_models(llm="strict-extractor")

        if answer_type in strict_non_stream_types:
            if not answer:
                collector.set_generation_mode("single_shot")
                # Strict-types path fallback: non-stream LLM answer.
                with collector.timed("llm"):
                    generated_text, _citations = await self._generator.generate(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=int(state["max_tokens"]),
                        collector=collector,
                        complexity=state.get("complexity", QueryComplexity.SIMPLE),
                        answer_type=answer_type,
                        prompt_hint=prompt_hint,
                    )
                answer = generated_text.strip()
        else:
            if not answer:
                build_structured_answer = getattr(self._generator, "build_structured_free_text_answer", None)
                if callable(build_structured_answer):
                    built_obj = build_structured_answer(
                        question=state["query"],
                        chunks=context_chunks,
                        doc_refs=state.get("doc_refs"),
                    )
                    if isinstance(built_obj, str) and built_obj.strip():
                        answer = built_obj.strip()
                        collector.set_generation_mode("single_shot")
                        collector.set_models(llm="structured-extractor")
                        collector.mark_first_token()
                        writer({"type": "token", "text": answer})
            if not answer:
                collector.set_generation_mode("stream")
                answer_parts: list[str] = []
                first_token = True
                # Adaptive max_tokens: boost for multi-entity queries to prevent truncation.
                effective_max_tokens = int(state["max_tokens"])
                doc_ref_count = len([r for r in (state.get("doc_refs") or []) if str(r).strip()])
                if doc_ref_count >= 2:
                    effective_max_tokens = min(int(effective_max_tokens * 1.5), 1800)
                with collector.timed("llm"):
                    async for token in self._generator.generate_stream(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=effective_max_tokens,
                        collector=collector,
                        complexity=state.get("complexity", QueryComplexity.SIMPLE),
                        answer_type=answer_type,
                        prompt_hint=prompt_hint,
                    ):
                        if first_token:
                            collector.mark_first_token()
                        if first_token:
                            first_token = False
                        writer({"type": "token", "text": token})
                        answer_parts.append(token)
                streamed = True
                answer = "".join(answer_parts).strip()
                if answer:
                    streamed_raw = answer
                    cleanup_obj = getattr(self._generator, "cleanup_truncated_answer", None)
                    if callable(cleanup_obj):
                        cleaned_obj = cleanup_obj(answer)
                        if isinstance(cleaned_obj, str):
                            malformed_tail_detected = cleaned_obj.strip() != answer.strip()
                            if malformed_tail_detected:
                                collector.set_llm_diagnostics(malformed_tail_detected=True)
                        if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                            answer = cleaned_obj.strip()
                    strip_neg = getattr(self._generator, "strip_negative_subclaims", None)
                    should_strip_neg = answer_type == "free_text" and (
                        _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
                    )
                    if callable(strip_neg) and should_strip_neg:
                        stripped_obj = strip_neg(answer)
                        if isinstance(stripped_obj, str) and stripped_obj.strip():
                            answer = stripped_obj.strip()
                    if streamed_raw and not answer.strip():
                        answer = streamed_raw
                else:
                    # Rare provider anomaly: stream completes without tokens. Fall back once to non-stream generation.
                    collector.set_generation_mode("single_shot")
                    with collector.timed("llm"):
                        generated_text, _citations = await self._generator.generate(
                            state["query"],
                            context_chunks,
                            model=state["model"],
                            max_tokens=effective_max_tokens,
                            collector=collector,
                            complexity=state.get("complexity", QueryComplexity.SIMPLE),
                            answer_type=answer_type,
                            prompt_hint=prompt_hint,
                        )
                    answer = generated_text.strip()

        if answer_type in strict_types:
            if not answer:
                answer = self._strict_type_fallback(answer_type, tuple(context_chunk_ids[:1]))

            # Coerce strict formats (parse-safe; citations handled via telemetry "used pages").
            cited_ids_raw = strict_cited_ids or list(context_chunk_ids)
            coerced, extracted_ok = self._coerce_strict_type_format(answer, answer_type, cited_ids_raw)
            answer = coerced.strip()

            # Rare second-pass "repair" if the first LLM output was not parseable.
            if (
                not extracted_ok
                and strict_repair_enabled
                and not extracted
                and not self._is_unanswerable_strict_answer(answer)
                and answer_type in {"boolean", "number", "date", "name", "names"}
            ):
                repair_hint = _STRICT_REPAIR_HINT_TEMPLATE.format(answer_type=answer_type)
                with collector.timed("llm"):
                    repaired_text, _ = await self._generator.generate(
                        state["query"],
                        context_chunks,
                        model=state["model"],
                        max_tokens=min(int(state["max_tokens"]), 64),
                        collector=collector,
                        complexity=QueryComplexity.SIMPLE,
                        answer_type=answer_type,
                        prompt_hint=repair_hint,
                    )
                repaired, extracted_ok_2 = self._coerce_strict_type_format(repaired_text, answer_type, cited_ids_raw)
                if extracted_ok_2:
                    answer = repaired.strip()
                    logger.info(
                        "strict_repair_succeeded",
                        extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
                    )
                else:
                    logger.warning(
                        "strict_repair_failed",
                        extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
                    )

            if not answer:
                answer = self._strict_type_fallback(answer_type, tuple(context_chunk_ids[:1]))

            if self._is_unanswerable_strict_answer(answer):
                used_ids = []
                cited_ids = []
                collector.set_retrieved_ids([])
                collector.set_context_ids([])
            else:
                cited_ids = (
                    list(strict_cited_ids)
                    if (extracted and strict_cited_ids)
                    else self._localize_strict_support_chunk_ids(
                        answer_type=answer_type,
                        answer=answer,
                        query=state["query"],
                        context_chunks=context_chunks,
                    )
                )
                used_ids = self._expand_page_spanning_support_chunk_ids(
                    chunk_ids=cited_ids,
                    context_chunks=context_chunks,
                )
                if not cited_ids:
                    logger.warning(
                        "strict_support_localization_failed",
                        extra={
                            "request_id": state.get("request_id"),
                            "question_id": state.get("question_id"),
                            "answer_type": answer_type,
                        },
                    )
            shaped_used_ids, support_shape_flags = self._apply_support_shape_policy(
                answer_type=answer_type,
                answer=answer,
                query=state["query"],
                context_chunks=context_chunks,
                cited_ids=cited_ids,
                support_ids=[],
            )
            if support_shape_flags:
                logger.warning(
                    "support_shape_flags_detected",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "answer_type": answer_type,
                        "flags": support_shape_flags,
                    },
                )
            collector.set_cited_ids(cited_ids)
            collector.set_used_ids(shaped_used_ids if shaped_used_ids else used_ids)
            citations: list[Citation] = []
            cited_ids = list(cited_ids)
        else:
            # Sanitize citations: remove any chunk IDs not present in context
            sanitize_citations = getattr(self._generator, "sanitize_citations", None)
            if callable(sanitize_citations):
                sanitized_obj = sanitize_citations(answer, context_chunk_ids)
                if isinstance(sanitized_obj, str):
                    answer = sanitized_obj
            cleanup_list_preamble = getattr(self._generator, "cleanup_list_answer_preamble", None)
            if callable(cleanup_list_preamble) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_preamble(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_list_items = getattr(self._generator, "cleanup_numbered_list_items", None)
            if callable(cleanup_list_items) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_items(
                    answer,
                    question=state["query"],
                    common_elements=_is_common_elements_query(state["query"]),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_titles_only = getattr(self._generator, "cleanup_broad_enumeration_titles_only", None)
            query_title_refs = _extract_question_title_refs(state["query"])
            named_ref_query = len(query_title_refs) >= 2 and any(
                term in str(state["query"]).lower() for term in ("mention", "mentions", "reference", "references")
            )
            if (
                callable(cleanup_titles_only)
                and _is_broad_enumeration_query(state["query"])
                and not _is_registrar_enumeration_query(state["query"])
                and not named_ref_query
            ):
                cleaned_obj = cleanup_titles_only(answer, question=state["query"], chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_interpretative_items = getattr(self._generator, "cleanup_interpretative_provisions_enumeration_items", None)
            if (
                callable(cleanup_interpretative_items)
                and _is_broad_enumeration_query(state["query"])
                and "interpretative provisions" in str(state["query"]).lower()
            ):
                cleaned_obj = cleanup_interpretative_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_ref_items = getattr(self._generator, "cleanup_named_ref_enumeration_items", None)
            if callable(cleanup_named_ref_items) and named_ref_query and _is_broad_enumeration_query(state["query"]):
                cleaned_obj = cleanup_named_ref_items(answer, question=state["query"], chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_ruler_enactment_items = getattr(self._generator, "cleanup_ruler_enactment_enumeration_items", None)
            if callable(cleanup_ruler_enactment_items) and _is_ruler_enactment_query(state["query"]):
                cleaned_obj = cleanup_ruler_enactment_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_registrar_items = getattr(self._generator, "cleanup_registrar_enumeration_items", None)
            if callable(cleanup_registrar_items) and (
                _is_registrar_enumeration_query(state["query"])
                or (_is_citation_title_query(state["query"]) and _is_enumeration_query(state["query"]))
            ):
                cleaned_obj = cleanup_registrar_items(answer, chunks=context_chunks)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_list_postamble = getattr(self._generator, "cleanup_list_answer_postamble", None)
            if callable(cleanup_list_postamble) and (
                _is_broad_enumeration_query(state["query"]) or _is_common_elements_query(state["query"])
            ):
                cleaned_obj = cleanup_list_postamble(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            strip_neg = getattr(self._generator, "strip_negative_subclaims", None)
            if callable(strip_neg) and _is_common_elements_query(state["query"]):
                stripped_obj = strip_neg(answer)
                if isinstance(stripped_obj, str) and stripped_obj.strip():
                    answer = stripped_obj.strip()
            cleanup_common_elements_answer = getattr(self._generator, "cleanup_common_elements_canonical_answer", None)
            if callable(cleanup_common_elements_answer) and _is_common_elements_query(state["query"]):
                cleaned_obj = cleanup_common_elements_answer(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_commencement = getattr(self._generator, "cleanup_named_commencement_answer", None)
            if callable(cleanup_named_commencement):
                cleaned_obj = cleanup_named_commencement(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_administration = getattr(self._generator, "cleanup_named_administration_answer", None)
            if callable(cleanup_named_administration):
                cleaned_obj = cleanup_named_administration(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_penalty = getattr(self._generator, "cleanup_named_penalty_answer", None)
            if callable(cleanup_named_penalty):
                cleaned_obj = cleanup_named_penalty(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_multi_title_lookup = getattr(self._generator, "cleanup_named_multi_title_lookup_answer", None)
            if callable(cleanup_named_multi_title_lookup):
                cleaned_obj = cleanup_named_multi_title_lookup(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_named_amendment = getattr(self._generator, "cleanup_named_amendment_answer", None)
            if callable(cleanup_named_amendment):
                cleaned_obj = cleanup_named_amendment(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_account_effective_dates = getattr(self._generator, "cleanup_account_effective_dates_answer", None)
            if callable(cleanup_account_effective_dates):
                cleaned_obj = cleanup_account_effective_dates(
                    answer,
                    question=state["query"],
                    chunks=context_chunks,
                    doc_refs=state.get("doc_refs"),
                )
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    answer = cleaned_obj.strip()
            cleanup_final_answer = getattr(self._generator, "cleanup_final_answer", None)
            if callable(cleanup_final_answer):
                cleaned_obj = cleanup_final_answer(answer)
                if isinstance(cleaned_obj, str) and cleaned_obj.strip():
                    if cleaned_obj.strip() != answer.strip():
                        collector.set_llm_diagnostics(malformed_tail_detected=True)
                    answer = cleaned_obj.strip()

            citations = self._generator.extract_citations(answer, context_chunks)
            cited_ids = self._generator.extract_cited_chunk_ids(answer)
            is_unanswerable_free_text = self._is_unanswerable_free_text_answer(answer)
            support_ids: list[str] = []
            prefer_citation_trace = _is_named_multi_title_lookup_query(state["query"]) or _is_named_amendment_query(
                state["query"]
            )
            if not cited_ids and not is_unanswerable_free_text:
                support_ids = self._localize_free_text_support_chunk_ids(
                    answer=answer,
                    query=state["query"],
                    context_chunks=context_chunks,
                )
                cited_ids = list(support_ids)
                if support_ids and not citations:
                    citations = self._citations_from_chunk_ids(
                        chunk_ids=support_ids,
                        context_chunks=context_chunks,
                    )
                elif not support_ids:
                    logger.warning(
                        "free_text_support_localization_failed",
                        extra={
                            "request_id": state.get("request_id"),
                            "question_id": state.get("question_id"),
                            "answer_type": answer_type,
                        },
                    )
            elif not is_unanswerable_free_text and not prefer_citation_trace:
                support_ids = self._localize_free_text_support_chunk_ids(
                    answer=answer,
                    query=state["query"],
                    context_chunks=context_chunks,
                )
            if is_unanswerable_free_text:
                citations = []
                cited_ids = []
                support_ids = []
                collector.set_retrieved_ids([])
                collector.set_context_ids([])
            elif support_ids:
                support_ids = self._suppress_named_administration_family_orphan_support_ids(
                    query=state["query"],
                    cited_ids=cited_ids,
                    support_ids=support_ids,
                    context_chunks=context_chunks,
                )
            collector.set_cited_ids(cited_ids)
            shaped_used_ids, support_shape_flags = self._apply_support_shape_policy(
                answer_type=answer_type,
                answer=answer,
                query=state["query"],
                context_chunks=context_chunks,
                cited_ids=cited_ids,
                support_ids=support_ids,
            )
            if support_shape_flags:
                logger.warning(
                    "support_shape_flags_detected",
                    extra={
                        "request_id": state.get("request_id"),
                        "question_id": state.get("question_id"),
                        "answer_type": answer_type,
                        "flags": support_shape_flags,
                    },
                )
            collector.set_used_ids(shaped_used_ids)
            if answer_type == "free_text" and streamed and answer.strip():
                writer({"type": "answer_final", "text": answer})

        logger.info(
            "Generated answer %d chars with %d citations (strict_extracted=%s)",
            len(answer),
            len(citations),
            extracted,
            extra={"request_id": state.get("request_id"), "question_id": state.get("question_id")},
        )
        return {
            "answer": answer,
            "citations": citations,
            "cited_chunk_ids": cited_ids,
            "streamed": streamed,
        }

    async def _verify(self, state: RAGState) -> dict[str, object]:
        verifier_settings = getattr(self._settings, "verifier", None)
        if verifier_settings is not None and not bool(getattr(verifier_settings, "enabled", True)):
            return {}
        if self._verifier is None:
            return {}

        answer = str(state.get("answer") or "").strip()
        cited_ids = list(state.get("cited_chunk_ids", []))

        # Skip LLM verifier entirely for strict types — deterministic coerce already ran in _generate
        answer_type = str(state.get("answer_type") or "free_text").strip().lower()
        if answer_type in {"boolean", "number", "date", "name", "names"}:
            return {}
        should_verify = self._verifier.should_verify(answer, cited_ids, force=False)
        if not should_verify:
            return {}

        collector = state["collector"]
        context_chunks = state.get("context_chunks", [])
        with collector.timed("verify"):
            verification = await self._verifier.verify(
                state["query"],
                answer,
                context_chunks,
            )

        # For free_text, verifier is audit-only (post-hoc revision cannot affect already-streamed output).
        if answer_type == "free_text":
            logger.info(
                "Verifier audit grounded=%s unsupported_claims=%d",
                verification.is_grounded,
                len(verification.unsupported_claims),
            )
            return {}

        next_answer = answer
        if not verification.is_grounded:
            if verification.revised_answer:
                next_answer = verification.revised_answer.strip()
                logger.info(
                    "Verifier revised answer grounded=%s unsupported_claims=%d",
                    verification.is_grounded,
                    len(verification.unsupported_claims),
                )
            else:
                # Fail-safe: keep strict-type output format deterministic even when sources are insufficient.
                next_answer = self._strict_type_fallback(answer_type, cited_ids)

        effective_cited_ids = self._generator.extract_cited_chunk_ids(next_answer) or cited_ids
        next_answer, _ = self._coerce_strict_type_format(next_answer, answer_type, effective_cited_ids)
        next_answer = next_answer.strip()

        # Sanitize citations after coercion
        context_chunk_ids = [c.chunk_id for c in context_chunks]
        sanitize_citations = getattr(self._generator, "sanitize_citations", None)
        if callable(sanitize_citations):
            sanitized_obj = sanitize_citations(next_answer, context_chunk_ids)
            if isinstance(sanitized_obj, str):
                next_answer = sanitized_obj

        if next_answer == answer:
            return {}

        citations = self._generator.extract_citations(next_answer, context_chunks)
        next_cited_ids = self._generator.extract_cited_chunk_ids(next_answer)
        if self._is_unanswerable_strict_answer(next_answer):
            citations = []
            next_cited_ids = []
            collector.set_retrieved_ids([])
            collector.set_context_ids([])
        collector.set_cited_ids(next_cited_ids)
        return {
            "answer": next_answer,
            "citations": citations,
            "cited_chunk_ids": next_cited_ids,
        }

    async def _emit(self, state: RAGState) -> dict[str, object]:
        """Emit answer tokens for strict-types flows that generated non-streaming output."""
        if bool(state.get("streamed", False)):
            return {}

        answer = str(state.get("answer") or "")
        if not answer.strip():
            return {}

        collector = state["collector"]
        writer = self._get_stream_writer_or_noop()
        collector.mark_first_token()
        writer({"type": "token", "text": answer})
        writer({"type": "answer_final", "text": answer})
        return {"streamed": True}

    async def _finalize(self, state: RAGState) -> dict[str, object]:
        collector = state["collector"]
        telemetry = collector.finalize()
        writer = self._get_stream_writer_or_noop()
        writer({"type": "telemetry", "payload": telemetry.model_dump()})
        return {"telemetry": telemetry}

    @staticmethod
    def _get_stream_writer_or_noop() -> Callable[[dict[str, object]], None]:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return lambda _: None
        return cast("Callable[[dict[str, object]], None]", writer)

    def _expand_retry_query(self, state: RAGState) -> str:
        base_query = state["query"]
        anchors: list[str] = []
        for chunk in state.get("reranked", []):
            if chunk.section_path:
                anchors.append(chunk.section_path)
        if not anchors:
            return base_query
        unique_anchors = list(dict.fromkeys(anchors))
        max_anchors = max(0, int(self._settings.pipeline.retry_query_max_anchors))
        return f"{base_query} {' '.join(unique_anchors[:max_anchors])}".strip()

    @staticmethod
    def _raw_ranked(chunks: list[RetrievedChunk], *, top_n: int) -> list[RankedChunk]:
        if not chunks:
            return []
        sorted_chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
        return [
            RankedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                section_path=chunk.section_path,
                text=chunk.text,
                retrieval_score=chunk.score,
                rerank_score=chunk.score,
                page_number=chunk.page_number,
                page_type=chunk.page_type,
                heading_text=chunk.heading_text,
                doc_refs=list(chunk.doc_refs),
                law_no=chunk.law_no,
                law_year=chunk.law_year,
                article_refs=list(chunk.article_refs),
                has_caption_terms=chunk.has_caption_terms,
                has_order_terms=chunk.has_order_terms,
                doc_summary=chunk.doc_summary,
            )
            for chunk in sorted_chunks[: max(0, int(top_n))]
        ]

    @staticmethod
    def _citation_suffix(cited_ids: list[str] | tuple[str, ...], *, enabled: bool) -> str:
        if not enabled:
            return ""
        ids = [chunk_id.strip() for chunk_id in cited_ids if str(chunk_id).strip()]
        if not ids:
            return ""
        keep = ids[:3]
        return f" (cite: {', '.join(keep)})"

    def _strict_type_citation_suffix(self, cited_ids: list[str] | tuple[str, ...]) -> str:
        return self._citation_suffix(
            cited_ids,
            enabled=bool(getattr(self._settings.pipeline, "strict_types_append_citations", False)),
        )

    @staticmethod
    def _is_unanswerable_strict_answer(answer: str) -> bool:
        normalized = (answer or "").strip().lower()
        return normalized in {"null", "none", ""}

    @staticmethod
    def _is_unanswerable_free_text_answer(answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", (answer or "").strip().lower())
        return normalized.startswith("there is no information on this question") or "insufficient sources retrieved" in normalized

    def _strict_type_fallback(self, answer_type: str, cited_ids: list[str] | tuple[str, ...]) -> str:
        kind = answer_type.strip().lower()
        if kind in {"boolean", "number", "date", "name", "names"}:
            return _UNANSWERABLE_STRICT
        return self._insufficient_sources_answer(cited_ids)

    def _insufficient_sources_answer(self, cited_ids: list[str] | tuple[str, ...]) -> str:
        _ = cited_ids
        return _UNANSWERABLE_FREE_TEXT

    def _coerce_strict_type_format(
        self,
        answer: str,
        answer_type: str,
        cited_ids: list[str] | tuple[str, ...],
    ) -> tuple[str, bool]:
        kind = answer_type.strip().lower()
        text = answer.strip()
        if not text:
            return (self._strict_type_fallback(kind, cited_ids), False)
        normalized = text.lower()
        if (
            "insufficient sources" in normalized
            or "there is no information on this question" in normalized
            or normalized.strip() in {"null", "none"}
        ):
            return (self._strict_type_fallback(kind, cited_ids), False)

        stripped_text = _CITE_RE.sub("", text).strip()
        stripped_text = re.sub(r"\s+", " ", stripped_text).strip()
        suffix = self._strict_type_citation_suffix(cited_ids)

        if kind == "boolean":
            lowered = stripped_text.lower().lstrip()
            if lowered.startswith("yes"):
                return (f"Yes{suffix}".strip(), True)
            if lowered.startswith("no"):
                return (f"No{suffix}".strip(), True)
            if "yes" in lowered and "no" not in lowered:
                return (f"Yes{suffix}".strip(), True)
            if "no" in lowered and "yes" not in lowered:
                return (f"No{suffix}".strip(), True)
            return (self._strict_type_fallback(kind, cited_ids), False)

        if kind == "number":
            for match in _NUMBER_RE.finditer(stripped_text):
                start, end = match.span()
                before = stripped_text[max(0, start - 24) : start]
                after = stripped_text[end : min(len(stripped_text), end + 10)]
                if after.lstrip().startswith("/") and re.match(r"\s*/\s*\d{2,4}", after):
                    continue
                if re.search(r"(?:CA|CFI|ARB|SCT|TCD|ENF|DEC)\s*$", before, re.IGNORECASE):
                    continue
                return (f"{match.group(0)}{suffix}".strip(), True)
            return (self._strict_type_fallback(kind, cited_ids), False)

        if kind == "date":
            match = _ISO_DATE_RE.search(stripped_text) or _SLASH_DATE_RE.search(stripped_text) or _TEXTUAL_DATE_RE.search(stripped_text)
            if match is None:
                return (self._strict_type_fallback(kind, cited_ids), False)
            return (f"{match.group(0)}{suffix}".strip(), True)

        if kind == "name":
            # If the model included a DIFC case ID, prefer returning just that normalized ID.
            case_match = _DIFC_CASE_ID_RE.search(stripped_text)
            if case_match is not None:
                prefix = case_match.group(1).upper()
                num = int(case_match.group(2))
                year = case_match.group(3)
                return (f"{prefix} {num:03d}/{year}{suffix}".strip(), True)

            # Prefer full DIFC law titles that include the law number, e.g. "Strata Title Law, DIFC Law No. 5 of 2007".
            law_title_match = re.search(
                r"([A-Z][^\n]{0,180}?\b(?:DIFC\s+)?Law\s+No\.?\s*\d+\s+of\s+\d{4})",
                stripped_text,
            )
            if law_title_match is not None and law_title_match.group(1).strip():
                candidate = re.sub(r"\s+", " ", law_title_match.group(1).strip())
                candidate = re.sub(r"\bNo\.\s*", "No ", candidate)
                candidate = candidate.rstrip(" .;")
                return (f"{candidate}{suffix}".strip(), True)

            stripped = stripped_text
            for pattern in (
                r"(?:is|called|known as|referred to as|named)\s+[\"']?([A-Z][^\"'!?\n]{1,80})[\"']?",
                r"term\s+[\"']([^\"']+)[\"']",
            ):
                m = re.search(pattern, stripped, re.IGNORECASE)
                if m and m.group(1).strip():
                    stripped = m.group(1).strip()
                    break
            # Tighten "name" outputs aggressively: evaluators expect a short entity/title, not a clause.
            stripped = re.sub(r"[.!?]", "", stripped).strip()
            lowered = stripped.lower()
            for marker in (
                " subject to ",
                " provided that ",
                " pursuant to ",
                " in accordance with ",
                " as per ",
                " as provided ",
                " under ",
            ):
                idx = lowered.find(marker)
                if idx != -1:
                    stripped = stripped[:idx].strip()
                    break
            # Prefer the first phrase if the model returned a longer explanatory fragment.
            for sep in (" — ", " - ", ";", ":", ","):
                if sep in stripped:
                    stripped = stripped.split(sep, 1)[0].strip()
            words = stripped.split()
            if len(words) > 12:
                stripped = " ".join(words[:12]).strip()
            if not stripped:
                return (self._strict_type_fallback(kind, cited_ids), False)
            return (f"{stripped}{suffix}".strip(), True)

        if kind == "names":
            stripped = re.sub(
                r"^(?:the\s+)?(?:names?|parties|individuals?)\s+(?:are|is|include[s]?)\s*:?\s*",
                "",
                stripped_text,
                flags=re.IGNORECASE,
            ).strip().rstrip(".")
            stripped = _CASE_REF_PREFIX_RE.sub("", stripped).strip()
            if not stripped:
                return (self._strict_type_fallback(kind, cited_ids), False)
            return (f"{stripped}{suffix}".strip(), True)

        return (stripped_text, True)

    @staticmethod
    def _normalize_support_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip())

    @classmethod
    def _support_terms(cls, text: str) -> set[str]:
        return {
            token.lower()
            for token in _SUPPORT_TOKEN_RE.findall(text or "")
            if len(token) > 2 and token.lower() not in _SUPPORT_STOPWORDS
        }

    @classmethod
    def _support_question_refs(cls, query: str) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for ref in _extract_question_title_refs(query):
            normalized = cls._normalize_support_text(ref)
            if normalized and normalized.casefold() not in seen:
                seen.add(normalized.casefold())
                refs.append(normalized)
        for match in _LAW_NO_REF_RE.finditer(query or ""):
            ref = f"Law No. {int(match.group(1))} of {match.group(2)}"
            normalized = cls._normalize_support_text(ref)
            if normalized and normalized.casefold() not in seen:
                seen.add(normalized.casefold())
                refs.append(normalized)
        return refs

    @classmethod
    def _paired_support_question_refs(cls, query: str) -> list[str]:
        title_refs = [
            cls._normalize_support_text(ref)
            for ref in _extract_question_title_refs(query)
            if cls._normalize_support_text(ref)
        ]
        law_refs = [
            cls._normalize_support_text(f"Law No. {int(match.group(1))} of {match.group(2)}")
            for match in _LAW_NO_REF_RE.finditer(query or "")
        ]
        if len(title_refs) < 2 or len(title_refs) != len(law_refs):
            return cls._support_question_refs(query)

        paired_refs: list[str] = []
        seen: set[str] = set()
        for title_ref, law_ref in zip(title_refs, law_refs, strict=False):
            law_suffix = law_ref[4:] if law_ref.startswith("Law ") else law_ref
            combined = cls._normalize_support_text(f"{title_ref} {law_suffix}")
            key = combined.casefold()
            if not combined or key in seen:
                continue
            seen.add(key)
            paired_refs.append(combined)

        return paired_refs or cls._support_question_refs(query)

    @classmethod
    def _combined_named_refs(cls, *, query: str, doc_refs: Sequence[str]) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for ref in [*doc_refs, *cls._support_question_refs(query)]:
            normalized = cls._normalize_support_text(str(ref))
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            refs.append(normalized)
        return refs

    @staticmethod
    def _ordinal_suffix(day: int) -> str:
        if 10 <= day % 100 <= 20:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    @classmethod
    def _date_fragment_variants(cls, fragment: str) -> set[str]:
        normalized = cls._normalize_support_text(fragment).casefold().replace(",", "")
        if not normalized:
            return set()

        year = month = day = 0
        iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", normalized)
        if iso_match is not None:
            year = int(iso_match.group(1))
            month = int(iso_match.group(2))
            day = int(iso_match.group(3))
        else:
            slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", normalized)
            if slash_match is not None:
                first = int(slash_match.group(1))
                second = int(slash_match.group(2))
                year = int(slash_match.group(3))
                day, month = (first, second) if first > 12 else (second, first)
            else:
                textual_match = re.search(
                    r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+day\s+of)?\s+([a-z]+)\s+(\d{4})\b",
                    normalized,
                )
                if textual_match is None:
                    return set()
                month = _MONTH_NAME_TO_NUMBER.get(textual_match.group(2), 0)
                if month <= 0:
                    return set()
                day = int(textual_match.group(1))
                year = int(textual_match.group(3))

        if not (1 <= month <= 12 and 1 <= day <= 31 and year > 0):
            return set()

        month_name = _MONTH_NUMBER_TO_NAME[month]
        ordinal = cls._ordinal_suffix(day)
        variants = {
            f"{year:04d}-{month:02d}-{day:02d}",
            f"{day}/{month}/{year}",
            f"{day:02d}/{month:02d}/{year}",
            f"{day} {month_name} {year}",
            f"{day}{ordinal} {month_name} {year}",
            f"{day} day of {month_name} {year}",
            f"{day}{ordinal} day of {month_name} {year}",
        }
        return {variant.casefold() for variant in variants if variant}

    @classmethod
    def _matched_doc_chunks_for_ref(
        cls,
        *,
        ref: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        best_anchor_doc_id = ""
        best_anchor_score = 0
        for raw in retrieved:
            score = cls._named_commencement_title_match_score(ref, raw)
            if score > best_anchor_score:
                best_anchor_doc_id = raw.doc_id
                best_anchor_score = score
        if not best_anchor_doc_id or best_anchor_score <= 0:
            return []
        return [raw for raw in retrieved if raw.doc_id == best_anchor_doc_id]

    @classmethod
    def _ref_has_criterion_support(
        cls,
        *,
        query: str,
        ref: str,
        ref_chunks: Sequence[RetrievedChunk],
    ) -> bool:
        if not ref_chunks:
            return False
        query_lower = re.sub(r"\s+", " ", (query or "").strip()).lower()
        if _is_common_elements_query(query):
            interpretation_sections = _is_interpretation_sections_common_elements_query(query)
            return any(
                cls._common_elements_evidence_score(str(getattr(chunk, "text", "") or ""), interpretation_sections=interpretation_sections) > 0
                for chunk in ref_chunks
            )
        if "penalt" in query_lower:
            return any(
                cls._named_penalty_clause_score(
                    query=query,
                    ref=ref,
                    text=str(getattr(chunk, "text", "") or ""),
                )
                > 0
                for chunk in ref_chunks
            )
        if "administ" in query_lower:
            return any(
                cls._named_administration_clause_score(
                    ref=ref,
                    text=str(getattr(chunk, "text", "") or ""),
                )
                > 0
                for chunk in ref_chunks
            )
        return any(
            cls._named_multi_title_clause_score(query=query, text=str(getattr(chunk, "text", "") or "")) > 0
            for chunk in ref_chunks
        )

    @classmethod
    def _missing_named_ref_targets(
        cls,
        *,
        query: str,
        doc_refs: Sequence[str],
        retrieved: Sequence[RetrievedChunk],
    ) -> list[str]:
        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if len(refs) < 2:
            return []
        missing: list[str] = []
        for ref in refs:
            ref_chunks = cls._matched_doc_chunks_for_ref(ref=ref, retrieved=retrieved)
            if not cls._ref_has_criterion_support(query=query, ref=ref, ref_chunks=ref_chunks):
                missing.append(ref)
        return missing

    @classmethod
    def _targeted_named_ref_query(
        cls,
        *,
        query: str,
        ref: str,
        refs: Sequence[str],
    ) -> str:
        query_lower = re.sub(r"\s+", " ", (query or "").strip()).lower()
        base_query = query or ""
        for other_ref in refs:
            other_clean = str(other_ref).strip()
            if not other_clean or other_clean.casefold() == ref.casefold():
                continue
            base_query = re.sub(re.escape(other_clean), " ", base_query, flags=re.IGNORECASE)
        base_query = re.sub(r"\s+", " ", base_query).strip()

        if _is_common_elements_query(query):
            if _is_interpretation_sections_common_elements_query(query):
                return (
                    f"{ref} schedule 1 interpretation rules of interpretation "
                    "a statutory provision includes a reference reference to a person includes"
                )
            return f"{ref} schedule 1 interpretative provisions defined terms"
        if _is_account_effective_dates_query(query):
            return (
                f"{ref} pre-existing accounts new accounts effective date enactment notice "
                "hereby enact enacted on date specified in the enactment notice"
            )
        if _is_restriction_effectiveness_query(query):
            return (
                f"{ref} article 23 restriction on transfer security actual knowledge "
                "ineffective against any person other than a person who had actual knowledge "
                "uncertificated security registered owner notified of the restriction"
            )
        if "same year" in query_lower and "enact" in query_lower:
            return f"{ref} title law no year enacted enactment"
        if _is_named_commencement_query(query):
            return f"{ref} commencement effective date enactment notice come into force"
        if "penalt" in query_lower:
            return f"{ref} penalty offences illegal penalties appendix penalty for offences"
        if "administ" in query_lower:
            if "registrar" in query_lower:
                return (
                    f"{ref} may be cited as administration administered by the registrar "
                    "this law is administered by this law shall be administered by "
                    "shall administer this law administration of this law"
                )
            return (
                f"{ref} may be cited as administration administered by "
                "this law is administered by this law shall be administered by "
                "shall administer this law administration of this law"
            )
        if _is_citation_title_query(query):
            return f'{ref} citation title may be cited as "'
        if "updated" in query_lower:
            return f"{ref} updated amended effective from"
        return f"{ref} {base_query}".strip() if base_query else ref

    @classmethod
    def _should_apply_doc_shortlist_gating(
        cls,
        *,
        query: str,
        answer_type: str,
        doc_refs: Sequence[str],
    ) -> bool:
        q = re.sub(r"\s+", " ", (query or "").strip()).lower()
        if not q or _is_broad_enumeration_query(query):
            return False
        refs = [ref for ref in doc_refs if str(ref).strip()] or cls._support_question_refs(query)
        if not refs:
            return False
        if answer_type in {"boolean", "number", "date", "name", "names"} and refs:
            return any(
                term in q
                for term in (
                    "title",
                    "full title",
                    "law number",
                    "updated",
                    "citation title",
                    "commencement",
                    "effective date",
                    "enact",
                    "administ",
                )
            )
        return any(
            term in q
            for term in (
                "title of",
                "titles of",
                "last updated",
                "citation title",
                "citation titles",
                "commencement",
                "effective date",
                "enact",
                "administ",
                "amend",
            )
        )

    @staticmethod
    def _page_num(section_path: str | None) -> int:
        match = re.search(r"page:(\d+)", section_path or "", flags=re.IGNORECASE)
        if match is None:
            return 10_000
        try:
            return int(match.group(1))
        except ValueError:
            return 10_000

    @staticmethod
    def _page_text_looks_like_continuation_tail(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if len(normalized) < 48:
            return False
        if normalized.endswith(("...", "…")):
            return True
        last = normalized[-1]
        if last in {",", ";", ":", "-"}:
            return True
        if last in {".", "!", "?", '"', "'", "]", "}"}:
            return False
        return last.isalnum() or last == ")"

    @staticmethod
    def _page_text_looks_like_continuation_head(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if len(normalized) < 24:
            return False
        if normalized[:1].islower():
            return True
        lowered = normalized.casefold()
        return lowered.startswith(
            (
                "and ",
                "or ",
                "but ",
                "if ",
                "unless ",
                "provided ",
                "where ",
                "which ",
                "that ",
                "including ",
                "in addition ",
                "continued ",
                "continuation ",
            )
        )

    @staticmethod
    def _page_text_looks_like_new_section(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if not normalized:
            return False
        prefix = normalized[:96]
        if re.match(r"^(?:article|section|schedule|part|chapter)\b", prefix, re.IGNORECASE):
            return True
        return bool(re.match(r"^[A-Z0-9\s'\"()/-]{10,}$", prefix) and len(prefix.split()) <= 12)

    @classmethod
    def _expand_page_spanning_support_chunk_ids(
        cls,
        *,
        chunk_ids: Sequence[str],
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        ordered_ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()))
        if not ordered_ids or not context_chunks:
            return ordered_ids

        context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
        chunks_by_doc_page: dict[str, dict[int, list[RankedChunk]]] = {}
        for chunk in context_chunks:
            doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
            if not doc_id:
                continue
            page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
            if page_num == 10_000:
                continue
            chunks_by_doc_page.setdefault(doc_id, {}).setdefault(page_num, []).append(chunk)

        expanded: list[str] = []
        seen: set[str] = set()

        def _append(chunk_id: str) -> None:
            normalized = str(chunk_id).strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            expanded.append(normalized)

        for chunk_id in ordered_ids:
            _append(chunk_id)
            chunk = context_by_id.get(chunk_id)
            if chunk is None:
                continue

            doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
            page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
            if not doc_id or page_num == 10_000:
                continue

            current_text = str(getattr(chunk, "text", "") or "")
            doc_pages = chunks_by_doc_page.get(doc_id, {})
            previous_page_chunks = doc_pages.get(page_num - 1, [])
            next_page_chunks = doc_pages.get(page_num + 1, [])

            if previous_page_chunks:
                previous_chunk = previous_page_chunks[0]
                previous_text = str(getattr(previous_chunk, "text", "") or "")
                if (
                    cls._page_text_looks_like_continuation_tail(previous_text)
                    or (
                        cls._page_text_looks_like_continuation_head(current_text)
                        and not cls._page_text_looks_like_new_section(current_text)
                    )
                ):
                    _append(previous_chunk.chunk_id)

            if next_page_chunks:
                next_chunk = next_page_chunks[0]
                next_text = str(getattr(next_chunk, "text", "") or "")
                if (
                    cls._page_text_looks_like_continuation_tail(current_text)
                    or (
                        cls._page_text_looks_like_continuation_head(next_text)
                        and not cls._page_text_looks_like_new_section(next_text)
                    )
                ):
                    _append(next_chunk.chunk_id)

        return expanded

    @classmethod
    def _boolean_year_compare_chunk_score(cls, *, ref: str, chunk: RetrievedChunk | RankedChunk) -> int:
        base = cls._named_commencement_title_match_score(ref, chunk)
        if base <= 0:
            return 0

        text = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        score = base
        if page_num <= 4:
            score += 240
        elif page_num <= 8:
            score += 80
        if _LAW_NO_REF_RE.search(text):
            score += 180
        if _YEAR_RE.search(text):
            score += 60
        if "title" in text:
            score += 60
        if "enact" in text or "legislative authority" in text:
            score += 40
        return score

    @classmethod
    def _account_effective_clause_score(cls, *, text: str) -> int:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return 0

        score = 0
        if "pre-existing accounts" in normalized:
            score += 14
        if "new accounts" in normalized:
            score += 14
        if "effective date" in normalized:
            score += 12
        if "31 december" in normalized or "1 january" in normalized:
            score += 10
        return score

    @classmethod
    def _account_enactment_clause_score(cls, *, ref: str, raw: RetrievedChunk) -> int:
        normalized = re.sub(r"\s+", " ", str(getattr(raw, "text", "") or "")).strip().casefold()
        if not normalized:
            return 0
        if not any(marker in normalized for marker in ("hereby enact", "enactment notice", "enacted on")):
            return 0

        doc_title = re.sub(r"\s+", " ", str(getattr(raw, "doc_title", "") or "")).strip().casefold()
        explicit_notice_doc = "enactment notice" in doc_title or normalized.startswith("enactment notice")
        generic_notice_reference = any(
            phrase in normalized
            for phrase in (
                "date specified in the enactment notice",
                "comes into force on the date specified in the enactment notice",
            )
        )
        explicit_enactment_date = bool(re.search(r"\bhereby enact\s+on\s+(?:this\s+)?[0-9]{1,2}", normalized))

        score = cls._named_commencement_title_match_score(ref, raw)
        if score <= 0:
            ref_terms = {
                token
                for token in cls._support_terms(ref)
                if token not in _SUPPORT_STOPWORDS and len(token) > 2
            }
            overlap = len(ref_terms.intersection(cls._support_terms(normalized)))
            if overlap >= max(2, len(ref_terms) - 1):
                score += 180 + (overlap * 18)
        if "hereby enact" in normalized:
            score += 220
        if "enactment notice" in normalized:
            score += 120
        if "enacted on" in normalized:
            score += 100
        if explicit_enactment_date:
            score += 260
        if explicit_notice_doc:
            score += 320
        if generic_notice_reference and not explicit_notice_doc and not explicit_enactment_date:
            score -= 760
        if _YEAR_RE.search(normalized):
            score += 30
        if cls._page_num(str(getattr(raw, "section_path", "") or "")) == 1:
            score += 50
        return score

    @classmethod
    def _restriction_effectiveness_clause_score(
        cls,
        *,
        ref: str,
        chunk: RetrievedChunk | RankedChunk,
    ) -> int:
        normalized = re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
        if not normalized or "restriction" not in normalized or "actual knowledge" not in normalized:
            return 0

        score = cls._named_commencement_title_match_score(ref, chunk)
        if score <= 0:
            ref_terms = {
                token
                for token in cls._support_terms(ref)
                if token not in _SUPPORT_STOPWORDS and len(token) > 2
            }
            overlap = len(ref_terms.intersection(cls._support_terms(normalized)))
            if overlap >= max(2, len(ref_terms) - 1):
                score += 180 + (overlap * 18)
        if "ineffective against any person other than a person who had actual knowledge" in normalized:
            score += 460
        if "restriction on transfer" in normalized:
            score += 180
        if "actual knowledge" in normalized:
            score += 140
        if "uncertificated" in normalized:
            score += 90
        if "notified" in normalized:
            score += 70
        if "article 23" in normalized:
            score += 80
        return score

    @classmethod
    def _doc_shortlist_score(
        cls,
        *,
        query: str,
        ref: str,
        doc_chunks: Sequence[RetrievedChunk],
    ) -> int:
        if not doc_chunks:
            return 0

        normalized_ref = cls._normalize_support_text(ref).casefold()
        title_score = max(cls._named_commencement_title_match_score(ref, chunk) for chunk in doc_chunks)
        identity_blob = cls._normalize_support_text(
            " ".join(
                part
                for part in (
                    str(doc_chunks[0].doc_title or ""),
                    str(doc_chunks[0].doc_summary or ""),
                )
                if part
            )
        ).casefold()
        identity_score = 0
        if normalized_ref and normalized_ref in identity_blob:
            identity_score += 900
        law_ref_match = _LAW_NO_REF_RE.search(ref)
        if law_ref_match is not None:
            law_key = f"law no. {int(law_ref_match.group(1))} of {law_ref_match.group(2)}"
            if law_key in identity_blob:
                identity_score += 700
        ref_terms = cls._support_terms(ref)
        identity_terms = cls._support_terms(identity_blob)
        if ref_terms:
            overlap = len(ref_terms.intersection(identity_terms))
            if overlap >= min(2, len(ref_terms)):
                identity_score += overlap * 90

        query_lower = cls._normalize_support_text(query).casefold()
        surrogate_enabled = _is_named_commencement_query(query) or _is_account_effective_dates_query(query)
        enactment_surrogate = 0
        if surrogate_enabled:
            enactment_surrogate = max(
                (cls._account_enactment_clause_score(ref=ref, raw=chunk) for chunk in doc_chunks[:4]),
                default=0,
            )
        administration_surrogate = 0
        if "administ" in query_lower:
            administration_surrogate = max(
                (
                    cls._named_administration_clause_score(
                        ref=ref,
                        text=str(getattr(chunk, "text", "") or ""),
                    )
                    + (140 if cls._page_num(str(getattr(chunk, "section_path", "") or "")) <= 5 else 0)
                    + (
                        40
                        if "may be cited as"
                        in re.sub(r"\s+", " ", str(getattr(chunk, "text", "") or "")).strip().casefold()
                        else 0
                    )
                )
                for chunk in doc_chunks[:6]
            )
        if title_score <= 0:
            if enactment_surrogate <= 0 and administration_surrogate <= 0:
                return 0
            title_score = min(320, max(enactment_surrogate, administration_surrogate))
        if identity_score <= 0 and max(enactment_surrogate, administration_surrogate) > 0:
            identity_score = min(450, max(enactment_surrogate, administration_surrogate))
        if identity_score <= 0:
            return 0
        if administration_surrogate > 0:
            identity_score += min(620, administration_surrogate * 12)
        identity_score += cls._ref_doc_family_consistency_adjustment(ref=ref, chunk=doc_chunks[0])

        query_terms = cls._support_terms(query)
        best_overlap = 0
        best_retrieval_score = 0.0
        for chunk in doc_chunks[:4]:
            blob = cls._chunk_support_blob(
                RankedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    doc_type=chunk.doc_type,
                    section_path=chunk.section_path,
                    text=chunk.text,
                    retrieval_score=float(chunk.score),
                    rerank_score=float(chunk.score),
                    page_number=chunk.page_number,
                    page_type=chunk.page_type,
                    heading_text=chunk.heading_text,
                    doc_refs=list(chunk.doc_refs),
                    law_no=chunk.law_no,
                    law_year=chunk.law_year,
                    article_refs=list(chunk.article_refs),
                    has_caption_terms=chunk.has_caption_terms,
                    has_order_terms=chunk.has_order_terms,
                    doc_summary=chunk.doc_summary,
                )
            )
            overlap = len(query_terms.intersection(cls._support_terms(blob)))
            if ref_terms and ref_terms.issubset(cls._support_terms(blob)):
                overlap += 4
            best_overlap = max(best_overlap, overlap)
            best_retrieval_score = max(best_retrieval_score, float(chunk.score))

        return identity_score + title_score + (best_overlap * 10) + int(best_retrieval_score * 100)

    @classmethod
    def _apply_doc_shortlist_gating(
        cls,
        *,
        query: str,
        doc_refs: Sequence[str],
        retrieved: Sequence[RetrievedChunk],
        must_keep_chunk_ids: Sequence[str] = (),
    ) -> list[RetrievedChunk]:
        if not retrieved:
            return []

        refs = cls._combined_named_refs(query=query, doc_refs=doc_refs)
        if not refs:
            return list(retrieved)

        chunks_by_doc: dict[str, list[RetrievedChunk]] = {}
        ordered_docs: list[str] = []
        for chunk in retrieved:
            doc_id = str(chunk.doc_id or "").strip()
            if not doc_id:
                continue
            if doc_id not in chunks_by_doc:
                ordered_docs.append(doc_id)
            chunks_by_doc.setdefault(doc_id, []).append(chunk)

        selected_doc_ids: set[str] = set()
        for ref in refs[:4]:
            scored_docs: list[tuple[int, float, str]] = []
            for doc_id in ordered_docs:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                score = cls._doc_shortlist_score(query=query, ref=ref, doc_chunks=doc_chunks)
                if score <= 0:
                    continue
                best_score = max(float(chunk.score) for chunk in doc_chunks) if doc_chunks else 0.0
                scored_docs.append((score, best_score, doc_id))
            scored_docs.sort(reverse=True)
            for _score, _best_score, doc_id in scored_docs[:2]:
                selected_doc_ids.add(doc_id)

        if _is_account_effective_dates_query(query):
            best_notice_doc: tuple[int, float, str] | None = None
            primary_ref = refs[0]
            for doc_id in ordered_docs:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                surrogate = max(
                    (cls._account_enactment_clause_score(ref=primary_ref, raw=chunk) for chunk in doc_chunks[:4]),
                    default=0,
                )
                if surrogate <= 0:
                    continue
                best_score = max(float(chunk.score) for chunk in doc_chunks) if doc_chunks else 0.0
                candidate = (surrogate, best_score, doc_id)
                if best_notice_doc is None or candidate > best_notice_doc:
                    best_notice_doc = candidate
            if best_notice_doc is not None:
                selected_doc_ids.add(best_notice_doc[2])

        must_keep_ids = {chunk_id for chunk_id in must_keep_chunk_ids if str(chunk_id).strip()}
        if must_keep_ids:
            for chunk in retrieved:
                if chunk.chunk_id not in must_keep_ids:
                    continue
                doc_id = str(chunk.doc_id or "").strip()
                if doc_id:
                    selected_doc_ids.add(doc_id)

        if not selected_doc_ids:
            return list(retrieved)
        return [chunk for chunk in retrieved if str(chunk.doc_id or "").strip() in selected_doc_ids]

    @staticmethod
    def _normalize_numeric_text(text: str) -> str:
        return re.sub(r"[,\s]", "", (text or "").strip())

    @classmethod
    def _chunk_support_blob(cls, chunk: RankedChunk) -> str:
        return cls._normalize_support_text(
            " ".join(
                part
                for part in (
                    str(chunk.doc_title or ""),
                    str(chunk.doc_summary or ""),
                    str(chunk.text or ""),
                )
                if part
            )
        )

    @classmethod
    def _chunk_support_score(
        cls,
        *,
        answer_type: str,
        query: str,
        fragment: str,
        chunk: RankedChunk,
    ) -> int:
        blob = cls._chunk_support_blob(chunk)
        if not blob:
            return 0

        blob_lower = blob.casefold()
        fragment_clean = cls._normalize_support_text(_CITE_RE.sub("", fragment))
        fragment_lower = fragment_clean.casefold()
        query_lower = cls._normalize_support_text(query).casefold()
        score = 0

        if fragment_lower:
            if len(fragment_lower) >= 8 and fragment_lower in blob_lower:
                score += 80
            fragment_terms = cls._support_terms(fragment_clean)
            if fragment_terms:
                blob_terms = cls._support_terms(blob)
                score += len(fragment_terms.intersection(blob_terms)) * 8

        query_terms = cls._support_terms(query)
        if query_terms:
            score += len(query_terms.intersection(cls._support_terms(blob))) * 3

        for ref in cls._support_question_refs(query):
            normalized_ref = cls._normalize_support_text(ref).casefold()
            if normalized_ref and normalized_ref in blob_lower:
                score += 30

        kind = answer_type.strip().lower()
        if kind == "number":
            numeric_answer = cls._normalize_numeric_text(fragment_clean)
            if not numeric_answer or numeric_answer not in cls._normalize_numeric_text(blob):
                return 0
            score += 120
        elif kind == "date":
            date_variants = cls._date_fragment_variants(fragment_clean)
            if not date_variants or not any(variant in blob_lower for variant in date_variants):
                return 0
            score += 120
        elif kind in {"name", "names"}:
            if fragment_lower and fragment_lower in blob_lower:
                score += 100
            else:
                title_score = cls._named_commencement_title_match_score(fragment_clean, chunk)
                if title_score <= 0:
                    return 0
                score += max(80, min(title_score, 140))
        elif kind == "boolean":
            polarity_answer = fragment_clean.strip().lower()
            positive_hits = sum(
                marker in blob_lower
                for marker in (" may ", " can ", " shall ", " entitled ", " includes ", " must ", " effective ")
            )
            negative_hits = sum(
                marker in blob_lower
                for marker in (" not ", " no ", " may not ", " shall not ", " ineffective ", " prohibited ")
            )
            if polarity_answer.startswith("yes"):
                score += positive_hits * 4
            elif polarity_answer.startswith("no"):
                score += negative_hits * 4
            if query_lower and query_lower in blob_lower:
                score += 20

        if cls._is_notice_focus_query(query):
            explicit_notice_doc = "enactment notice" in blob_lower or "hereby enact" in blob_lower
            generic_notice_reference = "date specified in the enactment notice" in blob_lower
            if explicit_notice_doc:
                score += 140
            elif generic_notice_reference:
                score -= 120

        return score

    @classmethod
    def _best_support_chunk_id(
        cls,
        *,
        answer_type: str,
        query: str,
        fragment: str,
        context_chunks: Sequence[RankedChunk],
        allow_first_chunk_fallback: bool,
    ) -> str:
        best_chunk_id = ""
        best_score = -1
        for idx, chunk in enumerate(context_chunks):
            score = cls._chunk_support_score(
                answer_type=answer_type,
                query=query,
                fragment=fragment,
                chunk=chunk,
            )
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
            if score == best_score and best_chunk_id and idx == 0:
                best_chunk_id = chunk.chunk_id

        if best_score > 0 and best_chunk_id:
            return best_chunk_id
        if allow_first_chunk_fallback and context_chunks:
            return context_chunks[0].chunk_id
        return ""

    @staticmethod
    def _split_names(answer: str) -> list[str]:
        raw_parts = [part.strip() for part in re.split(r"[,\n;]+", answer) if part.strip()]
        parts: list[str] = []
        for part in raw_parts:
            split_once = re.split(r"\s+\band\b\s+", part, maxsplit=1, flags=re.IGNORECASE)
            if len(split_once) == 2:
                parts.extend(item.strip() for item in split_once if item.strip())
            else:
                parts.append(part)
        return parts

    @classmethod
    def _split_free_text_support_fragments(cls, answer: str) -> list[str]:
        cleaned = cls._normalize_support_text(_CITE_RE.sub("", answer))
        if not cleaned:
            return []

        numbered = [
            cls._normalize_support_text(match.group(1))
            for match in _NUMBERED_ITEM_RE.finditer(answer or "")
            if cls._normalize_support_text(match.group(1))
        ]
        if numbered:
            return numbered

        sentences = [
            cls._normalize_support_text(part)
            for part in re.split(r"(?<=[.!?])\s+", cleaned)
            if cls._normalize_support_text(part)
        ]
        if sentences:
            return sentences
        return [cleaned]

    @classmethod
    def _split_free_text_items(cls, answer: str) -> list[str]:
        numbered = [
            cls._normalize_support_text(match.group(1))
            for match in _NUMBERED_ITEM_RE.finditer(answer or "")
            if cls._normalize_support_text(match.group(1))
        ]
        if numbered:
            return numbered

        bullet_items = [
            cls._normalize_support_text(match.group(1))
            for match in _BULLET_ITEM_RE.finditer(answer or "")
            if cls._normalize_support_text(match.group(1))
        ]
        if bullet_items:
            return bullet_items

        return cls._split_free_text_support_fragments(answer)

    @classmethod
    def _free_text_item_title_slot(cls, item: str) -> str:
        title_field_match = _TITLE_FIELD_RE.search(item)
        if title_field_match is not None:
            return cls._normalize_support_text(title_field_match.group(1))

        prefix = re.split(r"\s+-\s+|:\s+", item, maxsplit=1)[0].strip(" ,.;:")
        if prefix and ("law" in prefix.casefold() or "regulation" in prefix.casefold()):
            return cls._normalize_support_text(prefix)
        return ""

    @classmethod
    def _group_context_chunks_by_doc(
        cls,
        context_chunks: Sequence[RankedChunk],
    ) -> tuple[list[str], dict[str, list[RankedChunk]]]:
        chunks_by_doc: dict[str, list[RankedChunk]] = {}
        doc_order: list[str] = []
        for chunk in context_chunks:
            doc_key = str(chunk.doc_id or chunk.chunk_id)
            if doc_key not in chunks_by_doc:
                doc_order.append(doc_key)
            chunks_by_doc.setdefault(doc_key, []).append(chunk)
        return doc_order, chunks_by_doc

    @classmethod
    def _free_text_doc_group_match_score(
        cls,
        *,
        ref: str,
        doc_chunks: Sequence[RankedChunk],
    ) -> int:
        normalized_ref = cls._normalize_support_text(ref).casefold()
        if not normalized_ref or not doc_chunks:
            return 0

        haystack = " ".join(
            part
            for part in (
                *(str(chunk.doc_title or "") for chunk in doc_chunks[:2]),
                *(str(chunk.doc_summary or "") for chunk in doc_chunks[:2]),
                *(str(chunk.text or "")[:1200] for chunk in doc_chunks[:2]),
            )
            if part
        )
        normalized_haystack = cls._normalize_support_text(haystack).casefold()
        if not normalized_haystack:
            return 0

        if normalized_ref in normalized_haystack:
            return 900 - min(normalized_haystack.find(normalized_ref), 600)

        ref_match = _LAW_NO_REF_RE.search(ref)
        if ref_match is not None:
            law_no_key = f"law no. {int(ref_match.group(1))} of {ref_match.group(2)}"
            if law_no_key in normalized_haystack:
                return 720

        ordered_ref_tokens = [
            token.casefold()
            for token in _SUPPORT_TOKEN_RE.findall(normalized_ref)
            if token.casefold() not in _SUPPORT_STOPWORDS and len(token) > 2
        ]
        if not ordered_ref_tokens:
            return 0

        haystack_tokens = {token.casefold() for token in _SUPPORT_TOKEN_RE.findall(normalized_haystack)}
        overlap = len(set(ordered_ref_tokens).intersection(haystack_tokens))
        if len(ordered_ref_tokens) >= 3 and overlap < len(set(ordered_ref_tokens)):
            ref_bigrams = [
                f"{ordered_ref_tokens[idx]} {ordered_ref_tokens[idx + 1]}"
                for idx in range(len(ordered_ref_tokens) - 1)
            ]
            bigram_overlap = sum(1 for bigram in ref_bigrams if bigram in normalized_haystack)
            if overlap >= max(1, len(set(ordered_ref_tokens)) - 1) and bigram_overlap < max(1, len(ref_bigrams) - 1):
                return 0

        if overlap == len(set(ordered_ref_tokens)):
            return 260 + overlap
        if overlap >= max(1, len(set(ordered_ref_tokens)) - 1):
            return 120 + overlap
        if overlap >= max(1, (len(set(ordered_ref_tokens)) + 1) // 2):
            return 50 + overlap
        return 0

    @classmethod
    def _free_text_item_candidate_chunks(
        cls,
        *,
        query: str,
        item: str,
        item_index: int,
        item_count: int,
        context_chunks: Sequence[RankedChunk],
    ) -> Sequence[RankedChunk]:
        if not context_chunks:
            return context_chunks

        refs: list[str] = []
        seen: set[str] = set()

        def _push(ref: str) -> None:
            normalized = cls._normalize_support_text(ref)
            if not normalized:
                return
            key = normalized.casefold()
            if key in seen:
                return
            seen.add(key)
            refs.append(normalized)

        query_refs = cls._support_question_refs(query)
        if item_count == len(query_refs) and item_index < len(query_refs):
            _push(query_refs[item_index])

        item_without_cites = cls._normalize_support_text(_CITE_RE.sub("", item))
        title_slot = cls._free_text_item_title_slot(item_without_cites)
        if title_slot:
            _push(title_slot)

        for title, year in _TITLE_REF_RE.findall(item_without_cites):
            _push(" ".join(part for part in (title.strip(), year.strip()) if part).strip(" ,.;:"))

        for match in _LAW_NO_REF_RE.finditer(item_without_cites):
            _push(f"Law No. {int(match.group(1))} of {match.group(2)}")

        if not refs:
            return context_chunks

        doc_order, chunks_by_doc = cls._group_context_chunks_by_doc(context_chunks)
        best_doc_id = ""
        best_score = 0
        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            score = max(
                (cls._free_text_doc_group_match_score(ref=ref, doc_chunks=doc_chunks) for ref in refs),
                default=0,
            )
            if score > best_score:
                best_score = score
                best_doc_id = doc_id

        if not best_doc_id or best_score <= 0:
            return context_chunks
        return chunks_by_doc.get(best_doc_id, context_chunks)

    @classmethod
    def _free_text_slot_full_context_priority(
        cls,
        *,
        query: str,
        item_slots: Sequence[str],
        primary_slot_ids: Sequence[str],
    ) -> bool:
        if _is_account_effective_dates_query(query) or _is_named_amendment_query(query):
            return True

        non_empty_slots = [slot for slot in item_slots if str(slot).strip()]
        if len(non_empty_slots) < 2:
            return False

        unique_primary_ids = {chunk_id for chunk_id in primary_slot_ids if str(chunk_id).strip()}
        return len(unique_primary_ids) < min(2, len(non_empty_slots))

    @classmethod
    def _extract_free_text_item_slots(cls, *, query: str, item: str) -> list[str]:
        normalized_item = cls._normalize_support_text(_CITE_RE.sub("", item))
        if not normalized_item:
            return []

        query_lower = cls._normalize_support_text(query).casefold()
        slots: list[str] = []

        title_slot = cls._free_text_item_title_slot(normalized_item)
        if title_slot:
            slots.append(title_slot)

        if "updated" in query_lower:
            updated_match = _LAST_UPDATED_FIELD_RE.search(normalized_item)
            if updated_match is not None:
                slots.append(cls._normalize_support_text(updated_match.group(1)))

        if "enact" in query_lower:
            enacted_match = _ENACTED_ON_FIELD_RE.search(normalized_item)
            if enacted_match is not None:
                slots.append(cls._normalize_support_text(enacted_match.group(1)))

        if any(term in query_lower for term in ("commencement", "come into force", "effective date")):
            commencement_match = _COMMENCEMENT_FIELD_RE.search(normalized_item)
            if commencement_match is not None:
                slots.append(cls._normalize_support_text(commencement_match.group(1)))

        if "administ" in query_lower and ":" in normalized_item:
            remainder = re.split(r":\s+", normalized_item, maxsplit=1)[1].strip()
            if remainder:
                slots.append(remainder)

        bullet_lines = [
            cls._normalize_support_text(match.group(1))
            for match in _BULLET_ITEM_RE.finditer(item or "")
            if cls._normalize_support_text(match.group(1))
        ]
        for bullet in bullet_lines:
            bullet_title = cls._free_text_item_title_slot(bullet)
            slots.append(bullet_title or bullet)

        for title, year in _TITLE_REF_RE.findall(normalized_item):
            ref = " ".join(part for part in (title.strip(), year.strip()) if part).strip(" ,.;:")
            normalized_ref = cls._normalize_support_text(ref)
            if normalized_ref:
                slots.append(normalized_ref)

        if not slots:
            slots.append(normalized_item)

        deduped: list[str] = []
        seen: set[str] = set()
        for slot in slots:
            normalized_slot = cls._normalize_support_text(slot)
            if not normalized_slot:
                continue
            key = normalized_slot.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized_slot)
        return deduped

    @classmethod
    def _best_title_support_chunk_id(
        cls,
        *,
        title: str,
        context_chunks: Sequence[RankedChunk],
    ) -> str:
        normalized_title = cls._normalize_support_text(title).casefold()
        if not normalized_title:
            return ""

        best_chunk_id = ""
        best_score = -1
        for idx, chunk in enumerate(context_chunks):
            doc_title = cls._normalize_support_text(str(getattr(chunk, "doc_title", "") or "")).casefold()
            doc_summary = cls._normalize_support_text(str(getattr(chunk, "doc_summary", "") or "")).casefold()
            text = cls._normalize_support_text(str(getattr(chunk, "text", "") or "")).casefold()
            score = 0
            if normalized_title and normalized_title in doc_title:
                score += 300
            if normalized_title and normalized_title in doc_summary:
                score += 120
            if normalized_title and normalized_title in text:
                score += 60
            text_raw = str(getattr(chunk, "text", "") or "")
            if "may be cited as" in text_raw.casefold() or "title:" in text_raw.casefold():
                score += 80
            if score > best_score or (score == best_score and idx == 0):
                best_score = score
                best_chunk_id = chunk.chunk_id

        if best_score <= 0:
            return ""
        return best_chunk_id

    @classmethod
    def _doc_ids_for_chunk_ids(
        cls,
        *,
        chunk_ids: Sequence[str],
        context_chunks: Sequence[RankedChunk],
    ) -> set[str]:
        context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
        doc_ids: set[str] = set()
        for raw_chunk_id in chunk_ids:
            chunk = context_by_id.get(str(raw_chunk_id).strip())
            if chunk is None:
                continue
            doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
            if doc_id:
                doc_ids.add(doc_id)
        return doc_ids

    @classmethod
    def _context_family_chunk_ids(
        cls,
        *,
        doc_ids: set[str],
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        if not doc_ids:
            return []
        ordered: list[str] = []
        seen_pages: set[tuple[str, str]] = set()
        for chunk in context_chunks:
            doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
            if doc_id not in doc_ids:
                continue
            page_key = (doc_id, str(getattr(chunk, "section_path", "") or "").strip())
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            ordered.append(chunk.chunk_id)
        return ordered

    @classmethod
    def _case_party_anchor_terms(cls, query: str) -> tuple[str, ...]:
        q = re.sub(r"\s+", " ", (query or "").strip()).lower()
        ordered_terms = (
            "claimant",
            "defendant",
            "appellant",
            "respondent",
            "applicant",
            "party",
        )
        matched = tuple(term for term in ordered_terms if term in q)
        return matched or ("party",)

    @classmethod
    def _case_party_anchor_chunk_score(
        cls,
        *,
        query: str,
        chunk: RetrievedChunk | RankedChunk,
        ref: str | None = None,
        fragment: str | None = None,
    ) -> int:
        text = cls._normalize_support_text(str(getattr(chunk, "text", "") or "")).casefold()
        doc_title = cls._normalize_support_text(str(getattr(chunk, "doc_title", "") or "")).casefold()
        if not text and not doc_title:
            return 0

        page_type = str(getattr(chunk, "page_type", "") or "").strip().casefold()
        page_num = getattr(chunk, "page_number", None)
        if page_num is None:
            page_num = cls._page_num(str(getattr(chunk, "section_path", "") or ""))
        if page_num != 1 and page_type not in {"title_anchor", "caption_anchor"}:
            return 0

        score = 0
        if ref is not None:
            ref_score = cls._named_commencement_title_match_score(ref, chunk)
            if ref_score <= 0:
                return 0
            score += ref_score

        if fragment is not None:
            fragment_score = cls._chunk_support_score(
                answer_type="name",
                query=query,
                fragment=fragment,
                chunk=cast("RankedChunk", chunk),
            )
            if fragment_score <= 0:
                return 0
            score += fragment_score

        role_terms = cls._case_party_anchor_terms(query)
        blob = f"{doc_title} {text}"
        if any(term in blob for term in role_terms):
            score += 120
        if page_num == 1:
            score += 160
        if page_type == "title_anchor":
            score += 180
        if page_type == "caption_anchor":
            score += 240
        if bool(getattr(chunk, "has_caption_terms", False)):
            score += 80
        if " v " in blob or " and " in blob:
            score += 20
        return score

    @classmethod
    def _best_case_party_name_support_chunk_id(
        cls,
        *,
        query: str,
        fragment: str,
        context_chunks: Sequence[RankedChunk],
    ) -> str:
        best_chunk_id = ""
        best_score = 0
        for chunk in context_chunks:
            score = cls._case_party_anchor_chunk_score(
                query=query,
                chunk=chunk,
                fragment=fragment,
            )
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
        return best_chunk_id if best_score > 0 else ""

    @classmethod
    def _best_case_party_compare_chunk_id(
        cls,
        *,
        query: str,
        ref: str,
        context_chunks: Sequence[RankedChunk],
        seen_doc_ids: set[str],
    ) -> tuple[str, str]:
        best_chunk_id = ""
        best_doc_id = ""
        best_score = 0
        for chunk in context_chunks:
            doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
            if doc_id in seen_doc_ids:
                continue
            score = cls._case_party_anchor_chunk_score(
                query=query,
                chunk=chunk,
                ref=ref,
            )
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id
                best_doc_id = doc_id
        if best_score <= 0:
            return ("", "")
        return (best_chunk_id, best_doc_id)

    @classmethod
    def _is_named_metadata_support_query(cls, query: str) -> bool:
        q = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        if not q or _is_broad_enumeration_query(query):
            return False
        ref_count = len(_extract_question_title_refs(query)) + len(_LAW_NO_REF_RE.findall(query or ""))
        if ref_count < 1:
            return False
        return any(
            term in q
            for term in (
                "title",
                "citation title",
                "updated",
                "consolidated version",
                "published",
                "enact",
                "effective date",
                "commencement",
                "administ",
                "made by",
                "who made",
            )
        )

    @classmethod
    def _named_metadata_requires_support_union(cls, query: str) -> bool:
        q = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        if not cls._is_named_metadata_support_query(query):
            return False

        atoms = 0
        if any(term in q for term in ("citation title", "what is the title")):
            atoms += 1
        if any(term in q for term in ("official law number", "official difc law number")):
            atoms += 1
        if any(term in q for term in ("updated", "consolidated version", "published")):
            atoms += 1
        if any(term in q for term in ("enact", "effective date", "commencement")):
            atoms += 1
        if "administ" in q:
            atoms += 1
        if "made by" in q or "who made" in q:
            atoms += 1

        if "and any regulations made under it" in q:
            return False

        multiple_named_refs = (
            " and " in q
            and (
                len(_LAW_NO_REF_RE.findall(query or "")) >= 2
                or len(_extract_question_title_refs(query)) >= 2
                or len(_DIFC_CASE_ID_RE.findall(query or "")) >= 2
            )
        )
        return atoms >= 2 or (atoms >= 1 and multiple_named_refs)

    @classmethod
    def _apply_support_shape_policy(
        cls,
        *,
        answer_type: str,
        answer: str,
        query: str,
        context_chunks: Sequence[RankedChunk],
        cited_ids: Sequence[str],
        support_ids: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        ordered_ids = list(
            dict.fromkeys(
                str(chunk_id).strip()
                for chunk_id in [*cited_ids, *support_ids]
                if str(chunk_id).strip()
            )
        )
        if not ordered_ids or not context_chunks:
            return ordered_ids, []

        kind = answer_type.strip().lower()
        q_lower = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        extras: list[str] = []
        seen_ids = set(ordered_ids)
        flags: list[str] = []

        def _push(chunk_id: str) -> None:
            normalized = str(chunk_id).strip()
            if not normalized or normalized in seen_ids:
                return
            seen_ids.add(normalized)
            extras.append(normalized)

        compare_refs = cls._paired_support_question_refs(query)
        if len(compare_refs) < 2:
            case_refs: list[str] = []
            seen_case_refs: set[str] = set()
            for prefix, number, year in _DIFC_CASE_ID_RE.findall(query or ""):
                ref = f"{prefix.upper()} {int(number):03d}/{year}"
                if ref not in seen_case_refs:
                    seen_case_refs.add(ref)
                    case_refs.append(ref)
            if len(case_refs) >= 2:
                compare_refs = case_refs
        compare_shape = len(compare_refs) >= 2 and kind in {"boolean", "name", "number", "date"} and (
            kind == "boolean"
            or _is_case_issue_date_name_compare_query(query, answer_type=answer_type)
            or _is_case_monetary_claim_compare_query(query, answer_type=answer_type)
            or "same year" in q_lower
            or "administ" in q_lower
            or "same party" in q_lower
            or "appeared in both" in q_lower
            or ("judge" in q_lower and "both" in q_lower)
        )
        compare_doc_ids: set[str] = set()
        if compare_shape:
            if kind == "boolean":
                for chunk_id in cls._localize_boolean_compare_support_chunk_ids(
                    query=query,
                    context_chunks=context_chunks,
                ):
                    _push(chunk_id)
            for ref in compare_refs[:2]:
                title_chunk_id = cls._best_title_support_chunk_id(title=ref, context_chunks=context_chunks)
                if not title_chunk_id:
                    continue
                _push(title_chunk_id)
                compare_doc_ids.update(
                    cls._doc_ids_for_chunk_ids(chunk_ids=[title_chunk_id], context_chunks=context_chunks)
                )
            for chunk_id in cls._context_family_chunk_ids(
                doc_ids=compare_doc_ids,
                context_chunks=context_chunks,
            ):
                _push(chunk_id)

        metadata_query = cls._named_metadata_requires_support_union(query)
        metadata_doc_ids: set[str] = set()
        if metadata_query:
            for ref in cls._support_question_refs(query)[:4]:
                title_chunk_id = cls._best_title_support_chunk_id(title=ref, context_chunks=context_chunks)
                if not title_chunk_id:
                    continue
                _push(title_chunk_id)
                metadata_doc_ids.update(
                    cls._doc_ids_for_chunk_ids(chunk_ids=[title_chunk_id], context_chunks=context_chunks)
                )
            for chunk_id in cls._context_family_chunk_ids(
                doc_ids=metadata_doc_ids,
                context_chunks=context_chunks,
            ):
                _push(chunk_id)

        costs_query = kind == "free_text" and _is_case_outcome_query(query) and (
            "cost" in q_lower or "final ruling" in q_lower
        )
        if costs_query:
            for fragment in ("no order as to costs", "costs", "cost"):
                cost_chunk_id = cls._best_support_chunk_id(
                    answer_type="free_text",
                    query=query,
                    fragment=fragment,
                    context_chunks=context_chunks,
                    allow_first_chunk_fallback=False,
                )
                if cost_chunk_id:
                    _push(cost_chunk_id)

        shaped_ids = cls._expand_page_spanning_support_chunk_ids(
            chunk_ids=[*ordered_ids, *extras],
            context_chunks=context_chunks,
        )

        shaped_doc_ids = cls._doc_ids_for_chunk_ids(chunk_ids=shaped_ids, context_chunks=context_chunks)
        if compare_doc_ids and len(shaped_doc_ids.intersection(compare_doc_ids)) < min(2, len(compare_doc_ids)):
            flags.append("comparison_support_missing_side")
        if metadata_doc_ids and not shaped_doc_ids.intersection(metadata_doc_ids):
            flags.append("named_metadata_title_missing")
        if costs_query:
            context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
            if not any(
                re.search(r"\bcosts?\b|\bno order as to costs\b", str(context_by_id[chunk_id].text or ""), re.IGNORECASE)
                for chunk_id in shaped_ids
                if chunk_id in context_by_id
            ):
                flags.append("outcome_costs_support_missing")

        return shaped_ids, flags

    @classmethod
    def _citations_from_chunk_ids(
        cls,
        *,
        chunk_ids: Sequence[str],
        context_chunks: Sequence[RankedChunk],
    ) -> list[Citation]:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}
        citations: list[Citation] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(str(chunk_id).strip())
            if chunk is None:
                continue
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_title=str(chunk.doc_title or ""),
                    section_path=str(chunk.section_path or "") or None,
                )
            )
        return citations

    @classmethod
    def _localize_strict_support_chunk_ids(
        cls,
        *,
        answer_type: str,
        answer: str,
        query: str,
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        if not context_chunks:
            return []

        kind = answer_type.strip().lower()
        if kind == "names":
            fragments = cls._split_names(_CITE_RE.sub("", answer))
        elif kind == "boolean":
            return cls._localize_boolean_support_chunk_ids(
                answer=answer,
                query=query,
                context_chunks=context_chunks,
            )
        else:
            fragments = [answer]

        localized: list[str] = []
        seen: set[str] = set()
        for fragment in fragments:
            chunk_id = ""
            if _is_case_party_role_name_query(query, answer_type=answer_type):
                chunk_id = cls._best_case_party_name_support_chunk_id(
                    query=query,
                    fragment=fragment,
                    context_chunks=context_chunks,
                )
            if not chunk_id:
                chunk_id = cls._best_support_chunk_id(
                    answer_type=kind,
                    query=query,
                    fragment=fragment,
                    context_chunks=context_chunks,
                    allow_first_chunk_fallback=False,
                )
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                localized.append(chunk_id)

        return localized

    @classmethod
    def _localize_boolean_support_chunk_ids(
        cls,
        *,
        answer: str,
        query: str,
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        if not context_chunks:
            return []

        compare_localized = cls._localize_boolean_compare_support_chunk_ids(
            query=query,
            context_chunks=context_chunks,
        )
        if compare_localized:
            return compare_localized

        polarity = cls._normalize_support_text(answer).casefold()
        query_terms = cls._support_terms(query)
        query_lower = cls._normalize_support_text(query).casefold()
        exception_query = any(
            marker in query_lower
            for marker in (
                " if ",
                " unless ",
                " except ",
                " provided ",
                " notwithstanding ",
                " bad faith",
                " good faith",
                " liable",
                " liability",
            )
        )

        ranked: list[tuple[int, int, str, set[str], bool]] = []
        for idx, chunk in enumerate(context_chunks):
            base_score = cls._chunk_support_score(
                answer_type="boolean",
                query=query,
                fragment=query or answer,
                chunk=chunk,
            )
            blob = cls._chunk_support_blob(chunk)
            blob_lower = blob.casefold()
            matched_terms = query_terms.intersection(cls._support_terms(blob))
            has_exception_clause = bool(
                re.search(
                    r"\b(?:except|unless|provided\s+that|notwithstanding|bad\s+faith|good\s+faith|liable|liability|"
                    r"does\s+not\s+apply|nothing\s+in)\b",
                    blob_lower,
                )
            )
            if polarity.startswith("yes") and has_exception_clause:
                base_score += 18
            if polarity.startswith("no") and bool(
                re.search(
                    r"\b(?:not\s+liable|no\s+liability|shall\s+not|may\s+not|is\s+not\s+liable|immune)\b",
                    blob_lower,
                )
            ):
                base_score += 18
            if base_score <= 0:
                continue
            ranked.append((base_score, -idx, chunk.chunk_id, matched_terms, has_exception_clause))

        if not ranked:
            return []

        ranked.sort(reverse=True)
        primary_chunk_id = ranked[0][2]
        truncated_ranked = ranked[: min(len(ranked), 6)]
        max_term_overlap = max(len(matched_terms) for _score, _order, _chunk_id, matched_terms, _has_exception in truncated_ranked)
        exception_available = any(has_exception for _score, _order, _chunk_id, _matched_terms, has_exception in truncated_ranked)

        def _candidate_score(indices: tuple[int, ...]) -> tuple[int, int, int]:
            selected = [truncated_ranked[idx] for idx in indices]
            total_score = sum(score for score, _order, _chunk_id, _matched_terms, _has_exception in selected)
            covered: set[str] = set()
            for _score, _order, _chunk_id, matched_terms, _has_exception in selected:
                covered.update(matched_terms)
            exception_covered = any(has_exception for _score, _order, _chunk_id, _matched_terms, has_exception in selected)
            completeness_penalty = 0
            if exception_query and exception_available and not exception_covered:
                completeness_penalty -= 10_000
            if exception_query and len(covered) < max_term_overlap:
                completeness_penalty -= (max_term_overlap - len(covered)) * 40
            return (completeness_penalty + total_score + (len(covered) * 12), -len(indices), -indices[0])

        best_indices = (0,)
        best_tuple = _candidate_score(best_indices)

        for idx in range(len(truncated_ranked)):
            candidate = (idx,)
            score_tuple = _candidate_score(candidate)
            if score_tuple > best_tuple:
                best_tuple = score_tuple
                best_indices = candidate

        for left in range(len(truncated_ranked)):
            for right in range(left + 1, len(truncated_ranked)):
                candidate = (left, right)
                score_tuple = _candidate_score(candidate)
                if score_tuple > best_tuple:
                    best_tuple = score_tuple
                    best_indices = candidate

        localized = [truncated_ranked[idx][2] for idx in best_indices]
        if not localized:
            return [primary_chunk_id]
        return localized

    @classmethod
    def _localize_boolean_compare_support_chunk_ids(
        cls,
        *,
        query: str,
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        refs = cls._paired_support_question_refs(query)
        if len(refs) < 2 or not context_chunks:
            return []

        query_lower = cls._normalize_support_text(query).casefold()
        if "same year" in query_lower:
            def scorer(ref: str, chunk: RankedChunk) -> int:
                return cls._boolean_year_seed_chunk_score(ref=ref, chunk=chunk)
        elif "administ" in query_lower:
            def scorer(ref: str, chunk: RankedChunk) -> int:
                clause_score = cls._named_administration_clause_score(
                    ref=ref,
                    text=str(getattr(chunk, "text", "") or ""),
                )
                if clause_score <= 0:
                    return 0
                return cls._boolean_admin_seed_chunk_score(ref=ref, chunk=chunk) + clause_score
        elif _is_case_party_overlap_compare_query(query, answer_type="boolean"):
            localized: list[str] = []
            seen_chunk_ids: set[str] = set()
            seen_doc_ids: set[str] = set()
            for ref in refs[:2]:
                best_chunk_id, best_doc_id = cls._best_case_party_compare_chunk_id(
                    query=query,
                    ref=ref,
                    context_chunks=context_chunks,
                    seen_doc_ids=seen_doc_ids,
                )
                if not best_chunk_id:
                    continue
                if best_chunk_id not in seen_chunk_ids:
                    localized.append(best_chunk_id)
                    seen_chunk_ids.add(best_chunk_id)
                if best_doc_id:
                    seen_doc_ids.add(best_doc_id)
            return localized if len(localized) >= 2 else []
        else:
            return []

        localized: list[str] = []
        seen_chunk_ids: set[str] = set()
        seen_doc_ids: set[str] = set()
        for ref in refs:
            best_chunk_id = ""
            best_doc_id = ""
            best_score = 0
            for chunk in context_chunks:
                doc_id = str(getattr(chunk, "doc_id", "") or chunk.chunk_id).strip()
                if doc_id in seen_doc_ids:
                    continue
                score = scorer(ref, chunk)
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id
                    best_doc_id = doc_id
            if not best_chunk_id:
                continue
            if best_chunk_id not in seen_chunk_ids:
                localized.append(best_chunk_id)
                seen_chunk_ids.add(best_chunk_id)
            if best_doc_id:
                seen_doc_ids.add(best_doc_id)

        return localized if len(localized) >= 2 else []

    @classmethod
    def _localize_free_text_support_chunk_ids(
        cls,
        *,
        answer: str,
        query: str,
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        if not context_chunks:
            return []

        items = cls._split_free_text_items(answer)
        if not items:
            return []

        localized: list[str] = []
        seen: set[str] = set()
        bounded_items = items[:8]
        for item_index, item in enumerate(bounded_items):
            item_chunks = cls._free_text_item_candidate_chunks(
                query=query,
                item=item,
                item_index=item_index,
                item_count=len(bounded_items),
                context_chunks=context_chunks,
            )
            item_slots = cls._extract_free_text_item_slots(query=query, item=item)
            primary_slot_ids: list[str] = []
            title_slot = cls._free_text_item_title_slot(cls._normalize_support_text(_CITE_RE.sub("", item)))
            if title_slot:
                chunk_id = cls._best_title_support_chunk_id(
                    title=title_slot,
                    context_chunks=item_chunks,
                )
                if chunk_id:
                    primary_slot_ids.append(chunk_id)
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    localized.append(chunk_id)
            for slot in item_slots:
                chunk_id = cls._best_support_chunk_id(
                    answer_type="free_text",
                    query=query,
                    fragment=slot,
                    context_chunks=item_chunks,
                    allow_first_chunk_fallback=False,
                )
                if chunk_id:
                    primary_slot_ids.append(chunk_id)
                if chunk_id and chunk_id not in seen:
                    seen.add(chunk_id)
                    localized.append(chunk_id)

            if cls._free_text_slot_full_context_priority(
                query=query,
                item_slots=item_slots,
                primary_slot_ids=primary_slot_ids,
            ):
                if title_slot:
                    expanded_title_chunk_id = cls._best_title_support_chunk_id(
                        title=title_slot,
                        context_chunks=context_chunks,
                    )
                    if expanded_title_chunk_id and expanded_title_chunk_id not in seen:
                        seen.add(expanded_title_chunk_id)
                        localized.append(expanded_title_chunk_id)
                for slot in item_slots:
                    chunk_id = cls._best_support_chunk_id(
                        answer_type="free_text",
                        query=query,
                        fragment=slot,
                        context_chunks=context_chunks,
                        allow_first_chunk_fallback=False,
                    )
                    if chunk_id and chunk_id not in seen:
                        seen.add(chunk_id)
                        localized.append(chunk_id)

        if _is_named_multi_title_lookup_query(query):
            localized_doc_ids = {
                str(chunk.doc_id or chunk.chunk_id)
                for chunk in context_chunks
                if chunk.chunk_id in seen
            }
            for ref in cls._support_question_refs(query):
                ref_chunk_id = cls._best_title_support_chunk_id(
                    title=ref,
                    context_chunks=context_chunks,
                )
                if not ref_chunk_id:
                    continue
                ref_chunk = next((chunk for chunk in context_chunks if chunk.chunk_id == ref_chunk_id), None)
                ref_doc_id = str(ref_chunk.doc_id or ref_chunk.chunk_id) if ref_chunk is not None else ""
                if ref_doc_id and ref_doc_id in localized_doc_ids:
                    continue
                if ref_chunk_id not in seen:
                    seen.add(ref_chunk_id)
                    localized.append(ref_chunk_id)
                if ref_doc_id:
                    localized_doc_ids.add(ref_doc_id)

        return localized

    @classmethod
    def _suppress_named_administration_family_orphan_support_ids(
        cls,
        *,
        query: str,
        cited_ids: Sequence[str],
        support_ids: Sequence[str],
        context_chunks: Sequence[RankedChunk],
    ) -> list[str]:
        normalized_query = re.sub(r"\s+", " ", (query or "").strip()).casefold()
        if (
            not support_ids
            or not cited_ids
            or "administ" not in normalized_query
            or _is_broad_enumeration_query(query)
        ):
            return list(dict.fromkeys(str(chunk_id).strip() for chunk_id in support_ids if str(chunk_id).strip()))

        refs = cls._support_question_refs(query)
        if len(refs) < 2:
            return list(dict.fromkeys(str(chunk_id).strip() for chunk_id in support_ids if str(chunk_id).strip()))

        context_by_id = {chunk.chunk_id: chunk for chunk in context_chunks}

        def _canonical_cited_for_ref(ref: str) -> bool:
            for raw_chunk_id in cited_ids:
                chunk = context_by_id.get(str(raw_chunk_id).strip())
                if chunk is None:
                    continue
                if cls._boolean_admin_seed_chunk_score(ref=ref, chunk=chunk) <= 0:
                    continue
                if not cls._is_consolidated_or_amended_family_chunk(chunk=chunk):
                    return True
            return False

        canonical_refs = {ref for ref in refs if _canonical_cited_for_ref(ref)}
        if not canonical_refs:
            return list(dict.fromkeys(str(chunk_id).strip() for chunk_id in support_ids if str(chunk_id).strip()))

        filtered: list[str] = []
        seen: set[str] = set()
        for raw_chunk_id in support_ids:
            chunk_id = str(raw_chunk_id).strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id in cited_ids:
                filtered.append(chunk_id)
                continue

            chunk = context_by_id.get(chunk_id)
            if chunk is None or not cls._is_consolidated_or_amended_family_chunk(chunk=chunk):
                filtered.append(chunk_id)
                continue

            drop_surrogate = False
            for ref in canonical_refs:
                if cls._boolean_admin_seed_chunk_score(ref=ref, chunk=chunk) <= 0:
                    continue
                drop_surrogate = True
                break
            if not drop_surrogate:
                filtered.append(chunk_id)

        return filtered
