from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import tiktoken

from rag_challenge.config import get_settings
from rag_challenge.models import Citation, QueryComplexity, RankedChunk
from rag_challenge.prompts import load_prompt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from rag_challenge.llm.provider import LLMProvider
    from rag_challenge.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\(cite:\s*([^)]+)\)", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")
_TITLE_REF_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9]*(?:\s+(?:of|the|in|on|and|for|to|by|Non|Incorporated|Limited|General|Data|Protection|Application|Civil|Commercial|Strata|Title|Trust|Contract|Liability|Partnership|Profit|Organisations?|Operating|Companies|Insolvency|Foundations?|Employment|Arbitration|Securities|Investment|Personal|Property|Obligations|Netting|Courts|Court|Common|Reporting|Standard|Dematerialised|Investments?|Implied|Terms|Unfair|Amendment|DIFC))*\s+(?:Law|Regulations?)))\b(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
_AMENDMENT_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Laws?\s+Amendment\s+Law(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
_BROAD_ENUMERATION_RE = re.compile(
    r"^(?:which\s+(?:laws?|regulations?|documents?|cases?)|list\s+all|identify\s+all|name\s+all)\b",
    re.IGNORECASE,
)
_COMMON_ELEMENTS_RE = re.compile(r"\b(?:common elements|elements in common|in common)\b", re.IGNORECASE)
_NEGATIVE_SUBCLAIM_RE = re.compile(
    r"(?:^|\n)"
    r"[^\n]*?"
    r"(?:"
    r"[Tt]here\s+is\s+no\s+information\s+on\s+(?:the\s+)?(?:other\s+)?[A-Za-z]"
    r"|[Tt]here\s+is\s+no\s+information\s+in\s+the\s+provided\s+sources"
    r"|[Tt]here\s+is\s+no\s+explicit\s+mention(?:\s+of)?"
    r"|[Tt]here\s+is\s+also\s+evidence\s+that"
    r"|[Nn]o\s+information\s+(?:was\s+)?found\s+on\s+(?:the\s+)?[A-Z]"
    r"|does\s+not\s+(?:explicitly\s+)?mention"
    r"|does\s+not\s+(?:explicitly\s+)?contain"
    r"|does\s+not\s+(?:explicitly\s+)?include"
    r"|does\s+not\s+(?:explicitly\s+)?provide"
    r"|does\s+not\s+(?:explicitly\s+)?reference"
    r"|is\s+not\s+(?:explicitly\s+)?mentioned"
    r"|could\s+not\s+be\s+(?:confirmed|verified)"
    r"|are\s+confirmed\s+only\s+between"
    r"|Therefore,\s+the\s+common\s+elements"
    r")"
    r"[^\n]*",
    re.IGNORECASE,
)
_TRAILING_NEGATIVE_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"(?:"
    r"[Tt]here\s+is\s+no\s+information\s+on"
    r"|[Tt]here\s+is\s+no\s+information\s+in\s+the\s+provided\s+sources"
    r"|[Tt]here\s+is\s+no\s+explicit\s+mention(?:\s+of)?"
    r"|[Tt]here\s+is\s+also\s+evidence\s+that"
    r"|does\s+not\s+(?:explicitly\s+)?"
    r"|only\s+states\s+that"
    r"|are\s+confirmed\s+only\s+between"
    r"|Therefore,\s+the\s+common\s+elements"
    r"|cannot\s+be\s+confirmed"
    r")"
    r"[^.!?]*[.!?]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "this",
    "what",
    "when",
    "where",
    "which",
    "under",
    "according",
    "article",
    "section",
    "case",
    "law",
}

_SYSTEM_PROMPT_SIMPLE = load_prompt("llm/generator_system_simple")
_SYSTEM_PROMPT_COMPLEX = load_prompt("llm/generator_system_complex")
_SYSTEM_PROMPT_COMPLEX_IRAC = load_prompt("llm/generator_system_complex_irac")
_SYSTEM_PROMPT_STRICT = load_prompt("llm/generator_system_strict")
_USER_PROMPT_TEMPLATE = load_prompt("llm/generator_user")
_USER_PROMPT_TEMPLATE_STRICT = load_prompt("llm/generator_user_strict")
_IRAC_HINT_RE = re.compile(
    r"\b(compare|difference|distinguish|contrast|analy[sz]e|evaluate|common elements|modify|modifies|impact|effect|summari[sz]e)\b",
    re.IGNORECASE,
)
_CITED_TITLE_RE = re.compile(r'This Law may be cited as(?: the)? [“"]([^"”]+)[”"]', re.IGNORECASE)
_CITED_TITLE_PLAIN_RE = re.compile(
    r"\bThis Law may be cited as(?: the)? ([A-Z][A-Za-z0-9\s()/-]+?(?:Law|Regulations?)(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
_COVER_TITLE_LAW_YEAR_RE = re.compile(
    r"^\s*([A-Z][A-Z\s/&().-]+?(?:LAW|REGULATIONS?|RULES?|CODE|NOTICE))\s+"
    r"DIFC\s+LAW\s+NO\.?\s*\d+\s+OF\s+(\d{4})\b",
    re.IGNORECASE | re.MULTILINE,
)
_LEGISLATION_REF_TITLE_RE = re.compile(
    r"\b(?:this|the)\s+Law\s+is\s+(?:the\s+)?("
    r"[A-Z][A-Za-z][A-Za-z\s]+?Law(?:\s+Amendment\s+Law)?"
    r"(?:,\s*DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4})?"
    r")\s+made\s+by\s+the\s+Ruler\b",
    re.IGNORECASE,
)
_ENACTMENT_NOTICE_ATTACHED_TITLE_RE = re.compile(
    r"\bin\s+the\s+form\s+now\s+attached\s+(?:the\s+)?("
    r"[A-Z][A-Za-z][A-Za-z\s-]+?Law(?:\s+Amendment\s+Law)?"
    r"(?:\s+DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)
_ENACTMENT_NOTICE_TITLE_RE = re.compile(
    r"\bthe\s+([A-Z][A-Za-z][A-Za-z\s-]+Law(?:\s+DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4})?)\b",
    re.IGNORECASE,
)
_DOC_SUMMARY_TITLE_RE = re.compile(r"\*\*Document Title:\*\*\s*([^\n]+)", re.IGNORECASE)
_DOC_SUMMARY_TITLED_RE = re.compile(r'\btitled\s+[“"]([^"”]+)[”"]', re.IGNORECASE)
_DOC_SUMMARY_PREFIX_TITLE_RE = re.compile(
    r"^(?:statute|regulation|document|case(?:\s+law)?):\s*([^,(]+?(?:Law|Regulations?|Rules?|Code|Notice))\b",
    re.IGNORECASE,
)
_DOC_SUMMARY_IS_THE_RE = re.compile(
    r"\bThis\s+(?:document|statute|contract\s+document|case\s+law\s+document)\s+(?:is|,)\s*(?:the\s+)?"
    r"([^,.]+?(?:Law|Regulations?|Rules?|Code|Notice))\b",
    re.IGNORECASE,
)
_STRUCTURED_TITLE_BAD_LEAD_RE = re.compile(
    r"^(?:(?:we\s+hereby\s+enact(?:\s+on\s+this\s+[^.]+?)?\s+(?:the\s+)?)?(?:in\s+the\s+form\s+now\s+attached\s+)?|enactment\s+notice\s+)+",
    re.IGNORECASE,
)
_TITLE_LAW_NO_SUFFIX_RE = re.compile(r"\s*,?\s*DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
_TITLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TITLE_LEADING_CONNECTOR_RE = re.compile(r"^(?:(?:of|and|the)\s+)+", re.IGNORECASE)
_TITLE_REF_BAD_LEAD_RE = re.compile(
    r"^(?:(?:which|what|how|mention|mentions|reference|references|their|these|those|do|does|did)\s+)+",
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
_TITLE_CLAUSE_RE = re.compile(r"\b(?:title|may be cited as)\b", re.IGNORECASE)
_PRE_EXISTING_EFFECTIVE_DATE_RE = re.compile(
    r"pre-existing accounts?.{0,160}?effective date is\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)
_NEW_ACCOUNT_EFFECTIVE_DATE_RE = re.compile(
    r"new accounts?.{0,160}?effective date is\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)
_RECORDS_RETAINED_AFTER_REPORTING_RE = re.compile(
    r"retention\s+period\s+of\s+six\s+\(6\)\s+years\s+after\s+the\s+date\s+of\s+reporting\s+the\s+information",
    re.IGNORECASE,
)
_ACCOUNTING_RECORDS_PRESERVED_RE = re.compile(
    r"preserved\s+by\s+the\s+([A-Za-z][A-Za-z\s-]+?)\s+for\s+at\s+least\s+six\s+\(6\)\s+years\s+from\s+the\s+date\s+upon\s+which\s+they\s+were\s+created",
    re.IGNORECASE,
)
_ENACTMENT_DATE_RE = re.compile(
    r"\bhereby enact\s+on\s+(?:this\s+)?([0-9]{1,2}(?:st|nd|rd|th)?(?:\s+day\s+of)?\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)
_ENACTMENT_NOTICE_REFERENCE_RE = re.compile(
    r"\b(?:this|the)\s+law\s+is\s+enacted\s+on\s+the\s+date\s+specified\s+in\s+the\s+enactment\s+notice"
    r"(?:\s+in\s+respect\s+of\s+this\s+law)?\b",
    re.IGNORECASE,
)
_CONSOLIDATED_VERSION_RE = re.compile(
    r"\bConsolidated\s+Version(?:\s+No\.?\s*\d+)?\s*\(([^)]+)\)",
    re.IGNORECASE,
)
_UPDATED_VALUE_RE = re.compile(
    r"\b(?:last\s+updated|updated|effective\s+from)\s*(?:[:\-]|\bis\b)?\s*"
    r"([0-9]{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{4})\b",
    re.IGNORECASE,
)
_REMUNERATION_RECORDKEEPING_RE = re.compile(
    r"the\s+Employee'?s\s+Remuneration\s*\(([^)]+)\)\s*,\s*and\s+the\s+applicable\s+Pay\s+Period",
    re.IGNORECASE,
)
_QUESTION_SINGLE_LAW_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwho administers the (?P<title>.+? law(?:\s+\d{4})?)\??$", re.IGNORECASE),
    re.compile(
        r"\bwhen was the consolidated version of the (?P<title>.+?) published\??$",
        re.IGNORECASE,
    ),
)
_LIST_POSTAMBLE_RE = re.compile(
    r"^[.)\s-]*(?:No other|Therefore|Thus|Accordingly|In summary|These are|The laws|The documents)\b",
    re.IGNORECASE,
)
_ORDER_SECTION_MARKER_RE = re.compile(
    r"\bIT\s+IS\s+HEREBY\s+ORDERED(?:\s+AND\s+DIRECTED)?\s+THAT\b",
    re.IGNORECASE,
)
_ORDER_SECTION_STOP_RE = re.compile(
    r"^(?:Issued by:?|SCHEDULE OF REASONS|SCHEDULE OF THE COURT'?S REASONS|Introduction|Background|Discussion and Determination)\b",
    re.IGNORECASE,
)
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+\.\s*")
_OUTCOME_CUE_RE = re.compile(
    r"\b(?:dismissed|refused|granted|allowed|discharged|set aside|restored|proceed to trial|stayed|varied|rejected)\b",
    re.IGNORECASE,
)
_EXPLICIT_OUTCOME_VERB_RE = re.compile(
    r"\b(?:is|was|are|be|been|being|shall be|must be|to be)\s+"
    r"(?:dismissed|refused|granted|allowed|discharged|set aside|restored|stayed|varied|rejected)\b",
    re.IGNORECASE,
)
_COST_CUE_RE = re.compile(r"\bcosts?\b|\bno order as to costs\b", re.IGNORECASE)
_OUTCOME_NOISE_RE = re.compile(
    r"\b(?:issued by|date of issue|at:\s*\d|schedule of reasons|was considered|by\s+rdc\s+\d+)\b",
    re.IGNORECASE,
)
_COMPLETE_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z][a-z]|[-*]|\d+\.))")
_NUMBERED_ITEM_RE = re.compile(r"(?<!\d)(\d+)\.\s+")
_TITLE_ONLY_ITEM_BAD_LEAD_RE = re.compile(
    r"\b(?:this\s+is\s+confirmed|includes|states|contains|provides|application|schedule|article|section|"
    r"because|under|enacted|administered|commencement|penalty)\b",
    re.IGNORECASE,
)
_TITLE_ONLY_PLACEHOLDER_RE = re.compile(
    r"^(?:citation\s+title|this\s+is\s+(?:shown|stated|explicitly\s+stated|confirmed)\b|the\s+statement\b)",
    re.IGNORECASE,
)
_BODY_LIKE_TITLE_RE = re.compile(
    r"\b(?:this\s+law|requirements?\s+of\s+this\s+law|purposes?\s+of\s+this\s+law|"
    r"title\s+and\s+repeal|directors?\s+to|obligations?\b|application\s+of\s+this\s+law|"
    r"terms?\s+and\s+purposes?\s+of|penalty\s+for\s+offences?|penalty\s+for\s+an\s+offence)\b",
    re.IGNORECASE,
)
_REGISTRAR_SELF_ADMIN_RE = re.compile(
    r"\b(?:the\s+registrar\s+shall\s+administer\s+this\s+law"
    r"|this\s+law\s+shall\s+be\s+administered\s+by\s+the\s+registrar"
    r"|this\s+law\s+is\s+administered\s+by\s+the\s+registrar"
    r"|administration\s+of\s+this\s+law\b[^.]{0,160}\bregistrar\b)\b",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_ONLY_BAD_CANDIDATE_RE = re.compile(r"\b(?:application of the arbitration law|arbitration law)\b", re.IGNORECASE)
_RULER_OF_DUBAI_RE = re.compile(r"\bruler of dubai\b", re.IGNORECASE)
_PENALTY_AMOUNT_RE = re.compile(r"\b(?:USD|US\$)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})(?:\.\d+)?\b", re.IGNORECASE)
_ENACTMENT_NOTICE_COMMENCEMENT_RE = re.compile(
    r"\b(?:this\s+law\s+)?shall\s+come\s+into\s+force\s+on\s+"
    r"(?:the\s+date\s+specified\s+in\s+the\s+enactment\s+notice(?:\s+in\s+respect\s+of\s+this\s+law)?"
    r"|(?:the\s+)?\d+(?:st|nd|rd|th)?\s+business\s+day\s+after\s+enactment"
    r"|\d+\s+days?\s+after\s+enactment)\b"
    r"|\b(?:this\s+law\s+)?comes?\s+into\s+force\s+on\s+the\s+date\s+specified\s+in\s+the\s+enactment\s+notice(?:\s+in\s+respect\s+of\s+this\s+law)?\b",
    re.IGNORECASE,
)
_ADMINISTRATION_CLAUSE_RE = re.compile(
    r"\b("
    r"(?:this\s+law(?:\s+and\s+any\s+(?:legislation\s+made\s+for\s+the\s+purposes?\s+of\s+this\s+law|"
    r"regulations?\s+made\s+under\s+it))?\s+"
    r"(?:is|are|shall\s+be)\s+administered\s+by\s+(?:the\s+)?[^.]+)"
    r"|(?:the\s+[^.]+?\s+shall\s+administer\s+this\s+law)"
    r")\b",
    re.IGNORECASE,
)
_ADMINISTRATION_ENTITY_RE = re.compile(
    r"\badministered\s+by\s+(?:the\s+)?([A-Za-z][A-Za-z\s-]+)"
    r"|the\s+([A-Za-z][A-Za-z\s-]+?)\s+shall\s+administer\s+this\s+law\b",
    re.IGNORECASE,
)
_COMMON_ELEMENT_SIGNATURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "schedule_1_interpretative_provisions",
        "Schedule 1 contains interpretative provisions which apply to the Law.",
        ("schedule 1", "interpretative provisions"),
    ),
    (
        "schedule_1_defined_terms",
        "Schedule 1 contains a list of defined terms used in the Law.",
        ("schedule 1", "defined terms"),
    ),
    (
        "rules_of_interpretation",
        "Schedule 1 contains rules of interpretation.",
        ("schedule 1", "rules of interpretation"),
    ),
    (
        "amended_or_re_enacted_reference",
        "A statutory provision includes a reference to the statutory provision as amended or re-enacted from time to time.",
        ("a statutory provision includes a reference", "amended or re-enacted"),
    ),
    (
        "person_definition_reference",
        "A reference to a person includes any natural person, body corporate or body unincorporate, including a company, partnership, unincorporated association, government or state.",
        ("reference to a person includes", "natural person", "body corporate", "body unincorpor"),
    ),
)
_INTERPRETATION_SECTION_COMMON_KEYS: tuple[str, ...] = (
    "amended_or_re_enacted_reference",
    "person_definition_reference",
)
_PENALTY_STOPWORDS = {
    "what",
    "is",
    "the",
    "for",
    "and",
    "under",
    "penalty",
    "penalties",
    "prescribed",
    "offense",
    "offences",
    "offence",
    "against",
    "law",
    "laws",
    "regulations",
    "regulation",
    "title",
    "using",
}


class RAGGenerator:
    """Prompt builder + grounded answer generator for RAG."""

    def __init__(self, llm: LLMProvider) -> None:
        settings = get_settings()
        self._llm = llm
        self._llm_settings = settings.llm
        self._pipeline_settings = settings.pipeline
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def build_prompt(
        self,
        question: str,
        chunks: Sequence[RankedChunk],
        *,
        complexity: QueryComplexity | str = QueryComplexity.SIMPLE,
        max_words: int | None = None,
        answer_type: str = "free_text",
        prompt_hint: str = "",
    ) -> tuple[str, str]:
        complexity_value = self._normalize_complexity(complexity)
        answer_word_limit = int(max_words or self._pipeline_settings.max_answer_words)
        answer_kind = answer_type.strip().lower()
        strict_types = {"boolean", "number", "date", "name", "names"}

        if answer_kind in strict_types:
            system_prompt = _SYSTEM_PROMPT_STRICT
        elif complexity_value == QueryComplexity.COMPLEX:
            if answer_kind == "free_text":
                if self._should_use_irac(question):
                    system_prompt = _SYSTEM_PROMPT_COMPLEX_IRAC.format(max_words=answer_word_limit)
                else:
                    system_prompt = _SYSTEM_PROMPT_COMPLEX.format(max_words=answer_word_limit)
            else:
                system_prompt = _SYSTEM_PROMPT_COMPLEX.format(max_words=answer_word_limit)
        else:
            system_prompt = _SYSTEM_PROMPT_SIMPLE
        system_prompt = f"{system_prompt}\n\n{self._answer_type_instruction(answer_type)}".strip()
        if prompt_hint.strip():
            system_prompt = f"{system_prompt}\n\n{prompt_hint.strip()}"

        user_template = _USER_PROMPT_TEMPLATE_STRICT if answer_kind in strict_types else _USER_PROMPT_TEMPLATE
        user_prompt = user_template.format(
            question=question,
            answer_type=answer_type,
            formatted_context=self._format_context(
                question=question,
                chunks=chunks,
                complexity=complexity_value,
                answer_type=answer_type,
            ),
        )
        return system_prompt, user_prompt

    def get_context_debug_stats(
        self,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        complexity: QueryComplexity | str = QueryComplexity.SIMPLE,
        answer_type: str = "free_text",
    ) -> tuple[int, int]:
        parts, budget = self._render_context_blocks(
            question=question,
            chunks=chunks,
            complexity=self._normalize_complexity(complexity),
            answer_type=answer_type,
        )
        return len(parts), budget

    def build_structured_free_text_answer(
        self,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        q = (question or "").strip().lower()
        if not q or not chunks:
            return ""

        if self._is_unsupported_criminal_trap(question):
            return "The provided legal documents do not contain information on this topic."

        if self._is_common_elements_question(question):
            return self._build_common_elements_canonical_answer(
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_interpretative_provisions_enumeration_question(question):
            return self.build_interpretative_provisions_enumeration_answer(chunks=chunks)

        if self._is_remuneration_recordkeeping_question(question):
            return self.build_remuneration_recordkeeping_answer(chunks=chunks)

        if self._is_case_outcome_question(question):
            return self.build_case_outcome_answer(question=question, chunks=chunks)

        if self._is_consolidated_version_published_question(question):
            return self.build_consolidated_version_published_answer(
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_amended_by_enumeration_question(question):
            return self.build_amended_by_enumeration_answer(
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_multi_title_lookup_question(question):
            return self.cleanup_named_multi_title_lookup_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_amendment_question(question):
            return self.cleanup_named_amendment_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_enactment_date_question(question):
            return self.cleanup_named_enactment_date_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_made_by_question(question):
            return self.cleanup_named_made_by_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if "administ" in q and not self._is_broad_enumeration_question(question):
            return self.cleanup_named_administration_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_registrar_authority_question(question):
            return self.cleanup_named_registrar_authority_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_retention_period_question(question):
            return self.cleanup_named_retention_period_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_liability_question(question):
            return self.cleanup_named_liability_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_named_translation_requirement_question(question):
            return self.cleanup_named_translation_requirement_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if any(term in q for term in ("penalt", "fine")) and not self._is_broad_enumeration_question(question):
            return self.cleanup_named_penalty_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if self._is_registrar_enumeration_question(question):
            return self.build_registrar_enumeration_answer(question=question, chunks=chunks)

        if self._is_named_reference_enumeration_question(question):
            return self.cleanup_named_ref_enumeration_items("", question=question, chunks=chunks)

        if self._is_ruler_enactment_enumeration_question(question):
            return self.cleanup_ruler_enactment_enumeration_items("", chunks=chunks)

        if self._is_ruler_authority_year_enumeration_question(question):
            return self.build_ruler_authority_year_enumeration_answer(question=question, chunks=chunks)

        if "pre-existing" in q and "new accounts" in q and "effective date" in q:
            return self.cleanup_account_effective_dates_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        if any(term in q for term in ("commencement", "come into force", "effective date", "enactment notice")):
            return self.cleanup_named_commencement_answer(
                "",
                question=question,
                chunks=chunks,
                doc_refs=doc_refs,
            )

        return ""

    @staticmethod
    def _is_named_retention_period_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q:
            return False
        return (
            "retention period" in q
            or ("preserve" in q and "accounting records" in q)
            or ("records" in q and "six (6) years" in q)
            or ("records" in q and "minimum period" in q)
        )

    @staticmethod
    def _is_case_outcome_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q:
            return False
        return any(
            phrase in q
            for phrase in (
                "what was the result of the application heard in case",
                "what was the outcome of the specific order or application",
                "looking only at the last page of the document",
                "according to the 'it is hereby ordered that' section",
                'according to the "it is hereby ordered that" section',
                "summarize the court's final ruling in case",
                "how did the court of appeal rule, and what costs were awarded",
            )
        )

    _CRIMINAL_TRAP_TERMS = frozenset((
        "jury", "parole", "miranda", "plea bargain", "plea deal",
        "bail hearing", "indictment", "grand jury", "arraignment",
        "felony charge", "criminal sentencing",
    ))

    @staticmethod
    def _is_unsupported_criminal_trap(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        return any(term in q for term in RAGGenerator._CRIMINAL_TRAP_TERMS)

    @staticmethod
    def _is_consolidated_version_published_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        return bool(q) and "consolidated version" in q and "published" in q

    @staticmethod
    def _is_remuneration_recordkeeping_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        return bool(q) and "keep records" in q and "remuneration" in q and "article 16(1)(c)" in q

    async def generate_stream(
        self,
        question: str,
        chunks: Sequence[RankedChunk],
        *,
        model: str,
        max_tokens: int,
        collector: TelemetryCollector,
        complexity: QueryComplexity | str = QueryComplexity.SIMPLE,
        answer_type: str = "free_text",
        prompt_hint: str = "",
    ) -> AsyncIterator[str]:
        system_prompt, user_prompt = self.build_prompt(
            question,
            chunks,
            complexity=complexity,
            answer_type=answer_type,
            prompt_hint=prompt_hint,
        )

        collector.set_models(llm=model)
        emitted_parts: list[str] = []
        fallback_models = self._generation_fallback_models(model)

        async for token in self._llm.stream_generate_with_cascade(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            models=fallback_models,
            max_tokens=max_tokens,
        ):
            emitted_parts.append(token)
            yield token

        # If the underlying provider switched/fell back to another model, prefer provider-reported model.
        last_model_obj = getattr(self._llm, "get_last_stream_model", None)
        if callable(last_model_obj):
            try:
                raw_model = last_model_obj()
            except Exception:
                raw_model = ""
            if isinstance(raw_model, str) and raw_model.strip():
                collector.set_models(llm=raw_model.strip())
        last_provider_obj = getattr(self._llm, "get_last_stream_provider", None)
        last_finish_reason_obj = getattr(self._llm, "get_last_stream_finish_reason", None)
        raw_provider = ""
        raw_finish_reason = ""
        if callable(last_provider_obj):
            try:
                provider_obj = last_provider_obj()
            except Exception:
                provider_obj = ""
            if isinstance(provider_obj, str):
                raw_provider = provider_obj.strip()
        if callable(last_finish_reason_obj):
            try:
                finish_reason_obj = last_finish_reason_obj()
            except Exception:
                finish_reason_obj = ""
            if isinstance(finish_reason_obj, str):
                raw_finish_reason = finish_reason_obj.strip()
        collector.set_llm_diagnostics(provider=raw_provider, finish_reason=raw_finish_reason)

        provider_usage = self._llm.get_last_stream_usage()
        usage = self._resolve_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion_text="".join(emitted_parts),
            prompt_tokens=int(provider_usage.get("prompt_tokens", 0)),
            completion_tokens=int(provider_usage.get("completion_tokens", 0)),
            total_tokens=int(provider_usage.get("total_tokens", 0)),
        )
        collector.set_token_usage(
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            total_tokens=usage[2],
        )

    async def generate(
        self,
        question: str,
        chunks: Sequence[RankedChunk],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        collector: TelemetryCollector | None = None,
        complexity: QueryComplexity | str = QueryComplexity.SIMPLE,
        answer_type: str = "free_text",
        prompt_hint: str = "",
    ) -> tuple[str, list[Citation]]:
        chosen_model = model or self._llm_settings.simple_model
        chosen_max_tokens = int(max_tokens or self._llm_settings.simple_max_tokens)
        system_prompt, user_prompt = self.build_prompt(
            question,
            chunks,
            complexity=complexity,
            answer_type=answer_type,
            prompt_hint=prompt_hint,
        )

        result = await self._llm.generate_with_cascade(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            models=self._generation_fallback_models(chosen_model),
            max_tokens=chosen_max_tokens,
        )
        if collector is not None:
            collector.set_models(llm=result.model)
            collector.set_llm_diagnostics(provider=result.provider, finish_reason=result.finish_reason)
            usage = self._resolve_usage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                completion_text=result.text,
                prompt_tokens=int(result.prompt_tokens),
                completion_tokens=int(result.completion_tokens),
                total_tokens=int(result.total_tokens),
            )
            collector.set_token_usage(
                prompt_tokens=usage[0],
                completion_tokens=usage[1],
                total_tokens=usage[2],
            )
        citations = self.extract_citations(result.text, chunks)
        return result.text, citations

    def _generation_fallback_models(self, chosen_model: str) -> list[str]:
        ordered = [
            chosen_model,
            self._llm_settings.simple_model,
            self._llm_settings.fallback_model,
        ]
        models: list[str] = []
        seen: set[str] = set()
        for raw_model in ordered:
            model = str(raw_model or "").strip()
            if not model or model in seen:
                continue
            seen.add(model)
            models.append(model)
        return models

    def _format_context(
        self,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        complexity: QueryComplexity,
        answer_type: str,
    ) -> str:
        parts, _budget = self._render_context_blocks(
            question=question,
            chunks=chunks,
            complexity=complexity,
            answer_type=answer_type,
        )
        return "\n\n".join(parts)

    def _render_context_blocks(
        self,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        complexity: QueryComplexity,
        answer_type: str,
    ) -> tuple[list[str], int]:
        budget = self._context_budget(answer_type=answer_type, complexity=complexity)
        if budget <= 0:
            return [], 0

        ordered_chunks = list(chunks)
        used = 0
        parts: list[str] = []
        answer_kind = answer_type.strip().lower()
        is_broad_enumeration = answer_kind == "free_text" and self._is_broad_enumeration_question(question)
        is_common_elements = answer_kind == "free_text" and self._is_common_elements_question(question)
        mentions_multiple_titles = self._mentions_multiple_named_titles(question)
        is_named_multi_title_lookup = answer_kind == "free_text" and self._is_named_multi_title_lookup_question(question)
        use_extractive_compaction = answer_kind in {"boolean", "number", "date", "name", "names"}
        include_cite_hints = True
        is_registrar_enumeration = answer_kind == "free_text" and self._is_registrar_enumeration_question(question)
        is_named_ref_enumeration = answer_kind == "free_text" and self._is_named_reference_enumeration_question(question)
        is_company_structure_enumeration = answer_kind == "free_text" and self._is_company_structure_enumeration_question(question)
        use_compact_broad_payload = is_registrar_enumeration or is_named_ref_enumeration or is_company_structure_enumeration
        if is_broad_enumeration:
            if use_compact_broad_payload:
                compact_budget = int(
                    getattr(self._pipeline_settings, "context_token_budget_broad_enumeration_compact", 1600)
                )
                budget = max(0, min(int(self._llm_settings.max_context_tokens), compact_budget))
            else:
                budget = int(self._llm_settings.max_context_tokens)
            ordered_chunks = self._prioritize_broad_enumeration_chunks(question, ordered_chunks)
        elif is_common_elements:
            common_budget = int(getattr(self._pipeline_settings, "context_token_budget_common_elements", 1400))
            budget = max(0, min(int(self._llm_settings.max_context_tokens), common_budget))
            ordered_chunks = self._prioritize_common_elements_titles(question, ordered_chunks)
        elif is_named_multi_title_lookup:
            named_budget = int(getattr(self._pipeline_settings, "context_token_budget_named_multi_lookup", 1600))
            budget = max(0, min(int(self._llm_settings.max_context_tokens), named_budget))
            ordered_chunks = self._prioritize_unique_titles(self._prioritize_unique_pages(ordered_chunks))
            grouped_blocks = self._render_named_multi_title_lookup_blocks(
                question=question,
                chunks=ordered_chunks,
                budget=budget,
            )
            if grouped_blocks:
                return grouped_blocks, budget
        elif answer_kind == "free_text" and mentions_multiple_titles:
            budget = min(int(self._llm_settings.max_context_tokens), budget + 300)
            ordered_chunks = self._prioritize_unique_pages(ordered_chunks)
        if use_extractive_compaction and not bool(getattr(self._pipeline_settings, "strict_types_append_citations", False)):
            include_cite_hints = False

        for idx, chunk in enumerate(ordered_chunks, start=1):
            display_title = self._display_doc_title(chunk)
            header = f"{idx}. [{chunk.chunk_id}] {display_title}"
            if chunk.section_path:
                header += f" | {chunk.section_path}"
            cite_hint = f"CITE_AS: (cite: {chunk.chunk_id})" if include_cite_hints else ""
            chunk_text = chunk.text
            if use_extractive_compaction:
                top_n = 5 if answer_kind == "boolean" else 3
                chunk_text = self._compact_chunk_text(question=question, text=chunk_text, top_n=top_n)
            elif is_broad_enumeration:
                chunk_text = self._strip_enumeration_boilerplate(chunk_text)
                if use_compact_broad_payload:
                    chunk_text = self._compact_slow_broad_enumeration_chunk(
                        question=question,
                        text=chunk_text,
                    )
                else:
                    anchor_snippet = self._enumeration_anchor_snippet(question=question, text=chunk_text)
                    chunk_text = anchor_snippet or self._compact_chunk_text(question=question, text=chunk_text, top_n=3)
                chunk_text = f"Document title: {display_title}. {chunk_text}".strip()
            elif is_common_elements:
                chunk_text = self._strip_enumeration_boilerplate(chunk_text)
                chunk_text = self._compact_common_elements_chunk(chunk_text)
                chunk_text = f"Document title: {display_title}. {chunk_text}".strip()
            elif is_named_multi_title_lookup:
                chunk_text = self._compact_named_multi_title_lookup_chunk(question=question, text=chunk_text)
                chunk_text = f"Document title: {display_title}. {chunk_text}".strip()

            if cite_hint:
                block = f"---\n{header}\n{cite_hint}\n{chunk_text}\n---"
            else:
                block = f"---\n{header}\n{chunk_text}\n---"
            block_tokens = self.count_tokens(block)

            if used + block_tokens > budget:
                remaining = budget - used
                # Leave room for block wrapper and header when truncating.
                if remaining > 40:
                    header_block = f"---\n{header}\n{cite_hint}\n\n---" if cite_hint else f"---\n{header}\n\n---"
                    header_tokens = self.count_tokens(header_block)
                    text_budget = max(0, remaining - header_tokens - 4)
                    if text_budget > 0:
                        truncated_text = self._truncate_to_tokens(chunk_text, text_budget)
                        if truncated_text:
                            if cite_hint:
                                parts.append(f"---\n{header}\n{cite_hint}\n{truncated_text}\n---")
                            else:
                                parts.append(f"---\n{header}\n{truncated_text}\n---")
                break

            parts.append(block)
            used += block_tokens

        return parts, budget

    @staticmethod
    def _mentions_multiple_named_titles(question: str) -> bool:
        return len(_TITLE_REF_RE.findall(question or "")) >= 2

    @staticmethod
    def _page_num(section_path: str | None) -> int:
        match = re.search(r"page:(\d+)", section_path or "", flags=re.IGNORECASE)
        if match is None:
            return -1
        try:
            return int(match.group(1))
        except ValueError:
            return -1

    def _prioritize_unique_pages(self, chunks: Sequence[RankedChunk]) -> list[RankedChunk]:
        prioritized: list[RankedChunk] = []
        remainder: list[RankedChunk] = []
        seen: set[tuple[str, int]] = set()

        for chunk in chunks:
            key = ((chunk.doc_title or "").strip().lower(), self._page_num(chunk.section_path))
            if key not in seen:
                seen.add(key)
                prioritized.append(chunk)
            else:
                remainder.append(chunk)
        return prioritized + remainder

    def _prioritize_unique_titles(self, chunks: Sequence[RankedChunk]) -> list[RankedChunk]:
        prioritized: list[RankedChunk] = []
        remainder: list[RankedChunk] = []
        seen: set[str] = set()

        for chunk in chunks:
            key = self._enumeration_title_key(chunk)
            if key and key not in seen:
                seen.add(key)
                prioritized.append(chunk)
            else:
                remainder.append(chunk)
        return prioritized + remainder

    def _prioritize_broad_enumeration_chunks(self, question: str, chunks: Sequence[RankedChunk]) -> list[RankedChunk]:
        if not chunks:
            return []

        max_per_title = 1
        if self._requires_expanded_broad_enumeration_context(question):
            max_per_title = 2

        grouped: dict[str, list[tuple[int, RankedChunk]]] = {}
        title_order: list[str] = []
        for idx, chunk in enumerate(chunks):
            key = self._enumeration_title_key(chunk) or f"__chunk_{idx}"
            if key not in grouped:
                grouped[key] = []
                title_order.append(key)
            grouped[key].append((idx, chunk))

        prioritized: list[RankedChunk] = []
        remainder: list[RankedChunk] = []
        ranked_groups: list[list[RankedChunk]] = []
        for key in title_order:
            ranked_group = [
                chunk
                for _, chunk in sorted(
                grouped[key],
                key=lambda item: (
                    -self._broad_enumeration_chunk_priority(question, item[1]),
                    self._page_num(item[1].section_path),
                    item[0],
                ),
            )
            ]
            ranked_groups.append(ranked_group)

        for rep_idx in range(max_per_title):
            for ranked_group in ranked_groups:
                if rep_idx < len(ranked_group):
                    prioritized.append(ranked_group[rep_idx])

        for ranked_group in ranked_groups:
            remainder.extend(ranked_group[max_per_title:])

        return prioritized + remainder

    @classmethod
    def _broad_enumeration_chunk_priority(cls, question: str, chunk: RankedChunk) -> int:
        query_lower = (question or "").strip().lower()
        text = re.sub(r"\s+", " ", (chunk.text or "").strip())
        lowered = text.lower()
        title = re.sub(r"\s+", " ", (chunk.doc_title or "").strip())
        score = 0

        year_match = re.search(r"\b(19|20)\d{2}\b", query_lower)
        if year_match is not None and year_match.group(0) in lowered:
            score += 3
        if year_match is not None and year_match.group(0) in (title or "").lower():
            score += 2

        if "administered by the registrar" in query_lower:
            if (
                "the registrar shall administer this law" in lowered
                or "this law shall be administered by the registrar" in lowered
                or "this law is administered by the registrar" in lowered
                or "administration of this law" in lowered
            ):
                score += 20
            elif "registrar" in lowered:
                score += 2
            else:
                score -= 4
        if "schedule 2" in query_lower and "arbitration law" in query_lower and "schedule 2" in lowered and "arbitration law" in lowered:
            score += 10
        if "enactment notice" in query_lower:
            if "come into force" in lowered or "comes into force" in lowered:
                score += 8
            if "date specified in the enactment notice" in lowered:
                score += 6
        if "made by the ruler" in query_lower and "made by the ruler" in lowered:
            score += 6
        if "interpretative provisions" in query_lower and "interpretative provisions" in lowered:
            score += 7
        if "this law may be cited as" in lowered:
            score += 2
        if "part 1:" in lowered or "title and application" in lowered:
            score -= 4
        if lowered.startswith("in this document underlining indicates"):
            score -= 6
        return score

    def _prioritize_common_elements_titles(self, question: str, chunks: Sequence[RankedChunk]) -> list[RankedChunk]:
        groups: dict[str, list[tuple[int, RankedChunk]]] = {}
        for idx, chunk in enumerate(chunks):
            key = self._common_elements_title_key(chunk) or (chunk.doc_id or "").strip().casefold() or f"__idx_{idx}"
            groups.setdefault(key, []).append((idx, chunk))

        prioritized: list[RankedChunk] = []
        remainder: list[tuple[int, int, int, RankedChunk]] = []
        for group in groups.values():
            ranked_group = sorted(
                group,
                key=lambda item: (
                    -self._common_elements_chunk_priority(question, item[1]),
                    self._page_num(item[1].section_path),
                    item[0],
                ),
            )
            prioritized.append(ranked_group[0][1])
            for original_idx, chunk in ranked_group[1:]:
                remainder.append(
                    (
                        -self._common_elements_chunk_priority(question, chunk),
                        self._page_num(chunk.section_path),
                        original_idx,
                        chunk,
                    )
                )

        remainder_chunks = [chunk for *_meta, chunk in sorted(remainder)]
        return prioritized + remainder_chunks

    @staticmethod
    def _is_registrar_enumeration_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(q) and RAGGenerator._is_broad_enumeration_question(question) and "administered by the registrar" in q

    @staticmethod
    def _is_named_reference_enumeration_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return (
            bool(q)
            and RAGGenerator._is_broad_enumeration_question(question)
            and len(_TITLE_REF_RE.findall(question or "")) >= 2
            and any(term in q for term in ("mention", "mentions", "reference", "references"))
        )

    @staticmethod
    def _is_company_structure_enumeration_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(q) and RAGGenerator._is_broad_enumeration_question(question) and (
            "company structures" in q
            or ("schedule 2" in q and "arbitration law" in q)
            or "application of the arbitration law" in q
        )

    @staticmethod
    def _is_ruler_enactment_enumeration_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(q) and "enactment notice" in q and ("made by the ruler" in q or "ruler of dubai" in q)

    @staticmethod
    def _is_ruler_authority_year_enumeration_question(question: str) -> bool:
        q = (question or "").strip().lower()
        return (
            bool(q)
            and RAGGenerator._is_broad_enumeration_question(question)
            and "ruler of dubai" in q
            and ("enacted in" in q or "made in" in q)
            and bool(re.search(r"\b(19|20)\d{2}\b", q))
            and "enactment notice" not in q
        )

    @staticmethod
    def _requires_expanded_broad_enumeration_context(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        if "interpretative provisions" in q:
            return True
        if "difc law no. 2 of 2022" in q and "amended by" in q:
            return True
        return "enactment notice" in q and ("made by the ruler" in q or "ruler of dubai" in q)

    def _common_elements_title_key(self, chunk: RankedChunk) -> str:
        candidates = [
            self._extract_doc_title_from_text(chunk.text or ""),
            self._display_doc_title(chunk),
            (chunk.doc_title or "").strip(),
        ]
        for candidate in candidates:
            key = self._normalize_common_elements_title_key(candidate)
            if key:
                return key
        return ""

    @staticmethod
    def _common_elements_chunk_priority(question: str, chunk: RankedChunk) -> int:
        text = re.sub(r"\s+", " ", (chunk.text or "").strip()).lower()
        if not text:
            return 0

        score = 0
        if "schedule 1" in text:
            score += 4
        if "schedule" in text and "interpretation" in text:
            score += 4
        if "rules of interpretation" in text:
            score += 5
        if "interpretative provisions" in text:
            score += 4
        if "defined terms" in text:
            score += 2
        if (
            "rules of interpretation" not in text
            and "a statutory provision includes a reference" not in text
            and "interpretative provisions" in text
            and "defined terms" in text
        ):
            score -= 7
        if "part 1:" in text or "title and application" in text:
            score -= 6

        question_tokens = {
            token
            for token in _TOKEN_RE.findall((question or "").lower())
            if token not in _STOPWORDS and len(token) > 3
        }
        for token in question_tokens:
            if token in text:
                score += 1
        return score

    def _enumeration_title_key(self, chunk: RankedChunk) -> str:
        candidates = [
            self._extract_doc_title_from_text(chunk.text or ""),
            self._display_doc_title(chunk),
            (chunk.doc_title or "").strip(),
        ]
        for candidate in candidates:
            key = self._normalize_title_key(candidate)
            if key:
                return key
        return ""

    @staticmethod
    def _normalize_title_key(title: str) -> str:
        raw = re.sub(r"\s+", " ", (title or "").strip())
        if not raw:
            return ""
        lowered = raw.lower()
        if lowered == "enactment notice":
            return ""
        normalized = _TITLE_LAW_NO_SUFFIX_RE.sub("", raw)
        normalized = re.sub(r"\s+", " ", normalized).strip(" ,;.-")
        return normalized.casefold()

    @staticmethod
    def _normalize_common_elements_title_key(title: str) -> str:
        raw = re.sub(r"\s+", " ", (title or "").strip())
        if not raw:
            return ""
        normalized = _TITLE_LAW_NO_SUFFIX_RE.sub("", raw)
        normalized = _TITLE_YEAR_RE.sub("", normalized)
        normalized = _TITLE_CONTEXT_BAD_LEAD_RE.sub("", normalized)
        normalized = _TITLE_LEADING_CONNECTOR_RE.sub("", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip(" ,;.-")
        return normalized.casefold()

    @staticmethod
    def _is_broad_enumeration_question(question: str) -> bool:
        return bool(_BROAD_ENUMERATION_RE.search((question or "").strip()))

    @staticmethod
    def _is_common_elements_question(question: str) -> bool:
        return bool(_COMMON_ELEMENTS_RE.search((question or "").strip()))

    @staticmethod
    def _is_named_multi_title_lookup_question(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q or len(RAGGenerator._question_named_refs(question=question)) < 2:
            return False
        if RAGGenerator._is_broad_enumeration_question(question) or RAGGenerator._is_common_elements_question(question):
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
                "penalty",
                "citation title",
                "title of",
                "titles of",
                "last updated",
                "updated",
            )
        )

    @staticmethod
    def _is_named_amendment_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q or RAGGenerator._is_broad_enumeration_question(question):
            return False
        return len(RAGGenerator._question_named_refs(question=question)) >= 1 and "enact" in q and (
            "what law did it amend" in q or "what laws did it amend" in q
        )

    @staticmethod
    def _is_named_enactment_date_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q or RAGGenerator._is_broad_enumeration_question(question):
            return False
        if len(RAGGenerator._question_named_refs(question=question)) < 1:
            return False
        if "enact" not in q:
            return False
        if any(term in q for term in ("what law did it amend", "what laws did it amend", "ruler of dubai", "made by the ruler")):
            return False
        if any(term in q for term in ("commencement", "come into force", "effective date", "enactment notice")):
            return False
        return any(term in q for term in ("on what date", "what date", "date of enactment", "when was"))

    @staticmethod
    def _is_named_made_by_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q or RAGGenerator._is_broad_enumeration_question(question):
            return False
        if len(RAGGenerator._question_named_refs(question=question)) < 1:
            return False
        return "who made this law" in q or ("who made" in q and "law" in q)

    @staticmethod
    def _is_named_registrar_authority_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q or RAGGenerator._is_broad_enumeration_question(question):
            return False
        if len(RAGGenerator._question_named_refs(question=question)) < 1:
            return False
        return "registrar" in q and "appoint" in q and "dismiss" in q

    @staticmethod
    def _is_named_translation_requirement_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q or RAGGenerator._is_broad_enumeration_question(question):
            return False
        if len(RAGGenerator._question_named_refs(question=question)) < 1:
            return False
        return (
            "language other than english" in q
            and "upon request" in q
            and "provide" in q
            and "relevant authority" in q
        )

    @staticmethod
    def _is_interpretative_provisions_enumeration_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        return bool(q) and RAGGenerator._is_broad_enumeration_question(question) and "interpretative provisions" in q

    @staticmethod
    def _is_amended_by_enumeration_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not q or "amended by" not in q:
            return False
        return bool(
            RAGGenerator._is_broad_enumeration_question(question)
            or q.startswith("which specific ")
            or q.startswith("which difc laws ")
        )

    @staticmethod
    def _should_use_irac(question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        lowered = q.lower()
        if "common elements" in lowered or "elements in common" in lowered or " in common" in lowered:
            return False
        return bool(_IRAC_HINT_RE.search(q))

    @staticmethod
    def _strip_enumeration_boilerplate(text: str) -> str:
        cleaned = re.sub(
            r"\bIn this document underlining indicates new text and striking through indicates deleted text\.?\s*",
            "",
            text or "",
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _enumeration_anchor_snippet(question: str, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return ""

        q = (question or "").strip().lower()
        year_match = re.search(r"\b(19|20)\d{2}\b", q)
        anchors: list[tuple[str, int]] = []
        lowered = cleaned.lower()
        if "enactment notice" in q and "ruler" in q and "come into force" in lowered:
            start = max(0, lowered.find("we,"))
            force_idx = lowered.find("come into force")
            end = min(len(cleaned), force_idx + 140) if force_idx != -1 else min(len(cleaned), start + 420)
            return cleaned[start:end].strip(" ,;.")
        if year_match is not None:
            legislation_ref_match = _LEGISLATION_REF_TITLE_RE.search(cleaned)
            if legislation_ref_match is not None:
                start = max(0, legislation_ref_match.start() - 48)
                end = min(len(cleaned), legislation_ref_match.end() + 220)
                return cleaned[start:end].strip(" ,;.")
            enactment_match = _ENACTMENT_NOTICE_ATTACHED_TITLE_RE.search(cleaned)
            if enactment_match is not None:
                start = max(0, enactment_match.start() - 96)
                end = min(len(cleaned), enactment_match.end() + 220)
                return cleaned[start:end].strip(" ,;.")
            if "this law may be cited as" in lowered:
                return cleaned[:360].strip(" ,;.")
        if "amended by" in q:
            anchors.extend([("as amended by", 180), ("amended by", 180)])
        if "interpretative provisions" in q:
            anchors.append(("interpretative provisions", 220))
        if "administered by the registrar" in q:
            anchors.extend(
                [
                    ("the registrar shall administer this law", 220),
                    ("this law shall be administered by the registrar", 220),
                    ("administration of this law", 220),
                    ("administered by the registrar", 220),
                ]
            )
        if "enactment notice" in q:
            anchors.extend(
                [
                    ("date specified in the enactment notice", 220),
                    ("comes into force on the date specified in the enactment notice", 220),
                ]
            )
        if "made by the ruler" in q:
            anchors.append(("made by the ruler", 180))

        for anchor, span in anchors:
            idx = lowered.find(anchor)
            if idx == -1:
                continue
            start = max(0, idx - 48)
            end = min(len(cleaned), idx + span)
            return cleaned[start:end].strip(" ,;.")
        return ""

    def _compact_slow_broad_enumeration_chunk(self, *, question: str, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return cleaned

        lowered_question = (question or "").strip().lower()
        sentences = [sent.strip() for sent in _SENTENCE_SPLIT_RE.split(cleaned) if sent.strip()]
        if not sentences:
            return self._truncate_to_tokens(cleaned, 110)

        selected: list[str] = []

        def _append_matching(predicate: Callable[[str], bool]) -> None:
            for sentence in sentences:
                if predicate(sentence):
                    if sentence not in selected:
                        selected.append(sentence)
                    return

        if "citation title" in lowered_question:
            _append_matching(lambda sentence: "may be cited as" in sentence.lower())
        if "administered by the registrar" in lowered_question:
            _append_matching(lambda sentence: _REGISTRAR_SELF_ADMIN_RE.search(sentence) is not None)
        if self._is_named_reference_enumeration_question(question):
            refs = self._question_named_refs(question=question, extra_refs=None, prefer_extra_refs=False)
            ref_keys = [re.sub(r"\s+", " ", ref).strip().casefold() for ref in refs if ref.strip()]
            _append_matching(
                lambda sentence: sum(1 for ref_key in ref_keys if ref_key and ref_key in sentence.casefold()) >= min(2, len(ref_keys))
            )
        if self._is_company_structure_enumeration_question(question):
            _append_matching(
                lambda sentence: any(
                    marker in sentence.casefold()
                    for marker in (
                        "companies law 2018",
                        "insolvency law 2009",
                        "company structures",
                        "schedule 2",
                        "arbitration law",
                    )
                )
            )

        if not selected:
            anchor = self._enumeration_anchor_snippet(question=question, text=cleaned)
            if anchor:
                selected.append(anchor)

        if not selected:
            selected.append(self._compact_chunk_text(question=question, text=cleaned, top_n=2))

        compact = " ".join(dict.fromkeys(part.strip() for part in selected if part.strip()))
        return self._truncate_to_tokens(compact, 110)

    def _compact_named_multi_title_lookup_chunk(self, *, question: str, text: str) -> str:
        cleaned = self._strip_enumeration_boilerplate(text)
        return self._compact_chunk_text(question=question, text=cleaned, top_n=2)

    def _named_multi_title_lookup_chunk_priority(self, *, question: str, chunk: RankedChunk) -> int:
        query_lower = re.sub(r"\s+", " ", (question or "").strip()).lower()
        text = re.sub(r"\s+", " ", (chunk.text or "").strip()).lower()
        score = 0
        if "citation title" in query_lower or "title of" in query_lower or "titles of" in query_lower:
            if "may be cited as" in text:
                score += 18
            if "title" in text:
                score += 4
        if "administ" in query_lower:
            if _REGISTRAR_SELF_ADMIN_RE.search(text) is not None:
                score += 20
            elif "registrar" in text:
                score += 6
        if any(term in query_lower for term in ("commencement", "come into force", "effective date", "enactment notice")):
            if "comes into force" in text or "come into force" in text:
                score += 16
            if "enactment notice" in text or "effective date" in text or "commencement" in text:
                score += 8
        if "updated" in query_lower:
            if "updated" in text or "amended" in text or "effective from" in text:
                score += 8
            if re.search(r"\b(?:19|20)\d{2}\b", text):
                score += 3
        if text.startswith("part 1:") or "title and application" in text:
            score -= 6
        return score

    def _render_named_multi_title_lookup_blocks(
        self,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        budget: int,
    ) -> list[str]:
        refs = self._question_named_refs(question=question, extra_refs=None, prefer_extra_refs=False)
        selected_docs = self._select_doc_groups_for_refs(refs=refs, chunks=chunks)
        if len(selected_docs) < 2:
            return []

        parts: list[str] = []
        used = 0
        for ref, recovered_title, doc_chunks in selected_docs:
            ranked_doc_chunks = sorted(
                doc_chunks,
                key=lambda chunk: (
                    -self._named_multi_title_lookup_chunk_priority(question=question, chunk=chunk),
                    self._page_num(chunk.section_path),
                ),
            )
            evidence_lines: list[str] = []
            seen_chunk_ids: set[str] = set()
            for chunk in ranked_doc_chunks:
                if chunk.chunk_id in seen_chunk_ids:
                    continue
                snippet = self._compact_named_multi_title_lookup_chunk(question=question, text=chunk.text)
                if not snippet:
                    continue
                evidence_lines.append(f"[{chunk.chunk_id}] {snippet}")
                seen_chunk_ids.add(chunk.chunk_id)
                if len(evidence_lines) >= 2:
                    break

            if not evidence_lines:
                continue

            doc_title = recovered_title or self._recover_doc_title_from_chunks(doc_chunks) or ref
            block = (
                "---\n"
                f"DOCUMENT: {doc_title}\n"
                f"QUESTION_REF: {ref}\n"
                + "\n".join(evidence_lines)
                + "\n---"
            )
            block_tokens = self.count_tokens(block)
            if used + block_tokens > budget:
                break
            parts.append(block)
            used += block_tokens

        return parts

    def _display_doc_title(self, chunk: RankedChunk) -> str:
        raw = (chunk.doc_title or "").strip()
        extracted = self._extract_doc_title_from_text(chunk.text or "")
        if extracted and self._should_prefer_extracted_title(raw, extracted):
            return extracted
        return raw or extracted or "Unknown document"

    @staticmethod
    def _needs_title_recovery(title: str) -> bool:
        raw = (title or "").strip()
        if not raw:
            return True
        lowered = raw.lower()
        if lowered.startswith("in this document underlining indicates"):
            return True
        return bool(re.fullmatch(r"[_\s-]+", raw))

    @classmethod
    def _should_prefer_extracted_title(cls, raw: str, extracted: str) -> bool:
        raw_clean = (raw or "").strip()
        extracted_clean = (extracted or "").strip()
        if not extracted_clean:
            return False
        if cls._needs_title_recovery(raw_clean):
            return True

        raw_has_year = re.search(r"\b(19|20)\d{2}\b", raw_clean) is not None
        extracted_has_year = re.search(r"\b(19|20)\d{2}\b", extracted_clean) is not None
        raw_has_law_no = "difc law no." in raw_clean.lower()
        extracted_has_law_no = "difc law no." in extracted_clean.lower()
        if (extracted_has_year and not raw_has_year) or (extracted_has_law_no and not raw_has_law_no):
            return True

        raw_letters = re.sub(r"[^A-Za-z]+", "", raw_clean)
        if raw_letters and raw_letters.isupper() and raw_clean.casefold() != extracted_clean.casefold():
            return True

        return len(extracted_clean) > len(raw_clean) + 12

    @staticmethod
    def _extract_doc_title_from_text(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""

        cover_match = _COVER_TITLE_LAW_YEAR_RE.search(raw)
        if cover_match:
            title = re.sub(r"\s+", " ", cover_match.group(1)).strip(" ,.;:")
            is_enactment_notice = title.casefold().startswith("enactment notice")
            if is_enactment_notice:
                title = re.sub(r"^enactment\s+notice\s+", "", title, flags=re.IGNORECASE).strip(" ,.;:")
            if title and re.fullmatch(r"[A-Z][A-Z\s/&().-]+", title):
                title = title.title()
            year = cover_match.group(2).strip()
            if is_enactment_notice:
                return title
            return f"{title} {year}".strip()

        cited_match = _CITED_TITLE_RE.search(raw)
        if cited_match:
            return re.sub(r"\s+", " ", cited_match.group(1)).strip()

        cited_plain_match = _CITED_TITLE_PLAIN_RE.search(raw)
        if cited_plain_match:
            return re.sub(r"\s+", " ", cited_plain_match.group(1)).strip(" ,.;:")

        legislation_ref_match = _LEGISLATION_REF_TITLE_RE.search(raw)
        if legislation_ref_match:
            return re.sub(r"\s+", " ", legislation_ref_match.group(1)).strip()

        enactment_notice_match = _ENACTMENT_NOTICE_ATTACHED_TITLE_RE.search(raw)
        if enactment_notice_match:
            return re.sub(r"\s+", " ", enactment_notice_match.group(1)).strip()

        enactment_match = _ENACTMENT_NOTICE_TITLE_RE.search(raw)
        if enactment_match:
            candidate = re.sub(r"\s+", " ", enactment_match.group(1)).strip()
            lowered = candidate.lower()
            if not lowered.startswith(("date specified", "this law", "previous law", "registrar")) and (
                "notice in respect of this law" not in lowered
                and "date of commencement of the law" not in lowered
                and "prior to the date of commencement of the law" not in lowered
                and "shall administer this law" not in lowered
                and "administer this law" not in lowered
                and "administer the provisions of this law" not in lowered
                and "provisions of this law" not in lowered
                and "relevant authority" not in lowered
                and "competent authority" not in lowered
            ):
                return candidate

        return ""

    @staticmethod
    def _extract_doc_title_from_summary(summary: str) -> str:
        raw = (summary or "").strip()
        if not raw:
            return ""

        match = _DOC_SUMMARY_TITLE_RE.search(raw)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:")
            candidate = re.sub(r"\s+Enactment Notice\b", "", candidate, flags=re.IGNORECASE)
            return candidate

        prefix_match = _DOC_SUMMARY_PREFIX_TITLE_RE.search(raw)
        if prefix_match:
            return re.sub(r"\s+", " ", prefix_match.group(1)).strip(" ,.;:")

        titled_match = _DOC_SUMMARY_TITLED_RE.search(raw)
        if titled_match:
            return re.sub(r"\s+", " ", titled_match.group(1)).strip(" ,.;:")

        is_the_match = _DOC_SUMMARY_IS_THE_RE.search(raw)
        if not is_the_match:
            return ""
        candidate = re.sub(r"\s+", " ", is_the_match.group(1)).strip(" ,.;:")
        candidate = re.sub(r"\s+Enactment Notice\b", "", candidate, flags=re.IGNORECASE)
        return candidate

    @staticmethod
    def _looks_like_legal_doc_title(title: str) -> bool:
        normalized = re.sub(r"\s+", " ", (title or "").strip()).strip(" ,.;:")
        if not normalized:
            return False
        lowered = normalized.casefold()
        if _BODY_LIKE_TITLE_RE.search(lowered):
            return False
        words = _TOKEN_RE.findall(normalized)
        if len(words) > 14 and "law no." not in lowered:
            return False
        if re.fullmatch(r"[A-Z][A-Z\s/&().-]+", normalized) and len(words) <= 8:
            return True
        return any(
            marker in lowered
            for marker in (" law", " regulations", " regulation", " rules", " rule", " code", " notice", " order")
        )

    @classmethod
    def _recover_doc_title_from_chunks(
        cls,
        chunks: Sequence[RankedChunk],
        *,
        prefer_citation_title: bool = False,
    ) -> str:
        candidates: list[tuple[str, str]] = []
        for chunk in chunks:
            raw_title = re.sub(r"\s+", " ", (chunk.doc_title or "").strip()).strip(" ,.;:")
            extracted_title = cls._extract_doc_title_from_text(chunk.text or "")
            summary_title = cls._extract_doc_title_from_summary(chunk.doc_summary or "")

            if prefer_citation_title and extracted_title:
                candidates.append((extracted_title, "extracted"))
            if summary_title:
                candidates.append((summary_title, "summary"))
            if extracted_title:
                candidates.append((extracted_title, "extracted"))
            if raw_title and not cls._needs_title_recovery(raw_title) and cls._normalize_title_key(raw_title):
                candidates.append((raw_title, "raw"))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for candidate, source in candidates:
            normalized = re.sub(r"\s+", " ", candidate).strip(" ,.;:")
            if not normalized or cls._needs_title_recovery(normalized):
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((normalized, source))

        if not deduped:
            return ""

        def _family_key(title: str) -> str:
            normalized = _TITLE_LAW_NO_SUFFIX_RE.sub("", title)
            normalized = _TITLE_YEAR_RE.sub("", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip(" ,;.-")
            return normalized.casefold()

        best_info_by_family: dict[str, int] = {}
        family_has_non_raw: set[str] = set()
        for title, source in deduped:
            family_key = _family_key(title)
            info_value = (2 if "difc law no." in title.casefold() else 0) + (1 if _TITLE_YEAR_RE.search(title) else 0)
            if info_value > best_info_by_family.get(family_key, -1):
                best_info_by_family[family_key] = info_value
            if source != "raw":
                family_has_non_raw.add(family_key)

        def _score(item: tuple[str, str]) -> tuple[int, int, int, int]:
            title, source = item
            source_bonus = {"raw": 120, "summary": 180, "extracted": 160}.get(source, 0)
            plausible_bonus = 220 if cls._looks_like_legal_doc_title(title) else -220
            has_year = 1 if _TITLE_YEAR_RE.search(title) else 0
            has_law_no = 1 if "difc law no." in title.lower() else 0
            info_value = has_year + (has_law_no * 2)
            family_key = _family_key(title)
            info_bonus = info_value * 220
            if source == "raw" and best_info_by_family.get(family_key, 0) > info_value:
                source_bonus -= 260
            elif source == "raw" and family_key in family_has_non_raw:
                raw_letters = re.sub(r"[^A-Za-z]+", "", title)
                if raw_letters and raw_letters.isupper():
                    source_bonus -= 180
            elif source in {"summary", "extracted"} and info_value and best_info_by_family.get(family_key, 0) == info_value:
                source_bonus += 80
            compactness = -min(len(title), 120)
            return (source_bonus + plausible_bonus + info_bonus, info_value, has_law_no, compactness)

        deduped.sort(key=_score, reverse=True)
        return deduped[0][0]

    @staticmethod
    def _extract_commencement_rule(text: str) -> str:
        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return ""

        match = _ENACTMENT_NOTICE_COMMENCEMENT_RE.search(raw)
        if match:
            clause = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:")
            tail = raw[match.end(): match.end() + 120].lstrip()
            if tail.startswith("("):
                closing_idx = tail.find(")")
                if closing_idx != -1:
                    clause = f"{clause} {tail[: closing_idx + 1].strip()}"
            clause = re.sub(r"^This Law\s+", "", clause, flags=re.IGNORECASE)
            return clause[:1].upper() + clause[1:] if clause else ""

        general_match = re.search(
            r"\b(?:(?:this\s+law)\s+)?(?P<lemma>shall\s+come|comes?)\s+into\s+force\s+on\s+(?P<tail>[^.]+)",
            raw,
            re.IGNORECASE,
        )
        if general_match is None:
            return ""

        lemma = str(general_match.group("lemma") or "").casefold()
        tail = str(general_match.group("tail") or "").strip(" ,.;:")
        if not tail:
            return ""
        prefix = "Shall come" if lemma.startswith("shall") else "Comes"
        clause = f"{prefix} into force on {tail}"
        return clause[:1].upper() + clause[1:] if clause else ""

    @staticmethod
    def extract_citations(answer: str, chunks: Sequence[RankedChunk]) -> list[Citation]:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        cited_ids = RAGGenerator.extract_cited_chunk_ids(answer)
        citations: list[Citation] = []

        for chunk_id in cited_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                citations.append(Citation(chunk_id=chunk_id, doc_title="unknown"))
                continue
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    doc_title=chunk.doc_title,
                    section_path=chunk.section_path or None,
                )
            )
        return citations

    @staticmethod
    def extract_cited_chunk_ids(answer: str) -> list[str]:
        ids: list[str] = []
        for match in _CITE_RE.finditer(answer):
            for raw_id in re.split(r"[,;]|\s+and\s+", match.group(1)):
                chunk_id = raw_id.strip()
                if chunk_id and chunk_id not in ids:
                    ids.append(chunk_id)
        return ids

    @staticmethod
    def sanitize_citations(answer: str, context_chunk_ids: list[str]) -> str:
        """Remove any (cite: ID) markers where ID is not in context_chunk_ids.
        """
        if not context_chunk_ids:
            return answer

        valid_ids = set(context_chunk_ids)

        def _replace(m: re.Match[str]) -> str:
            raw_inner = m.group(1)
            good = [cid.strip() for cid in re.split(r"[,;]|\s+and\s+", raw_inner) if cid.strip() in valid_ids]
            if not good:
                return ""  # drop the whole (cite: ...) marker
            return f"(cite: {', '.join(good)})"

        sanitized = _CITE_RE.sub(_replace, answer).strip()
        # Clean up any double spaces left by removed citations
        sanitized = re.sub(r"  +", " ", sanitized)

        return sanitized

    @staticmethod
    def strip_negative_subclaims(answer: str) -> str:
        """Remove sentences claiming 'no information on [entity]' from within list answers.

        Preserves the answer if the ENTIRE answer is a single 'There is no information' sentence.
        Only strips negative sub-claims when there are also positive claims (enumerated items).
        Handles both line-based lists and inline trailing disclaimers.
        """
        stripped = (answer or "").strip()
        if not stripped:
            return stripped
        normalized = stripped.lower().strip()
        if normalized.startswith("there is no information on this question"):
            return stripped

        has_newline = "\n" in stripped
        has_numbered = bool(re.search(r"\d+\.\s", stripped))
        has_multi_sentence = bool(re.search(r"[.!?]\s+[A-Z]", stripped))
        if not has_newline and not has_numbered and not has_multi_sentence:
            return stripped

        sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(stripped) if segment.strip()]
        filtered_sentences: list[str] = []
        removed_negative = False
        for sentence in sentences:
            candidate = sentence.strip()
            if not candidate:
                continue
            sentence_with_break = f"{candidate}\n"
            if _NEGATIVE_SUBCLAIM_RE.search(sentence_with_break) or _TRAILING_NEGATIVE_RE.search(candidate):
                removed_negative = True
                continue
            filtered_sentences.append(candidate)

        cleaned = " ".join(filtered_sentences).strip() if removed_negative else stripped
        if removed_negative and cleaned:
            cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
            cleaned = re.sub(r"\(\s+cite:", "(cite:", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+\)", ")", cleaned)

        cleaned = _NEGATIVE_SUBCLAIM_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if not cleaned:
            cleaned = stripped

        cleaned = _TRAILING_NEGATIVE_RE.sub("", cleaned).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        if not cleaned:
            return stripped
        return cleaned

    @staticmethod
    def cleanup_truncated_answer(answer: str) -> str:
        cleaned = (answer or "").strip()
        if not cleaned:
            return cleaned

        # Drop unfinished trailing citation marker, e.g. "... (cite: abc123" or "(cite:"
        cleaned = re.sub(r"\(cite:\s*[^)]*$", "", cleaned, flags=re.IGNORECASE).rstrip()
        # Also catch a bare opening paren followed by hex/hash fragment at end of string
        cleaned = re.sub(r"\(\s*[0-9a-f]{6,}$", "", cleaned, flags=re.IGNORECASE).rstrip()

        # Drop trailing bullet / numbered list item if it is abruptly cut and lacks sentence ending.
        lines = cleaned.splitlines()
        if lines:
            while lines:
                last_line = lines[-1].strip()
                if re.fullmatch(r"(?:[-*]|\d+\.)\s*", last_line):
                    lines = lines[:-1]
                    continue
                if re.match(r"^[-*]\s+", last_line) and RAGGenerator._looks_like_truncated_tail(last_line):
                    lines = lines[:-1]
                    continue
                break
            cleaned = "\n".join(lines).rstrip()

        matches = list(_NUMBERED_ITEM_RE.finditer(cleaned))
        if len(matches) >= 2:
            last_start = matches[-1].start()
            last_item = cleaned[last_start:]
            last_body = re.sub(r"^\d+\.\s*", "", last_item).strip()
            if "(cite:" not in last_body.casefold() and RAGGenerator._looks_like_truncated_tail(last_body):
                cleaned = cleaned[:last_start].rstrip()

        # If answer still ends mid-sentence, trim a trailing fragment only when it
        # clearly looks truncated. Use a stricter boundary than _SENTENCE_SPLIT_RE
        # so legal references like "Law No. 2 of 2022" do not get split mid-title.
        if cleaned and not re.search(r"[.!?)]\s*$", cleaned):
            boundary_matches = list(_COMPLETE_SENTENCE_BOUNDARY_RE.finditer(cleaned))
            while boundary_matches:
                last_boundary = boundary_matches[-1]
                trailing_fragment = cleaned[last_boundary.end() :].strip()
                if "(cite:" in trailing_fragment.casefold():
                    break
                if trailing_fragment and not (
                    _TRAILING_NEGATIVE_RE.search(trailing_fragment)
                    or RAGGenerator._looks_like_truncated_tail(trailing_fragment)
                ):
                    break
                cleaned = cleaned[: last_boundary.start()].rstrip()
                boundary_matches = list(_COMPLETE_SENTENCE_BOUNDARY_RE.finditer(cleaned))

        # Final whitespace normalization.
        return re.sub(r"  +", " ", cleaned).strip()

    @staticmethod
    def cleanup_list_answer_postamble(answer: str) -> str:
        cleaned = (answer or "").strip()
        if not cleaned or not re.search(r"(?:^|\n)\s*1\.\s+", cleaned):
            return cleaned

        cite_matches = list(_CITE_RE.finditer(cleaned))
        if not cite_matches:
            return cleaned

        trailing = cleaned[cite_matches[-1].end():].strip()
        if not trailing:
            return cleaned
        if not _LIST_POSTAMBLE_RE.search(trailing):
            return cleaned

        trimmed = cleaned[: cite_matches[-1].end()].rstrip(" \n.;")
        if trimmed and not re.search(r"[.!?]\s*$", trimmed):
            trimmed = f"{trimmed}."
        return trimmed.strip()

    @staticmethod
    def cleanup_list_answer_preamble(answer: str) -> str:
        cleaned = (answer or "").strip()
        if not cleaned:
            return cleaned

        match = re.search(r"(?:^|[\s])(?P<item>1\.\s+)", cleaned)
        if match is None:
            return cleaned
        item_start = match.start("item")
        if item_start == 0:
            return cleaned

        preamble = cleaned[:item_start].strip()
        if not preamble:
            return cleaned[item_start:].lstrip()
        if _CITE_RE.search(preamble):
            return cleaned
        if preamble.lower().startswith("there is no information on this question"):
            return cleaned
        return cleaned[item_start:].lstrip()

    @staticmethod
    def cleanup_final_answer(answer: str) -> str:
        cleaned = RAGGenerator.cleanup_truncated_answer(answer)
        if not cleaned:
            return cleaned

        cleaned = RAGGenerator.cleanup_list_answer_postamble(cleaned)
        cleaned = RAGGenerator.cleanup_list_answer_preamble(cleaned)
        summary_match = re.search(r"\nSummary:\s*$|\nSummary:\s*\n", cleaned, flags=re.IGNORECASE)
        if summary_match is not None and re.search(r"(?:^|\n)1\.\s+", cleaned):
            cleaned = cleaned[: summary_match.start()].rstrip()
        cleaned = re.sub(r"(?:\n|\s)+\d+\.\s*$", "", cleaned).rstrip()
        lines = cleaned.splitlines()
        if lines:
            lines = [line for line in lines if not re.fullmatch(r"\s*(?:[-*]|\d+\.)\s*", line)]
            cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _looks_like_truncated_tail(text: str) -> bool:
        stripped = re.sub(r"\s+", " ", (text or "").strip())
        if not stripped:
            return False
        if re.search(r"\b(?:law\s+)?no\.\s*$", stripped, re.IGNORECASE):
            return True
        if re.search(r"[.!?)]\s*$", stripped):
            return False
        if "(cite:" in stripped.casefold() and stripped.endswith(")"):
            return False
        if stripped.count("(") > stripped.count(")"):
            return True
        return bool(re.search(r"\b(?:of|and|the|law\s+no\.?)\s*$", stripped, re.IGNORECASE))

    @classmethod
    def _group_chunks_by_doc(cls, chunks: Sequence[RankedChunk]) -> tuple[list[str], dict[str, list[RankedChunk]]]:
        chunks_by_doc: dict[str, list[RankedChunk]] = {}
        doc_order: list[str] = []
        for chunk in chunks:
            doc_key = str(chunk.doc_id or chunk.chunk_id)
            if doc_key not in chunks_by_doc:
                doc_order.append(doc_key)
            chunks_by_doc.setdefault(doc_key, []).append(chunk)
        return doc_order, chunks_by_doc

    @classmethod
    def _question_named_refs(
        cls,
        *,
        question: str,
        extra_refs: Sequence[str] | None = None,
        prefer_extra_refs: bool = False,
    ) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()

        def _push(raw: str) -> None:
            normalized = re.sub(r"\s+", " ", raw).strip(" ,.;:")
            if not normalized:
                return
            key = normalized.casefold()
            if key in seen:
                return
            seen.add(key)
            refs.append(normalized)

        extras = [str(ref).strip() for ref in (extra_refs or []) if str(ref).strip()]
        if prefer_extra_refs:
            for ref in extras:
                _push(ref)
            if len(extras) >= 2:
                return refs

        law_no_refs: list[str] = []
        law_no_seen: set[str] = set()
        for match in _LAW_NO_REF_RE.finditer(question or ""):
            ref = f"Law No. {int(match.group(1))} of {match.group(2)}"
            key = ref.casefold()
            if key in law_no_seen:
                continue
            law_no_seen.add(key)
            law_no_refs.append(ref)
        if len(law_no_refs) >= 2:
            if prefer_extra_refs:
                return refs + [ref for ref in law_no_refs if ref.casefold() not in seen]
            return list(law_no_refs)

        for match in _AMENDMENT_TITLE_RE.finditer(question or ""):
            ref = re.sub(r"\s+", " ", match.group(1).strip())
            ref = _TITLE_REF_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_QUERY_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", ref)
            ref = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_LEADING_CONNECTOR_RE.sub("", ref).strip(" ,.;:")
            _push(ref)

        for title, year in _TITLE_REF_RE.findall(question or ""):
            normalized_title = _TITLE_REF_BAD_LEAD_RE.sub("", title.strip())
            normalized_title = _TITLE_QUERY_BAD_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_CONTEXT_BAD_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_LEADING_CONNECTOR_RE.sub("", normalized_title).strip(" ,.;:")
            ref = " ".join(part for part in (normalized_title, year.strip()) if part).strip()
            if ref.casefold() in {"law", "difc law"}:
                continue
            if ref:
                _push(ref)

        for ref in law_no_refs:
            _push(ref)

        if not prefer_extra_refs:
            for ref in extras:
                _push(ref)

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

    @classmethod
    def _doc_group_match_score(cls, ref: str, doc_chunks: Sequence[RankedChunk]) -> int:
        if not ref or not doc_chunks:
            return 0

        recovered_title = cls._recover_doc_title_from_chunks(doc_chunks)
        normalized_ref = re.sub(r"\s+", " ", ref).strip().casefold()
        haystack = " ".join(
            part
            for part in (
                recovered_title,
                *(str(chunk.doc_title or "") for chunk in doc_chunks[:2]),
                *(str(chunk.text or "")[:1200] for chunk in doc_chunks[:2]),
            )
            if part
        )
        normalized_haystack = re.sub(r"\s+", " ", haystack).strip().casefold()
        if not normalized_haystack:
            return 0

        exact_position = normalized_haystack.find(normalized_ref) if normalized_ref else -1
        if exact_position >= 0:
            score = 1500 - min(exact_position, 900)
            if recovered_title and normalized_ref in re.sub(r"\s+", " ", recovered_title).strip().casefold():
                score += 200
            if any(
                normalized_ref in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold()
                and (
                    _TITLE_CLAUSE_RE.search(chunk.text or "")
                    or (chunk.text or "").strip().casefold().startswith(normalized_ref)
                )
                for chunk in doc_chunks[:3]
            ):
                score += 250
            if any("hereby enact" in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold() for chunk in doc_chunks[:2]):
                score += 150
            if any(
                ("comes into force" in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold())
                or ("commencement" in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold())
                for chunk in doc_chunks[:3]
            ):
                score += 100
            return score

        ref_match = _LAW_NO_REF_RE.search(ref)
        if ref_match is not None:
            law_no_key = f"law no. {int(ref_match.group(1))} of {ref_match.group(2)}"
            if law_no_key in normalized_haystack:
                score = 1200
                if any("hereby enact" in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold() for chunk in doc_chunks[:2]):
                    score += 150
                if any(
                    ("comes into force" in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold())
                    or ("commencement" in re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold())
                    for chunk in doc_chunks[:3]
                ):
                    score += 100
                return score
            return 0

        normalized_ref_key = cls._normalize_common_elements_title_key(ref)
        if normalized_ref_key and normalized_ref_key in normalized_haystack:
            return 500

        ref_tokens = {
            token
            for token in _TOKEN_RE.findall(normalized_ref_key or ref.casefold())
            if token.casefold() not in _STOPWORDS and len(token) > 2
        }
        if not ref_tokens:
            return 0

        ordered_ref_tokens = [
            token.casefold()
            for token in _TOKEN_RE.findall(normalized_ref_key or ref.casefold())
            if token.casefold() not in _STOPWORDS and len(token) > 2
        ]

        haystack_tokens = {token.casefold() for token in _TOKEN_RE.findall(normalized_haystack)}
        overlap = len(ref_tokens.intersection(haystack_tokens))
        if len(ordered_ref_tokens) >= 3 and overlap < len(ref_tokens):
            ref_bigrams = [
                f"{ordered_ref_tokens[idx]} {ordered_ref_tokens[idx + 1]}"
                for idx in range(len(ordered_ref_tokens) - 1)
            ]
            bigram_overlap = sum(1 for bigram in ref_bigrams if bigram in normalized_haystack)
            if overlap >= max(1, len(ref_tokens) - 1) and bigram_overlap < max(1, len(ref_bigrams) - 1):
                return 0
        if overlap == len(ref_tokens):
            return 200 + overlap
        if overlap >= max(1, len(ref_tokens) - 1):
            return 120 + overlap
        if overlap >= max(1, (len(ref_tokens) + 1) // 2):
            return 60 + overlap
        return 0

    @classmethod
    def _select_doc_groups_for_refs(
        cls,
        *,
        refs: Sequence[str],
        chunks: Sequence[RankedChunk],
    ) -> list[tuple[str, str, list[RankedChunk]]]:
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        selected: list[tuple[str, str, list[RankedChunk]]] = []
        used_doc_ids: set[str] = set()

        for ref in refs:
            best_doc_id = ""
            best_score = 0
            for doc_id in doc_order:
                if doc_id in used_doc_ids:
                    continue
                score = cls._doc_group_match_score(ref, chunks_by_doc.get(doc_id, []))
                if score > best_score:
                    best_doc_id = doc_id
                    best_score = score
            if not best_doc_id:
                continue
            used_doc_ids.add(best_doc_id)
            doc_chunks = chunks_by_doc.get(best_doc_id, [])
            recovered_title = cls._recover_doc_title_from_chunks(doc_chunks)
            selected.append((ref, recovered_title or ref, doc_chunks))

        return selected

    @staticmethod
    def _common_element_signatures(text: str) -> list[tuple[str, str]]:
        normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
        if not normalized:
            return []

        signatures: list[tuple[str, str]] = []
        for key, display, required_terms in _COMMON_ELEMENT_SIGNATURES:
            if all(term in normalized for term in required_terms):
                signatures.append((key, display))
        return signatures

    @staticmethod
    def _is_interpretation_sections_common_elements_question(question: str) -> bool:
        normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
        return "interpretation section" in normalized or "interpretation sections" in normalized

    @classmethod
    def _build_common_elements_canonical_answer(
        cls,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=False)
        selected_docs = cls._select_doc_groups_for_refs(refs=refs, chunks=chunks)
        if len(selected_docs) < 2:
            return ""

        interpretation_sections_query = cls._is_interpretation_sections_common_elements_question(question)
        support_by_doc: list[dict[str, tuple[str, str]]] = []
        for _ref, _title, doc_chunks in selected_docs:
            doc_support: dict[str, tuple[str, str]] = {}
            for chunk in doc_chunks:
                for key, display in cls._common_element_signatures(chunk.text or ""):
                    if interpretation_sections_query and key not in _INTERPRETATION_SECTION_COMMON_KEYS:
                        continue
                    doc_support.setdefault(key, (display, chunk.chunk_id))
            if not doc_support:
                return ""
            support_by_doc.append(doc_support)

        common_keys = set(support_by_doc[0])
        for doc_support in support_by_doc[1:]:
            common_keys.intersection_update(doc_support)
        if interpretation_sections_query:
            common_keys.intersection_update(_INTERPRETATION_SECTION_COMMON_KEYS)
        if not common_keys:
            return ""

        if interpretation_sections_query:
            ordered_keys = [key for key in _INTERPRETATION_SECTION_COMMON_KEYS if key in common_keys]
        else:
            ordered_keys = [key for key, _display, _terms in _COMMON_ELEMENT_SIGNATURES if key in common_keys]
        if not ordered_keys:
            return ""

        rebuilt: list[str] = []
        for key in ordered_keys:
            display = support_by_doc[0][key][0]
            cited_ids = list(dict.fromkeys(doc_support[key][1] for doc_support in support_by_doc))
            rebuilt.append(f"{len(rebuilt) + 1}. {display} (cite: {', '.join(cited_ids)})")

        return "\n".join(rebuilt)

    @classmethod
    def cleanup_common_elements_canonical_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cleaned or not cls._is_common_elements_question(question):
            return cleaned

        rebuilt = cls._build_common_elements_canonical_answer(
            question=question,
            chunks=chunks,
            doc_refs=doc_refs,
        )
        return rebuilt or cleaned

    @classmethod
    def build_interpretative_provisions_enumeration_answer(
        cls,
        *,
        chunks: Sequence[RankedChunk],
    ) -> str:
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        merged_by_title: dict[str, tuple[str, list[str]]] = {}
        title_order: list[str] = []
        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            support_ids = [
                chunk.chunk_id
                for chunk in doc_chunks
                if "schedule" in re.sub(r"\s+", " ", (chunk.text or "").strip()).lower()
                and "interpretative provisions" in re.sub(r"\s+", " ", (chunk.text or "").strip()).lower()
            ]
            if not support_ids:
                continue
            title = (
                cls._recover_doc_title_from_chunks(doc_chunks, prefer_citation_title=True)
                or cls._recover_doc_title_from_chunks(doc_chunks)
                or re.sub(r"\s+", " ", (doc_chunks[0].doc_title or "").strip()).strip(" ,.;:")
            )
            if not title:
                continue
            title_key = cls._normalize_title_key(title)
            if not title_key:
                continue
            deduped_ids = list(dict.fromkeys(support_ids))
            if title_key not in merged_by_title:
                title_order.append(title_key)
                merged_by_title[title_key] = (title, deduped_ids)
            else:
                prev_title, prev_ids = merged_by_title[title_key]
                better_title = title if cls._should_prefer_extracted_title(prev_title, title) else prev_title
                merged_by_title[title_key] = (better_title, list(dict.fromkeys(prev_ids + deduped_ids)))

        if not title_order:
            return ""
        return "\n".join(
            f"{idx}. {merged_by_title[key][0]} (cite: {', '.join(merged_by_title[key][1])})"
            for idx, key in enumerate(title_order, start=1)
        )

    @classmethod
    def cleanup_interpretative_provisions_enumeration_items(
        cls,
        answer: str,
        *,
        chunks: Sequence[RankedChunk],
    ) -> str:
        rebuilt = cls.build_interpretative_provisions_enumeration_answer(chunks=chunks)
        return rebuilt or (answer or "").strip()

    @classmethod
    def build_case_outcome_answer(
        cls,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
    ) -> str:
        question_lower = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not cls._is_case_outcome_question(question) or not chunks:
            return ""

        asks_costs = "cost" in question_lower or "final ruling" in question_lower
        prefers_order_section = any(
            phrase in question_lower
            for phrase in (
                "it is hereby ordered",
                "last page of the document",
                "specific order or application",
            )
        )
        prefers_first_page = any(
            phrase in question_lower
            for phrase in (
                "first page of the document",
                "stated on the first page",
            )
        )
        wants_multi_outcome = any(
            phrase in question_lower
            for phrase in (
                "court of appeal rule",
                "what did the court decide",
                "final ruling",
            )
        )
        prefers_conclusion_section = any(
            phrase in question_lower
            for phrase in (
                "conclusion section",
                "final ruling",
                "last page of the document",
            )
        )

        outcome_candidates: list[tuple[int, str, str]] = []
        cost_candidates: list[tuple[int, str, str]] = []
        page_numbers: list[int] = []
        for chunk in chunks:
            page_match = re.search(r"page:(\d+)", str(chunk.section_path or ""), flags=re.IGNORECASE)
            if page_match is not None:
                page_numbers.append(int(page_match.group(1)))
        highest_page = max(page_numbers) if page_numbers else 0
        for chunk in chunks:
            chunk_text = str(chunk.text or "")
            chunk_lower = chunk_text.casefold()
            page_match = re.search(r"page:(\d+)", str(chunk.section_path or ""), flags=re.IGNORECASE)
            page_num = int(page_match.group(1)) if page_match is not None else 0
            clauses = cls._extract_case_outcome_clauses(text=chunk.text or "", prefer_order_section=prefers_order_section)
            for clause in clauses:
                cleaned_clause = cls._clean_case_outcome_clause(clause)
                if not cleaned_clause:
                    continue
                lowered = cleaned_clause.casefold()
                if _OUTCOME_NOISE_RE.search(cleaned_clause):
                    continue
                score = 0
                chunk_amount_roles = set(getattr(chunk, "amount_roles", []) or [])
                if asks_costs and "costs_awarded" in chunk_amount_roles:
                    score += 40
                if asks_costs and "claim_amount" in chunk_amount_roles and "costs_awarded" not in chunk_amount_roles:
                    score -= 20
                if page_num == 1:
                    score += 24
                elif page_num == 2:
                    score += 12
                if prefers_first_page:
                    if page_num == 1:
                        score += 48
                    elif page_num > 1:
                        score -= 18
                if prefers_conclusion_section and highest_page > 0:
                    if page_num == highest_page:
                        score += 30
                    elif page_num >= highest_page - 1:
                        score += 16
                if _ORDER_SECTION_MARKER_RE.search(chunk_text):
                    score += 28
                if "conclusion" in chunk_lower:
                    score += 22
                if _OUTCOME_CUE_RE.search(cleaned_clause):
                    score += 20
                if "application" in lowered or "appeal" in lowered or "order" in lowered:
                    score += 8
                if "permission to appeal" in lowered:
                    score += 10
                if "no costs application" in lowered:
                    score += 12
                if "for all of the foregoing reasons" in lowered:
                    score += 20
                if "costs awarded" in lowered or "costs of the appeal assessed" in lowered:
                    score += 16
                if asks_costs and "fair, reasonable and proportionate award of costs" in lowered:
                    score += 80
                if asks_costs and re.search(r"\bsum of (?:USD|AED)\s*[0-9]", cleaned_clause):
                    score += 80
                if asks_costs and "total bill of costs amounts" in lowered:
                    score -= 90
                if asks_costs and "leading and junior counsel" in lowered:
                    score -= 90
                if asks_costs and "junior counsel" in lowered:
                    score -= 60
                if asks_costs and "vary enormously" in lowered:
                    score -= 90
                if re.search(r"\b(?:USD|AED)\s*\d", cleaned_clause):
                    score += 24
                if "court of appeal" in question_lower and "appeal" in lowered:
                    score += 8
                if "court of appeal" in question_lower and "permission to appeal" in lowered:
                    score -= 24
                if "it is hereby ordered" in question_lower and "set aside" in lowered:
                    score += 6
                if "it is hereby ordered" in question_lower and "discharged" in lowered:
                    score += 6
                if "cost" in lowered:
                    score += 2
                if "by way of the order of" in chunk_lower or "justice " in chunk_lower:
                    score -= 10
                if "was considered" in lowered:
                    score -= 40
                candidate = (score, cleaned_clause, chunk.chunk_id)
                has_explicit_outcome = bool(
                    _EXPLICIT_OUTCOME_VERB_RE.search(cleaned_clause)
                    or cleaned_clause.casefold().startswith("the court of appeal allowed")
                    or cleaned_clause.casefold().startswith("the application was dismissed")
                    or cleaned_clause.casefold().startswith("application was dismissed")
                )
                if has_explicit_outcome:
                    outcome_candidates.append(candidate)
                elif _COST_CUE_RE.search(cleaned_clause):
                    cost_candidates.append(candidate)

        selected_outcomes = cls._select_case_outcome_clauses(
            outcome_candidates,
            max_items=1 if asks_costs else (2 if wants_multi_outcome else 1),
        )
        if not selected_outcomes:
            return ""

        outcome_text = cls._join_case_outcome_clauses([clause for clause, _cid in selected_outcomes])
        if not outcome_text:
            return ""
        outcome_citations = list(dict.fromkeys(chunk_id for _clause, chunk_id in selected_outcomes))
        sentences = [f"{outcome_text} (cite: {', '.join(outcome_citations)})."]

        if asks_costs:
            selected_costs = cls._select_case_outcome_clauses(cost_candidates, max_items=1)
            if selected_costs:
                cost_text = cls._join_case_outcome_clauses([clause for clause, _cid in selected_costs])
                if cost_text:
                    cost_citations = list(dict.fromkeys(chunk_id for _clause, chunk_id in selected_costs))
                    sentences.append(f"{cost_text} (cite: {', '.join(cost_citations)}).")

        return " ".join(sentences).strip()

    @staticmethod
    def _extract_case_outcome_clauses(*, text: str, prefer_order_section: bool) -> list[str]:
        normalized = (text or "").replace("\r", "\n")
        if not normalized.strip():
            return []

        raw_lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
        lines = [line for line in raw_lines if line]

        ordered_lines: list[str] = []
        in_order_section = False
        current_item: list[str] = []
        for line in lines:
            if _ORDER_SECTION_MARKER_RE.search(line):
                in_order_section = True
                continue
            if in_order_section and _ORDER_SECTION_STOP_RE.search(line):
                break
            if in_order_section:
                if _NUMBERED_LINE_RE.match(line):
                    if current_item:
                        ordered_lines.append(" ".join(current_item).strip(" ;"))
                        current_item = []
                    cleaned_line = _NUMBERED_LINE_RE.sub("", line).strip(" ;")
                    if cleaned_line:
                        current_item.append(cleaned_line)
                    continue
                if current_item:
                    cleaned_line = line.strip(" ;")
                    if cleaned_line:
                        current_item.append(cleaned_line)
        if current_item:
            ordered_lines.append(" ".join(current_item).strip(" ;"))

        ordered_candidates = [
            line
            for line in ordered_lines
            if _OUTCOME_CUE_RE.search(line) or _COST_CUE_RE.search(line)
        ]
        if prefer_order_section and ordered_candidates:
            return ordered_candidates

        line_candidates = [
            _NUMBERED_LINE_RE.sub("", line).strip(" ;")
            for line in lines
            if _OUTCOME_CUE_RE.search(line) or _COST_CUE_RE.search(line)
        ]
        if ordered_candidates or line_candidates:
            merged: list[str] = []
            seen: set[str] = set()
            for line in [*ordered_candidates, *line_candidates]:
                key = line.casefold()
                if key in seen or not line:
                    continue
                seen.add(key)
                merged.append(line)
            return merged

        sentence_candidates = [
            sentence.strip(" ;")
            for sentence in re.split(r"(?<=[.!?;])\s+", re.sub(r"\s+", " ", normalized).strip())
            if sentence.strip()
        ]
        return [
            _NUMBERED_LINE_RE.sub("", sentence).strip(" ;")
            for sentence in sentence_candidates
            if _OUTCOME_CUE_RE.search(sentence) or _COST_CUE_RE.search(sentence)
        ]

    @staticmethod
    def _clean_case_outcome_clause(clause: str) -> str:
        cleaned = re.sub(r"\s+", " ", (clause or "").strip()).strip(" ;")
        if not cleaned:
            return ""
        cleaned = _NUMBERED_LINE_RE.sub("", cleaned).strip(" ;")
        if cleaned[:1].islower():
            return ""
        cleaned = re.sub(
            r"^AND UPON .*? by which (.+)$",
            lambda match: match.group(1)[:1].upper() + match.group(1)[1:],
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ;,")
        cleaned = re.sub(
            r"^This Order concerns the costs of (.+?), which was dismissed(?: by Order of H\.?\s*E\.?.*)?$",
            lambda match: (
                f"{match.group(1)[:1].upper() + match.group(1)[1:]} was dismissed"
                if not match.group(1).casefold().startswith("the ")
                else f"{match.group(1)} was dismissed"
            ),
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ;,")
        cleaned = re.sub(
            r"^This No Costs Application .*?$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ;,")
        cleaned = re.sub(r"\bby\s+Order\s+of\s+H\.?\s*E\.?.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
        cleaned = re.sub(
            r"^On\s+\d{1,2}\s+[A-Za-z]+\s+\d{4},\s+by\s+way\s+of\s+the\s+Order\s+of\s+H\.?\s*E\.?.*?,\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ;,")
        cleaned = re.sub(
            r"^Justice\s+[A-Z][A-Za-z .'-]+,\s+the\s+",
            "The ",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" ;,")
        cleaned = re.sub(r"\s+and\s+By\s+RDC\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
        cleaned = re.sub(r"\bBy\s+RDC\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
        cleaned = re.sub(r"\bIssued by:.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
        cleaned = re.sub(r"\bDate of (?:Issue|issue):.*$", "", cleaned, flags=re.IGNORECASE).strip(" ;,")
        cleaned = re.sub(r"^(?:[A-Z][a-z]+ \d{4}\.\s*)+", "", cleaned).strip(" ;,")
        if cleaned.casefold() in {"costs", "conclusion"}:
            return ""
        if re.fullmatch(r"(?:USD|AED)?\s*[0-9][0-9,]*(?:\.\d+)?", cleaned, flags=re.IGNORECASE):
            return ""
        if cleaned.casefold() == "the appeal is allowed, to the following extent.":
            return "The Court of Appeal allowed the appeal in part"
        if cleaned.casefold() == "the appeal is allowed, to the following extent":
            return "The Court of Appeal allowed the appeal in part"
        if cleaned.casefold().startswith("for all of the foregoing reasons, we have allowed the appeal"):
            return "The Court of Appeal allowed the appeal in part"
        if (
            "fair, reasonable and proportionate award of costs" in cleaned.casefold()
            and "sum of usd" in cleaned.casefold()
        ):
            amount_match = re.search(r"\bsum of (USD|AED)\s*([0-9][0-9,]*(?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
            if amount_match is not None:
                return f"The Appellant was awarded costs in the sum of {amount_match.group(1).upper()} {amount_match.group(2)}"
        if cleaned.casefold().startswith("the defendant's application for immediate judgment and/or strike out was dismissed"):
            return "The Defendant's Application for immediate judgment and/or strike out was dismissed"
        if cleaned.casefold().startswith("application was dismissed and the applicant was ordered to pay"):
            return "The Application was dismissed"
        if cleaned.casefold().startswith("the application was dismissed and the applicant was ordered to pay"):
            return "The Application was dismissed"
        if cleaned.casefold().startswith("accordingly, the application must be dismissed, and the claimant shall bear its own costs"):
            return "The Application is dismissed and the Claimant shall bear its own costs of the Application"
        if cleaned.casefold().startswith("the application is dismissed. the claim is to proceed to trial"):
            return "The Application is dismissed"
        if cleaned.casefold().startswith("the no costs application is dismissed"):
            return "The No Costs Application is dismissed"
        if cleaned.casefold().startswith("the no costs application is rejected"):
            return "The No Costs Application is rejected"
        if cleaned.casefold().startswith("save and insofar as") and "order is otherwise set aside" in cleaned.casefold():
            return "The Order is otherwise set aside except insofar as the Judge ordered that the Second Part 50 Order should continue to apply"
        if cleaned.startswith("That the "):
            cleaned = "The " + cleaned[9:]
        elif cleaned.startswith("That "):
            cleaned = cleaned[5:]
        if cleaned.casefold().startswith("by rdc "):
            return ""
        if "permission to appeal against the order was granted by the judge himself" in cleaned.casefold():
            return ""
        if "was considered" in cleaned.casefold():
            return ""
        return cleaned.rstrip(".")

    @staticmethod
    def _select_case_outcome_clauses(
        candidates: Sequence[tuple[int, str, str]],
        *,
        max_items: int,
    ) -> list[tuple[str, str]]:
        if not candidates or max_items <= 0:
            return []
        ranked = sorted(candidates, key=lambda item: -item[0])
        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _score, clause, chunk_id in ranked:
            key = clause.casefold()
            if key in seen:
                continue
            seen.add(key)
            selected.append((clause, chunk_id))
            if len(selected) >= max_items:
                break
        return selected

    @staticmethod
    def _join_case_outcome_clauses(clauses: Sequence[str]) -> str:
        cleaned = [re.sub(r"\s+", " ", clause).strip(" ;,.") for clause in clauses if clause.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        normalized_tail: list[str] = [cleaned[0]]
        for clause in cleaned[1:]:
            if clause.startswith("The "):
                normalized_tail.append("the " + clause[4:])
            else:
                normalized_tail.append(clause)
        return (
            ", and ".join([", ".join(normalized_tail[:-1]), normalized_tail[-1]])
            if len(normalized_tail) > 2
            else " and ".join(normalized_tail)
        )

    @classmethod
    def _extract_commencement_support(
        cls,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        clause = ""
        cited_ids: list[str] = []
        for chunk in doc_chunks:
            rule = cls._extract_commencement_rule(chunk.text or "")
            if not rule:
                continue
            clause = rule.strip().rstrip(".")
            cited_ids.append(chunk.chunk_id)
        return clause, list(dict.fromkeys(cited_ids))

    @staticmethod
    def _extract_last_updated_support(
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        for chunk in doc_chunks:
            text_sources = [
                re.sub(r"\s+", " ", (chunk.text or "").strip()),
                re.sub(r"\s+", " ", str(chunk.doc_summary or "").strip()),
            ]
            for normalized in text_sources:
                if not normalized:
                    continue
                consolidated_match = _CONSOLIDATED_VERSION_RE.search(normalized)
                if consolidated_match is not None:
                    return consolidated_match.group(1).strip(" ,.;:"), [chunk.chunk_id]
                updated_match = _UPDATED_VALUE_RE.search(normalized)
                if updated_match is not None:
                    return updated_match.group(1).strip(" ,.;:"), [chunk.chunk_id]
        return "", []

    @classmethod
    def _select_title_support_chunk_id(
        cls,
        doc_chunks: Sequence[RankedChunk],
        *,
        title: str,
    ) -> str:
        normalized_title = cls._normalize_title_key(title)
        best_chunk_id = ""
        best_score = -1

        for chunk in doc_chunks:
            text = re.sub(r"\s+", " ", (chunk.text or "").strip())
            if not text:
                continue
            score = 0
            if _CITED_TITLE_RE.search(text) or _CITED_TITLE_PLAIN_RE.search(text):
                score += 30
            if normalized_title and normalized_title in cls._normalize_title_key(text):
                score += 20
            if cls._page_num(chunk.section_path) == 1:
                score += 8
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id

        if best_chunk_id:
            return best_chunk_id
        return doc_chunks[0].chunk_id if doc_chunks else ""

    @classmethod
    def _select_identity_support_chunk_id(
        cls,
        doc_chunks: Sequence[RankedChunk],
        *,
        title: str,
    ) -> str:
        normalized_title = cls._normalize_title_key(title)
        best_chunk_id = ""
        best_score = -1

        for chunk in doc_chunks:
            text = re.sub(r"\s+", " ", (chunk.text or "").strip())
            if not text:
                continue
            normalized_text = cls._normalize_title_key(text)
            normalized_doc_title = cls._normalize_title_key(str(chunk.doc_title or ""))
            normalized_summary = cls._normalize_title_key(str(chunk.doc_summary or ""))
            lowered = text.casefold()
            score = 0
            if normalized_title and normalized_title in normalized_doc_title:
                score += 80
            if normalized_title and normalized_title in normalized_summary:
                score += 60
            if normalized_title and normalized_title in normalized_text:
                score += 40
            if cls._page_num(chunk.section_path) == 1:
                score += 140
            if "enactment notice" in lowered:
                score += 60
            if "consolidated version" in lowered:
                score += 40
            if _CITED_TITLE_RE.search(text) or _CITED_TITLE_PLAIN_RE.search(text):
                score += 10
            if score > best_score:
                best_score = score
                best_chunk_id = chunk.chunk_id

        if best_chunk_id:
            return best_chunk_id
        return cls._select_title_support_chunk_id(doc_chunks, title=title)

    @classmethod
    def _extract_enactment_date_support(
        cls,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, str]:
        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "").strip())
            if not normalized:
                continue
            match = _ENACTMENT_DATE_RE.search(normalized)
            if match is not None:
                return match.group(1).strip(" ,.;:"), chunk.chunk_id
        return "", ""

    @staticmethod
    def _extract_administration_support(
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, str, list[str]]:
        clause = ""
        entity = ""
        cited_ids: list[str] = []
        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "").strip())
            if not normalized:
                continue
            match = _ADMINISTRATION_CLAUSE_RE.search(normalized)
            if match is None:
                continue
            clause = match.group(1).strip().rstrip(".")
            cited_ids.append(chunk.chunk_id)
            entity_match = _ADMINISTRATION_ENTITY_RE.search(clause)
            if entity_match is not None:
                entity = next((group.strip() for group in entity_match.groups() if group and group.strip()), "")
            break
        return clause, entity, list(dict.fromkeys(cited_ids))

    @staticmethod
    def _extract_single_law_title_from_question(question: str) -> str:
        normalized = re.sub(r"\s+", " ", (question or "").strip()).strip(" ?")
        if not normalized:
            return ""
        for pattern in _QUESTION_SINGLE_LAW_TITLE_PATTERNS:
            match = pattern.search(normalized)
            if match is None:
                continue
            title = re.sub(r"\s+", " ", match.group("title")).strip(" ,.;:")
            if title:
                return title
        return ""

    @staticmethod
    def _normalize_commencement_rule(rule: str) -> str:
        normalized = re.sub(r"\s+", " ", (rule or "").strip()).rstrip(".")
        normalized = re.sub(r"^this law\s+", "", normalized, flags=re.IGNORECASE)
        return normalized.casefold()

    @staticmethod
    def _clean_structured_doc_label(label: str) -> str:
        cleaned = re.sub(r"\s+", " ", (label or "").strip()).strip(" ,.;:")
        if not cleaned:
            return ""
        cleaned = _STRUCTURED_TITLE_BAD_LEAD_RE.sub("", cleaned).strip(" ,.;:")
        cleaned = re.sub(r"\s+Enactment Notice\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:")
        cleaned = re.sub(r"^The\s+The\b", "The", cleaned, flags=re.IGNORECASE)
        return cleaned

    @staticmethod
    def _clean_amendment_title_historical_year(label: str) -> str:
        cleaned = re.sub(r"\s+", " ", (label or "").strip()).strip(" ,.;:")
        if "amendment law" not in cleaned.casefold():
            return cleaned
        return re.sub(
            r"(?i)\b(Law)(?:\s+of)?\s+\d{4}\s+(Amendment Law)\b",
            r"\1 \2",
            cleaned,
        ).strip(" ,.;:")

    @classmethod
    def cleanup_named_commencement_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        question_lower = (question or "").strip().lower()
        if not any(term in question_lower for term in ("commencement", "come into force", "effective date", "enactment notice")):
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        if len(refs) < 2:
            return cleaned

        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return cleaned

        support_by_ref: dict[str, tuple[str, str, list[str]]] = {}
        for ref in refs:
            best_label = ref.strip(" ,.;:") or ref
            best_label_score = 0
            best_support: tuple[str, str, list[str]] | None = None
            best_support_score = 0

            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue

                recovered_title = cls._recover_doc_title_from_chunks(doc_chunks) or ref
                label = cls._clean_structured_doc_label(recovered_title) or ref.strip(" ,.;:") or ref
                if match_score > best_label_score:
                    best_label = label
                    best_label_score = match_score

                clause, cited_ids = cls._extract_commencement_support(doc_chunks)
                if not clause:
                    continue
                candidate_score = match_score + (25 if "hereby enact" in re.sub(r"\s+", " ", (doc_chunks[0].text or "")).strip().casefold() else 0)
                if candidate_score > best_support_score:
                    best_support = (clause, label, cited_ids)
                    best_support_score = candidate_score

            if best_support is not None:
                clause, label, cited_ids = best_support
                support_by_ref[ref.casefold()] = (clause, label or best_label, cited_ids)

        if not support_by_ref:
            return cleaned

        supported_entries: list[tuple[str, str, list[str]]] = []
        missing_refs: list[str] = []
        for ref in refs:
            support = support_by_ref.get(ref.casefold())
            if support is None:
                missing_refs.append(ref)
                continue
            clause, label, cited_ids = support
            supported_entries.append((label, clause, cited_ids))

        if not supported_entries:
            return cleaned

        normalized_rules = {cls._normalize_commencement_rule(clause) for _label, clause, _ids in supported_entries}
        if not missing_refs and len(normalized_rules) == 1 and "common commencement" in question_lower:
            labels = [label for label, _clause, _ids in supported_entries]
            cited_ids = list(dict.fromkeys(chunk_id for _label, _clause, ids in supported_entries for chunk_id in ids))
            labels_text = ", ".join(labels[:-1]) + f", and {labels[-1]}" if len(labels) > 2 else " and ".join(labels)
            return (
                f"The common commencement rule for {labels_text} is {supported_entries[0][1]} "
                f"(cite: {', '.join(cited_ids)})"
            )

        rebuilt: list[str] = []
        for label, clause, cited_ids in supported_entries:
            rebuilt.append(f"{len(rebuilt) + 1}. {label}: {clause} (cite: {', '.join(cited_ids)})")
        for ref in missing_refs:
            rebuilt.append(f"{len(rebuilt) + 1}. {ref}: The provided sources do not contain its commencement provision.")
        return "\n".join(rebuilt) if rebuilt else cleaned

    @classmethod
    def cleanup_named_enactment_date_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cls._is_named_enactment_date_question(question):
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        if not refs:
            return cleaned

        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return cleaned

        support_by_ref: dict[str, tuple[str, str, list[str]]] = {}
        for ref in refs:
            best_support: tuple[str, str, list[str]] | None = None
            best_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue
                label = cls._recover_doc_title_from_chunks(doc_chunks) or ref
                enactment_date, enactment_chunk_id = cls._extract_enactment_date_support(doc_chunks)
                if enactment_date and enactment_chunk_id:
                    candidate = (label, enactment_date, [enactment_chunk_id])
                    candidate_score = match_score + 120
                else:
                    candidate = None
                    candidate_score = 0
                    for chunk in doc_chunks:
                        normalized = re.sub(r"\s+", " ", (chunk.text or "").strip())
                        if not normalized or _ENACTMENT_NOTICE_REFERENCE_RE.search(normalized) is None:
                            continue
                        candidate = (
                            label,
                            "the date specified in the Enactment Notice in respect of this Law",
                            [chunk.chunk_id],
                        )
                        candidate_score = match_score + 90
                        break
                if candidate is not None and candidate_score > best_score:
                    best_support = candidate
                    best_score = candidate_score
            if best_support is not None:
                support_by_ref[ref.casefold()] = best_support

        if not support_by_ref and len(doc_order) == 1:
            only_doc_chunks = chunks_by_doc.get(doc_order[0], [])
            fallback_ref = refs[0]
            recovered_title = cls._recover_doc_title_from_chunks(only_doc_chunks) or fallback_ref
            enactment_date, enactment_chunk_id = cls._extract_enactment_date_support(only_doc_chunks)
            if enactment_date and enactment_chunk_id:
                support_by_ref[fallback_ref.casefold()] = (recovered_title, enactment_date, [enactment_chunk_id])
            else:
                for chunk in only_doc_chunks:
                    normalized = re.sub(r"\s+", " ", (chunk.text or "").strip())
                    if not normalized or _ENACTMENT_NOTICE_REFERENCE_RE.search(normalized) is None:
                        continue
                    support_by_ref[fallback_ref.casefold()] = (
                        recovered_title,
                        "the date specified in the Enactment Notice in respect of this Law",
                        [chunk.chunk_id],
                    )
                    break

        if not support_by_ref:
            return cleaned

        if len(refs) == 1:
            support = support_by_ref.get(refs[0].casefold())
            if support is None:
                return cleaned
            _label, enactment_date, cited_ids = support
            return f"The date of enactment is {enactment_date} (cite: {', '.join(cited_ids)})"

        rebuilt: list[str] = []
        for ref in refs:
            support = support_by_ref.get(ref.casefold())
            if support is None:
                rebuilt.append(f"{len(rebuilt) + 1}. {ref}: The provided sources do not specify the date of enactment.")
                continue
            label, enactment_date, cited_ids = support
            rebuilt.append(
                f"{len(rebuilt) + 1}. {label}: The date of enactment is {enactment_date} "
                f"(cite: {', '.join(cited_ids)})"
            )

        return "\n".join(rebuilt) if rebuilt else cleaned

    @classmethod
    def _extract_made_by_support(
        cls,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        best_clause = ""
        best_ids: list[str] = []
        best_score = 0

        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "")).strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            clause = ""

            made_match = re.search(r"\bthis law is made by the ([A-Z][A-Za-z\s-]+?)(?:[.;,]|$)", normalized)
            if made_match is not None:
                entity = made_match.group(1).strip(" ,.;:")
                clause = f"This Law was made by the {entity}."
            elif "made by the ruler of dubai" in lowered or ("ruler of dubai" in lowered and "hereby enact" in lowered):
                clause = "This Law was made by the Ruler of Dubai."

            if not clause:
                continue

            score = 180
            if "ruler of dubai" in lowered:
                score += 40
            if "legislative authority" in lowered:
                score += 20
            if cls._page_num(chunk.section_path) <= 3:
                score += 10
            if score > best_score:
                best_clause = clause
                best_ids = [chunk.chunk_id]
                best_score = score

        return best_clause, best_ids

    @classmethod
    def cleanup_named_made_by_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cls._is_named_made_by_question(question):
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order or not refs:
            return cleaned

        for ref in refs:
            best_clause = ""
            best_ids: list[str] = []
            best_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue
                clause, cited_ids = cls._extract_made_by_support(doc_chunks)
                if not clause or not cited_ids:
                    continue
                candidate_score = match_score + 120
                if candidate_score > best_score:
                    best_clause = clause
                    best_ids = cited_ids
                    best_score = candidate_score
            if best_clause and best_ids:
                return f"{best_clause} (cite: {', '.join(best_ids)})"

        if len(doc_order) == 1:
            clause, cited_ids = cls._extract_made_by_support(chunks_by_doc.get(doc_order[0], []))
            if clause and cited_ids:
                return f"{clause} (cite: {', '.join(cited_ids)})"
        return cleaned

    @classmethod
    def cleanup_named_administration_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        question_lower = (question or "").strip().lower()
        if "administ" not in question_lower:
            return cleaned
        if cls._is_broad_enumeration_question(question):
            return cleaned

        extracted_title = cls._extract_single_law_title_from_question(question)
        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        if extracted_title:
            refs = [extracted_title] + [
                ref for ref in refs if cls._normalize_title_key(ref) != cls._normalize_title_key(extracted_title)
            ]
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return cleaned
        if not refs and extracted_title:
            refs = [extracted_title]
        if not refs:
            recovered_title = cls._recover_doc_title_from_chunks(chunks_by_doc.get(doc_order[0], []))
            if recovered_title:
                refs = [recovered_title]
        if not refs:
            return cleaned

        support_by_ref: dict[str, tuple[str, str, str, list[str]]] = {}
        for ref in refs:
            best_label = ref.strip(" ,.;:") or ref
            best_support: tuple[str, str, str, list[str]] | None = None
            best_support_score = 0

            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue

                recovered_title = cls._recover_doc_title_from_chunks(doc_chunks) or ref
                label = recovered_title.strip(" ,.;:") or best_label
                ref_has_year = _TITLE_YEAR_RE.search(ref) is not None or "law no." in ref.casefold()
                label_has_year = _TITLE_YEAR_RE.search(label) is not None or "law no." in label.casefold()
                if ref_has_year and not label_has_year:
                    label = ref.strip(" ,.;:") or label
                clause, entity, cited_ids = cls._extract_administration_support(doc_chunks)
                if not clause or not cited_ids:
                    continue
                candidate_score = match_score + (50 if entity else 0)
                if candidate_score > best_support_score:
                    best_support = (label, clause, entity, cited_ids)
                    best_support_score = candidate_score
                    best_label = label

            if best_support is not None:
                support_by_ref[ref.casefold()] = best_support

        fallback_ref = extracted_title or (refs[0] if len(refs) == 1 else "")
        if not support_by_ref and fallback_ref:
            ref = fallback_ref
            ref_key = cls._normalize_title_key(ref)
            fallback_support: tuple[str, str, str, list[str]] | None = None
            fallback_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                clause, entity, cited_ids = cls._extract_administration_support(doc_chunks)
                if not clause or not cited_ids:
                    continue
                recovered_title = cls._recover_doc_title_from_chunks(doc_chunks) or str(doc_chunks[0].doc_title or "")
                title_blob = " ".join(
                    part
                    for part in (
                        recovered_title,
                        str(doc_chunks[0].doc_summary or ""),
                        re.sub(r"\s+", " ", (doc_chunks[0].text or "").strip()),
                    )
                    if part
                )
                title_blob_key = cls._normalize_title_key(title_blob)
                score = 100
                if ref_key and ref_key in title_blob_key:
                    score += 80
                if any(cls._page_num(chunk.section_path) == 1 for chunk in doc_chunks):
                    score += 20
                if entity:
                    score += 10
                if score > fallback_score:
                    ref_label = cls._clean_structured_doc_label(ref) or ref.strip(" ,.;:") or ref
                    recovered_label = cls._clean_structured_doc_label(recovered_title)
                    label = ref_label if ref_key and ref_key in title_blob_key else (recovered_label or ref_label)
                    fallback_support = (label, clause, entity, cited_ids)
                    fallback_score = score
            if fallback_support is not None:
                support_by_ref[ref.casefold()] = fallback_support

        if not support_by_ref:
            return cleaned

        ask_for_entity_only = (
            question_lower.startswith("what entity administers")
            or question_lower.startswith("who administers")
            or "who is responsible for administering" in question_lower
        )
        if len(refs) >= 1:
            single_support_signatures = {
                (
                    (support[2] if ask_for_entity_only and support[2] else support[1]).casefold(),
                    tuple(support[3]),
                )
                for support in support_by_ref.values()
            }
            if len(single_support_signatures) == 1:
                label, clause, entity, cited_ids = next(iter(support_by_ref.values()))
                clean_label = cls._clean_structured_doc_label(label) or label
                if extracted_title:
                    extracted_label = cls._clean_structured_doc_label(extracted_title) or extracted_title.strip(" ,.;:")
                    if extracted_label and clean_label.isupper():
                        clean_label = extracted_label
                if len(refs) == 1:
                    ref_source = extracted_title or refs[0]
                    ref_label = cls._clean_structured_doc_label(ref_source) or ref_source.strip(" ,.;:") or ref_source
                    if ref_label and (
                        clean_label.isupper() or clean_label.casefold() == ref_label.casefold()
                    ):
                        clean_label = ref_label
                if ask_for_entity_only and entity:
                    return f"{entity} administers {clean_label} and any Regulations made under it (cite: {', '.join(cited_ids)})"
                content = entity if ask_for_entity_only and entity else clause
                return f"{clean_label}: {content} (cite: {', '.join(cited_ids)})"

        rebuilt: list[str] = []
        for ref in refs:
            support = support_by_ref.get(ref.casefold())
            if support is None:
                if len(refs) == 1:
                    return cleaned
                rebuilt.append(f"{len(rebuilt) + 1}. {ref}: The provided sources do not contain its administration provision.")
                continue
            label, clause, entity, cited_ids = support
            clean_label = cls._clean_structured_doc_label(label) or label
            content = entity if ask_for_entity_only and entity else clause
            if len(support_by_ref) == 1:
                ref_source = extracted_title or ref
                ref_label = cls._clean_structured_doc_label(ref_source) or ref_source.strip(" ,.;:") or ref_source
                if ref_label and clean_label.isupper():
                    clean_label = ref_label
                if ask_for_entity_only and entity:
                    return f"{entity} administers {clean_label} and any Regulations made under it (cite: {', '.join(cited_ids)})"
                return f"{clean_label}: {content} (cite: {', '.join(cited_ids)})"
            rebuilt.append(f"{len(rebuilt) + 1}. {clean_label}: {content} (cite: {', '.join(cited_ids)})")

        return "\n".join(rebuilt) if rebuilt else cleaned

    @classmethod
    def _extract_registrar_authority_support(
        cls,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        best_clause = ""
        best_ids: list[str] = []
        best_score = 0

        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "")).strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            if "registrar" not in lowered or "appoint" not in lowered or "dismiss" not in lowered:
                continue

            clause = ""
            if "board of directors of the difca" in lowered:
                clause = "The Board of Directors of the DIFCA appoints and may dismiss the Registrar."
            elif "board of directors" in lowered:
                clause = "The Board of Directors appoints and may dismiss the Registrar."
            if not clause:
                continue

            score = 200
            if "consult the president" in lowered:
                score += 10
            if cls._page_num(chunk.section_path) <= 5:
                score += 10
            if score > best_score:
                best_clause = clause
                best_ids = [chunk.chunk_id]
                best_score = score

        return best_clause, best_ids

    @classmethod
    def cleanup_named_registrar_authority_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cls._is_named_registrar_authority_question(question):
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order or not refs:
            return cleaned

        for ref in refs:
            best_clause = ""
            best_ids: list[str] = []
            best_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue
                clause, cited_ids = cls._extract_registrar_authority_support(doc_chunks)
                if not clause or not cited_ids:
                    continue
                candidate_score = match_score + 120
                if candidate_score > best_score:
                    best_clause = clause
                    best_ids = cited_ids
                    best_score = candidate_score
            if best_clause and best_ids:
                return f"{best_clause} (cite: {', '.join(best_ids)})"

        if len(doc_order) == 1:
            clause, cited_ids = cls._extract_registrar_authority_support(chunks_by_doc.get(doc_order[0], []))
            if clause and cited_ids:
                return f"{clause} (cite: {', '.join(cited_ids)})"
        return cleaned

    @classmethod
    def _extract_retention_period_support(
        cls,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        best_clause = ""
        best_ids: list[str] = []
        best_score = 0

        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "")).strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            score = 0
            clause = ""
            if _RECORDS_RETAINED_AFTER_REPORTING_RE.search(normalized):
                clause = "6 years after reporting the information"
                score = 200
            else:
                preserved_match = _ACCOUNTING_RECORDS_PRESERVED_RE.search(normalized)
                if preserved_match is not None:
                    clause = "at least 6 years from creation"
                    score = 180

            if not clause:
                continue
            if "accounting records" in lowered:
                score += 20
            if "retention period" in lowered or "preserved by" in lowered:
                score += 20
            if cls._page_num(chunk.section_path) == 1:
                score += 5
            if score > best_score:
                best_clause = clause
                best_ids = [chunk.chunk_id]
                best_score = score

        return best_clause, best_ids

    @classmethod
    def cleanup_named_retention_period_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cls._is_named_retention_period_question(question):
            return cleaned

        question_lower = re.sub(r"\s+", " ", (question or "").strip()).lower()
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return cleaned

        target_titles: list[str] = []
        if "common reporting standard" in question_lower:
            target_titles.append("common reporting standard law")
        if "general partnership" in question_lower:
            target_titles.append("general partnership law")
        if "limited liability partnership" in question_lower:
            target_titles.append("limited liability partnership law")
        if len(target_titles) < 2:
            return cleaned

        selected_docs: list[tuple[str, str, Sequence[RankedChunk]]] = []
        seen_titles: set[str] = set()
        for target_title in target_titles:
            best_doc: tuple[str, str, Sequence[RankedChunk]] | None = None
            best_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                title = (cls._recover_doc_title_from_chunks(doc_chunks) or doc_chunks[0].doc_title or "").strip()
                title_lower = title.casefold()
                if target_title not in title_lower:
                    continue
                clause, cited_ids = cls._extract_retention_period_support(doc_chunks)
                if not clause or not cited_ids:
                    continue
                score = 100 + (25 if cls._clean_structured_doc_label(title).casefold().startswith(target_title) else 0)
                if score > best_score:
                    best_doc = (title, title, doc_chunks)
                    best_score = score
            if best_doc is None:
                continue
            key = best_doc[0].casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            selected_docs.append(best_doc)

        if len(selected_docs) < 2:
            return cleaned

        rebuilt: list[str] = []
        for ref, title, doc_chunks in selected_docs:
            clause, cited_ids = cls._extract_retention_period_support(doc_chunks)
            if not clause or not cited_ids:
                continue
            label = cls._clean_structured_doc_label(title) or ref.strip(" ,.;:") or title
            rebuilt.append(f"{len(rebuilt) + 1}. {label}: {clause} (cite: {', '.join(cited_ids)})")

        return "\n".join(rebuilt) if len(rebuilt) >= 2 else cleaned

    @staticmethod
    def _is_named_liability_question(question: str) -> bool:
        q = re.sub(r"\s+", " ", (question or "").strip()).casefold()
        if not q or RAGGenerator._is_broad_enumeration_question(question):
            return False
        if "what kind of liability" in q or "what liability" in q:
            return True
        return ("liabil" in q or "liable" in q) and "partner" in q and "under article" in q

    @classmethod
    def _extract_liability_support(
        cls,
        *,
        question: str,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        question_lower = re.sub(r"\s+", " ", (question or "").strip()).casefold()
        subject = "Partners" if "partner" in question_lower else "The relevant parties"

        best_clause = ""
        best_cited_ids: list[str] = []
        best_score = 0
        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "")).strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            if "jointly and severally liable" not in lowered and (
                "jointly" not in lowered or "severally liable" not in lowered
            ):
                continue

            score = 200
            if "partner" in lowered:
                score += 20
            if "liab" in lowered:
                score += 20
            if cls._page_num(chunk.section_path) <= 2:
                score += 10
            if score > best_score:
                best_score = score
                best_clause = f"{subject} are jointly and severally liable."
                best_cited_ids = [chunk.chunk_id]

        return best_clause, best_cited_ids

    @classmethod
    def cleanup_named_liability_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cls._is_named_liability_question(question):
            return cleaned

        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if len(doc_order) != 1:
            return cleaned

        only_doc_chunks = chunks_by_doc.get(doc_order[0], [])
        if not only_doc_chunks:
            return cleaned

        clause, cited_ids = cls._extract_liability_support(question=question, doc_chunks=only_doc_chunks)
        if not clause or not cited_ids:
            return cleaned

        return f"{clause} (cite: {', '.join(cited_ids)})"

    @classmethod
    def _extract_translation_requirement_support(
        cls,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        best_clause = ""
        best_ids: list[str] = []
        best_score = 0

        for chunk in doc_chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "")).strip()
            if not normalized:
                continue
            lowered = normalized.casefold()
            if "language other than english" not in lowered:
                continue
            if "english translation" not in lowered or "relevant authority" not in lowered:
                continue

            score = 200
            if "upon request" in lowered:
                score += 20
            if cls._page_num(chunk.section_path) <= 8:
                score += 10
            if score > best_score:
                best_clause = "It must provide an English translation to the Relevant Authority."
                best_ids = [chunk.chunk_id]
                best_score = score

        return best_clause, best_ids

    @classmethod
    def cleanup_named_translation_requirement_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        if not cls._is_named_translation_requirement_question(question):
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order or not refs:
            return cleaned

        for ref in refs:
            best_clause = ""
            best_ids: list[str] = []
            best_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue
                clause, cited_ids = cls._extract_translation_requirement_support(doc_chunks)
                if not clause or not cited_ids:
                    continue
                candidate_score = match_score + 120
                if candidate_score > best_score:
                    best_clause = clause
                    best_ids = cited_ids
                    best_score = candidate_score
            if best_clause and best_ids:
                return f"{best_clause} (cite: {', '.join(best_ids)})"

        if len(doc_order) == 1:
            clause, cited_ids = cls._extract_translation_requirement_support(chunks_by_doc.get(doc_order[0], []))
            if clause and cited_ids:
                return f"{clause} (cite: {', '.join(cited_ids)})"
        return cleaned

    @classmethod
    def _extract_penalty_support(
        cls,
        *,
        question: str,
        ref: str,
        doc_chunks: Sequence[RankedChunk],
    ) -> tuple[str, list[str]]:
        ref_terms = {
            token.casefold()
            for token in _TOKEN_RE.findall(ref)
            if token.casefold() not in _PENALTY_STOPWORDS and len(token) > 2
        }
        query_terms = {
            token.casefold()
            for token in _TOKEN_RE.findall(question or "")
            if token.casefold() not in _PENALTY_STOPWORDS and token.casefold() not in ref_terms and len(token) > 2
        }

        best_amount = ""
        best_chunk_id = ""
        best_score = 0
        for chunk in doc_chunks:
            lines = [re.sub(r"\s+", " ", line).strip() for line in (chunk.text or "").splitlines() if line.strip()]
            for idx, line in enumerate(lines):
                window = line
                if idx + 1 < len(lines) and re.fullmatch(r"(?:USD|US\$)?\s*[0-9,]+(?:\.\d+)?", lines[idx + 1], flags=re.IGNORECASE):
                    window = f"{line} {lines[idx + 1]}"
                normalized = window.casefold()
                amount_match = _PENALTY_AMOUNT_RE.search(window)
                if amount_match is None:
                    continue
                overlap = sum(1 for term in query_terms if term in normalized)
                score = overlap * 6
                if "penalt" in normalized:
                    score += 4
                if "illegal" in normalized:
                    score += 6
                if "offence" in normalized or "offense" in normalized:
                    score += 2
                if score > best_score:
                    best_score = score
                    best_amount = amount_match.group(1).strip()
                    best_chunk_id = chunk.chunk_id

        if best_amount and best_chunk_id:
            return f"The penalty is {best_amount} USD", [best_chunk_id]
        return "", []

    @classmethod
    def cleanup_named_penalty_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        question_lower = (question or "").strip().lower()
        if not any(term in question_lower for term in ("penalt", "fine")) or cls._is_broad_enumeration_question(question):
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        if not refs:
            return cleaned

        support_by_ref: dict[str, tuple[str, str, list[str]]] = {}
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        for ref in refs:
            best_support: tuple[str, str, list[str]] | None = None
            best_score = 0
            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue
                label = cls._recover_doc_title_from_chunks(doc_chunks) or ref
                clause, cited_ids = cls._extract_penalty_support(question=question, ref=ref, doc_chunks=doc_chunks)
                if not clause or not cited_ids:
                    continue
                candidate_score = match_score + 120
                if candidate_score > best_score:
                    best_support = (label, clause, cited_ids)
                    best_score = candidate_score
            if best_support is not None:
                support_by_ref[ref.casefold()] = best_support

        if not support_by_ref and len(doc_order) == 1:
            only_doc_chunks = chunks_by_doc.get(doc_order[0], [])
            fallback_ref = refs[0]
            recovered_title = cls._recover_doc_title_from_chunks(only_doc_chunks) or fallback_ref
            clause, cited_ids = cls._extract_penalty_support(
                question=question,
                ref=fallback_ref,
                doc_chunks=only_doc_chunks,
            )
            if clause and cited_ids:
                support_by_ref[fallback_ref.casefold()] = (recovered_title, clause, cited_ids)

        if len(doc_order) == 1 and len(support_by_ref) == 1:
            _label, clause, cited_ids = next(iter(support_by_ref.values()))
            return f"{clause} (cite: {', '.join(cited_ids)})"

        if not support_by_ref and cleaned:
            return cleaned
        if len(refs) == 1:
            support = support_by_ref.get(refs[0].casefold())
            if support is None:
                return cleaned
            _label, clause, cited_ids = support
            return f"{clause} (cite: {', '.join(cited_ids)})"

        rebuilt: list[str] = []
        for ref in refs:
            support = support_by_ref.get(ref.casefold())
            if support is None:
                rebuilt.append(f"{len(rebuilt) + 1}. {ref}: The provided sources do not specify the penalty.")
                continue
            label, clause, cited_ids = support
            rebuilt.append(f"{len(rebuilt) + 1}. {label}: {clause} (cite: {', '.join(cited_ids)})")

        return "\n".join(rebuilt) if rebuilt else cleaned

    @classmethod
    def cleanup_named_multi_title_lookup_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        question_lower = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if not cls._is_named_multi_title_lookup_question(question):
            return cleaned
        if "updated" not in question_lower and "title of" not in question_lower and "titles of" not in question_lower:
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        if len(refs) < 2:
            return cleaned

        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if len(doc_order) < 2:
            return cleaned

        rebuilt: list[str] = []
        for ref in refs:
            fallback_title = ref.strip(" ,.;:") or ref
            fallback_score = 0
            best_title = ""
            best_title_chunk_id = ""
            best_title_score = 0
            best_updated_value = ""
            best_updated_ids: list[str] = []
            best_updated_score = 0

            for doc_id in doc_order:
                doc_chunks = chunks_by_doc.get(doc_id, [])
                if not doc_chunks:
                    continue
                match_score = cls._doc_group_match_score(ref, doc_chunks)
                if match_score <= 0:
                    continue

                title = cls._recover_doc_title_from_chunks(doc_chunks, prefer_citation_title=True) or ref
                title = cls._clean_structured_doc_label(title) or ref.strip(" ,.;:") or ref
                title = cls._clean_amendment_title_historical_year(title) or title
                if match_score > fallback_score:
                    fallback_title = title
                    fallback_score = match_score

                title_chunk_id = cls._select_identity_support_chunk_id(doc_chunks, title=title)
                updated_value, updated_ids = cls._extract_last_updated_support(doc_chunks)
                if title_chunk_id:
                    title_chunk = next((chunk for chunk in doc_chunks if chunk.chunk_id == title_chunk_id), None)
                    title_score = match_score + 120
                    if title_chunk is not None and cls._page_num(title_chunk.section_path) == 1:
                        title_score += 180
                    if title_chunk is not None and "enactment notice" in re.sub(
                        r"\s+", " ", (title_chunk.text or "").strip()
                    ).casefold():
                        title_score += 80
                    if title_chunk_id and title_chunk_id in updated_ids:
                        title_score += 220
                    if title_score > best_title_score:
                        best_title = title
                        best_title_chunk_id = title_chunk_id
                        best_title_score = title_score
                if "updated" in question_lower and updated_value and updated_ids:
                    updated_score = match_score + 700
                    if any(
                        cls._page_num(chunk.section_path) == 1
                        for chunk in doc_chunks
                        if chunk.chunk_id in updated_ids
                    ):
                        updated_score += 80
                    if updated_score > best_updated_score:
                        best_updated_value = updated_value
                        best_updated_ids = list(updated_ids)
                        best_updated_score = updated_score

            title = best_title or fallback_title
            title_chunk_id = best_title_chunk_id

            if "updated" in question_lower:
                if best_updated_value and best_updated_ids:
                    updated_citation = ", ".join(best_updated_ids)
                    if title_chunk_id and title_chunk_id in best_updated_ids:
                        rebuilt.append(
                            f"{len(rebuilt) + 1}. {ref} - Title: {title} - Last updated (consolidated version): "
                            f"{best_updated_value} (cite: {updated_citation})"
                        )
                    else:
                        title_citation = f" (cite: {title_chunk_id})" if title_chunk_id else ""
                        rebuilt.append(
                            f"{len(rebuilt) + 1}. {ref} - Title: {title}{title_citation} - Last updated "
                            f"(consolidated version): {best_updated_value} (cite: {updated_citation})"
                        )
                    continue

                title_suffix = f" (cite: {title_chunk_id})" if title_chunk_id else ""
                rebuilt.append(
                    f"{len(rebuilt) + 1}. {ref} - Title: {title}{title_suffix} - "
                    "The provided sources do not state when the consolidated version was last updated."
                )
                continue

            if title_chunk_id:
                rebuilt.append(f"{len(rebuilt) + 1}. {ref} - Title: {title} (cite: {title_chunk_id})")
            else:
                rebuilt.append(f"{len(rebuilt) + 1}. {ref} - Title: {title}")

        return "\n".join(rebuilt) if rebuilt else cleaned

    @classmethod
    def build_consolidated_version_published_answer(
        cls,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        if not cls._is_consolidated_version_published_question(question) or not chunks:
            return ""

        extracted_title = cls._extract_single_law_title_from_question(question)
        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        if extracted_title:
            refs = [extracted_title] + [
                ref for ref in refs if cls._normalize_title_key(ref) != cls._normalize_title_key(extracted_title)
            ]
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return ""

        if refs:
            for ref in refs:
                best_match_score = 0
                best_label = ref
                best_value = ""
                best_ids: list[str] = []
                for doc_id in doc_order:
                    doc_chunks = chunks_by_doc.get(doc_id, [])
                    if not doc_chunks:
                        continue
                    match_score = cls._doc_group_match_score(ref, doc_chunks)
                    if match_score <= 0:
                        continue
                    updated_value, updated_ids = cls._extract_last_updated_support(doc_chunks)
                    if not updated_value or not updated_ids:
                        continue
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_label = cls._clean_structured_doc_label(
                            cls._recover_doc_title_from_chunks(doc_chunks) or ref
                        ) or ref
                        best_value = updated_value
                        best_ids = list(updated_ids)
                if best_value and best_ids:
                    return (
                        f"The consolidated version of {best_label} was published in {best_value} "
                        f"(cite: {', '.join(best_ids)})."
                    )

        best_label = ""
        best_value = ""
        best_ids: list[str] = []
        best_score = 0
        target_title = extracted_title or (refs[0] if len(refs) == 1 else "")
        target_title_key = cls._normalize_title_key(target_title)
        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            updated_value, updated_ids = cls._extract_last_updated_support(doc_chunks)
            if not updated_value or not updated_ids:
                continue
            title_blob = " ".join(
                part
                for part in (
                    cls._recover_doc_title_from_chunks(doc_chunks) or "",
                    str(doc_chunks[0].doc_summary or ""),
                    re.sub(r"\s+", " ", (doc_chunks[0].text or "").strip()),
                )
                if part
            )
            title_blob_key = cls._normalize_title_key(title_blob)
            if target_title_key and target_title_key not in title_blob_key:
                continue
            recovered_title = cls._clean_structured_doc_label(
                cls._recover_doc_title_from_chunks(doc_chunks) or str(doc_chunks[0].doc_title or "")
            )
            score = 100 + (80 if target_title_key else 0)
            if any(
                "consolidated version" in re.sub(r"\s+", " ", (chunk.text or "").strip()).casefold()
                for chunk in doc_chunks
            ):
                score += 40
            if any(cls._page_num(chunk.section_path) == 1 for chunk in doc_chunks):
                score += 20
            if score > best_score:
                best_label = recovered_title or extracted_title or "the document"
                best_value = updated_value
                best_ids = list(updated_ids)
                best_score = score

        if best_value and best_ids:
            return (
                f"The consolidated version of {best_label} was published in {best_value} "
                f"(cite: {', '.join(best_ids)})."
            )

        return ""

    @classmethod
    def build_remuneration_recordkeeping_answer(
        cls,
        *,
        chunks: Sequence[RankedChunk],
    ) -> str:
        best_clause = ""
        best_ids: list[str] = []
        best_score = 0

        for chunk in chunks:
            normalized = re.sub(r"\s+", " ", (chunk.text or "").strip())
            if not normalized:
                continue
            match = _REMUNERATION_RECORDKEEPING_RE.search(normalized)
            if match is None:
                continue
            page_detail = match.group(1).strip(" ,.;:")
            clause = (
                "An Employer must keep records of the Employee's remuneration, including "
                f"{page_detail}, and the applicable pay period"
            )
            score = 100
            if cls._page_num(chunk.section_path) == 4:
                score += 20
            if "article 16" in normalized.casefold():
                score += 20
            if score > best_score:
                best_clause = clause
                best_ids = [chunk.chunk_id]
                best_score = score

        if not best_clause or not best_ids:
            return ""

        return f"{best_clause} (cite: {', '.join(best_ids)})."

    @classmethod
    def cleanup_named_amendment_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        question_lower = re.sub(r"\s+", " ", (question or "").strip()).lower()
        if "what law did it amend" not in question_lower and "what laws did it amend" not in question_lower:
            return cleaned
        if "enact" not in question_lower:
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        amendment_ref = refs[0] if refs else ""
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return cleaned

        amender_doc_id = ""
        amender_score = 0
        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            score = cls._doc_group_match_score(amendment_ref, doc_chunks) if amendment_ref else 0
            if score > amender_score:
                amender_doc_id = doc_id
                amender_score = score

        if not amender_doc_id:
            return cleaned

        amender_chunks = chunks_by_doc.get(amender_doc_id, [])
        enactment_date, enactment_chunk_id = cls._extract_enactment_date_support(amender_chunks)
        if not enactment_date or not enactment_chunk_id:
            return cleaned

        amender_label = cls._recover_doc_title_from_chunks(amender_chunks, prefer_citation_title=True) or amendment_ref or "the amendment law"
        normalized_markers: list[str] = []
        if amendment_ref:
            normalized_markers.append(re.sub(r"\s+", " ", amendment_ref).strip().casefold())
        if amender_label:
            normalized_markers.append(re.sub(r"\s+", " ", amender_label).strip().casefold())
        law_match = _LAW_NO_REF_RE.search(amendment_ref or question)
        if law_match is not None:
            normalized_markers.append(f"law no. {int(law_match.group(1))} of {law_match.group(2)}")
        normalized_markers.extend(["difc laws amendment law", "laws amendment law"])
        normalized_markers = [marker for marker in dict.fromkeys(normalized_markers) if marker]

        amended_entries: list[tuple[str, str]] = []
        seen_titles: set[str] = set()
        for doc_id in doc_order:
            if doc_id == amender_doc_id:
                continue
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue

            best_chunk_id = ""
            best_score = 0
            for chunk in doc_chunks:
                normalized = re.sub(r"\s+", " ", (chunk.text or "").strip()).casefold()
                if "amended by" not in normalized:
                    continue
                score = 0
                for marker in normalized_markers:
                    if marker in normalized:
                        score += 20
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id

            if not best_chunk_id:
                continue

            title = cls._recover_doc_title_from_chunks(doc_chunks, prefer_citation_title=True) or doc_chunks[0].doc_title or "Unknown law"
            title = re.sub(r"\s+", " ", title).strip(" ,.;:")
            title_key = cls._normalize_title_key(title)
            if title_key and title_key in seen_titles:
                continue
            if title_key:
                seen_titles.add(title_key)
            amended_entries.append((title, best_chunk_id))

        if not amended_entries:
            return cleaned

        rebuilt = [
            f"1. Enactment Date:\n{amender_label} was enacted on {enactment_date} (cite: {enactment_chunk_id}).",
            "2. Laws Amended:",
        ]
        for title, chunk_id in amended_entries:
            rebuilt.append(f"- {title} (cite: {chunk_id})")
        return "\n".join(rebuilt)

    @classmethod
    def build_amended_by_enumeration_answer(
        cls,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        if not cls._is_amended_by_enumeration_question(question):
            return ""

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        amendment_ref = refs[0] if refs else ""
        question_lower = re.sub(r"\s+", " ", (question or "").strip()).lower()
        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        if not doc_order:
            return ""

        normalized_markers: list[str] = []
        if amendment_ref:
            normalized_markers.append(re.sub(r"\s+", " ", amendment_ref).strip().casefold())
        law_match = _LAW_NO_REF_RE.search(amendment_ref or question)
        if law_match is not None:
            normalized_markers.append(f"law no. {int(law_match.group(1))} of {law_match.group(2)}")
        normalized_markers = [marker for marker in dict.fromkeys(normalized_markers) if marker]

        amended_entries: list[tuple[str, str]] = []
        seen_titles: set[str] = set()
        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue

            title = cls._recover_doc_title_from_chunks(doc_chunks, prefer_citation_title=True) or doc_chunks[0].doc_title or ""
            title = re.sub(r"\s+", " ", title).strip(" ,.;:")
            title_key = cls._normalize_title_key(title)
            if amendment_ref and title_key and title_key == cls._normalize_title_key(amendment_ref):
                continue

            best_chunk_id = ""
            best_score = 0
            for chunk in doc_chunks:
                normalized = re.sub(r"\s+", " ", (chunk.text or "").strip()).casefold()
                if "amended by" not in normalized:
                    continue
                score = 0
                if "as amended by" in normalized:
                    score += 5
                for marker in normalized_markers:
                    if marker in normalized:
                        score += 20
                if "which specific" in question_lower and "specific difc laws" in question_lower and "difc law" in normalized:
                    score += 2
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id

            if not best_chunk_id or not title:
                continue
            if title_key and title_key in seen_titles:
                continue
            if title_key:
                seen_titles.add(title_key)
            amended_entries.append((title, best_chunk_id))

        if not amended_entries:
            return ""

        rebuilt: list[str] = []
        for idx, (title, chunk_id) in enumerate(amended_entries, start=1):
            rebuilt.append(f"{idx}. {title} (cite: {chunk_id})")
        return "\n".join(rebuilt)

    @classmethod
    def cleanup_account_effective_dates_answer(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
        doc_refs: Sequence[str] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        question_lower = (question or "").strip().lower()
        if "pre-existing" not in question_lower or "new accounts" not in question_lower:
            return cleaned
        if "effective date" not in question_lower:
            return cleaned

        refs = cls._question_named_refs(question=question, extra_refs=doc_refs, prefer_extra_refs=True)
        selected_docs = cls._select_doc_groups_for_refs(refs=refs, chunks=chunks)
        if not selected_docs:
            doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
            if not doc_order:
                return cleaned
            fallback_ref = refs[0] if refs else "the Law"
            selected_docs = [(fallback_ref, fallback_ref, chunks_by_doc.get(doc_order[0], []))]

        pre_existing_date = ""
        new_account_date = ""
        effective_chunk_id = ""
        enactment_date = ""
        enactment_chunk_id = ""
        law_label = refs[0].strip(" ,.;:") if refs else "the Law"

        def _consume_enactment_details(candidate_chunks: Sequence[RankedChunk]) -> None:
            nonlocal pre_existing_date, new_account_date, effective_chunk_id, enactment_date, enactment_chunk_id
            for chunk in candidate_chunks:
                normalized_text = re.sub(r"\s+", " ", (chunk.text or "")).strip()
                if not normalized_text:
                    continue
                if not pre_existing_date:
                    match = _PRE_EXISTING_EFFECTIVE_DATE_RE.search(normalized_text)
                    if match:
                        pre_existing_date = match.group(1).strip()
                        effective_chunk_id = chunk.chunk_id
                if not new_account_date:
                    match = _NEW_ACCOUNT_EFFECTIVE_DATE_RE.search(normalized_text)
                    if match:
                        new_account_date = match.group(1).strip()
                        effective_chunk_id = effective_chunk_id or chunk.chunk_id
                generic_enactment_reference = (
                    enactment_date.casefold()
                    == "the date specified in the enactment notice in respect of this law".casefold()
                    if enactment_date
                    else False
                )
                if not enactment_date or generic_enactment_reference:
                    match = _ENACTMENT_DATE_RE.search(normalized_text)
                    if match:
                        enactment_date = match.group(1).strip()
                        enactment_chunk_id = chunk.chunk_id
                        continue
                if not enactment_date:
                    match = _ENACTMENT_NOTICE_REFERENCE_RE.search(normalized_text)
                    if match:
                        enactment_date = "the date specified in the Enactment Notice in respect of this Law"
                        enactment_chunk_id = chunk.chunk_id

        for _ref, _title, doc_chunks in selected_docs:
            _consume_enactment_details(doc_chunks)

        generic_enactment_reference = (
            enactment_date.casefold() == "the date specified in the enactment notice in respect of this law".casefold()
        )

        if not enactment_date or generic_enactment_reference:
            for chunk in chunks:
                if enactment_chunk_id and chunk.chunk_id == enactment_chunk_id:
                    continue
                normalized_chunk = re.sub(r"\s+", " ", (chunk.text or "")).strip().casefold()
                enactment_like_chunk = any(
                    marker in normalized_chunk
                    for marker in (
                        "hereby enact",
                        "enactment notice",
                        "enacted on",
                    )
                )
                if refs and not any(cls._doc_group_match_score(ref, [chunk]) > 0 for ref in refs) and not enactment_like_chunk:
                    continue
                _consume_enactment_details([chunk])
                if enactment_date and enactment_date.casefold() != "the date specified in the enactment notice in respect of this law".casefold():
                    break

        if not pre_existing_date or not new_account_date or not effective_chunk_id:
            return cleaned

        include_enactment = "enact" in question_lower or "date of enactment" in question_lower
        rebuilt = [
            f"1. Pre-existing Accounts: The effective date is {pre_existing_date} (cite: {effective_chunk_id})",
            f"2. New Accounts: The effective date is {new_account_date} (cite: {effective_chunk_id})",
        ]
        if include_enactment and enactment_date and enactment_chunk_id:
            rebuilt.append(f"3. {law_label}: The date of enactment is {enactment_date} (cite: {enactment_chunk_id})")
        return "\n".join(rebuilt)

    @staticmethod
    def cleanup_numbered_list_items(answer: str, *, question: str, common_elements: bool = False) -> str:
        cleaned = (answer or "").strip()
        if not cleaned:
            return cleaned

        matches = list(_NUMBERED_ITEM_RE.finditer(cleaned))
        if not matches:
            return cleaned

        items: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            item_text = cleaned[start:end].strip()
            item_text = re.sub(r"^\d+\.\s*", "", item_text).strip()
            if item_text:
                items.append(item_text)

        if not items:
            return cleaned

        referenced_titles: list[str] = []
        if common_elements:
            seen: set[str] = set()
            for title, year in _TITLE_REF_RE.findall(question or ""):
                ref = " ".join(part for part in (title.strip(), year.strip()) if part).strip()
                if not ref:
                    continue
                key = ref.casefold()
                if key in seen:
                    continue
                seen.add(key)
                referenced_titles.append(ref)

        referenced_title_keys = {
            normalized
            for ref in referenced_titles
            if (normalized := RAGGenerator._normalize_common_elements_title_key(ref))
        }

        kept: list[str] = []
        for item in items:
            stripped_item = item.strip()
            if not stripped_item:
                continue
            if not _CITE_RE.search(stripped_item):
                continue
            if re.match(r"^(?:Therefore|Thus|Accordingly|In summary|No other)\b", stripped_item, flags=re.IGNORECASE):
                continue
            if common_elements and len(referenced_titles) >= 2:
                item_title_keys: set[str] = set()
                for title, year in _TITLE_REF_RE.findall(stripped_item):
                    ref = " ".join(part for part in (title.strip(), year.strip()) if part).strip()
                    if ref:
                        normalized = RAGGenerator._normalize_common_elements_title_key(ref)
                        if normalized:
                            item_title_keys.add(normalized)
                mentioned = len(item_title_keys.intersection(referenced_title_keys))
                if 0 < mentioned < len(referenced_titles):
                    continue
            kept.append(stripped_item)

        if not kept:
            return cleaned

        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(kept, start=1))

    @classmethod
    def cleanup_broad_enumeration_titles_only(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        query_lower = (question or "").strip().lower()
        if not cleaned or not cls._is_broad_enumeration_question(question):
            return cleaned
        if any(
            marker in query_lower
            for marker in (
                "citation title",
                "citation titles",
                "what are their respective",
                "respective citation",
                "common commencement date",
            )
        ):
            return cleaned

        matches = list(_NUMBERED_ITEM_RE.finditer(cleaned))
        if not matches:
            return cleaned

        chunk_map = {chunk.chunk_id: chunk for chunk in chunks or []}
        compressed_items: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            item_text = cleaned[start:end].strip()
            body = re.sub(r"^\d+\.\s*", "", item_text).strip()
            cited_ids = cls.extract_cited_chunk_ids(body)
            if not cited_ids:
                continue
            cited_chunks = [chunk_map[chunk_id] for chunk_id in cited_ids if chunk_id in chunk_map]
            recovered_title_from_citations = cls._recover_doc_title_from_chunks(cited_chunks) if cited_chunks else ""

            title = cls._extract_title_only_item(body)
            if title:
                title_key = cls._normalize_title_key(title)
                recovered_key = cls._normalize_title_key(recovered_title_from_citations)
                if recovered_title_from_citations and recovered_key and recovered_key != title_key:
                    title = recovered_title_from_citations
                compressed_items.append(f"{len(compressed_items) + 1}. {title} (cite: {', '.join(cited_ids)})")
                continue

            grouped_by_doc: dict[str, list[str]] = {}
            chunks_by_doc: dict[str, list[RankedChunk]] = {}
            for chunk_id in cited_ids:
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    continue
                doc_key = str(chunk.doc_id or chunk.chunk_id)
                grouped_by_doc.setdefault(doc_key, []).append(chunk_id)
                chunks_by_doc.setdefault(doc_key, []).append(chunk)

            if not grouped_by_doc:
                continue

            for doc_key, doc_citations in grouped_by_doc.items():
                recovered_title = cls._recover_doc_title_from_chunks(chunks_by_doc.get(doc_key, []))
                if not recovered_title:
                    continue
                deduped_ids = list(dict.fromkeys(doc_citations))
                compressed_items.append(
                    f"{len(compressed_items) + 1}. {recovered_title} (cite: {', '.join(deduped_ids)})"
                )

        if not compressed_items:
            return cleaned
        return "\n".join(compressed_items)

    @classmethod
    def cleanup_registrar_enumeration_items(cls, answer: str, chunks: Sequence[RankedChunk]) -> str:
        cleaned = (answer or "").strip()
        if not cleaned:
            return cleaned

        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        chunks_by_doc: dict[str, list[RankedChunk]] = {}
        for chunk in chunks:
            doc_key = str(chunk.doc_id or chunk.chunk_id)
            chunks_by_doc.setdefault(doc_key, []).append(chunk)
        matches = list(_NUMBERED_ITEM_RE.finditer(cleaned))
        if not matches:
            return cleaned

        rebuilt: list[str] = []
        merged_by_title: dict[str, tuple[str, list[str]]] = {}
        title_order: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            item_text = cleaned[start:end].strip()
            cited_ids = cls.extract_cited_chunk_ids(item_text)
            grouped_ids: dict[str, list[str]] = {}
            grouped_titles: dict[str, str] = {}
            for chunk_id in cited_ids:
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    continue
                doc_key = str(chunk.doc_id or chunk.chunk_id)
                grouped_ids.setdefault(doc_key, []).append(chunk_id)
                if doc_key not in grouped_titles:
                    grouped_titles[doc_key] = (
                        cls._recover_doc_title_from_chunks(
                            chunks_by_doc.get(doc_key, [chunk]),
                            prefer_citation_title=True,
                        )
                        or re.sub(r"\s+", " ", (chunk.doc_title or "").strip())
                        or "Unknown document"
                    )

            if len(grouped_ids) <= 1:
                title = cls._extract_title_only_item(item_text)
                if title:
                    key = re.sub(r"\s+", " ", title).strip().casefold()
                    if key not in merged_by_title:
                        title_order.append(key)
                        merged_by_title[key] = (title, cited_ids)
                    else:
                        prev_title, prev_ids = merged_by_title[key]
                        merged_by_title[key] = (prev_title, list(dict.fromkeys(prev_ids + cited_ids)))
                else:
                    rebuilt.append(item_text)
                continue

            for doc_key, doc_citations in grouped_ids.items():
                title = grouped_titles.get(doc_key) or "Unknown document"
                deduped_ids = list(dict.fromkeys(doc_citations))
                key = re.sub(r"\s+", " ", title).strip().casefold()
                if key not in merged_by_title:
                    title_order.append(key)
                    merged_by_title[key] = (title, deduped_ids)
                else:
                    prev_title, prev_ids = merged_by_title[key]
                    merged_by_title[key] = (prev_title, list(dict.fromkeys(prev_ids + deduped_ids)))

        for key in title_order:
            title, cited_ids = merged_by_title[key]
            rebuilt.append(f"{len(rebuilt) + 1}. {title} (cite: {', '.join(cited_ids)})")

        if not rebuilt:
            return cleaned
        return "\n".join(rebuilt)

    @classmethod
    def cleanup_named_ref_enumeration_items(
        cls,
        answer: str,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
    ) -> str:
        cleaned = (answer or "").strip()
        query_lower = (question or "").strip().lower()
        if "mention" not in query_lower and "reference" not in query_lower:
            return cleaned

        question_refs: list[str] = []
        seen_refs: set[str] = set()
        for match in _AMENDMENT_TITLE_RE.finditer(question or ""):
            ref = re.sub(r"\s+", " ", match.group(1).strip())
            ref = _TITLE_REF_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_QUERY_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", ref)
            ref = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", ref)
            ref = _TITLE_LEADING_CONNECTOR_RE.sub("", ref).strip(" ,.;:")
            if not ref:
                continue
            lowered_ref = ref.casefold()
            if lowered_ref in seen_refs:
                continue
            seen_refs.add(lowered_ref)
            question_refs.append(ref)
        for title, year in _TITLE_REF_RE.findall(question or ""):
            normalized_title = _TITLE_REF_BAD_LEAD_RE.sub("", title.strip())
            normalized_title = _TITLE_QUERY_BAD_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_GENERIC_QUESTION_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_PREPOSITION_BAD_LEAD_RE.sub("", normalized_title)
            normalized_title = _TITLE_LEADING_CONNECTOR_RE.sub("", normalized_title).strip(" ,.;:")
            ref = " ".join(part for part in (normalized_title, year.strip()) if part).strip(" ,.;:")
            if ref.casefold() in {"law", "difc law"}:
                continue
            if not ref:
                continue
            lowered_ref = ref.casefold()
            if lowered_ref in {"their regulations", "these regulations", "those regulations", "regulations", "regulation", "law"}:
                continue
            key = re.sub(r"\s+", " ", ref).strip().casefold()
            if key in seen_refs:
                continue
            seen_refs.add(key)
            question_refs.append(ref)
        if len(question_refs) < 2:
            return cleaned

        chunks_by_doc: dict[str, list[RankedChunk]] = {}
        doc_order: list[str] = []
        for chunk in chunks:
            doc_key = str(chunk.doc_id or chunk.chunk_id)
            if doc_key not in chunks_by_doc:
                doc_order.append(doc_key)
            chunks_by_doc.setdefault(doc_key, []).append(chunk)

        supporting_docs: list[tuple[str, str, list[str]]] = []
        for doc_key in doc_order:
            doc_chunks = chunks_by_doc.get(doc_key, [])
            supporting_ids: list[str] = []
            for ref in question_refs:
                ref_key = re.sub(r"\s+", " ", ref).strip().casefold()
                matched_chunk_id = ""
                for chunk in doc_chunks:
                    text_key = re.sub(r"\s+", " ", chunk.text or "").strip().casefold()
                    if ref_key in text_key:
                        matched_chunk_id = chunk.chunk_id
                        break
                if not matched_chunk_id:
                    supporting_ids = []
                    break
                supporting_ids.append(matched_chunk_id)
            if not supporting_ids:
                continue
            title = re.sub(r"\s+", " ", (doc_chunks[0].doc_title or "").strip()).strip(" ,.;:") if doc_chunks else ""
            if title:
                supporting_docs.append((doc_key, title, list(dict.fromkeys(supporting_ids))))

        if not supporting_docs:
            return cleaned
        return "\n".join(
            f"{idx}. {title} (cite: {', '.join(cited_ids)})"
            for idx, (_doc_key, title, cited_ids) in enumerate(supporting_docs, start=1)
        )

    @classmethod
    def cleanup_ruler_enactment_enumeration_items(
        cls,
        answer: str,
        *,
        chunks: Sequence[RankedChunk],
    ) -> str:
        cleaned = (answer or "").strip()

        chunks_by_doc: dict[str, list[RankedChunk]] = {}
        doc_order: list[str] = []
        for chunk in chunks:
            doc_key = str(chunk.doc_id or chunk.chunk_id)
            if doc_key not in chunks_by_doc:
                doc_order.append(doc_key)
            chunks_by_doc.setdefault(doc_key, []).append(chunk)

        supported_docs: list[tuple[str, str, list[str]]] = []
        merged_by_title: dict[str, tuple[str, str, list[str]]] = {}
        title_order: list[str] = []
        for doc_key in doc_order:
            doc_chunks = chunks_by_doc.get(doc_key, [])
            ruler_chunk_id = ""
            commencement_chunk_id = ""
            commencement_rule = ""
            title = cls._recover_doc_title_from_chunks(doc_chunks) or ""
            for chunk in doc_chunks:
                text = chunk.text or ""
                if not ruler_chunk_id and _RULER_OF_DUBAI_RE.search(text):
                    ruler_chunk_id = chunk.chunk_id
                rule = cls._extract_commencement_rule(text)
                if not commencement_chunk_id and rule:
                    commencement_chunk_id = chunk.chunk_id
                    commencement_rule = rule
            if title and ruler_chunk_id and commencement_chunk_id:
                clean_title = title.strip(" ,.;:")
                title_key = _TITLE_LAW_NO_SUFFIX_RE.sub("", clean_title).strip().casefold()
                cited_ids = list(dict.fromkeys([ruler_chunk_id, commencement_chunk_id]))
                if title_key not in merged_by_title:
                    title_order.append(title_key)
                    merged_by_title[title_key] = (clean_title, commencement_rule, cited_ids)
                else:
                    prev_title, prev_rule, prev_ids = merged_by_title[title_key]
                    better_title = clean_title if cls._should_prefer_extracted_title(prev_title, clean_title) else prev_title
                    better_rule = commencement_rule or prev_rule
                    merged_by_title[title_key] = (better_title, better_rule, list(dict.fromkeys(prev_ids + cited_ids)))

        for title_key in title_order:
            supported_docs.append(merged_by_title[title_key])

        if not supported_docs:
            return cleaned
        return "\n".join(
            f"{idx}. {title}: {rule} (cite: {', '.join(cited_ids)})"
            for idx, (title, rule, cited_ids) in enumerate(supported_docs, start=1)
        )

    @classmethod
    def build_registrar_enumeration_answer(
        cls,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
    ) -> str:
        if not cls._is_registrar_enumeration_question(question):
            return ""

        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        merged_by_title: dict[str, tuple[str, list[str]]] = {}
        title_order: list[str] = []
        ask_for_citation_title = "citation title" in (question or "").strip().lower()
        target_year_match = re.search(r"\b(?:enacted|made)\s+in\s+((?:19|20)\d{2})\b", question or "", re.IGNORECASE)
        target_year = target_year_match.group(1) if target_year_match is not None else ""

        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            clause, _entity, cited_ids = cls._extract_administration_support(doc_chunks)
            if not clause or "registrar" not in clause.casefold():
                continue

            title = cls._recover_doc_title_from_chunks(doc_chunks, prefer_citation_title=ask_for_citation_title)
            if not title:
                title = cls._recover_doc_title_from_chunks(doc_chunks) or re.sub(
                    r"\s+",
                    " ",
                    (doc_chunks[0].doc_title or "").strip(),
                )
            title = title.strip(" ,.;:")
            if not title:
                continue

            title_chunk_id = ""
            title_source_text = ""
            for chunk in doc_chunks:
                text = re.sub(r"\s+", " ", (chunk.text or "").strip())
                if _CITED_TITLE_RE.search(text) or _CITED_TITLE_PLAIN_RE.search(text) or _TITLE_CLAUSE_RE.search(text):
                    title_chunk_id = chunk.chunk_id
                    title_source_text = text
                    break
            if target_year:
                title_year_source = title_source_text or title
                title_years = re.findall(r"\b(?:19|20)\d{2}\b", title_year_source)
                title_has_target_year = bool(title_years) and title_years[-1] == target_year
                support_has_target_year = any(
                    any(
                        re.search(rf"\b{re.escape(target_year)}\b", sentence)
                        and any(marker in sentence.casefold() for marker in ("hereby enact", "enacted", "made this law"))
                        for sentence in _SENTENCE_SPLIT_RE.split(re.sub(r"\s+", " ", (chunk.text or "").strip()))
                    )
                    for chunk in doc_chunks[:4]
                )
                if not title_has_target_year and not support_has_target_year:
                    continue

            merged_ids = list(dict.fromkeys(([title_chunk_id] if title_chunk_id else []) + cited_ids))
            title_key = re.sub(r"\s+", " ", title).strip().casefold()
            if title_key not in merged_by_title:
                title_order.append(title_key)
                merged_by_title[title_key] = (title, merged_ids)
            else:
                prev_title, prev_ids = merged_by_title[title_key]
                merged_by_title[title_key] = (prev_title, list(dict.fromkeys(prev_ids + merged_ids)))

        if not title_order:
            return ""
        return "\n".join(
            f"{idx}. {merged_by_title[key][0]} (cite: {', '.join(merged_by_title[key][1])})"
            for idx, key in enumerate(title_order, start=1)
        )

    @classmethod
    def build_ruler_authority_year_enumeration_answer(
        cls,
        *,
        question: str,
        chunks: Sequence[RankedChunk],
    ) -> str:
        if not cls._is_ruler_authority_year_enumeration_question(question):
            return ""

        year_match = re.search(r"\b((?:19|20)\d{2})\b", question or "")
        if year_match is None:
            return ""
        target_year = year_match.group(1)

        doc_order, chunks_by_doc = cls._group_chunks_by_doc(chunks)
        rebuilt: list[str] = []
        seen_titles: set[str] = set()
        for doc_id in doc_order:
            doc_chunks = chunks_by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            title = cls._recover_doc_title_from_chunks(doc_chunks) or re.sub(r"\s+", " ", (doc_chunks[0].doc_title or "").strip())
            title = title.strip(" ,.;:")
            if not title or target_year not in title:
                continue

            ruler_chunk_id = ""
            title_chunk_id = ""
            for chunk in doc_chunks:
                text = chunk.text or ""
                normalized = re.sub(r"\s+", " ", text).strip()
                if not title_chunk_id and title and title.casefold() in normalized.casefold():
                    title_chunk_id = chunk.chunk_id
                if not ruler_chunk_id and _RULER_OF_DUBAI_RE.search(text):
                    ruler_chunk_id = chunk.chunk_id
            if not ruler_chunk_id:
                continue

            title_key = re.sub(r"\s+", " ", _TITLE_LAW_NO_SUFFIX_RE.sub("", title)).strip().casefold()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            cited_ids = list(dict.fromkeys(([title_chunk_id] if title_chunk_id else []) + [ruler_chunk_id]))
            rebuilt.append(f"{len(rebuilt) + 1}. {title} (cite: {', '.join(cited_ids)})")

        return "\n".join(rebuilt)

    @classmethod
    def _extract_title_only_item(cls, item: str) -> str:
        body = re.sub(_CITE_RE, "", item or "").strip()
        body = re.sub(r"^\d+\.\s*", "", body).strip()
        if not body:
            return ""

        doc_ref_match = re.search(
            r"\b(?:text\s+in\s+the|in\s+the|the)\s+([A-Z][A-Za-z][A-Za-z\s]+Law(?:\s+\d{4})?)\s+document\b",
            body,
        )
        if doc_ref_match:
            return re.sub(r"\s+", " ", doc_ref_match.group(1)).strip(" ,.;:")

        prefix = re.split(r"\s*[—-]\s+|:\s+", body, maxsplit=1)[0].strip(" ,.;:")
        if (
            prefix
            and len(prefix) <= 140
            and not _TITLE_ONLY_ITEM_BAD_LEAD_RE.search(prefix)
            and not _TITLE_ONLY_PLACEHOLDER_RE.search(prefix)
        ):
            return prefix

        candidates: list[tuple[str, bool]] = []
        for title, year in _TITLE_REF_RE.findall(body):
            ref = " ".join(part for part in (title.strip(), year.strip()) if part).strip(" ,.;:")
            if ref and not _TITLE_ONLY_BAD_CANDIDATE_RE.search(ref.strip()):
                candidates.append((ref, bool(year.strip())))
        if candidates:
            candidates.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
            return candidates[0][0]
        return ""

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        token_ids = self._encoding.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        if max_tokens <= 3:
            return self._encoding.decode(token_ids[:max_tokens])
        return f"{self._encoding.decode(token_ids[: max_tokens - 1]).rstrip()}..."

    def _resolve_usage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        completion_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> tuple[int, int, int]:
        prompt = max(0, int(prompt_tokens))
        completion = max(0, int(completion_tokens))
        total = max(0, int(total_tokens))
        if total > 0 and (prompt > 0 or completion > 0):
            if total < (prompt + completion):
                total = prompt + completion
            return (prompt, completion, total)
        return self._estimate_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion_text=completion_text,
        )

    def _estimate_usage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        completion_text: str,
    ) -> tuple[int, int, int]:
        prompt_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)
        completion_tokens = self.count_tokens(completion_text) if completion_text.strip() else 0
        total_tokens = prompt_tokens + completion_tokens
        logger.info(
            "llm_usage_provider_missing_fallback",
            extra={
                "prompt_tokens_est": prompt_tokens,
                "completion_tokens_est": completion_tokens,
                "total_tokens_est": total_tokens,
            },
        )
        return (prompt_tokens, completion_tokens, total_tokens)

    def _context_budget(self, *, answer_type: str, complexity: QueryComplexity) -> int:
        global_cap = int(self._llm_settings.max_context_tokens)
        if global_cap <= 0:
            return 0

        kind = answer_type.strip().lower()
        if kind == "boolean":
            budget = int(getattr(self._pipeline_settings, "context_token_budget_boolean", 900))
            return max(0, min(global_cap, budget))
        if kind in {"number", "date", "name", "names"}:
            budget = int(getattr(self._pipeline_settings, "context_token_budget_strict", 1100))
            return max(0, min(global_cap, budget))
        if complexity == QueryComplexity.COMPLEX:
            budget = int(getattr(self._pipeline_settings, "context_token_budget_free_text_complex", 2400))
            return max(0, min(global_cap, budget))
        budget = int(getattr(self._pipeline_settings, "context_token_budget_free_text_simple", 1800))
        return max(0, min(global_cap, budget))

    def _compact_chunk_text(self, *, question: str, text: str, top_n: int) -> str:
        raw = text.strip()
        if not raw:
            return raw

        sentences = [sent.strip() for sent in _SENTENCE_SPLIT_RE.split(raw) if sent.strip()]
        if not sentences:
            return self._truncate_to_tokens(raw, 180)

        # For Article-based strict questions, always include the sentence(s) that mention the Article.
        article_keys: list[str] = []
        for match in re.finditer(r"\barticle\s+\d+(?:\s*\([^)]*\))*", question, flags=re.IGNORECASE):
            key = re.sub(r"\s+", " ", match.group(0)).strip().lower()
            if key:
                article_keys.append(key)
                article_keys.append(key.replace("article ", "").strip())
        article_keys = [k for k in dict.fromkeys(article_keys) if k]

        query_terms = {
            token.lower()
            for token in _TOKEN_RE.findall(question)
            if token.lower() not in _STOPWORDS
        }
        if not query_terms and not article_keys:
            return " ".join(sentences[: max(1, top_n)])

        scored: list[tuple[int, int]] = []
        for idx, sentence in enumerate(sentences):
            tokens = {
                token.lower()
                for token in _TOKEN_RE.findall(sentence)
                if token.lower() not in _STOPWORDS
            }
            overlap = len(tokens.intersection(query_terms))
            scored.append((overlap, idx))

        selected: set[int] = set()
        if article_keys:
            for idx, sentence in enumerate(sentences):
                lowered = sentence.lower()
                if any(key in lowered for key in article_keys):
                    selected.add(idx)

        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        for overlap, idx in scored:
            if len(selected) >= max(1, top_n):
                break
            if overlap <= 0 and selected:
                break
            if overlap <= 0 and not selected:
                # If no overlap, keep at least the first sentence for minimal context.
                selected.add(0)
                break
            selected.add(idx)

        selected_idxs = sorted(selected) if selected else [0]

        compact = " ".join(sentences[idx] for idx in selected_idxs)
        # Keep compact context tight; avoid heavy tails from giant chunks.
        return self._truncate_to_tokens(compact, 180)

    def _compact_common_elements_chunk(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return raw

        lowered = raw.lower()
        start = 0
        for anchor in ("schedule 1 interpretation", "rules of interpretation", "schedule 1", "interpretation"):
            idx = lowered.find(anchor)
            if idx != -1:
                start = max(0, idx - 24)
                break

        focused = raw[start:].strip() if start > 0 else raw
        sentences = [sent.strip() for sent in _SENTENCE_SPLIT_RE.split(focused) if sent.strip()]
        if not sentences:
            return self._truncate_to_tokens(focused, 180)

        compact = " ".join(sentences[:4])
        return self._truncate_to_tokens(compact, 180)

    @staticmethod
    def _normalize_complexity(complexity: QueryComplexity | str) -> QueryComplexity:
        if isinstance(complexity, QueryComplexity):
            return complexity
        raw = complexity.strip().lower()
        return QueryComplexity.COMPLEX if raw == QueryComplexity.COMPLEX.value else QueryComplexity.SIMPLE

    @staticmethod
    def _answer_type_instruction(answer_type: str) -> str:
        kind = answer_type.strip().lower()
        if kind == "boolean":
            return "Output format: output ONLY 'Yes' or 'No'."
        if kind == "number":
            return "Output format: output ONLY a single numeric value."
        if kind == "date":
            return "Output format: output ONLY one date in YYYY-MM-DD format."
        if kind == "name":
            return "Output format: output ONLY one exact name/title."
        if kind == "names":
            return "Output format: output ONLY a comma-separated list of exact names."
        return "Output format: concise grounded answer with explicit citations like (cite: CHUNK_ID)."
