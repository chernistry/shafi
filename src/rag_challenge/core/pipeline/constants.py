# pyright: reportMissingTypeStubs=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportTypedDictNotRequiredAccess=false, reportPrivateUsage=false, reportUnusedImport=false
from __future__ import annotations

import re

from rag_challenge.prompts.loader import load_prompt

_NUMBER_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")

_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

_SLASH_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

_TEXTUAL_DATE_RE = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b")

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

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

