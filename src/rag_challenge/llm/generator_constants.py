"""Shared constants and prompt templates for the generator package."""

from __future__ import annotations

import re

from rag_challenge.prompts import load_prompt

CITE_RE = re.compile(r"\(cite:\s*([^)]+)\)", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")
TITLE_REF_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9]*(?:\s+(?:of|the|in|on|and|for|to|by|Non|Incorporated|Limited|General|Data|Protection|Application|Civil|Commercial|Strata|Title|Trust|Contract|Liability|Partnership|Profit|Organisations?|Operating|Companies|Insolvency|Foundations?|Employment|Arbitration|Securities|Investment|Personal|Property|Obligations|Netting|Courts|Court|Common|Reporting|Standard|Dematerialised|Investments?|Implied|Terms|Unfair|Amendment|DIFC))*\s+(?:Law|Regulations?)))\b(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)
AMENDMENT_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+Laws?\s+Amendment\s+Law(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
BROAD_ENUMERATION_RE = re.compile(
    r"^(?:which\s+(?:laws?|regulations?|documents?|cases?)|list\s+all|identify\s+all|name\s+all)\b",
    re.IGNORECASE,
)
COMMON_ELEMENTS_RE = re.compile(r"\b(?:common elements|elements in common|in common)\b", re.IGNORECASE)
NEGATIVE_SUBCLAIM_RE = re.compile(
    r"(?:^|\n)"
    r"[^\n]*?"
    r"(?:"
    r"[Tt]here\s+is\s+no\s+information\s+on\s+(?:the\s+)?(?:other\s+)?[A-Za-z]"
    r"|[Tt]here\s+is\s+no\s+information\s+in\s+the\s+provided\s+sources"
    r"|[Tt]here\s+is\s+no\s+explicit\s+mention(?:\s+of)?"
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
TRAILING_NEGATIVE_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"(?:"
    r"[Tt]here\s+is\s+no\s+information\s+on"
    r"|[Tt]here\s+is\s+no\s+information\s+in\s+the\s+provided\s+sources"
    r"|[Tt]here\s+is\s+no\s+explicit\s+mention(?:\s+of)?"
    r"|does\s+not\s+(?:explicitly\s+)?(?:mention|contain|include|provide|reference|address|specify)"
    r"|are\s+confirmed\s+only\s+between"
    r"|Therefore,\s+the\s+common\s+elements"
    r"|cannot\s+be\s+confirmed"
    r")"
    r"[^.!?]*[.!?]?\s*$",
    re.IGNORECASE | re.DOTALL,
)
STOPWORDS = {
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

SYSTEM_PROMPT_SIMPLE = load_prompt("llm/generator_system_simple")
SYSTEM_PROMPT_COMPLEX = load_prompt("llm/generator_system_complex")
SYSTEM_PROMPT_COMPLEX_IRAC = load_prompt("llm/generator_system_complex_irac")
SYSTEM_PROMPT_STRICT = load_prompt("llm/generator_system_strict")
SYSTEM_PROMPT_COMPLEX_INTERLEAVED = load_prompt("llm/generator_system_complex_interleaved")
SYSTEM_PROMPT_COMPLEX_IRAC_INTERLEAVED = load_prompt("llm/generator_system_complex_irac_interleaved")
USER_PROMPT_TEMPLATE = load_prompt("llm/generator_user")
USER_PROMPT_TEMPLATE_STRICT = load_prompt("llm/generator_user_strict")
IRAC_HINT_RE = re.compile(
    r"\b(compare|difference|distinguish|contrast|analy[sz]e|evaluate|common elements|modify|modifies|impact|effect|summari[sz]e)\b",
    re.IGNORECASE,
)
CITED_TITLE_RE = re.compile(r'This Law may be cited as(?: the)? [“"]([^"”]+)[”"]', re.IGNORECASE)
CITED_TITLE_PLAIN_RE = re.compile(
    r"\bThis Law may be cited as(?: the)? ([A-Z][A-Za-z0-9\s()/-]+?(?:Law|Regulations?)(?:\s+\d{4})?)\b",
    re.IGNORECASE,
)
COVER_TITLE_LAW_YEAR_RE = re.compile(
    r"^\s*([A-Z][A-Z\s/&().-]+?(?:LAW|REGULATIONS?|RULES?|CODE|NOTICE))\s+"
    r"DIFC\s+LAW\s+NO\.?\s*\d+\s+OF\s+(\d{4})\b",
    re.IGNORECASE | re.MULTILINE,
)
LEGISLATION_REF_TITLE_RE = re.compile(
    r"\b(?:this|the)\s+Law\s+is\s+(?:the\s+)?("
    r"[A-Z][A-Za-z][A-Za-z\s]+?Law(?:\s+Amendment\s+Law)?"
    r"(?:,\s*DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4})?"
    r")\s+made\s+by\s+the\s+Ruler\b",
    re.IGNORECASE,
)
ENACTMENT_NOTICE_ATTACHED_TITLE_RE = re.compile(
    r"\bin\s+the\s+form\s+now\s+attached\s+(?:the\s+)?("
    r"[A-Z][A-Za-z][A-Za-z\s-]+?Law(?:\s+Amendment\s+Law)?"
    r"(?:\s+DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)
ENACTMENT_NOTICE_TITLE_RE = re.compile(
    r"\bthe\s+([A-Z][A-Za-z][A-Za-z\s-]+Law(?:\s+DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4})?)\b",
    re.IGNORECASE,
)
DOC_SUMMARY_TITLE_RE = re.compile(r"\*\*Document Title:\*\*\s*([^\n]+)", re.IGNORECASE)
DOC_SUMMARY_TITLED_RE = re.compile(r'\btitled\s+[“"]([^"”]+)[”"]', re.IGNORECASE)
DOC_SUMMARY_PREFIX_TITLE_RE = re.compile(
    r"^(?:statute|regulation|document|case(?:\s+law)?):\s*([^,(]+?(?:Law|Regulations?|Rules?|Code|Notice))\b",
    re.IGNORECASE,
)
DOC_SUMMARY_IS_THE_RE = re.compile(
    r"\bThis\s+(?:document|statute|contract\s+document|case\s+law\s+document)\s+(?:is|,)\s*(?:the\s+)?"
    r"([^,.]+?(?:Law|Regulations?|Rules?|Code|Notice))\b",
    re.IGNORECASE,
)
STRUCTURED_TITLE_BAD_LEAD_RE = re.compile(
    r"^(?:(?:we\s+hereby\s+enact(?:\s+on\s+this\s+[^.]+?)?\s+(?:the\s+)?)?(?:in\s+the\s+form\s+now\s+attached\s+)?|enactment\s+notice\s+)+",
    re.IGNORECASE,
)
TITLE_LAW_NO_SUFFIX_RE = re.compile(r"\s*,?\s*DIFC\s+Law\s+No\.?\s*\d+\s+of\s+\d{4}\b", re.IGNORECASE)
TITLE_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
TITLE_LEADING_CONNECTOR_RE = re.compile(r"^(?:(?:of|and|the)\s+)+", re.IGNORECASE)
TITLE_REF_BAD_LEAD_RE = re.compile(
    r"^(?:(?:which|what|how|mention|mentions|reference|references|their|these|those|do|does|did)\s+)+",
    re.IGNORECASE,
)
TITLE_QUERY_BAD_LEAD_RE = re.compile(r"^(?:(?:is|are)\s+)?(?:the\s+)?titles?\s+of\s+", re.IGNORECASE)
TITLE_GENERIC_QUESTION_LEAD_RE = re.compile(
    r"^(?:(?:on\s+what\s+date|in\s+what\s+year|what|which|when|where|who|how|was|were|is|are|did|does|do)\s+)+"
    r"(?:(?:the|its)\s+)?(?:(?:citation\s+)?titles?\s+of\s+)?",
    re.IGNORECASE,
)
TITLE_CONTEXT_BAD_LEAD_RE = re.compile(
    r"^(?:(?:interpretation\s+sections?|sections?|section\s+\d+|schedule\s+\d+)\s+of\s+)+",
    re.IGNORECASE,
)
TITLE_PREPOSITION_BAD_LEAD_RE = re.compile(
    r"^(?:(?:under|for|to|about|regarding|concerning|within|as|than)\s+)+(?:the\s+)?",
    re.IGNORECASE,
)
LAW_NO_REF_RE = re.compile(r"\blaw\s+no\.?\s*(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)
TITLE_CLAUSE_RE = re.compile(r"\b(?:title|may be cited as)\b", re.IGNORECASE)
PRE_EXISTING_EFFECTIVE_DATE_RE = re.compile(
    r"pre-existing accounts?.{0,160}?effective date is\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)
NEW_ACCOUNT_EFFECTIVE_DATE_RE = re.compile(
    r"new accounts?.{0,160}?effective date is\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)
RECORDS_RETAINED_AFTER_REPORTING_RE = re.compile(
    r"retention\s+period\s+of\s+six\s+\(6\)\s+years\s+after\s+the\s+date\s+of\s+reporting\s+the\s+information",
    re.IGNORECASE,
)
ACCOUNTING_RECORDS_PRESERVED_RE = re.compile(
    r"preserved\s+by\s+the\s+([A-Za-z][A-Za-z\s-]+?)\s+for\s+at\s+least\s+six\s+\(6\)\s+years\s+from\s+the\s+date\s+upon\s+which\s+they\s+were\s+created",
    re.IGNORECASE,
)
ENACTMENT_DATE_RE = re.compile(
    r"\bhereby enact\s+on\s+(?:this\s+)?([0-9]{1,2}(?:st|nd|rd|th)?(?:\s+day\s+of)?\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)
ENACTMENT_NOTICE_REFERENCE_RE = re.compile(
    r"\b(?:this|the)\s+law\s+is\s+enacted\s+on\s+the\s+date\s+specified\s+in\s+the\s+enactment\s+notice"
    r"(?:\s+in\s+respect\s+of\s+this\s+law)?\b",
    re.IGNORECASE,
)
CONSOLIDATED_VERSION_RE = re.compile(
    r"\bConsolidated\s+Version(?:\s+No\.?\s*\d+)?\s*\(([^)]+)\)",
    re.IGNORECASE,
)
UPDATED_VALUE_RE = re.compile(
    r"\b(?:last\s+updated|updated|effective\s+from)\s*(?:[:\-]|\bis\b)?\s*"
    r"([0-9]{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{4})\b",
    re.IGNORECASE,
)
REMUNERATION_RECORDKEEPING_RE = re.compile(
    r"the\s+Employee'?s\s+Remuneration\s*\(([^)]+)\)\s*,\s*and\s+the\s+applicable\s+Pay\s+Period",
    re.IGNORECASE,
)
QUESTION_SINGLE_LAW_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwho administers the (?P<title>.+? law(?:\s+\d{4})?)\??$", re.IGNORECASE),
    re.compile(
        r"\bwhen was the consolidated version of the (?P<title>.+?) published\??$",
        re.IGNORECASE,
    ),
)
LIST_POSTAMBLE_RE = re.compile(
    r"^[.)\s-]*(?:No other|Therefore|Thus|Accordingly|In summary|These are|The laws|The documents)\b",
    re.IGNORECASE,
)
ORDER_SECTION_MARKER_RE = re.compile(
    r"\bIT\s+IS\s+HEREBY\s+ORDERED(?:\s+AND\s+DIRECTED)?\s+THAT\b",
    re.IGNORECASE,
)
ORDER_SECTION_STOP_RE = re.compile(
    r"^(?:Issued by:?|SCHEDULE OF REASONS|SCHEDULE OF THE COURT'?S REASONS|Introduction|Background|Discussion and Determination)\b",
    re.IGNORECASE,
)
NUMBERED_LINE_RE = re.compile(r"^\s*\d+\.\s*")
OUTCOME_CUE_RE = re.compile(
    r"\b(?:dismissed|refused|granted|allowed|discharged|set aside|restored|proceed to trial|stayed|varied|rejected)\b",
    re.IGNORECASE,
)
EXPLICIT_OUTCOME_VERB_RE = re.compile(
    r"\b(?:is|was|are|be|been|being|shall be|must be|to be)\s+"
    r"(?:dismissed|refused|granted|allowed|discharged|set aside|restored|stayed|varied|rejected)\b",
    re.IGNORECASE,
)
COST_CUE_RE = re.compile(r"\bcosts?\b|\bno order as to costs\b", re.IGNORECASE)
OUTCOME_NOISE_RE = re.compile(
    r"\b(?:issued by|date of issue|at:\s*\d|schedule of reasons|was considered|by\s+rdc\s+\d+)\b",
    re.IGNORECASE,
)
COMPLETE_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z][a-z]|[-*]|\d+\.))")
NUMBERED_ITEM_RE = re.compile(r"(?<!\d)(\d+)\.\s+")
TITLE_ONLY_ITEM_BAD_LEAD_RE = re.compile(
    r"\b(?:this\s+is\s+confirmed|includes|states|contains|provides|application|schedule|article|section|"
    r"because|under|enacted|administered|commencement|penalty)\b",
    re.IGNORECASE,
)
TITLE_ONLY_PLACEHOLDER_RE = re.compile(
    r"^(?:citation\s+title|this\s+is\s+(?:shown|stated|explicitly\s+stated|confirmed)\b|the\s+statement\b)",
    re.IGNORECASE,
)
BODY_LIKE_TITLE_RE = re.compile(
    r"\b(?:this\s+law|requirements?\s+of\s+this\s+law|purposes?\s+of\s+this\s+law|"
    r"title\s+and\s+repeal|directors?\s+to|obligations?\b|application\s+of\s+this\s+law|"
    r"terms?\s+and\s+purposes?\s+of|penalty\s+for\s+offences?|penalty\s+for\s+an\s+offence)\b",
    re.IGNORECASE,
)
REGISTRAR_SELF_ADMIN_RE = re.compile(
    r"\b(?:the\s+registrar\s+shall\s+administer\s+this\s+law"
    r"|this\s+law\s+shall\s+be\s+administered\s+by\s+the\s+registrar"
    r"|this\s+law\s+is\s+administered\s+by\s+the\s+registrar"
    r"|administration\s+of\s+this\s+law\b[^.]{0,160}\bregistrar\b)\b",
    re.IGNORECASE | re.DOTALL,
)
TITLE_ONLY_BAD_CANDIDATE_RE = re.compile(r"\b(?:application of the arbitration law|arbitration law)\b", re.IGNORECASE)
RULER_OF_DUBAI_RE = re.compile(r"\bruler of dubai\b", re.IGNORECASE)
PENALTY_AMOUNT_RE = re.compile(r"\b(?:USD|US\$)?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})(?:\.\d+)?\b", re.IGNORECASE)
ENACTMENT_NOTICE_COMMENCEMENT_RE = re.compile(
    r"\b(?:this\s+law\s+)?shall\s+come\s+into\s+force\s+on\s+"
    r"(?:the\s+date\s+specified\s+in\s+the\s+enactment\s+notice(?:\s+in\s+respect\s+of\s+this\s+law)?"
    r"|(?:the\s+)?\d+(?:st|nd|rd|th)?\s+business\s+day\s+after\s+enactment"
    r"|\d+\s+days?\s+after\s+enactment)\b"
    r"|\b(?:this\s+law\s+)?comes?\s+into\s+force\s+on\s+the\s+date\s+specified\s+in\s+the\s+enactment\s+notice(?:\s+in\s+respect\s+of\s+this\s+law)?\b",
    re.IGNORECASE,
)
ADMINISTRATION_CLAUSE_RE = re.compile(
    r"\b("
    r"(?:this\s+law(?:\s+and\s+any\s+(?:legislation\s+made\s+for\s+the\s+purposes?\s+of\s+this\s+law|"
    r"regulations?\s+made\s+under\s+it))?\s+"
    r"(?:is|are|shall\s+be)\s+administered\s+by\s+(?:the\s+)?[^.]+)"
    r"|(?:the\s+[^.]+?\s+shall\s+administer\s+this\s+law)"
    r")\b",
    re.IGNORECASE,
)
ADMINISTRATION_ENTITY_RE = re.compile(
    r"\badministered\s+by\s+(?:the\s+)?([A-Za-z][A-Za-z\s-]+)"
    r"|the\s+([A-Za-z][A-Za-z\s-]+?)\s+shall\s+administer\s+this\s+law\b",
    re.IGNORECASE,
)
COMMON_ELEMENT_SIGNATURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
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
INTERPRETATION_SECTION_COMMON_KEYS: tuple[str, ...] = (
    "amended_or_re_enacted_reference",
    "person_definition_reference",
)
PENALTY_STOPWORDS = {
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
CRIMINAL_TRAP_TERMS = frozenset(
    (
        "jury",
        "parole",
        "miranda",
        "plea bargain",
        "plea deal",
        "bail hearing",
        "indictment",
        "grand jury",
        "arraignment",
        "felony charge",
        "criminal sentencing",
    )
)

