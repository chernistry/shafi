"""Typed query-contract compilation for structured execution routing."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from shafi.ingestion.canonical_entities import EntityAliasResolver

_ENUMERATION_RE = re.compile(r"^\s*(?:which|list|identify|name)\b", re.IGNORECASE)
_SPECIFIC_DATE_RE = re.compile(
    r"\b(?:on|as at|as of|before|after)\s+(\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
    re.IGNORECASE,
)
_CASE_NUMBER_RE = re.compile(
    r"\b(?P<case_number>(?:CFI|CA|ARB|SCT|TCD|ENF|DEC)\s*\d{1,4}/\d{4})\b",
    re.IGNORECASE,
)
_LOW_SIGNAL_ENTITY_TOKENS = frozenset(
    {
        "a",
        "all",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "being",
        "both",
        "by",
        "case",
        "cases",
        "date",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "is",
        "it",
        "its",
        "law",
        "laws",
        "main",
        "of",
        "on",
        "or",
        "other",
        "parties",
        "party",
        "same",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "those",
        "under",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "with",
    }
)


def _resolved_entities_factory() -> list[ResolvedContractEntity]:
    """Build a typed empty resolved-entity list for Pydantic defaults."""

    return []


def _comparison_axes_factory() -> list[str]:
    """Build a typed empty comparison-axis list for Pydantic defaults."""

    return []


def _support_slots_factory() -> list[str]:
    """Build a typed empty support-slot list for Pydantic defaults."""

    return []


def _execution_plan_factory() -> list[ExecutionEngine]:
    """Build a typed default execution-plan list for Pydantic defaults."""

    return [ExecutionEngine.STANDARD_RAG]


def _normalize_query_text(query_text: str) -> str:
    return re.sub(r"\s+", " ", query_text.strip()).lower()


def _has_compare_signal(query_text: str) -> bool:
    q = _normalize_query_text(query_text)
    if not q:
        return False
    markers = (
        "compare",
        "compared with",
        "contrast",
        "difference between",
        "earlier decision date",
        "later decision date",
        "earlier date of issue",
        "later date of issue",
        "earlier enactment date",
        "later enactment date",
        "issued earlier",
        "issued later",
        "which case was issued",
        "which law has a later",
        "which law has an earlier",
        "different from",
        "same legal entities",
        "same legal entity",
        "same individual",
        "same individuals",
        "same parties",
        "same judge",
        "same party",
        "same authority",
        "in common",
        "appeared in both",
        "common to ",
        "common to both",
        "have in common",
        "share a judge",
        "share an entity",
        "common party",
        "common judge",
        "which of the two",
        "which came first",
    )
    return any(marker in q for marker in markers)


def _has_temporal_signal(query_text: str) -> bool:
    q = _normalize_query_text(query_text)
    if not q:
        return False
    markers = (
        "when did",
        "when was",
        "come into force",
        "came into force",
        "currently in force",
        "in force on",
        "effective date",
        "effective on",
        "commencement date",
        "commencement",
        "amendment",
        "amended",
        "superseded",
        "replaced",
        "repealed",
        "current version",
        "historical version",
    )
    return any(marker in q for marker in markers)


def _has_lookup_provision_signal(query_text: str) -> bool:
    q = _normalize_query_text(query_text)
    if not q:
        return False
    if not any(label in q for label in ("article ", "section ", "schedule ", "definition", "clause ")):
        return False
    return any(
        marker in q
        for marker in (
            "what does",
            "what is",
            "what are",
            "provide",
            "provides",
            "say",
            "says",
            "mean",
            "definition of",
            "defined as",
            "set out in",
        )
    )


def _detect_field_name(query_text: str) -> str:
    q = _normalize_query_text(query_text)
    if not q:
        return ""
    field_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("issued_by", ("who issued", "issued by", "issuing authority", "made by")),
        ("title", ("title of", "citation title", "citation titles", "what is the title")),
        (
            "commencement_date",
            (
                "commencement date",
                "date of commencement",
                "come into force",
                "came into force",
                "coming into force",
                "enactment date",
                "date of enactment",
            ),
        ),
        ("law_number", ("law number", "number of the law")),
        ("case_number", ("case number", "claim number")),
        ("judge", ("which judge", "who was the judge", "who was the presiding judge", "presided over")),
        ("claimant", ("claimant", "claimants")),
        ("respondent", ("respondent", "respondents")),
        ("party", ("party", "parties")),
        ("outcome", ("outcome", "result of", "held that", "final ruling")),
        ("date", ("date of issue", "issue date", "judgment date", "decision date", "what date")),
        ("authority", ("administered by", "authority administers", "which authority")),
    )
    for field_name, patterns in field_patterns:
        if any(pattern in q for pattern in patterns):
            return field_name
    return ""


_COUNT_QUESTION_RE: re.Pattern[str] = re.compile(
    r"\b(how many|total number of|number of distinct|count of|how many unique)\b"
)


def _has_lookup_field_signal(query_text: str, *, answer_type: str) -> bool:
    q = _normalize_query_text(query_text)
    normalized_answer_type = answer_type.strip().lower()
    if not q:
        return False
    # Count questions ("how many claimants", "total number of parties") ask for
    # aggregation — DB lookup returns a field value (name/date), not a count.
    # Route to RAG instead of LOOKUP_FIELD to avoid type-mismatch null answers.
    if normalized_answer_type == "number" and _COUNT_QUESTION_RE.search(q):
        return False
    detected_field = _detect_field_name(q)
    if detected_field:
        # commencement_date routing only makes sense for date-type questions.
        # name/names questions about cases that mention "came into force" should
        # not be mis-routed to the commencement_date field — they expect case refs.
        # commencement_date routing only makes sense for date-type questions.
        return not (detected_field == "commencement_date" and normalized_answer_type not in {"date"})
    return normalized_answer_type in {"name", "names", "date", "number"} and any(
        lead in q for lead in ("who is", "who was", "what is the", "what was the", "which is the", "which was the")
    )


def _detect_comparison_axes(query_text: str) -> list[str]:
    q = _normalize_query_text(query_text)
    axes: list[str] = []
    axis_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("judge", ("judge", "judges", "presided")),
        ("party", ("party", "parties", "claimant", "respondent", "appellant", "applicant")),
        ("authority", ("issued by", "issuing authority", "administered by", "authority")),
        ("date", ("date", "issued first", "issued earlier", "year")),
        ("title", ("title", "citation title")),
        ("outcome", ("outcome", "result", "held")),
        ("entity", ("entity", "entities", "in common", "common elements")),
    )
    for axis_name, patterns in axis_patterns:
        if any(pattern in q for pattern in patterns):
            axes.append(axis_name)
    return axes


def _has_negative_polarity(query_text: str) -> bool:
    q = f" {_normalize_query_text(query_text)} "
    if not q.strip():
        return False
    return any(
        marker in q
        for marker in (
            " not ",
            " no ",
            " without ",
            " except ",
            " did not ",
            " does not ",
            " was not ",
            " were not ",
            " didn't ",
            " doesn't ",
        )
    )


def _extract_specific_date(query_text: str) -> str:
    match = _SPECIFIC_DATE_RE.search(query_text)
    if match is None:
        return ""
    return re.sub(r"\s+", " ", match.group(1).strip())


def _normalize_case_number(case_number: str) -> str:
    """Normalize a case number into a stable lookup token.

    Args:
        case_number: Raw case-number surface form.

    Returns:
        str: Compact canonical token suitable for entity IDs.
    """

    return re.sub(r"[^a-z0-9]+", "", case_number.casefold())


def _normalize_case_doc_id(case_number: str) -> str:
    """Normalize a case number into the registry doc-id convention.

    Args:
        case_number: Raw case-number surface form.

    Returns:
        str: Registry doc-id token used by compiled case objects.
    """

    doc_id = re.sub(r"[^a-z0-9]+", "_", case_number.casefold())
    return doc_id.strip("_")


def _extract_explicit_case_entities(query_text: str) -> list[ResolvedContractEntity]:
    """Extract explicit case-number entities from the query text.

    Args:
        query_text: Raw user query.

    Returns:
        list[ResolvedContractEntity]: Deterministic case-number entities in
        query order.
    """

    entities: list[ResolvedContractEntity] = []
    seen_doc_ids: set[str] = set()
    for match in _CASE_NUMBER_RE.finditer(query_text or ""):
        case_number = re.sub(r"\s+", " ", match.group("case_number").strip())
        doc_id = _normalize_case_doc_id(case_number)
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        entities.append(
            ResolvedContractEntity(
                canonical_id=f"case_number:{_normalize_case_number(case_number)}",
                canonical_form=case_number,
                entity_type="case_number",
                source_doc_ids=[doc_id],
                mention_text=case_number,
                mention_start=match.start("case_number"),
                mention_end=match.end("case_number"),
            )
        )
    return entities


def _is_low_signal_entity_mention(mention_text: str) -> bool:
    """Detect entity mentions that are too generic to route on safely.

    Args:
        mention_text: Raw mention span from the query text.

    Returns:
        bool: True when the mention is a stopword-like or otherwise low-signal span.
    """

    normalized = _normalize_query_text(mention_text)
    if not normalized:
        return True
    if normalized in _LOW_SIGNAL_ENTITY_TOKENS:
        return True
    if len(normalized) < 3:
        return True
    return not any(char.isdigit() for char in normalized) and len(normalized.split()) == 1


def _entity_priority(
    *,
    entity_type: str,
    field_name: str,
    predicate: PredicateType,
    answer_type: str,
) -> int:
    """Rank resolved entity kinds for safer field-routing.

    Args:
        entity_type: Resolved canonical entity type.
        field_name: Normalized target field name.
        predicate: Compiled predicate family.
        answer_type: Requested answer type.

    Returns:
        int: Higher values should be preferred as primary entities.
    """

    if predicate is PredicateType.COMPARE:
        return {
            "case_number": 100,
            "law_title": 95,
            "judge": 70,
            "authority": 65,
            "party": 60,
        }.get(entity_type, 10)

    if predicate is not PredicateType.LOOKUP_FIELD:
        return {
            "case_number": 100,
            "law_title": 95,
            "judge": 70,
            "authority": 65,
            "party": 60,
        }.get(entity_type, 10)

    case_fields = {"claimant", "respondent", "party", "judge", "outcome", "case_number"}
    law_fields = {"title", "issued_by", "authority", "law_number", "commencement_date"}
    temporal_fields = {"date"}
    if field_name in case_fields:
        return {
            "case_number": 100,
            "law_title": 55,
            "judge": 35,
            "party": 25,
            "authority": 15,
        }.get(entity_type, 10)
    if field_name in law_fields:
        return {
            "law_title": 100,
            "case_number": 45,
            "authority": 35,
            "party": 15,
            "judge": 10,
        }.get(entity_type, 10)
    if field_name in temporal_fields and answer_type == "date":
        return {
            "case_number": 100,
            "law_title": 95,
            "authority": 25,
            "party": 20,
            "judge": 15,
        }.get(entity_type, 10)
    return {
        "case_number": 100,
        "law_title": 95,
        "judge": 70,
        "authority": 65,
        "party": 60,
    }.get(entity_type, 10)


def _should_promote_lookup_to_compare(
    *,
    field_name: str,
    answer_type: str,
    primary_entities: list[ResolvedContractEntity],
    query_text: str,
) -> bool:
    """Detect lookup-field queries that are actually compare/composite requests.

    Args:
        field_name: Detected lookup field name.
        answer_type: Requested answer type.
        primary_entities: Resolved primary entities.
        query_text: Raw query text.

    Returns:
        bool: True when the routing should prefer compare/composite execution.
    """

    if len(primary_entities) < 2:
        return False
    normalized_query = _normalize_query_text(query_text)
    if answer_type == "boolean":
        return True
    if field_name in {"party", "authority", "date", "title", "law_number"}:
        return True
    return any(marker in normalized_query for marker in (" and ", " or ", " both ", " earlier ", " later "))


def _has_historical_scope_signal(query_text: str) -> bool:
    q = _normalize_query_text(query_text)
    if not q:
        return False
    return any(
        marker in q
        for marker in (
            "historical",
            "at the time",
            "as originally enacted",
            "previous version",
            "former version",
            "before it was amended",
        )
    )


def _is_enumeration_query(query_text: str) -> bool:
    return bool(_ENUMERATION_RE.search(query_text))


class PredicateType(StrEnum):
    """Supported structured predicate families."""

    LOOKUP_FIELD = "lookup_field"
    LOOKUP_PROVISION = "lookup_provision"
    COMPARE = "compare"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    ENUMERATE = "enumerate"
    EXPLAIN = "explain"


class ExecutionEngine(StrEnum):
    """Execution engines selected by the compiled contract."""

    FIELD_LOOKUP = "field_lookup"
    COMPARE_JOIN = "compare_join"
    TEMPORAL_QUERY = "temporal_query"
    STANDARD_RAG = "standard_rag"
    COMPOSITE = "composite"


class Polarity(StrEnum):
    """Assertion polarity inferred from the query."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class TimeScopeType(StrEnum):
    """Temporal scope classifications for routing and support obligations."""

    CURRENT = "current"
    HISTORICAL = "historical"
    SPECIFIC_DATE = "specific_date"
    AMENDMENT_PERIOD = "amendment_period"


class TimeScope(BaseModel):
    """Typed temporal scope carried inside the query contract."""

    model_config = ConfigDict(frozen=True)

    scope_type: TimeScopeType = TimeScopeType.CURRENT
    reference_date: str = ""


class ResolvedContractEntity(BaseModel):
    """Resolved canonical entity mention used by the contract."""

    model_config = ConfigDict(frozen=True)

    canonical_id: str
    canonical_form: str = ""
    entity_type: str = ""
    source_doc_ids: list[str] = Field(default_factory=list)
    mention_text: str = ""
    mention_start: int = 0
    mention_end: int = 0


class QueryContract(BaseModel):
    """Compiled execution contract for one query."""

    model_config = ConfigDict(frozen=True)

    query_text: str
    answer_type: str
    primary_entities: list[ResolvedContractEntity] = Field(default_factory=_resolved_entities_factory)
    constraint_entities: list[ResolvedContractEntity] = Field(default_factory=_resolved_entities_factory)
    predicate: PredicateType = PredicateType.EXPLAIN
    field_name: str = ""
    comparison_axes: list[str] = Field(default_factory=_comparison_axes_factory)
    polarity: Polarity = Polarity.UNKNOWN
    time_scope: TimeScope = Field(default_factory=TimeScope)
    required_support_slots: list[str] = Field(default_factory=_support_slots_factory)
    execution_plan: list[ExecutionEngine] = Field(default_factory=_execution_plan_factory)
    confidence: float = 0.0


class QueryContractCompiler:
    """Compile raw queries into typed routing contracts."""

    def __init__(self, *, alias_resolver: EntityAliasResolver | None = None) -> None:
        """Initialize the compiler.

        Args:
            alias_resolver: Optional preloaded canonical alias resolver.
        """

        self._alias_resolver = alias_resolver
        self._cluster_by_id = (
            {cluster.canonical_id: cluster for cluster in alias_resolver.iter_clusters()}
            if alias_resolver is not None
            else {}
        )

    @classmethod
    def from_alias_registry_path(cls, path: str | Path) -> QueryContractCompiler:
        """Load a compiler from a persisted alias registry path.

        Args:
            path: Alias registry JSON path.

        Returns:
            QueryContractCompiler: Compiler backed by the loaded resolver.
        """

        if not str(path).strip():
            return cls()
        resolved_path = Path(path)
        if not resolved_path.exists():
            return cls()
        return cls(alias_resolver=EntityAliasResolver.load(resolved_path))

    def compile(
        self,
        query_text: str,
        *,
        answer_type: str = "free_text",
    ) -> QueryContract:
        """Compile a raw query into a structured contract.

        Args:
            query_text: Raw user query.
            answer_type: Answer-type hint from the request payload.

        Returns:
            QueryContract: Typed routing contract.
        """

        normalized_answer_type = answer_type.strip().lower() or "free_text"
        predicate = self.detect_predicate(query_text, answer_type=normalized_answer_type)
        field_name = _detect_field_name(query_text)
        primary_entities: list[ResolvedContractEntity]
        constraint_entities: list[ResolvedContractEntity]
        primary_entities, constraint_entities = self.resolve_entities(
            query_text,
            predicate=predicate,
            field_name=field_name,
            answer_type=normalized_answer_type,
        )
        if predicate is PredicateType.LOOKUP_FIELD and _should_promote_lookup_to_compare(
            field_name=field_name,
            answer_type=normalized_answer_type,
            primary_entities=primary_entities,
            query_text=query_text,
        ):
            predicate = PredicateType.COMPARE
        comparison_axes = _detect_comparison_axes(query_text)
        polarity = self.detect_polarity(query_text)
        time_scope = self.detect_time_scope(query_text)

        provisional = QueryContract(
            query_text=query_text,
            answer_type=normalized_answer_type,
            primary_entities=primary_entities,
            constraint_entities=constraint_entities,
            predicate=predicate,
            field_name=field_name,
            comparison_axes=comparison_axes,
            polarity=polarity,
            time_scope=time_scope,
            confidence=0.0,
        )
        confidence = self._estimate_confidence(provisional)
        execution_plan = self.plan_execution(provisional, confidence=confidence)
        required_support_slots = self.infer_support_slots(
            provisional.model_copy(update={"execution_plan": execution_plan, "confidence": confidence})
        )
        return provisional.model_copy(
            update={
                "required_support_slots": required_support_slots,
                "execution_plan": execution_plan,
                "confidence": confidence,
            }
        )

    def detect_predicate(self, query_text: str, *, answer_type: str) -> PredicateType:
        """Detect the dominant predicate family for a query.

        Args:
            query_text: Raw user query.
            answer_type: Normalized answer type.

        Returns:
            PredicateType: Structured predicate family.
        """

        if _has_compare_signal(query_text):
            return PredicateType.COMPARE
        if _has_temporal_signal(query_text):
            return PredicateType.TEMPORAL
        if _has_lookup_provision_signal(query_text):
            return PredicateType.LOOKUP_PROVISION
        if _has_lookup_field_signal(query_text, answer_type=answer_type):
            return PredicateType.LOOKUP_FIELD
        if _is_enumeration_query(query_text):
            return PredicateType.ENUMERATE
        if answer_type == "boolean":
            return PredicateType.BOOLEAN
        return PredicateType.EXPLAIN

    def resolve_entities(
        self,
        query_text: str,
        *,
        predicate: PredicateType = PredicateType.EXPLAIN,
        field_name: str = "",
        answer_type: str = "free_text",
    ) -> tuple[list[ResolvedContractEntity], list[ResolvedContractEntity]]:
        """Resolve query entities and partition them into primary and constraint roles.

        Args:
            query_text: Raw user query.
            predicate: Provisional predicate family.
            field_name: Normalized target field name.
            answer_type: Requested answer type.

        Returns:
            tuple[list[ResolvedContractEntity], list[ResolvedContractEntity]]: Primary and constraint entities.
        """

        query_kind_is_compare = predicate is PredicateType.COMPARE or _has_compare_signal(query_text)
        primary_ids: set[str] = set()
        constraint_ids: set[str] = set()
        primary_entities: list[ResolvedContractEntity] = []
        constraint_entities: list[ResolvedContractEntity] = []
        candidates: list[ResolvedContractEntity] = []
        candidates.extend(_extract_explicit_case_entities(query_text))

        if self._alias_resolver is not None:
            for (start, end), canonical_id in self._alias_resolver.resolve_all(query_text):
                mention_text = query_text[start:end]
                if _is_low_signal_entity_mention(mention_text):
                    continue
                entity = self._build_resolved_entity(
                    canonical_id=canonical_id,
                    mention_text=mention_text,
                    mention_start=start,
                    mention_end=end,
                )
                candidates.append(entity)

        if not candidates:
            return [], []

        candidates.sort(
            key=lambda entity: (
                -_entity_priority(
                    entity_type=entity.entity_type,
                    field_name=field_name,
                    predicate=predicate,
                    answer_type=answer_type,
                ),
                entity.mention_start,
                -(entity.mention_end - entity.mention_start),
                entity.canonical_id,
            )
        )

        for entity in candidates:
            canonical_id = entity.canonical_id
            if query_kind_is_compare:
                if canonical_id not in primary_ids and len(primary_entities) < 2:
                    primary_ids.add(canonical_id)
                    primary_entities.append(entity)
                elif canonical_id not in primary_ids and canonical_id not in constraint_ids:
                    constraint_ids.add(canonical_id)
                    constraint_entities.append(entity)
                continue

            if canonical_id not in primary_ids and not primary_entities:
                primary_ids.add(canonical_id)
                primary_entities.append(entity)
                continue
            if canonical_id not in primary_ids and canonical_id not in constraint_ids:
                constraint_ids.add(canonical_id)
                constraint_entities.append(entity)

        return primary_entities, constraint_entities

    @staticmethod
    def detect_time_scope(query_text: str) -> TimeScope:
        """Detect the temporal scope embedded in the query.

        Args:
            query_text: Raw user query.

        Returns:
            TimeScope: Structured time scope classification.
        """

        specific_date = _extract_specific_date(query_text)
        if specific_date:
            return TimeScope(scope_type=TimeScopeType.SPECIFIC_DATE, reference_date=specific_date)
        if _has_historical_scope_signal(query_text):
            return TimeScope(scope_type=TimeScopeType.HISTORICAL)
        if "amendment period" in query_text.lower():
            return TimeScope(scope_type=TimeScopeType.AMENDMENT_PERIOD)
        return TimeScope(scope_type=TimeScopeType.CURRENT)

    @staticmethod
    def detect_polarity(query_text: str) -> Polarity:
        """Detect query polarity from lexical negation cues.

        Args:
            query_text: Raw user query.

        Returns:
            Polarity: Positive, negative, or unknown polarity.
        """

        if not query_text.strip():
            return Polarity.UNKNOWN
        if _has_negative_polarity(query_text):
            return Polarity.NEGATIVE
        return Polarity.POSITIVE

    def plan_execution(
        self,
        contract: QueryContract,
        *,
        confidence: float | None = None,
    ) -> list[ExecutionEngine]:
        """Map a compiled contract to its intended execution engine.

        Args:
            contract: Provisional contract.
            confidence: Optional precomputed confidence override.

        Returns:
            list[ExecutionEngine]: Planned execution engines.
        """

        resolved_confidence = contract.confidence if confidence is None else confidence
        if resolved_confidence < 0.55:
            return [ExecutionEngine.STANDARD_RAG]
        if contract.predicate is PredicateType.LOOKUP_FIELD:
            return [ExecutionEngine.FIELD_LOOKUP]
        if contract.predicate is PredicateType.COMPARE:
            if contract.time_scope.scope_type is not TimeScopeType.CURRENT:
                return [ExecutionEngine.COMPOSITE]
            return [ExecutionEngine.COMPARE_JOIN]
        if contract.predicate is PredicateType.TEMPORAL:
            return [ExecutionEngine.TEMPORAL_QUERY]
        return [ExecutionEngine.STANDARD_RAG]

    def infer_support_slots(self, contract: QueryContract) -> list[str]:
        """Infer the support obligations implied by a contract.

        Args:
            contract: Compiled contract.

        Returns:
            list[str]: Required support slot identifiers.
        """

        slots: list[str] = []
        if contract.predicate is PredicateType.LOOKUP_FIELD:
            if contract.field_name:
                slots.append(f"page_with_{contract.field_name}_field")
            if contract.primary_entities:
                slots.append("page_for_target_entity")
        elif contract.predicate is PredicateType.COMPARE:
            slots.append("page_for_each_primary_entity")
            for axis in contract.comparison_axes or ["entity"]:
                slots.append(f"page_with_{axis}_field")
        elif contract.predicate is PredicateType.TEMPORAL:
            temporal_slots = {
                TimeScopeType.AMENDMENT_PERIOD: "page_with_amendment_notice",
                TimeScopeType.SPECIFIC_DATE: "page_with_effective_date_context",
                TimeScopeType.HISTORICAL: "page_with_historical_version_context",
                TimeScopeType.CURRENT: "page_with_commencement_or_status",
            }
            slots.append(temporal_slots[contract.time_scope.scope_type])
        elif contract.predicate is PredicateType.LOOKUP_PROVISION:
            slots.append("page_with_provision_text")
        elif contract.predicate is PredicateType.ENUMERATE:
            slots.append("pages_covering_each_matching_item")
        unique_slots: list[str] = []
        seen: set[str] = set()
        for slot in slots:
            if slot in seen:
                continue
            seen.add(slot)
            unique_slots.append(slot)
        return unique_slots

    def _build_resolved_entity(
        self,
        *,
        canonical_id: str,
        mention_text: str,
        mention_start: int,
        mention_end: int,
    ) -> ResolvedContractEntity:
        """Build a resolved entity record from alias-cluster metadata.

        Args:
            canonical_id: Canonical alias-cluster ID.
            mention_text: Raw span from the query.
            mention_start: Inclusive start offset.
            mention_end: Exclusive end offset.

        Returns:
            ResolvedContractEntity: Typed resolved entity payload.
        """

        cluster = self._cluster_by_id.get(canonical_id)
        return ResolvedContractEntity(
            canonical_id=canonical_id,
            canonical_form="" if cluster is None else cluster.canonical_form,
            entity_type="" if cluster is None else cluster.entity_type.value,
            source_doc_ids=[] if cluster is None else list(cluster.source_doc_ids),
            mention_text=mention_text,
            mention_start=mention_start,
            mention_end=mention_end,
        )

    @staticmethod
    def _estimate_confidence(contract: QueryContract) -> float:
        """Estimate contract confidence from deterministic signals.

        Args:
            contract: Provisional contract.

        Returns:
            float: Confidence score in the `[0.0, 0.99]` range.
        """

        confidence = 0.25
        if contract.primary_entities:
            confidence += 0.25
        if contract.constraint_entities:
            confidence += 0.05
        if contract.predicate is not PredicateType.EXPLAIN:
            confidence += 0.15
        if contract.field_name:
            confidence += 0.15
        if contract.comparison_axes:
            confidence += 0.1
        if contract.time_scope.scope_type is not TimeScopeType.CURRENT:
            confidence += 0.05
        if contract.polarity is not Polarity.UNKNOWN:
            confidence += 0.05
        if contract.answer_type in {"boolean", "number", "date", "name", "names"}:
            confidence += 0.05
        return max(0.0, min(confidence, 0.99))
