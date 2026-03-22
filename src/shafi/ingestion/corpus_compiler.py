"""Offline compiler that converts parsed documents into typed legal objects."""

from __future__ import annotations

import re
from pathlib import Path

from shafi.core.law_notice_support import extract_enactment_authority, extract_enactment_date
from shafi.ingestion.manual_domain_overrides import apply_manual_domain_override, get_manual_domain_override
from shafi.ingestion.parser import DocumentParser
from shafi.models import DocType, ParsedDocument
from shafi.models.legal_objects import (
    AmendmentObject,
    ArticleNode,
    CaseObject,
    CaseParty,
    CorpusRegistry,
    LawObject,
    LegalDocType,
    LegalEntity,
    LegalEntityType,
    OrderObject,
    OtherLegalObject,
    PracticeDirectionObject,
    ProvenanceLink,
    ProvenanceLinkType,
)

type CompiledLegalObject = (
    LawObject | CaseObject | OrderObject | PracticeDirectionObject | AmendmentObject | OtherLegalObject
)

_LAW_NUMBER_RE = re.compile(r"\b(?:Law|Act|Regulation|Order)\s+No\.?\s*(\d+)\s+of\s+(\d{4})\b", re.IGNORECASE)
_YEAR_TITLE_RE = re.compile(r"\b(19|20)\d{2}\b")
_CASE_NUMBER_RE = re.compile(
    r"\b(?:CFI|CA|ARB|SCT|TCD|ENF|DEC)\s+\d{3}/\d{4}\b|\[\d{4}\]\s+[A-Z]{2,10}\s+\d+\b",
    re.IGNORECASE,
)
_JUDGE_LINE_HINT_RE = re.compile(
    r"\b(?:justice|judge|chief justice|h\.e\.|he\.|hon\.|honourable|sir|lady|lord)\b",
    re.IGNORECASE,
)
_JUDGE_LINE_BLOCKLIST_RE = re.compile(
    r"\b(?:assistant\s+registrar|registrar|claimant|claimants|respondent|respondents|appellant|appellants|"
    r"applicant|applicants|defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\b",
    re.IGNORECASE,
)
_ISSUED_BY_RE = re.compile(r"Issued by[:\s]+(?P<issuer>[^\n]+)", re.IGNORECASE)
_COMMENCEMENT_RE = re.compile(
    r"(?:Date of Issue|Date of Re-issue|effective date|comes into force on|commenc(?:e|ed|ement on))[:\s]+(?P<date>[^\n.]+)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_LAW_TITLE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z&,'()/-]+\s+)+(?:Law|Act|Regulations?|Order|Code)(?:\s+No\.?\s*\d+\s+of\s+\d{4}|\s+\d{4})?\b"
)
_JUDGE_RE = re.compile(
    r"\b(?:Justice|Judge|Chief Justice|Registrar|Sir)\s+[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}\b"
)
_COURT_RE = re.compile(
    r"\b(?:Supreme Court|Court of Appeal(?: of(?: the)? DIFC Courts?)?|Court of First Instance(?: of(?: the)? DIFC Courts?)?|"
    r"DIFC Courts?|Small Claims Tribunal|Technology and Construction Division|Arbitral Tribunal)\b",
    re.IGNORECASE,
)
_PARTY_ROLE_RE = re.compile(
    r"^(?:claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)(?:/[A-Za-z]+)?[:\s,-]+"
    r"([A-Z][A-Za-z0-9&.,'()/-]{2,90})$",
    re.IGNORECASE | re.MULTILINE,
)
_ROLE_ONLY_LINE_RE = re.compile(
    r"^(claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)(?:/[A-Za-z]+)?$",
    re.IGNORECASE,
)
_ROLE_VALUE_LINE_RE = re.compile(
    r"^(claimant|claimants|respondent|respondents|appellant|appellants|applicant|applicants|"
    r"defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)(?:/[A-Za-z]+)?[:\\s,-]+(.+)$",
    re.IGNORECASE,
)
_CAPTION_SPLIT_RE = re.compile(r"\b(?:v\.?|vs\.?|versus)\b", re.IGNORECASE)
_OUTCOME_RE = re.compile(
    r"\b(?:held that|ordered that|dismissed|allowed|granted|refused|enforceable|set aside)\b[^.]{0,200}\.",
    re.IGNORECASE,
)
_SECTION_LABEL_RE = re.compile(
    r"^(?P<label>(?:Article|Section|Schedule|Part|Chapter)\s+[A-Za-z0-9().-]+)\s*(?:[-:]\s*(?P<title>.+))?$",
    re.IGNORECASE,
)
_LOW_SIGNAL_PARTY_VALUES = frozenset(
    {
        "/",
        "and",
        "application",
        "claim",
        "costs",
        "creditor",
        "debtor",
        "defendant",
        "further",
        "from",
        "has",
        "hearing",
        "intends",
        "its",
        "order",
        "pursuant",
        "pondent",
        "relies",
        "respondent",
        "served",
        "shall",
        "submitted",
        "the",
        "was",
    }
)
_LOW_SIGNAL_PARTY_SUFFIXES = frozenset(
    {
        "appellant",
        "appellee",
        "applicant",
        "claimant",
        "defendant",
        "petitioner",
        "respondent",
    }
)
_CAPTION_ROLE_SUFFIX_RE = re.compile(
    r"\s*(?:\(|,)?\s*(?:claimant|claimants|respondent|respondents|appellant|appellants|appellee|appellees|"
    r"applicant|applicants|defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\s*\)?\s*$",
    re.IGNORECASE,
)
_ROLE_CANONICAL_MAP = {
    "claimants": "claimant",
    "respondents": "respondent",
    "appellants": "appellant",
    "appellees": "appellee",
    "applicants": "applicant",
    "defendants": "defendant",
    "plaintiffs": "plaintiff",
    "petitioners": "petitioner",
}


class CorpusCompiler:
    """Compile parsed legal documents into a normalized corpus registry."""

    def __init__(self, *, parser: DocumentParser | None = None) -> None:
        """Initialize the compiler.

        Args:
            parser: Optional parser instance for directory-based compilation.
        """
        self._parser = parser or DocumentParser()

    def compile_corpus(self, doc_dir: str | Path) -> CorpusRegistry:
        """Parse and compile every supported document under a directory.

        Args:
            doc_dir: Directory containing source documents.

        Returns:
            A compiled corpus registry.
        """
        docs = self._parser.parse_directory(Path(doc_dir))
        return self.compile_documents(docs)

    def compile_documents(self, docs: list[ParsedDocument]) -> CorpusRegistry:
        """Compile already-parsed documents into a corpus registry.

        Args:
            docs: Parsed documents to compile.

        Returns:
            A normalized corpus registry with entities and links.
        """
        registry = CorpusRegistry(source_doc_count=len(docs))
        for doc in docs:
            _legal_doc_type, compiled = self.compile_document(doc)
            self._store_object(registry, compiled)
        registry.entities = self._build_entities(registry)
        registry.links = self._build_links(registry)
        return registry

    def compile_document(self, parsed_doc: ParsedDocument) -> tuple[LegalDocType, CompiledLegalObject]:
        """Compile one parsed document into a typed legal object.

        Args:
            parsed_doc: Parsed document from the ingestion parser.

        Returns:
            A tuple of normalized legal doc type and compiled legal object.
        """
        legal_doc_type = self.resolve_document_type(parsed_doc)
        if legal_doc_type in {LegalDocType.LAW, LegalDocType.REGULATION, LegalDocType.SCHEDULE}:
            compiled = self.extract_law_metadata(parsed_doc, legal_doc_type=legal_doc_type)
            return legal_doc_type, self._apply_manual_override(parsed_doc, compiled)
        if legal_doc_type == LegalDocType.CASE:
            compiled = self.extract_case_metadata(parsed_doc)
            return legal_doc_type, self._apply_manual_override(parsed_doc, compiled)
        if legal_doc_type == LegalDocType.PRACTICE_DIRECTION:
            compiled = self.extract_practice_direction_metadata(parsed_doc)
            return legal_doc_type, self._apply_manual_override(parsed_doc, compiled)
        if legal_doc_type in {LegalDocType.ORDER, LegalDocType.ENACTMENT_NOTICE}:
            compiled = self.extract_order_metadata(parsed_doc, legal_doc_type=legal_doc_type)
            return legal_doc_type, self._apply_manual_override(parsed_doc, compiled)
        if legal_doc_type == LegalDocType.AMENDMENT:
            compiled = self.extract_amendment_metadata(parsed_doc)
            return legal_doc_type, self._apply_manual_override(parsed_doc, compiled)
        compiled = self.extract_other_metadata(parsed_doc)
        return legal_doc_type, self._apply_manual_override(parsed_doc, compiled)

    def resolve_document_type(self, parsed_doc: ParsedDocument) -> LegalDocType:
        """Resolve the normalized legal document type for a parsed document.

        Args:
            parsed_doc: Parsed document from the parser.

        Returns:
            The normalized legal document type.
        """
        title = parsed_doc.title.casefold()
        full_text = parsed_doc.full_text.casefold()
        if "practice direction" in title or "practice direction" in full_text:
            return LegalDocType.PRACTICE_DIRECTION
        if "enactment notice" in title or "enactment notice" in full_text:
            return LegalDocType.ENACTMENT_NOTICE
        if "amendment law" in title or "amends" in title:
            return LegalDocType.AMENDMENT
        if "schedule" in title:
            return LegalDocType.SCHEDULE
        if parsed_doc.doc_type == DocType.CASE_LAW or _CASE_NUMBER_RE.search(parsed_doc.title):
            return LegalDocType.CASE
        if parsed_doc.doc_type == DocType.REGULATION or "regulation" in title or "regulations" in title:
            return LegalDocType.REGULATION
        if parsed_doc.doc_type == DocType.STATUTE or any(token in title for token in ("law", "act", "code")):
            return LegalDocType.LAW
        if "order" in title or "it is hereby ordered" in full_text:
            return LegalDocType.ORDER
        if "policy" in title:
            return LegalDocType.POLICY
        return LegalDocType.OTHER

    def extract_law_metadata(
        self,
        parsed_doc: ParsedDocument,
        *,
        legal_doc_type: LegalDocType = LegalDocType.LAW,
    ) -> LawObject:
        """Extract structured law-like metadata.

        Args:
            parsed_doc: Parsed document to compile.
            legal_doc_type: Normalized legal document type to assign.

        Returns:
            A compiled law object.
        """
        title = parsed_doc.title.strip()
        law_number, year = self._extract_law_number_and_year(parsed_doc)
        short_title = self._extract_short_title(parsed_doc) or title
        issuing_authority = extract_enactment_authority(
            source_text=parsed_doc.full_text,
            fallback=self._first_group_match(_ISSUED_BY_RE, parsed_doc.full_text, "issuer"),
        )
        commencement_date = extract_enactment_date(
            source_text=parsed_doc.full_text,
            fallback=self._first_group_match(_COMMENCEMENT_RE, parsed_doc.full_text, "date"),
        )
        page_ids = self._page_ids(parsed_doc)
        page_texts = self._page_texts(parsed_doc)
        field_page_ids = self._field_page_ids(
            parsed_doc,
            {
                "title": [title, short_title],
                "law_number": [
                    value
                    for value in (law_number, f"Law No. {law_number} of {year}" if law_number and year else "")
                    if value
                ],
                "issued_by": issuing_authority,
                "authority": issuing_authority,
                "commencement_date": commencement_date,
                "date": commencement_date,
            },
            fallback_fields={"title", "law_number"},
        )
        return LawObject(
            object_id=f"{legal_doc_type.value}:{parsed_doc.doc_id}",
            doc_id=parsed_doc.doc_id,
            title=title,
            source_path=parsed_doc.source_path,
            page_ids=page_ids,
            source_text=parsed_doc.full_text.strip(),
            page_texts=page_texts,
            field_page_ids=field_page_ids,
            legal_doc_type=legal_doc_type,
            short_title=short_title,
            law_number=law_number,
            year=year,
            issuing_authority=issuing_authority,
            commencement_date=commencement_date,
            amendment_refs=self._extract_amendment_refs(parsed_doc),
            article_tree=self._extract_article_tree(parsed_doc),
        )

    def extract_case_metadata(self, parsed_doc: ParsedDocument) -> CaseObject:
        """Extract structured case metadata.

        Args:
            parsed_doc: Parsed document to compile.

        Returns:
            A compiled case object.
        """
        title = parsed_doc.title.strip()
        case_number = self._first_match(_CASE_NUMBER_RE, f"{parsed_doc.title}\n{parsed_doc.full_text}")
        caption_parties = self._extract_caption_parties(parsed_doc.title)
        role_parties = [
            CaseParty(name=name, role=role) for role, name in self._extract_role_parties(parsed_doc.full_text)
        ]
        merged_parties = self._merge_parties(caption_parties + role_parties)
        judges = self._extract_case_judges(parsed_doc.full_text)
        date = self._first_match(_DATE_RE, f"{parsed_doc.title}\n{parsed_doc.full_text}")
        outcome_summary = self._first_match(_OUTCOME_RE, parsed_doc.full_text)
        court = self._normalize_court_title(self._first_match(_COURT_RE, f"{parsed_doc.title}\n{parsed_doc.full_text}"))
        page_ids = self._page_ids(parsed_doc)
        page_texts = self._page_texts(parsed_doc)
        field_page_ids = self._field_page_ids(
            parsed_doc,
            {
                "title": title,
                "case_number": case_number or parsed_doc.doc_id,
                "judge": judges,
                "claimant": [party.name for party in merged_parties if party.role.casefold() == "claimant"],
                "respondent": [party.name for party in merged_parties if party.role.casefold() == "respondent"],
                "appellant": [party.name for party in merged_parties if party.role.casefold() == "appellant"],
                "appellee": [party.name for party in merged_parties if party.role.casefold() == "appellee"],
                "party": [party.name for party in merged_parties],
                "date": date,
                "outcome": outcome_summary,
                "court": court,
            },
            fallback_fields={"title", "case_number", "court"},
        )
        return CaseObject(
            object_id=f"{LegalDocType.CASE.value}:{parsed_doc.doc_id}",
            doc_id=parsed_doc.doc_id,
            title=title,
            source_path=parsed_doc.source_path,
            page_ids=page_ids,
            source_text=parsed_doc.full_text.strip(),
            page_texts=page_texts,
            field_page_ids=field_page_ids,
            legal_doc_type=LegalDocType.CASE,
            case_number=case_number or parsed_doc.doc_id,
            court=court,
            judges=judges,
            parties=merged_parties,
            date=date,
            outcome_summary=outcome_summary,
            cited_law_titles=self._unique(self._all_matches(_LAW_TITLE_RE, parsed_doc.full_text)),
            cited_case_numbers=self._unique(self._all_matches(_CASE_NUMBER_RE, parsed_doc.full_text)),
        )

    def extract_order_metadata(
        self,
        parsed_doc: ParsedDocument,
        *,
        legal_doc_type: LegalDocType = LegalDocType.ORDER,
    ) -> OrderObject:
        """Extract structured order-like metadata.

        Args:
            parsed_doc: Parsed document to compile.
            legal_doc_type: Normalized legal document type to assign.

        Returns:
            A compiled order object.
        """
        page_ids = self._page_ids(parsed_doc)
        page_texts = self._page_texts(parsed_doc)
        order_number = self._first_match(_CASE_NUMBER_RE, f"{parsed_doc.title}\n{parsed_doc.full_text}")
        issued_by = extract_enactment_authority(
            source_text=parsed_doc.full_text,
            fallback=self._first_group_match(_ISSUED_BY_RE, parsed_doc.full_text, "issuer"),
        )
        effective_date = extract_enactment_date(
            source_text=parsed_doc.full_text,
            fallback=self._first_group_match(_COMMENCEMENT_RE, parsed_doc.full_text, "date"),
        )
        scope = self._first_sentence(parsed_doc.full_text)
        return OrderObject(
            object_id=f"{legal_doc_type.value}:{parsed_doc.doc_id}",
            doc_id=parsed_doc.doc_id,
            title=parsed_doc.title.strip(),
            source_path=parsed_doc.source_path,
            page_ids=page_ids,
            source_text=parsed_doc.full_text.strip(),
            page_texts=page_texts,
            field_page_ids=self._field_page_ids(
                parsed_doc,
                {
                    "title": parsed_doc.title.strip(),
                    "case_number": order_number,
                    "issued_by": issued_by,
                    "authority": issued_by,
                    "date": effective_date,
                    "effective_date": effective_date,
                    "outcome": scope,
                },
                fallback_fields={"title", "case_number"},
            ),
            legal_doc_type=legal_doc_type,
            order_number=order_number,
            issued_by=issued_by,
            effective_date=effective_date,
            supersedes=self._extract_amendment_refs(parsed_doc),
            scope=scope,
        )

    def extract_practice_direction_metadata(self, parsed_doc: ParsedDocument) -> PracticeDirectionObject:
        """Extract practice direction metadata.

        Args:
            parsed_doc: Parsed document to compile.

        Returns:
            A compiled practice direction object.
        """
        page_ids = self._page_ids(parsed_doc)
        page_texts = self._page_texts(parsed_doc)
        number = self._first_match(_LAW_NUMBER_RE, parsed_doc.full_text)
        issued_by = extract_enactment_authority(
            source_text=parsed_doc.full_text,
            fallback=self._first_group_match(_ISSUED_BY_RE, parsed_doc.full_text, "issuer"),
        )
        effective_date = extract_enactment_date(
            source_text=parsed_doc.full_text,
            fallback=self._first_group_match(_COMMENCEMENT_RE, parsed_doc.full_text, "date"),
        )
        scope = self._first_sentence(parsed_doc.full_text)
        return PracticeDirectionObject(
            object_id=f"{LegalDocType.PRACTICE_DIRECTION.value}:{parsed_doc.doc_id}",
            doc_id=parsed_doc.doc_id,
            title=parsed_doc.title.strip(),
            source_path=parsed_doc.source_path,
            page_ids=page_ids,
            source_text=parsed_doc.full_text.strip(),
            page_texts=page_texts,
            field_page_ids=self._field_page_ids(
                parsed_doc,
                {
                    "title": parsed_doc.title.strip(),
                    "law_number": number,
                    "issued_by": issued_by,
                    "authority": issued_by,
                    "date": effective_date,
                    "effective_date": effective_date,
                },
                fallback_fields={"title", "law_number"},
            ),
            legal_doc_type=LegalDocType.PRACTICE_DIRECTION,
            number=number,
            issued_by=issued_by,
            effective_date=effective_date,
            scope=scope,
        )

    def extract_amendment_metadata(self, parsed_doc: ParsedDocument) -> AmendmentObject:
        """Extract amendment metadata.

        Args:
            parsed_doc: Parsed document to compile.

        Returns:
            A compiled amendment object.
        """
        page_ids = self._page_ids(parsed_doc)
        page_texts = self._page_texts(parsed_doc)
        amends_titles = self._extract_amendment_refs(parsed_doc)
        effective_date = self._first_group_match(_COMMENCEMENT_RE, parsed_doc.full_text, "date")
        return AmendmentObject(
            object_id=f"{LegalDocType.AMENDMENT.value}:{parsed_doc.doc_id}",
            doc_id=parsed_doc.doc_id,
            title=parsed_doc.title.strip(),
            source_path=parsed_doc.source_path,
            page_ids=page_ids,
            source_text=parsed_doc.full_text.strip(),
            page_texts=page_texts,
            field_page_ids=self._field_page_ids(
                parsed_doc,
                {
                    "title": parsed_doc.title.strip(),
                    "date": effective_date,
                },
                fallback_fields={"title"},
            ),
            legal_doc_type=LegalDocType.AMENDMENT,
            amends_titles=amends_titles,
            effective_date=effective_date,
        )

    def extract_other_metadata(self, parsed_doc: ParsedDocument) -> OtherLegalObject:
        """Extract fallback metadata for documents outside specific families.

        Args:
            parsed_doc: Parsed document to compile.

        Returns:
            A fallback compiled legal object.
        """
        page_ids = self._page_ids(parsed_doc)
        page_texts = self._page_texts(parsed_doc)
        return OtherLegalObject(
            object_id=f"{LegalDocType.OTHER.value}:{parsed_doc.doc_id}",
            doc_id=parsed_doc.doc_id,
            title=parsed_doc.title.strip(),
            source_path=parsed_doc.source_path,
            page_ids=page_ids,
            source_text=parsed_doc.full_text.strip(),
            page_texts=page_texts,
            field_page_ids=self._field_page_ids(
                parsed_doc,
                {"title": parsed_doc.title.strip()},
                fallback_fields={"title"},
            ),
            legal_doc_type=self.resolve_document_type(parsed_doc),
            summary=self._first_sentence(parsed_doc.full_text),
        )

    def merge_registries(self, registries: list[CorpusRegistry]) -> CorpusRegistry:
        """Merge multiple registries into one registry.

        Args:
            registries: Registries to merge.

        Returns:
            A single combined registry.
        """
        merged = CorpusRegistry()
        for registry in registries:
            merged.source_doc_count += registry.source_doc_count
            merged.laws.update(registry.laws)
            merged.cases.update(registry.cases)
            merged.orders.update(registry.orders)
            merged.practice_directions.update(registry.practice_directions)
            merged.amendments.update(registry.amendments)
            merged.other_documents.update(registry.other_documents)
            merged.entities.update(registry.entities)
            merged.links.extend(registry.links)
        return merged

    def write_registry(self, registry: CorpusRegistry, out_path: str | Path) -> None:
        """Write a compiled registry to disk as JSON.

        Args:
            registry: Registry to serialize.
            out_path: Output path for the JSON document.
        """
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            registry.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def _store_object(self, registry: CorpusRegistry, compiled: CompiledLegalObject) -> None:
        """Store a compiled object in the appropriate registry bucket.

        Args:
            registry: Registry being populated.
            compiled: Compiled legal object to store.
        """
        if isinstance(compiled, LawObject):
            registry.laws[compiled.doc_id] = compiled
        elif isinstance(compiled, CaseObject):
            registry.cases[compiled.doc_id] = compiled
        elif isinstance(compiled, OrderObject):
            registry.orders[compiled.doc_id] = compiled
        elif isinstance(compiled, PracticeDirectionObject):
            registry.practice_directions[compiled.doc_id] = compiled
        elif isinstance(compiled, AmendmentObject):
            registry.amendments[compiled.doc_id] = compiled
        else:
            registry.other_documents[compiled.doc_id] = compiled

    def _build_entities(self, registry: CorpusRegistry) -> dict[str, LegalEntity]:
        """Create a minimal entity registry from compiled objects.

        Args:
            registry: Compiled corpus registry.

        Returns:
            Canonical entities keyed by entity ID.
        """
        entities: dict[str, LegalEntity] = {}
        for law in registry.laws.values():
            entity_id = f"law:{law.doc_id}"
            aliases = self._unique(
                [
                    law.title,
                    law.short_title,
                    f"Law No. {law.law_number} of {law.year}" if law.law_number and law.year else "",
                ]
            )
            entities[entity_id] = LegalEntity(
                entity_id=entity_id,
                name=law.title,
                canonical_name=law.title,
                entity_type=LegalEntityType.LAW,
                aliases=self._unique([*aliases, *law.aliases]),
                source_doc_ids=[law.doc_id],
            )
            if law.issuing_authority:
                authority_id = f"authority:{self._slug(law.issuing_authority)}"
                entities[authority_id] = LegalEntity(
                    entity_id=authority_id,
                    name=law.issuing_authority,
                    canonical_name=law.issuing_authority,
                    entity_type=LegalEntityType.AUTHORITY,
                    aliases=[law.issuing_authority],
                    source_doc_ids=self._merge_doc_ids(entities.get(authority_id), law.doc_id),
                )
        for case in registry.cases.values():
            case_entity_id = f"case:{case.doc_id}"
            entities[case_entity_id] = LegalEntity(
                entity_id=case_entity_id,
                name=case.title,
                canonical_name=case.case_number or case.title,
                entity_type=LegalEntityType.CASE,
                aliases=self._unique([case.title, case.case_number, *case.aliases]),
                source_doc_ids=[case.doc_id],
            )
            for judge in case.judges:
                judge_id = f"judge:{self._slug(judge)}"
                entities[judge_id] = LegalEntity(
                    entity_id=judge_id,
                    name=judge,
                    canonical_name=judge,
                    entity_type=LegalEntityType.JUDGE,
                    aliases=[judge],
                    source_doc_ids=self._merge_doc_ids(entities.get(judge_id), case.doc_id),
                )
            for party in case.parties:
                party_id = f"party:{self._slug(party.name)}"
                entities[party_id] = LegalEntity(
                    entity_id=party_id,
                    name=party.name,
                    canonical_name=party.name,
                    entity_type=LegalEntityType.PARTY,
                    aliases=self._unique([party.name, party.role]),
                    source_doc_ids=self._merge_doc_ids(entities.get(party_id), case.doc_id),
                )
        return entities

    def _build_links(self, registry: CorpusRegistry) -> list[ProvenanceLink]:
        """Resolve inter-document links from compiled object metadata.

        Args:
            registry: Compiled corpus registry.

        Returns:
            Resolved provenance links across the corpus.
        """
        links: list[ProvenanceLink] = []
        title_lookup = {law.title.casefold(): law.doc_id for law in registry.laws.values()}
        title_lookup.update(
            {law.short_title.casefold(): law.doc_id for law in registry.laws.values() if law.short_title}
        )
        title_lookup.update(
            {alias.casefold(): law.doc_id for law in registry.laws.values() for alias in law.aliases if alias}
        )
        case_lookup = {case.case_number.casefold(): case.doc_id for case in registry.cases.values() if case.case_number}
        for case in registry.cases.values():
            for title in case.cited_law_titles:
                target_doc_id = title_lookup.get(title.casefold())
                if target_doc_id and target_doc_id != case.doc_id:
                    links.append(
                        ProvenanceLink(
                            source_doc_id=case.doc_id,
                            target_doc_id=target_doc_id,
                            link_type=ProvenanceLinkType.CITES_LAW,
                            evidence_page=case.page_ids[0] if case.page_ids else "",
                            evidence_text=title,
                        )
                    )
            for cited_case in case.cited_case_numbers:
                target_doc_id = case_lookup.get(cited_case.casefold())
                if target_doc_id and target_doc_id != case.doc_id:
                    links.append(
                        ProvenanceLink(
                            source_doc_id=case.doc_id,
                            target_doc_id=target_doc_id,
                            link_type=ProvenanceLinkType.CITES_CASE,
                            evidence_page=case.page_ids[0] if case.page_ids else "",
                            evidence_text=cited_case,
                        )
                    )
        for amendment in registry.amendments.values():
            for title in amendment.amends_titles:
                target_doc_id = title_lookup.get(title.casefold())
                if target_doc_id and target_doc_id != amendment.doc_id:
                    links.append(
                        ProvenanceLink(
                            source_doc_id=amendment.doc_id,
                            target_doc_id=target_doc_id,
                            link_type=ProvenanceLinkType.AMENDS,
                            evidence_page=amendment.page_ids[0] if amendment.page_ids else "",
                            evidence_text=title,
                        )
                    )
        return links

    def _apply_manual_override(
        self,
        parsed_doc: ParsedDocument,
        compiled: CompiledLegalObject,
    ) -> CompiledLegalObject:
        """Apply manual overrides and refresh field-page mappings.

        Args:
            parsed_doc: Source parsed document.
            compiled: Compiled legal object before overrides.

        Returns:
            CompiledLegalObject: Final compiled object after manual overrides.
        """

        override = get_manual_domain_override(parsed_doc.doc_id)
        if override is None:
            return compiled
        patched = apply_manual_domain_override(compiled, override)
        return self._refresh_field_page_ids(parsed_doc, patched)

    def _refresh_field_page_ids(
        self,
        parsed_doc: ParsedDocument,
        compiled: CompiledLegalObject,
    ) -> CompiledLegalObject:
        """Rebuild field-page mappings after a manual override changes values.

        Args:
            parsed_doc: Source parsed document.
            compiled: Patched compiled legal object.

        Returns:
            CompiledLegalObject: Compiled object with refreshed field-page IDs.
        """

        if isinstance(compiled, LawObject):
            law_refs = [
                compiled.law_number,
                f"Law No. {compiled.law_number} of {compiled.year}" if compiled.law_number and compiled.year else "",
            ]
            field_page_ids = self._field_page_ids(
                parsed_doc,
                {
                    "title": [compiled.title, compiled.short_title, *compiled.aliases],
                    "law_number": [value for value in law_refs if value],
                    "issued_by": compiled.issuing_authority,
                    "authority": compiled.issuing_authority,
                    "commencement_date": compiled.commencement_date,
                    "date": compiled.commencement_date,
                },
                fallback_fields={"title", "law_number"},
            )
            return compiled.model_copy(update={"field_page_ids": field_page_ids})
        if isinstance(compiled, CaseObject):
            field_page_ids = self._field_page_ids(
                parsed_doc,
                {
                    "title": [compiled.title, *compiled.aliases],
                    "case_number": compiled.case_number or parsed_doc.doc_id,
                    "judge": compiled.judges,
                    "claimant": [party.name for party in compiled.parties if party.role.casefold() == "claimant"],
                    "respondent": [party.name for party in compiled.parties if party.role.casefold() == "respondent"],
                    "appellant": [party.name for party in compiled.parties if party.role.casefold() == "appellant"],
                    "appellee": [party.name for party in compiled.parties if party.role.casefold() == "appellee"],
                    "party": [party.name for party in compiled.parties],
                    "date": compiled.date,
                    "outcome": compiled.outcome_summary,
                    "court": compiled.court,
                },
                fallback_fields={"title", "case_number", "court"},
            )
            return compiled.model_copy(update={"field_page_ids": field_page_ids})
        if isinstance(compiled, OrderObject):
            field_page_ids = self._field_page_ids(
                parsed_doc,
                {
                    "title": [compiled.title, *compiled.aliases],
                    "case_number": compiled.order_number,
                    "issued_by": compiled.issued_by,
                    "authority": compiled.issued_by,
                    "date": compiled.effective_date,
                    "effective_date": compiled.effective_date,
                    "outcome": compiled.scope,
                },
                fallback_fields={"title", "case_number"},
            )
            return compiled.model_copy(update={"field_page_ids": field_page_ids})
        if isinstance(compiled, PracticeDirectionObject):
            field_page_ids = self._field_page_ids(
                parsed_doc,
                {
                    "title": [compiled.title, *compiled.aliases],
                    "law_number": compiled.number,
                    "issued_by": compiled.issued_by,
                    "authority": compiled.issued_by,
                    "date": compiled.effective_date,
                    "effective_date": compiled.effective_date,
                },
                fallback_fields={"title", "law_number"},
            )
            return compiled.model_copy(update={"field_page_ids": field_page_ids})
        return compiled

    def _extract_law_number_and_year(self, parsed_doc: ParsedDocument) -> tuple[str, str]:
        """Extract a law number and year from a parsed document.

        Args:
            parsed_doc: Parsed document to inspect.

        Returns:
            A tuple of law number and year strings.
        """
        match = _LAW_NUMBER_RE.search(f"{parsed_doc.title}\n{parsed_doc.full_text}")
        if match:
            return match.group(1), match.group(2)
        year_match = _YEAR_TITLE_RE.search(parsed_doc.title)
        return "", year_match.group(0) if year_match else ""

    def _extract_short_title(self, parsed_doc: ParsedDocument) -> str:
        """Extract a short title from parsed sections or text.

        Args:
            parsed_doc: Parsed document to inspect.

        Returns:
            A short title if one is found.
        """
        for section in parsed_doc.sections:
            if "short title" in section.heading.casefold():
                sentence = self._first_sentence(section.text)
                if sentence:
                    return sentence.rstrip(".")
        return ""

    def _extract_article_tree(self, parsed_doc: ParsedDocument) -> list[ArticleNode]:
        """Extract a structured article tree from parser sections.

        Args:
            parsed_doc: Parsed document to inspect.

        Returns:
            Structured article nodes in source order.
        """
        nodes: list[ArticleNode] = []
        page_ids = self._page_ids(parsed_doc)
        fallback_page = page_ids[0] if page_ids else ""
        for index, section in enumerate(parsed_doc.sections, start=1):
            match = _SECTION_LABEL_RE.match(section.heading.strip())
            if not match:
                continue
            label = match.group("label").strip()
            title = (match.group("title") or "").strip()
            node_page_ids = (
                [self._section_page_id(parsed_doc, section.section_path)]
                if section.section_path.startswith("page:")
                else [fallback_page]
            )
            nodes.append(
                ArticleNode(
                    article_id=f"{parsed_doc.doc_id}:{index}",
                    label=label,
                    title=title,
                    page_ids=[page_id for page_id in node_page_ids if page_id],
                )
            )
        return nodes

    def _extract_amendment_refs(self, parsed_doc: ParsedDocument) -> list[str]:
        """Extract cited law titles likely to be amendment targets.

        Args:
            parsed_doc: Parsed document to inspect.

        Returns:
            Candidate amendment target titles.
        """
        full_text = f"{parsed_doc.title}\n{parsed_doc.full_text}"
        return self._unique(self._all_matches(_LAW_TITLE_RE, full_text))

    def _extract_case_judges(self, text: str) -> list[str]:
        """Extract judge names while filtering out registrar and party noise.

        Args:
            text: Full case text to inspect.

        Returns:
            list[str]: Normalized judge names in source order.
        """

        judges: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip(" ,.;:")
            if not line or not _JUDGE_LINE_HINT_RE.search(line):
                continue
            if _JUDGE_LINE_BLOCKLIST_RE.search(line):
                continue
            for match in _JUDGE_RE.finditer(line):
                judge = self._normalize_case_judge_name(match.group(0))
                if judge:
                    judges.append(judge)
        return self._unique(judges)

    @staticmethod
    def _page_ids(parsed_doc: ParsedDocument) -> list[str]:
        """Build stable page IDs from parser page sections.

        Args:
            parsed_doc: Parsed document to inspect.

        Returns:
            Stable page identifiers in source order.
        """
        return list(CorpusCompiler._page_texts(parsed_doc).keys())

    @staticmethod
    def _page_texts(parsed_doc: ParsedDocument) -> dict[str, str]:
        """Build a stable page-text mapping for a parsed document.

        Args:
            parsed_doc: Parsed document to inspect.

        Returns:
            dict[str, str]: Stable page IDs mapped to page text.
        """

        page_texts: dict[str, str] = {}
        for section in parsed_doc.sections:
            if not section.section_path.startswith("page:"):
                continue
            page_id = CorpusCompiler._section_page_id(parsed_doc, section.section_path)
            text = section.text.strip()
            if not page_id or not text:
                continue
            page_texts[page_id] = text
        if page_texts:
            return page_texts
        fallback_text = parsed_doc.full_text.strip()
        if not fallback_text:
            return {}
        return {f"{parsed_doc.doc_id}_1": fallback_text}

    @staticmethod
    def _field_page_ids(
        parsed_doc: ParsedDocument,
        field_values: dict[str, str | list[str]],
        *,
        fallback_fields: set[str] | None = None,
    ) -> dict[str, list[str]]:
        """Locate source pages for structured field values.

        Args:
            parsed_doc: Parsed document holding the field values.
            field_values: Mapping from field names to one or more extracted values.
            fallback_fields: Fields allowed to fall back to the first page when a
                direct text match is not found.

        Returns:
            dict[str, list[str]]: Field-to-source-page-id mapping.
        """

        page_texts = CorpusCompiler._page_texts(parsed_doc)
        page_order = list(page_texts.keys())
        allowed_fallbacks = fallback_fields or set()
        result: dict[str, list[str]] = {}
        for field_name, raw_value in field_values.items():
            values = [raw_value] if isinstance(raw_value, str) else list(raw_value)
            normalized_values = [
                CorpusCompiler._normalize_match_text(value)
                for value in values
                if CorpusCompiler._normalize_match_text(value)
            ]
            if not normalized_values:
                continue
            matches: list[str] = []
            for page_id, page_text in page_texts.items():
                normalized_page = CorpusCompiler._normalize_match_text(page_text)
                if any(value in normalized_page for value in normalized_values):
                    matches.append(page_id)
            if not matches and field_name in allowed_fallbacks and page_order:
                matches = [page_order[0]]
            if matches:
                result[field_name] = matches
        return result

    @staticmethod
    def _section_page_id(parsed_doc: ParsedDocument, section_path: str) -> str:
        """Map a section path to a stable page ID when possible.

        Args:
            parsed_doc: Parsed document containing the section.
            section_path: Section path value from the parser.

        Returns:
            The stable page ID or an empty string.
        """
        try:
            page_num = int(section_path.split(":", maxsplit=1)[1])
        except (IndexError, ValueError):
            return ""
        return f"{parsed_doc.doc_id}_{page_num}"

    @staticmethod
    def _normalize_match_text(value: str) -> str:
        """Normalize text for deterministic field-to-page matching.

        Args:
            value: Raw field or page text.

        Returns:
            str: Casefolded, whitespace-collapsed text.
        """

        return re.sub(r"\s+", " ", value.strip()).casefold()

    @staticmethod
    def _normalize_case_judge_name(value: str) -> str:
        """Normalize a judge name extracted from a case line.

        Args:
            value: Raw judge text from the document.

        Returns:
            str: Cleaned judge name.
        """

        normalized = re.sub(r"\s+", " ", value.strip()).strip(" ,.;:")
        normalized = re.sub(r"^(?:h\.e\.|he\.|hon\.|honourable)\s+", "", normalized, flags=re.IGNORECASE)
        return normalized.strip(" ,.;:")

    @staticmethod
    def _normalize_court_title(value: str) -> str:
        """Normalize court title text extracted from a case.

        Args:
            value: Raw court title text.

        Returns:
            str: Cleaned court title.
        """

        normalized = re.sub(r"\s+", " ", value.strip()).strip(" ,.;:")
        if not normalized:
            return ""
        normalized = re.sub(r"^(?:the\s+)+", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bof\s+(?:the\s+)?DIFC\s+Courts?\b", "", normalized, flags=re.IGNORECASE)
        return normalized.strip(" ,.;:")

    @staticmethod
    def _extract_caption_parties(title: str) -> list[CaseParty]:
        """Extract plaintiff/defendant style parties from a caption.

        Args:
            title: Case title or caption line.

        Returns:
            Candidate case parties from the caption.
        """
        cleaned = title.split("[", maxsplit=1)[0].strip()
        case_prefix = _CASE_NUMBER_RE.match(cleaned)
        if case_prefix is not None:
            cleaned = cleaned[case_prefix.end() :].strip(" -,:.")
        parts = _CAPTION_SPLIT_RE.split(cleaned, maxsplit=1)
        if len(parts) != 2:
            return []
        left_raw, right_raw = (part.strip(" -,:.") for part in parts)
        left = CorpusCompiler._strip_party_role_suffix(left_raw)
        right = CorpusCompiler._strip_party_role_suffix(right_raw)
        parties: list[CaseParty] = []
        if CorpusCompiler._is_plausible_party_value(left):
            parties.append(
                CaseParty(name=left, role=CorpusCompiler._caption_role_or_default(left_raw, default="claimant"))
            )
        if CorpusCompiler._is_plausible_party_value(right):
            parties.append(
                CaseParty(name=right, role=CorpusCompiler._caption_role_or_default(right_raw, default="respondent"))
            )
        return parties

    @staticmethod
    def _extract_role_parties(text: str) -> list[tuple[str, str]]:
        """Extract role-tagged parties from body text.

        Args:
            text: Full document text.

        Returns:
            Role/name pairs discovered in the document text.
        """
        pairs: list[tuple[str, str]] = []
        previous_line = ""
        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            role_value_match = _ROLE_VALUE_LINE_RE.match(line)
            if role_value_match is not None:
                role = CorpusCompiler._normalize_party_role(role_value_match.group(1))
                name = CorpusCompiler._strip_party_role_suffix(role_value_match.group(2).strip(" ,.;:"))
                if CorpusCompiler._is_plausible_party_value(name):
                    pairs.append((role, name))
                previous_line = line
                continue
            role_only_match = _ROLE_ONLY_LINE_RE.match(line)
            if role_only_match is not None and CorpusCompiler._is_plausible_party_value(previous_line):
                pairs.append(
                    (
                        CorpusCompiler._normalize_party_role(role_only_match.group(1)),
                        CorpusCompiler._strip_party_role_suffix(previous_line.strip(" ,.;:")),
                    )
                )
                previous_line = line
                continue
            previous_line = line
        for match in _PARTY_ROLE_RE.finditer(text):
            role = CorpusCompiler._normalize_party_role(match.group(0).split(maxsplit=1)[0])
            name = CorpusCompiler._strip_party_role_suffix(match.group(1).strip(" ,.;:"))
            if CorpusCompiler._is_plausible_party_value(name):
                pairs.append((role, name))
        return pairs

    @staticmethod
    def _merge_parties(parties: list[CaseParty]) -> list[CaseParty]:
        """Deduplicate party entries while preserving the earliest role.

        Args:
            parties: Candidate party records.

        Returns:
            Deduplicated party records.
        """
        seen: set[str] = set()
        merged: list[CaseParty] = []
        for party in parties:
            cleaned_name = party.name.strip(" -,:.")
            cleaned_name = CorpusCompiler._strip_party_role_suffix(cleaned_name)
            normalized_role = CorpusCompiler._normalize_party_role(party.role)
            if cleaned_name.casefold() in {"mr", "mrs", "ms", "dr"}:
                continue
            if not CorpusCompiler._is_plausible_party_value(cleaned_name):
                continue
            key = cleaned_name.casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(
                CaseParty(name=cleaned_name, role=normalized_role, canonical_entity_id=party.canonical_entity_id)
            )
        return merged

    @staticmethod
    def _is_plausible_party_value(value: str) -> bool:
        """Return whether extracted party text looks like a real party name.

        Args:
            value: Candidate party text.

        Returns:
            bool: True when the value is specific enough to keep.
        """

        cleaned = value.strip(" -,:.")
        if not cleaned:
            return False
        case_prefix = _CASE_NUMBER_RE.match(cleaned)
        if case_prefix is not None:
            cleaned = cleaned[case_prefix.end() :].strip(" -,:.")
        if not cleaned:
            return False
        normalized = cleaned.casefold()
        if normalized in _LOW_SIGNAL_PARTY_VALUES:
            return False
        if any(role.endswith(normalized) and role != normalized for role in _LOW_SIGNAL_PARTY_SUFFIXES):
            return False
        tokens = [token.strip(".,;:()[]{}") for token in re.split(r"\s+", normalized) if token]
        if len(tokens) > 12:
            return False
        low_signal_hits = sum(token in _LOW_SIGNAL_PARTY_VALUES for token in tokens)
        return low_signal_hits < 2

    @staticmethod
    def _strip_party_role_suffix(value: str) -> str:
        """Remove trailing role markers from a caption party string.

        Args:
            value: Raw caption value that may include a role suffix.

        Returns:
            str: Cleaned party text without a trailing role label.
        """

        cleaned = re.sub(r"\s*\([^)]+\)\s*$", "", value).strip(" -,:.")
        return _CAPTION_ROLE_SUFFIX_RE.sub("", cleaned).strip(" -,:.")

    @staticmethod
    def _normalize_party_role(role: str) -> str:
        """Normalize a role label to its canonical singular form.

        Args:
            role: Raw role label from a caption or body line.

        Returns:
            str: Canonical role label.
        """

        normalized = role.strip(" -,:.").casefold()
        return _ROLE_CANONICAL_MAP.get(normalized, normalized)

    @staticmethod
    def _caption_role_or_default(value: str, *, default: str) -> str:
        """Infer a canonical role from a caption fragment.

        Args:
            value: Raw caption fragment possibly containing a role suffix.
            default: Role to use when no explicit caption role is present.

        Returns:
            str: Canonical role name for the fragment.
        """

        match = re.search(
            r"(claimant|claimants|respondent|respondents|appellant|appellants|appellee|appellees|"
            r"applicant|applicants|defendant|defendants|plaintiff|plaintiffs|petitioner|petitioners)\s*\)?\s*$",
            value,
            re.IGNORECASE,
        )
        if match is None:
            return default
        return CorpusCompiler._normalize_party_role(match.group(1))

    @staticmethod
    def _merge_doc_ids(entity: LegalEntity | None, doc_id: str) -> list[str]:
        """Append a doc ID to an entity source list if needed.

        Args:
            entity: Existing entity or None.
            doc_id: Document ID to include.

        Returns:
            Updated source doc IDs.
        """
        if entity is None:
            return [doc_id]
        values = list(entity.source_doc_ids)
        if doc_id not in values:
            values.append(doc_id)
        return values

    @staticmethod
    def _slug(value: str) -> str:
        """Create a stable lowercase slug for IDs.

        Args:
            value: Raw string to normalize.

        Returns:
            Lowercase slug string.
        """
        return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        """Return values in first-seen order without empty duplicates.

        Args:
            values: Candidate values.

        Returns:
            Deduplicated values preserving order.
        """
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            cleaned = value.strip()
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Extract the first sentence-sized fragment from text.

        Args:
            text: Raw text to summarize.

        Returns:
            The first sentence-sized fragment.
        """
        stripped = re.sub(r"\s+", " ", text).strip()
        if not stripped:
            return ""
        parts = re.split(r"(?<=[.?!])\s+", stripped, maxsplit=1)
        return parts[0][:240]

    @staticmethod
    def _first_match(pattern: re.Pattern[str], text: str) -> str:
        """Return the first full regex match from text.

        Args:
            pattern: Compiled regex pattern.
            text: Text to search.

        Returns:
            The first match or an empty string.
        """
        match = pattern.search(text)
        return match.group(0).strip() if match else ""

    @staticmethod
    def _all_matches(pattern: re.Pattern[str], text: str) -> list[str]:
        """Return every regex match from text as strings.

        Args:
            pattern: Compiled regex pattern.
            text: Text to search.

        Returns:
            All matched strings in source order.
        """
        return [match.group(0).strip() for match in pattern.finditer(text)]

    @staticmethod
    def _first_group_match(pattern: re.Pattern[str], text: str, group: str) -> str:
        """Return the first named-group regex match from text.

        Args:
            pattern: Compiled regex pattern.
            text: Text to search.
            group: Named regex group to extract.

        Returns:
            The first captured group or an empty string.
        """
        match = pattern.search(text)
        if not match:
            return ""
        value = match.group(group) or ""
        return value.strip(" \n.:;")
