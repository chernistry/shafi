"""Typed legal object models for the closed-world corpus compiler."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


def _article_nodes_factory() -> list[ArticleNode]:
    """Build a typed empty article-node list for Pydantic defaults."""

    return []


def _case_parties_factory() -> list[CaseParty]:
    """Build a typed empty case-party list for Pydantic defaults."""

    return []


def _links_factory() -> list[ProvenanceLink]:
    """Build a typed empty provenance-link list for Pydantic defaults."""

    return []


def _page_text_map_factory() -> dict[str, str]:
    """Build a typed empty page-text mapping for Pydantic defaults."""

    return {}


def _field_page_map_factory() -> dict[str, list[str]]:
    """Build a typed empty field-to-page-id mapping for Pydantic defaults."""

    return {}


def _visual_region_map_factory() -> dict[str, list[VisualRegion]]:
    """Build a typed empty page-to-visual-region mapping for Pydantic defaults."""

    return {}


def _table_rows_factory() -> list[list[str]]:
    """Build a typed empty table-row list for Pydantic defaults."""

    return []


class LegalDocType(StrEnum):
    """Normalized legal document kinds emitted by the corpus compiler."""

    LAW = "law"
    REGULATION = "regulation"
    CASE = "case"
    ORDER = "order"
    PRACTICE_DIRECTION = "practice_direction"
    POLICY = "policy"
    SCHEDULE = "schedule"
    AMENDMENT = "amendment"
    ENACTMENT_NOTICE = "enactment_notice"
    OTHER = "other"


class LegalEntityType(StrEnum):
    """Supported entity categories in the compiled corpus registry."""

    LAW = "law"
    CASE = "case"
    PERSON = "person"
    ORGANIZATION = "organization"
    AUTHORITY = "authority"
    PARTY = "party"
    JUDGE = "judge"


class ProvenanceLinkType(StrEnum):
    """Relationship types captured between compiled legal objects."""

    CITES_CASE = "cites_case"
    CITES_LAW = "cites_law"
    AMENDS = "amends"
    ISSUED_BY = "issued_by"
    COMMENCES = "commences"


class RegionType(StrEnum):
    """Supported visual-region types for hard-page enrichment."""

    CAPTION = "caption"
    TITLE = "title"
    TABLE = "table"
    SIGNATURE = "signature"
    SCHEDULE = "schedule"
    OPERATIVE = "operative"
    HEADER = "header"
    FOOTER = "footer"


class BoundingBox(BaseModel):
    """Normalized page-region bounding box."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    w: float
    h: float


class TableData(BaseModel):
    """Structured table extraction metadata."""

    model_config = ConfigDict(frozen=True)

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=_table_rows_factory)


class SignatureData(BaseModel):
    """Structured signature/date extraction metadata."""

    model_config = ConfigDict(frozen=True)

    signer: str = ""
    authority: str = ""
    date: str = ""


class VisualRegion(BaseModel):
    """Extracted visual region from a hard page."""

    model_config = ConfigDict(frozen=True)

    region_type: RegionType
    bbox: BoundingBox
    extracted_text: str = ""
    confidence: float = 0.0
    page_id: str
    doc_id: str
    table_data: TableData | None = None
    signature_data: SignatureData | None = None


class ArticleNode(BaseModel):
    """Structured legal article or section emitted from a compiled document."""

    model_config = ConfigDict(frozen=True)

    article_id: str
    label: str
    title: str = ""
    page_ids: list[str] = Field(default_factory=list)
    child_labels: list[str] = Field(default_factory=list)


class LegalEntity(BaseModel):
    """Canonical entity entry stored in the compiled registry."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    name: str
    canonical_name: str
    entity_type: LegalEntityType
    aliases: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)


class CaseParty(BaseModel):
    """Normalized case party record."""

    model_config = ConfigDict(frozen=True)

    name: str
    role: str = ""
    canonical_entity_id: str = ""


class BaseLegalObject(BaseModel):
    """Shared fields for all compiled legal objects."""

    object_id: str
    doc_id: str
    title: str
    aliases: list[str] = Field(default_factory=list)
    source_path: str = ""
    page_ids: list[str] = Field(default_factory=list)
    source_text: str = ""
    page_texts: dict[str, str] = Field(default_factory=_page_text_map_factory)
    field_page_ids: dict[str, list[str]] = Field(default_factory=_field_page_map_factory)
    visual_regions: dict[str, list[VisualRegion]] = Field(default_factory=_visual_region_map_factory)
    legal_doc_type: LegalDocType


class LawObject(BaseLegalObject):
    """Compiled law or regulation metadata."""

    short_title: str = ""
    law_number: str = ""
    year: str = ""
    issuing_authority: str = ""
    commencement_date: str = ""
    amendment_refs: list[str] = Field(default_factory=list)
    article_tree: list[ArticleNode] = Field(default_factory=_article_nodes_factory)


class CaseObject(BaseLegalObject):
    """Compiled case metadata."""

    case_number: str = ""
    court: str = ""
    judges: list[str] = Field(default_factory=list)
    parties: list[CaseParty] = Field(default_factory=_case_parties_factory)
    date: str = ""
    outcome_summary: str = ""
    cited_law_titles: list[str] = Field(default_factory=list)
    cited_case_numbers: list[str] = Field(default_factory=list)


class OrderObject(BaseLegalObject):
    """Compiled order metadata."""

    order_number: str = ""
    issued_by: str = ""
    effective_date: str = ""
    supersedes: list[str] = Field(default_factory=list)
    scope: str = ""


class PracticeDirectionObject(BaseLegalObject):
    """Compiled practice direction metadata."""

    number: str = ""
    issued_by: str = ""
    effective_date: str = ""
    scope: str = ""


class AmendmentObject(BaseLegalObject):
    """Compiled amendment metadata."""

    amends_titles: list[str] = Field(default_factory=list)
    effective_date: str = ""


class OtherLegalObject(BaseLegalObject):
    """Fallback object for documents outside specific compiler families."""

    summary: str = ""


class ProvenanceLink(BaseModel):
    """Typed provenance edge between compiled legal objects."""

    model_config = ConfigDict(frozen=True)

    source_doc_id: str
    target_doc_id: str
    link_type: ProvenanceLinkType
    evidence_page: str = ""
    evidence_text: str = ""


class CorpusRegistry(BaseModel):
    """Top-level compiled registry for the legal corpus."""

    schema_version: int = 1
    source_doc_count: int = 0
    laws: dict[str, LawObject] = Field(default_factory=dict)
    cases: dict[str, CaseObject] = Field(default_factory=dict)
    orders: dict[str, OrderObject] = Field(default_factory=dict)
    practice_directions: dict[str, PracticeDirectionObject] = Field(default_factory=dict)
    amendments: dict[str, AmendmentObject] = Field(default_factory=dict)
    other_documents: dict[str, OtherLegalObject] = Field(default_factory=dict)
    entities: dict[str, LegalEntity] = Field(default_factory=dict)
    links: list[ProvenanceLink] = Field(default_factory=_links_factory)
    applicability_graph: dict[str, object] | None = None
