"""Bridge-fact generation for compiler-driven cross-field retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

from rag_challenge.models import BridgeFact, BridgeFactType

if TYPE_CHECKING:
    from pathlib import Path

    from rag_challenge.ingestion.canonical_entities import EntityAliasResolver
    from rag_challenge.models import LegalSegment
    from rag_challenge.models.applicability import ApplicabilityGraph
    from rag_challenge.models.legal_objects import CaseObject, CorpusRegistry, LawObject, LegalEntity


def _stable_fact_id(*parts: str) -> str:
    """Build a deterministic bridge-fact identifier.

    Args:
        *parts: Canonical text fragments participating in the ID.

    Returns:
        str: Stable bridge-fact ID.
    """

    payload = "||".join(part.strip() for part in parts if part.strip())
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"bridge:{digest}"


def _normalize_text(text: str) -> str:
    """Collapse whitespace for bridge-fact text.

    Args:
        text: Raw source text.

    Returns:
        str: Clean single-space text.
    """

    return " ".join(str(text or "").split()).strip()


_HEX_CASE_ID_RE = re.compile(r"^[0-9a-f]{32,}$", re.IGNORECASE)
_HONORIFIC_TOKEN_RE = re.compile(
    r"^(?:justice|judge|chief|sir|mr|mrs|ms|dr|registrar|the|kc|k\.c\.)$",
    re.IGNORECASE,
)
_LOW_SIGNAL_PERSON_TOKENS = frozenset({"and", "date", "order", "judgment", "hearing"})
_LOW_SIGNAL_PARTY_TOKENS = frozenset(
    {
        "and",
        "are",
        "filed",
        "from",
        "for",
        "has",
        "have",
        "hearing",
        "intends",
        "judgment",
        "order",
        "pursuant",
        "relies",
        "served",
        "shall",
        "that",
        "the",
        "under",
        "was",
        "were",
        "which",
        "who",
        "with",
    }
)


def _is_plausible_case_identifier(case_number: str) -> bool:
    """Check whether a case identifier is safe to expose in bridge facts.

    Args:
        case_number: Candidate case-number text.

    Returns:
        bool: True when the case identifier looks like a real case number.
    """

    normalized = _normalize_text(case_number)
    if not normalized:
        return False
    compact = normalized.replace(" ", "")
    if _HEX_CASE_ID_RE.fullmatch(compact):
        return False
    return "/" in normalized or "[" in normalized


def _normalize_judge_name(judge_name: str) -> str:
    """Normalize extracted judge text for bridge-fact quality checks.

    Args:
        judge_name: Raw extracted judge text.

    Returns:
        str: Cleaned one-line judge name.
    """

    return _normalize_text(judge_name)


def _is_plausible_judge_name(judge_name: str) -> bool:
    """Check whether an extracted judge name is specific enough to index.

    Args:
        judge_name: Raw or normalized judge text.

    Returns:
        bool: True when the judge name looks like a real person reference.
    """

    normalized = _normalize_judge_name(judge_name)
    if not normalized:
        return False
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    if len(tokens) < 2:
        return False
    meaningful = [token for token in tokens if not _HONORIFIC_TOKEN_RE.fullmatch(token)]
    if not meaningful:
        return False
    if any(token.casefold() in _LOW_SIGNAL_PERSON_TOKENS for token in meaningful):
        return False
    # Require at least one alphabetic token that is not just a title.
    return any(any(char.isalpha() for char in token) and len(token) > 2 for token in meaningful)


def _is_plausible_party_name(party_name: str) -> bool:
    """Check whether an extracted party string is specific enough to index.

    Args:
        party_name: Raw extracted party text.

    Returns:
        bool: True when the party string looks like a real party name.
    """

    normalized = _normalize_text(party_name)
    if not normalized:
        return False
    if normalized.casefold() in _LOW_SIGNAL_PARTY_TOKENS:
        return False
    if normalized.count(",") > 2:
        return False
    tokens = [token.strip(".,;:()[]{}") for token in re.split(r"\s+", normalized) if token]
    if len(tokens) > 10:
        return False
    low_signal_hits = sum(token.casefold() in _LOW_SIGNAL_PARTY_TOKENS for token in tokens)
    return not low_signal_hits >= 2


class BridgeFactGenerator:
    """Generate typed bridge facts from compiled corpus structures."""

    def generate_all(
        self,
        *,
        corpus_registry: CorpusRegistry,
        entity_resolver: EntityAliasResolver,
        segments: list[LegalSegment] | None = None,
        applicability_graph: ApplicabilityGraph | None = None,
    ) -> list[BridgeFact]:
        """Generate all bridge facts for the current compiled corpus slice.

        Args:
            corpus_registry: Compiled corpus registry.
            entity_resolver: Canonical alias resolver for source entities.
            segments: Optional compiler-driven legal segments.
            applicability_graph: Optional amendment/commencement graph.

        Returns:
            list[BridgeFact]: Deduplicated bridge-fact records.
        """

        segments_by_doc_id: dict[str, list[LegalSegment]] = {}
        for segment in segments or []:
            segments_by_doc_id.setdefault(segment.doc_id, []).append(segment)

        facts: list[BridgeFact] = []
        for case in corpus_registry.cases.values():
            facts.extend(self.generate_case_party_facts(case_obj=case, entity_resolver=entity_resolver))
            facts.extend(self.generate_case_judge_facts(case_obj=case, entity_resolver=entity_resolver))
            facts.extend(self.generate_case_law_facts(case_obj=case, entity_resolver=entity_resolver))
            facts.extend(self.generate_case_outcome_facts(case_obj=case, entity_resolver=entity_resolver))

        for law in corpus_registry.laws.values():
            law_segments = segments_by_doc_id.get(law.doc_id, [])
            facts.extend(self.generate_law_authority_facts(law_obj=law, entity_resolver=entity_resolver))
            facts.extend(
                self.generate_law_amendment_facts(
                    law_obj=law,
                    entity_resolver=entity_resolver,
                    applicability_graph=applicability_graph,
                )
            )
            facts.extend(self.generate_law_definition_facts(law_obj=law, entity_resolver=entity_resolver, segments=law_segments))
            facts.extend(
                self.generate_law_commencement_facts(
                    law_obj=law,
                    entity_resolver=entity_resolver,
                    applicability_graph=applicability_graph,
                )
            )
            facts.extend(self.generate_article_location_facts(law_obj=law, entity_resolver=entity_resolver))

        for entity in corpus_registry.entities.values():
            facts.extend(self.generate_entity_document_facts(entity=entity))

        return self.deduplicate(facts)

    def generate_case_party_facts(
        self,
        *,
        case_obj: CaseObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[BridgeFact]:
        """Generate case-party bridge facts.

        Args:
            case_obj: Compiled case object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[BridgeFact]: Generated case-party facts.
        """

        facts: list[BridgeFact] = []
        case_id = self._resolve_case_ids(case_obj=case_obj, entity_resolver=entity_resolver)
        case_label = case_obj.case_number if _is_plausible_case_identifier(case_obj.case_number) else ""
        if not case_label:
            return []
        for party in case_obj.parties:
            if not _is_plausible_party_name(party.name):
                continue
            entity_ids = sorted(
                {
                    *case_id,
                    *self._resolve_entity_ids(entity_resolver=entity_resolver, party_names=[party.name]),
                }
            )
            canonical_text = _normalize_text(
                f"Case {case_label} involves {party.name} as {party.role or 'party'}."
            )
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.CASE_PARTY.value, case_obj.doc_id, party.name, party.role),
                    fact_type=BridgeFactType.CASE_PARTY,
                    canonical_text=canonical_text,
                    source_entity_ids=entity_ids,
                    source_doc_ids=[case_obj.doc_id],
                    evidence_page_ids=list(case_obj.page_ids[:2]),
                    attributes={
                        "case_number": case_label,
                        "party_name": party.name,
                        "party_role": party.role,
                    },
                )
            )
        return facts

    def generate_case_judge_facts(
        self,
        *,
        case_obj: CaseObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[BridgeFact]:
        """Generate case-judge bridge facts.

        Args:
            case_obj: Compiled case object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[BridgeFact]: Generated case-judge facts.
        """

        facts: list[BridgeFact] = []
        case_id = self._resolve_case_ids(case_obj=case_obj, entity_resolver=entity_resolver)
        case_label = case_obj.case_number if _is_plausible_case_identifier(case_obj.case_number) else ""
        if not case_label:
            return []
        for judge in case_obj.judges:
            normalized_judge = _normalize_judge_name(judge)
            if not _is_plausible_judge_name(normalized_judge):
                continue
            entity_ids = sorted(
                {
                    *case_id,
                    *self._resolve_entity_ids(entity_resolver=entity_resolver, judge_names=[normalized_judge]),
                }
            )
            canonical_text = _normalize_text(
                f"Case {case_label} was decided by {normalized_judge}."
            )
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.CASE_JUDGE.value, case_obj.doc_id, normalized_judge),
                    fact_type=BridgeFactType.CASE_JUDGE,
                    canonical_text=canonical_text,
                    source_entity_ids=entity_ids,
                    source_doc_ids=[case_obj.doc_id],
                    evidence_page_ids=list(case_obj.page_ids[:2]),
                    attributes={"case_number": case_label, "judge_name": normalized_judge},
                )
            )
        return facts

    def generate_case_law_facts(
        self,
        *,
        case_obj: CaseObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[BridgeFact]:
        """Generate case-law bridge facts.

        Args:
            case_obj: Compiled case object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[BridgeFact]: Generated case-law facts.
        """

        facts: list[BridgeFact] = []
        case_id = self._resolve_case_ids(case_obj=case_obj, entity_resolver=entity_resolver)
        case_label = case_obj.case_number if _is_plausible_case_identifier(case_obj.case_number) else ""
        if not case_label:
            return []
        for law_title in case_obj.cited_law_titles:
            entity_ids = sorted(
                {
                    *case_id,
                    *self._resolve_entity_ids(entity_resolver=entity_resolver, law_titles=[law_title]),
                }
            )
            canonical_text = _normalize_text(
                f"Case {case_label} considers {law_title}."
            )
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.CASE_LAW.value, case_obj.doc_id, law_title),
                    fact_type=BridgeFactType.CASE_LAW,
                    canonical_text=canonical_text,
                    source_entity_ids=entity_ids,
                    source_doc_ids=[case_obj.doc_id],
                    evidence_page_ids=list(case_obj.page_ids[:2]),
                    attributes={"case_number": case_label, "law_title": law_title},
                )
            )
        return facts

    def generate_case_outcome_facts(
        self,
        *,
        case_obj: CaseObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[BridgeFact]:
        """Generate case-outcome bridge facts when summaries exist.

        Args:
            case_obj: Compiled case object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[BridgeFact]: Generated case-outcome facts.
        """

        summary = _normalize_text(case_obj.outcome_summary)
        case_label = case_obj.case_number if _is_plausible_case_identifier(case_obj.case_number) else ""
        if not summary or not case_label:
            return []
        return [
            BridgeFact(
                fact_id=_stable_fact_id(BridgeFactType.CASE_OUTCOME.value, case_obj.doc_id, summary),
                fact_type=BridgeFactType.CASE_OUTCOME,
                canonical_text=_normalize_text(
                    f"In case {case_label}, the court outcome was: {summary}."
                ),
                source_entity_ids=self._resolve_case_ids(case_obj=case_obj, entity_resolver=entity_resolver),
                source_doc_ids=[case_obj.doc_id],
                evidence_page_ids=list(case_obj.page_ids[:2]),
                attributes={"case_number": case_label, "outcome_summary": summary},
            )
        ]

    def generate_law_authority_facts(
        self,
        *,
        law_obj: LawObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[BridgeFact]:
        """Generate law-authority bridge facts.

        Args:
            law_obj: Compiled law object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[BridgeFact]: Generated law-authority facts.
        """

        if not law_obj.issuing_authority:
            return []
        entity_ids = self._resolve_entity_ids(
            entity_resolver=entity_resolver,
            law_titles=[law_obj.short_title or law_obj.title],
            authority_names=[law_obj.issuing_authority],
        )
        text = f"{law_obj.short_title or law_obj.title} was issued by {law_obj.issuing_authority}"
        if law_obj.year:
            text += f" in {law_obj.year}"
        text += "."
        return [
            BridgeFact(
                fact_id=_stable_fact_id(
                    BridgeFactType.LAW_AUTHORITY.value,
                    law_obj.doc_id,
                    law_obj.issuing_authority,
                ),
                fact_type=BridgeFactType.LAW_AUTHORITY,
                canonical_text=_normalize_text(text),
                source_entity_ids=entity_ids,
                source_doc_ids=[law_obj.doc_id],
                evidence_page_ids=list(law_obj.page_ids[:2]),
                attributes={
                    "law_title": law_obj.short_title or law_obj.title,
                    "authority": law_obj.issuing_authority,
                    "year": law_obj.year,
                },
            )
        ]

    def generate_law_amendment_facts(
        self,
        *,
        law_obj: LawObject,
        entity_resolver: EntityAliasResolver,
        applicability_graph: ApplicabilityGraph | None = None,
    ) -> list[BridgeFact]:
        """Generate law-amendment bridge facts.

        Args:
            law_obj: Compiled law object.
            entity_resolver: Canonical alias resolver.
            applicability_graph: Optional amendment graph.

        Returns:
            list[BridgeFact]: Generated amendment bridge facts.
        """

        amendment_refs = list(law_obj.amendment_refs)
        if applicability_graph is not None:
            amendment_refs.extend(edge.source_doc_id for edge in applicability_graph.get_amendments(law_obj.doc_id))
        facts: list[BridgeFact] = []
        for amendment_ref in sorted({value for value in amendment_refs if value}):
            text = _normalize_text(f"{amendment_ref} amends {law_obj.short_title or law_obj.title}.")
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.LAW_AMENDMENT.value, law_obj.doc_id, amendment_ref),
                    fact_type=BridgeFactType.LAW_AMENDMENT,
                    canonical_text=text,
                    source_entity_ids=self._resolve_entity_ids(
                        entity_resolver=entity_resolver,
                        law_titles=[law_obj.short_title or law_obj.title],
                    ),
                    source_doc_ids=[law_obj.doc_id],
                    evidence_page_ids=list(law_obj.page_ids[:2]),
                    attributes={"law_title": law_obj.short_title or law_obj.title, "amendment_ref": amendment_ref},
                )
            )
        return facts

    def generate_law_definition_facts(
        self,
        *,
        law_obj: LawObject,
        entity_resolver: EntityAliasResolver,
        segments: list[LegalSegment],
    ) -> list[BridgeFact]:
        """Generate definition bridge facts from definition segments or article titles.

        Args:
            law_obj: Compiled law object.
            entity_resolver: Canonical alias resolver.
            segments: Optional law segments for the same document.

        Returns:
            list[BridgeFact]: Generated definition bridge facts.
        """

        facts: list[BridgeFact] = []
        law_entity_ids = self._resolve_entity_ids(
            entity_resolver=entity_resolver,
            law_titles=[law_obj.short_title or law_obj.title],
        )
        for segment in segments:
            if segment.segment_type.value != "definition":
                continue
            excerpt = _normalize_text(segment.text[:220])
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.LAW_DEFINITION.value, law_obj.doc_id, segment.segment_id),
                    fact_type=BridgeFactType.LAW_DEFINITION,
                    canonical_text=_normalize_text(
                        f"{law_obj.short_title or law_obj.title} defines: {excerpt}"
                    ),
                    source_entity_ids=law_entity_ids,
                    source_doc_ids=[law_obj.doc_id],
                    evidence_page_ids=list(segment.page_ids),
                    attributes={"law_title": law_obj.short_title or law_obj.title, "segment_id": segment.segment_id},
                )
            )
        return facts

    def generate_law_commencement_facts(
        self,
        *,
        law_obj: LawObject,
        entity_resolver: EntityAliasResolver,
        applicability_graph: ApplicabilityGraph | None = None,
    ) -> list[BridgeFact]:
        """Generate law-commencement bridge facts.

        Args:
            law_obj: Compiled law object.
            entity_resolver: Canonical alias resolver.
            applicability_graph: Optional amendment/commencement graph.

        Returns:
            list[BridgeFact]: Generated commencement facts.
        """

        commencement_date = law_obj.commencement_date
        evidence_page_id = law_obj.page_ids[0] if law_obj.page_ids else ""
        if applicability_graph is not None:
            record = applicability_graph.get_commencement(law_obj.doc_id)
            if record is not None:
                commencement_date = record.commencement_date or commencement_date
                evidence_page_id = record.evidence_page_id or evidence_page_id
        if not commencement_date:
            return []
        return [
            BridgeFact(
                fact_id=_stable_fact_id(BridgeFactType.LAW_COMMENCEMENT.value, law_obj.doc_id, commencement_date),
                fact_type=BridgeFactType.LAW_COMMENCEMENT,
                canonical_text=_normalize_text(
                    f"{law_obj.short_title or law_obj.title} commenced on {commencement_date}."
                ),
                source_entity_ids=self._resolve_entity_ids(
                    entity_resolver=entity_resolver,
                    law_titles=[law_obj.short_title or law_obj.title],
                ),
                source_doc_ids=[law_obj.doc_id],
                evidence_page_ids=[evidence_page_id] if evidence_page_id else list(law_obj.page_ids[:1]),
                attributes={"law_title": law_obj.short_title or law_obj.title, "commencement_date": commencement_date},
            )
        ]

    def generate_entity_document_facts(self, *, entity: LegalEntity) -> list[BridgeFact]:
        """Generate entity-document membership facts.

        Args:
            entity: Canonical entity record.

        Returns:
            list[BridgeFact]: Generated entity-document facts.
        """

        facts: list[BridgeFact] = []
        if entity.entity_type.value == "judge" and not _is_plausible_judge_name(entity.canonical_name):
            return facts
        if entity.entity_type.value == "party" and not _is_plausible_party_name(entity.canonical_name):
            return facts
        for doc_id in entity.source_doc_ids:
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.ENTITY_DOCUMENT.value, entity.entity_id, doc_id),
                    fact_type=BridgeFactType.ENTITY_DOCUMENT,
                    canonical_text=_normalize_text(
                        f"{entity.canonical_name} appears in document {doc_id} as {entity.entity_type.value}."
                    ),
                    source_entity_ids=[entity.entity_id],
                    source_doc_ids=[doc_id],
                    evidence_page_ids=[],
                    attributes={"entity_name": entity.canonical_name, "entity_type": entity.entity_type.value},
                )
            )
        return facts

    def generate_article_location_facts(
        self,
        *,
        law_obj: LawObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[BridgeFact]:
        """Generate article-location bridge facts from the law article tree.

        Args:
            law_obj: Compiled law object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[BridgeFact]: Generated article-location facts.
        """

        facts: list[BridgeFact] = []
        law_entity_ids = self._resolve_entity_ids(
            entity_resolver=entity_resolver,
            law_titles=[law_obj.short_title or law_obj.title],
        )
        for article in law_obj.article_tree:
            if not article.page_ids:
                continue
            page_range = ", ".join(article.page_ids)
            facts.append(
                BridgeFact(
                    fact_id=_stable_fact_id(BridgeFactType.ARTICLE_LOCATION.value, law_obj.doc_id, article.article_id),
                    fact_type=BridgeFactType.ARTICLE_LOCATION,
                    canonical_text=_normalize_text(
                        f"{article.label} of {law_obj.short_title or law_obj.title} appears on pages {page_range}."
                    ),
                    source_entity_ids=law_entity_ids,
                    source_doc_ids=[law_obj.doc_id],
                    evidence_page_ids=list(article.page_ids),
                    attributes={
                        "law_title": law_obj.short_title or law_obj.title,
                        "article_id": article.article_id,
                        "article_label": article.label,
                    },
                )
            )
        return facts

    @staticmethod
    def deduplicate(facts: list[BridgeFact]) -> list[BridgeFact]:
        """Deduplicate bridge facts by stable semantic key.

        Args:
            facts: Raw generated bridge facts.

        Returns:
            list[BridgeFact]: Deduplicated fact list.
        """

        deduped: dict[tuple[str, str], BridgeFact] = {}
        for fact in facts:
            key = (fact.fact_type.value, fact.canonical_text.casefold())
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = fact
                continue
            merged_entity_ids = sorted({*existing.source_entity_ids, *fact.source_entity_ids})
            merged_doc_ids = sorted({*existing.source_doc_ids, *fact.source_doc_ids})
            merged_page_ids = sorted({*existing.evidence_page_ids, *fact.evidence_page_ids})
            merged_attributes = {**existing.attributes, **fact.attributes}
            deduped[key] = existing.model_copy(
                update={
                    "source_entity_ids": merged_entity_ids,
                    "source_doc_ids": merged_doc_ids,
                    "evidence_page_ids": merged_page_ids,
                    "attributes": merged_attributes,
                }
            )
        return [deduped[key] for key in sorted(deduped)]

    @staticmethod
    def write_facts(facts: list[BridgeFact], output_path: Path) -> Path:
        """Write bridge facts to JSON.

        Args:
            facts: Bridge facts to serialize.
            output_path: Output JSON path.

        Returns:
            Path: Written output path.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [fact.model_dump(mode="json") for fact in facts]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    @staticmethod
    def _resolve_case_ids(
        *,
        case_obj: CaseObject,
        entity_resolver: EntityAliasResolver,
    ) -> list[str]:
        """Resolve canonical IDs for a case number when available.

        Args:
            case_obj: Compiled case object.
            entity_resolver: Canonical alias resolver.

        Returns:
            list[str]: Canonical IDs linked to the case.
        """

        if not _is_plausible_case_identifier(case_obj.case_number):
            return []
        return entity_resolver.resolve_known_values(case_numbers=[case_obj.case_number])

    @staticmethod
    def _resolve_entity_ids(
        *,
        entity_resolver: EntityAliasResolver,
        law_titles: list[str] | None = None,
        case_numbers: list[str] | None = None,
        party_names: list[str] | None = None,
        authority_names: list[str] | None = None,
        judge_names: list[str] | None = None,
    ) -> list[str]:
        """Resolve structured values into canonical entity IDs.

        Args:
            entity_resolver: Canonical alias resolver.
            law_titles: Candidate law titles.
            case_numbers: Candidate case numbers.
            party_names: Candidate party names.
            authority_names: Candidate authority names.
            judge_names: Candidate judge names.

        Returns:
            list[str]: Canonical entity IDs.
        """

        return entity_resolver.resolve_known_values(
            law_titles=law_titles or [],
            case_numbers=case_numbers or [],
            party_names=party_names or [],
            authority_names=authority_names or [],
            judge_names=judge_names or [],
        )
