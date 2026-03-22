"""Offline applicability-graph extraction for compiled legal objects."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

from rag_challenge.models.applicability import (
    ApplicabilityEdge,
    ApplicabilityEdgeType,
    ApplicabilityGraph,
    CommencementRecord,
    GraphWarning,
)
from rag_challenge.models.legal_objects import CorpusRegistry, LawObject, LegalDocType

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from rag_challenge.models import ParsedDocument


_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    r"|date specified in the Enactment Notice)\b"
)
_SCOPE_RE = re.compile(r"\b(Article|Section|Schedule)\s+[A-Za-z0-9()]+", re.IGNORECASE)
_AMENDMENT_TRIGGER_RE = re.compile(r"\b(amends?|amended by|inserts?|substitutes?)\b", re.IGNORECASE)
_SUPERSESSION_TRIGGER_RE = re.compile(r"\b(replaces?|supersedes?|revokes?|repeals?)\b", re.IGNORECASE)
_COMMENCEMENT_TRIGGER_RE = re.compile(r"\b(comes into force|commencement)\b", re.IGNORECASE)


def build_corpus_registry_from_parsed_documents(parsed_docs: Sequence[ParsedDocument]) -> CorpusRegistry:
    """Build a minimal corpus registry from parsed documents.

    Args:
        parsed_docs: Parsed source documents.

    Returns:
        Minimal registry with law-like objects required for graph extraction.
    """

    laws: dict[str, LawObject] = {}
    for doc in parsed_docs:
        legal_doc_type = _resolve_legal_doc_type(doc.title, doc.full_text)
        if legal_doc_type not in {LegalDocType.LAW, LegalDocType.REGULATION, LegalDocType.AMENDMENT, LegalDocType.ENACTMENT_NOTICE}:
            continue
        page_texts = _extract_page_texts(doc)
        page_ids = sorted(page_texts)
        laws[doc.doc_id] = LawObject(
            object_id=doc.doc_id,
            doc_id=doc.doc_id,
            title=doc.title,
            source_path=doc.source_path,
            page_ids=page_ids,
            source_text=doc.full_text,
            page_texts=page_texts,
            legal_doc_type=legal_doc_type,
            short_title=doc.title,
            law_number=_extract_law_number(doc.title, doc.full_text),
            year=_extract_year(doc.title, doc.full_text),
            issuing_authority=_extract_issuing_authority(doc.full_text),
            commencement_date="",
            amendment_refs=[],
            article_tree=[],
        )
    return CorpusRegistry(schema_version=1, source_doc_count=len(parsed_docs), laws=laws)


def build_applicability_graph(corpus_registry: CorpusRegistry) -> ApplicabilityGraph:
    """Build the full applicability graph from a corpus registry."""

    laws = dict(corpus_registry.laws)
    edges: list[ApplicabilityEdge] = []
    for law in laws.values():
        edges.extend(extract_amendment_edges(law, laws.values()))
    edges.extend(extract_supersession_edges(corpus_registry))
    commencements = extract_commencement_dates(corpus_registry)
    graph = ApplicabilityGraph(
        nodes=sorted(laws),
        edges=_dedupe_edges(edges),
        commencements=_dedupe_commencements(commencements),
        laws=cast("dict[str, object]", laws),
    )
    graph.warnings = validate_graph(graph)
    return graph


def extract_amendment_edges(
    law_object: LawObject,
    all_docs: Iterable[LawObject],
) -> list[ApplicabilityEdge]:
    """Parse amendment references from a law object's text."""

    title_index = _title_index(all_docs)
    edges: list[ApplicabilityEdge] = []
    for page_id, page_text in law_object.page_texts.items():
        normalized_page = _normalize_text(page_text)
        if not _AMENDMENT_TRIGGER_RE.search(page_text):
            continue
        for normalized_title, target_doc_id in title_index.items():
            if target_doc_id == law_object.doc_id or normalized_title not in normalized_page:
                continue
            edge_type = ApplicabilityEdgeType.AMENDS
            if "insert" in normalized_page:
                edge_type = ApplicabilityEdgeType.INSERTS
            elif "substitut" in normalized_page:
                edge_type = ApplicabilityEdgeType.SUBSTITUTES
            edges.append(
                ApplicabilityEdge(
                    source_doc_id=law_object.doc_id,
                    target_doc_id=target_doc_id,
                    edge_type=edge_type,
                    effective_date=_extract_effective_date(page_text),
                    scope=_extract_scope(page_text),
                    evidence_text=_truncate_evidence(page_text),
                    evidence_page_id=page_id,
                )
            )
    return edges


def extract_commencement_dates(corpus_registry: CorpusRegistry) -> list[CommencementRecord]:
    """Parse commencement records from law and enactment-notice text."""

    records: list[CommencementRecord] = []
    for law in corpus_registry.laws.values():
        for page_id, page_text in law.page_texts.items():
            if not _COMMENCEMENT_TRIGGER_RE.search(page_text):
                continue
            date = _extract_effective_date(page_text)
            if not date:
                continue
            records.append(
                CommencementRecord(
                    law_id=law.doc_id,
                    commencement_date=date,
                    commencement_notice_id=law.doc_id if law.legal_doc_type is LegalDocType.ENACTMENT_NOTICE else "",
                    evidence_page_id=page_id,
                )
            )
            break
    return records


def extract_supersession_edges(corpus_registry: CorpusRegistry) -> list[ApplicabilityEdge]:
    """Parse supersession and replacement edges from law text."""

    title_index = _title_index(corpus_registry.laws.values())
    edges: list[ApplicabilityEdge] = []
    for law in corpus_registry.laws.values():
        for page_id, page_text in law.page_texts.items():
            normalized_page = _normalize_text(page_text)
            if not _SUPERSESSION_TRIGGER_RE.search(page_text):
                continue
            for normalized_title, target_doc_id in title_index.items():
                if target_doc_id == law.doc_id or normalized_title not in normalized_page:
                    continue
                edge_type = ApplicabilityEdgeType.SUPERSEDES
                if "replace" in normalized_page:
                    edge_type = ApplicabilityEdgeType.REPLACES
                elif "repeal" in normalized_page:
                    edge_type = ApplicabilityEdgeType.REPEALS
                elif "revok" in normalized_page:
                    edge_type = ApplicabilityEdgeType.REVOKES
                edges.append(
                    ApplicabilityEdge(
                        source_doc_id=law.doc_id,
                        target_doc_id=target_doc_id,
                        edge_type=edge_type,
                        effective_date=_extract_effective_date(page_text),
                        scope=_extract_scope(page_text),
                        evidence_text=_truncate_evidence(page_text),
                        evidence_page_id=page_id,
                    )
                )
    return edges


def validate_graph(graph: ApplicabilityGraph) -> list[GraphWarning]:
    """Validate graph integrity and emit non-fatal warnings."""

    warnings: list[GraphWarning] = []
    node_ids = set(graph.nodes)
    for edge in graph.edges:
        if edge.source_doc_id not in node_ids or edge.target_doc_id not in node_ids:
            warnings.append(
                GraphWarning(
                    warning_type="orphan_edge",
                    message="Edge references a document outside the graph node set.",
                    doc_ids=[edge.source_doc_id, edge.target_doc_id],
                )
            )
        if edge.edge_type in {ApplicabilityEdgeType.AMENDS, ApplicabilityEdgeType.SUPERSEDES, ApplicabilityEdgeType.REPLACES} and not edge.effective_date:
            warnings.append(
                GraphWarning(
                    warning_type="missing_effective_date",
                    message="Temporal edge is missing an effective date.",
                    doc_ids=[edge.source_doc_id, edge.target_doc_id],
                )
            )
    cycle_nodes = _find_supersession_cycle_nodes(graph.edges)
    if cycle_nodes:
        warnings.append(
            GraphWarning(
                warning_type="circular_supersession",
                message="Supersession/replacement edges form a cycle.",
                doc_ids=sorted(cycle_nodes),
            )
        )
    return warnings


def write_applicability_graph(
    *,
    graph: ApplicabilityGraph,
    output_path: Path,
) -> None:
    """Persist a graph as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _resolve_legal_doc_type(title: str, full_text: str) -> LegalDocType:
    blob = _normalize_text("\n".join([title, full_text[:2000]]))
    if "enactment notice" in blob:
        return LegalDocType.ENACTMENT_NOTICE
    if "amendment law" in blob:
        return LegalDocType.AMENDMENT
    if "regulation" in blob:
        return LegalDocType.REGULATION
    if "law" in blob:
        return LegalDocType.LAW
    return LegalDocType.OTHER


def _extract_page_texts(doc: ParsedDocument) -> dict[str, str]:
    page_texts: dict[str, str] = {}
    for section in doc.sections:
        if not section.section_path.startswith("page:"):
            continue
        page_num_text = section.section_path.split(":", maxsplit=1)[1]
        if not page_num_text.isdigit():
            continue
        page_texts[f"{doc.doc_id}_{page_num_text}"] = section.text
    if not page_texts and doc.full_text:
        page_texts[f"{doc.doc_id}_1"] = doc.full_text
    return page_texts


def _title_index(docs: Iterable[LawObject]) -> dict[str, str]:
    index: dict[str, str] = {}
    for doc in docs:
        normalized_title = _normalize_text(doc.title)
        if normalized_title:
            index[normalized_title] = doc.doc_id
    return index


def _extract_law_number(title: str, full_text: str) -> str:
    match = re.search(r"\bLaw\s+No\.?\s*([0-9]+)\b", f"{title}\n{full_text}", re.IGNORECASE)
    return match.group(1) if match is not None else ""


def _extract_year(title: str, full_text: str) -> str:
    match = re.search(r"\b(20\d{2}|19\d{2})\b", f"{title}\n{full_text}")
    return match.group(1) if match is not None else ""


def _extract_issuing_authority(full_text: str) -> str:
    if "Ruler of Dubai" in full_text:
        return "Ruler of Dubai"
    return ""


def _extract_effective_date(text: str) -> str | None:
    match = _DATE_RE.search(text)
    return match.group(1) if match is not None else None


def _extract_scope(text: str) -> str:
    match = _SCOPE_RE.search(text)
    return match.group(0) if match is not None else ""


def _truncate_evidence(text: str) -> str:
    return " ".join(text.split())[:240]


def _normalize_text(text: str) -> str:
    return " ".join(token.lower() for token in re.findall(r"[A-Za-z0-9]+", text))


def _dedupe_edges(edges: Sequence[ApplicabilityEdge]) -> list[ApplicabilityEdge]:
    deduped: dict[tuple[str, str, str, str, str], ApplicabilityEdge] = {}
    for edge in edges:
        key = (
            edge.source_doc_id,
            edge.target_doc_id,
            edge.edge_type.value,
            edge.scope,
            edge.evidence_page_id,
        )
        deduped[key] = edge
    return [deduped[key] for key in sorted(deduped)]


def _dedupe_commencements(records: Sequence[CommencementRecord]) -> list[CommencementRecord]:
    deduped: dict[tuple[str, str, str], CommencementRecord] = {}
    for record in records:
        key = (record.law_id, record.commencement_date, record.evidence_page_id)
        deduped[key] = record
    return [deduped[key] for key in sorted(deduped)]


def _find_supersession_cycle_nodes(edges: Sequence[ApplicabilityEdge]) -> set[str]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if edge.edge_type not in {ApplicabilityEdgeType.SUPERSEDES, ApplicabilityEdgeType.REPLACES}:
            continue
        adjacency.setdefault(edge.target_doc_id, []).append(edge.source_doc_id)

    cycle_nodes: set[str] = set()
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle_nodes.add(node)
            return
        visiting.add(node)
        for neighbor in adjacency.get(node, []):
            _visit(neighbor)
            if neighbor in cycle_nodes:
                cycle_nodes.add(node)
        visiting.remove(node)
        visited.add(node)

    for node in list(adjacency):
        _visit(node)
    return cycle_nodes
