from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from shafi.ingestion.applicability_graph import (
    build_corpus_registry_from_parsed_documents,
    extract_amendment_edges,
    extract_commencement_dates,
    validate_graph,
)
from shafi.models import DocType, DocumentSection, ParsedDocument
from shafi.models.applicability import ApplicabilityEdge, ApplicabilityEdgeType, ApplicabilityGraph
from shafi.models.legal_objects import CorpusRegistry, LawObject, LegalDocType

if TYPE_CHECKING:
    from pathlib import Path


def _law_doc(*, doc_id: str, title: str, pages: dict[int, str]) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        doc_type=DocType.STATUTE,
        source_path=f"/tmp/{doc_id}.pdf",
        full_text="\n".join(pages.values()),
        sections=[
            DocumentSection(section_path=f"page:{page_num}", text=text) for page_num, text in sorted(pages.items())
        ],
    )


def test_extract_amendment_edge_from_synthetic_law_text() -> None:
    target = LawObject(
        object_id="law-base",
        doc_id="law-base",
        title="Base Law 2018",
        legal_doc_type=LegalDocType.LAW,
        page_ids=["law-base_1"],
        source_text="",
        page_texts={"law-base_1": "Base Law 2018"},
    )
    amendment = LawObject(
        object_id="law-amendment",
        doc_id="law-amendment",
        title="Amendment Law 2020",
        legal_doc_type=LegalDocType.AMENDMENT,
        page_ids=["law-amendment_2"],
        source_text="",
        page_texts={
            "law-amendment_2": "This Amendment Law amends Base Law 2018 in Article 5 and comes into force on 1 January 2020."
        },
    )

    edges = extract_amendment_edges(amendment, [target, amendment])

    assert len(edges) == 1
    assert edges[0].target_doc_id == "law-base"
    assert edges[0].edge_type is ApplicabilityEdgeType.AMENDS
    assert edges[0].scope == "Article 5"
    assert edges[0].effective_date == "1 January 2020"


def test_extract_commencement_dates_reads_law_text() -> None:
    registry = CorpusRegistry(
        laws={
            "law-base": LawObject(
                object_id="law-base",
                doc_id="law-base",
                title="Base Law 2018",
                legal_doc_type=LegalDocType.LAW,
                page_ids=["law-base_1"],
                source_text="",
                page_texts={"law-base_1": "This Law comes into force on 1 January 2018."},
            )
        }
    )

    commencements = extract_commencement_dates(registry)

    assert len(commencements) == 1
    assert commencements[0].law_id == "law-base"
    assert commencements[0].commencement_date == "1 January 2018"
    assert commencements[0].evidence_page_id == "law-base_1"


def test_get_current_version_follows_supersession_chain() -> None:
    original = LawObject(object_id="law-2018", doc_id="law-2018", title="Law 2018", legal_doc_type=LegalDocType.LAW)
    replacement = LawObject(object_id="law-2022", doc_id="law-2022", title="Law 2022", legal_doc_type=LegalDocType.LAW)
    graph = ApplicabilityGraph(
        nodes=["law-2018", "law-2022"],
        edges=[
            ApplicabilityEdge(
                source_doc_id="law-2022",
                target_doc_id="law-2018",
                edge_type=ApplicabilityEdgeType.REPLACES,
                effective_date="1 January 2022",
            )
        ],
        laws={"law-2018": original, "law-2022": replacement},
    )

    current = graph.get_current_version("law-2018")

    assert current is not None
    assert current.doc_id == "law-2022"


def test_validate_graph_detects_orphans_and_cycles() -> None:
    graph = ApplicabilityGraph(
        nodes=["a", "b"],
        edges=[
            ApplicabilityEdge(source_doc_id="a", target_doc_id="missing", edge_type=ApplicabilityEdgeType.AMENDS),
            ApplicabilityEdge(source_doc_id="a", target_doc_id="b", edge_type=ApplicabilityEdgeType.REPLACES),
            ApplicabilityEdge(source_doc_id="b", target_doc_id="a", edge_type=ApplicabilityEdgeType.REPLACES),
        ],
    )

    warnings = validate_graph(graph)
    warning_types = {warning.warning_type for warning in warnings}

    assert "orphan_edge" in warning_types
    assert "circular_supersession" in warning_types


def test_get_amendment_history_orders_by_effective_date() -> None:
    graph = ApplicabilityGraph(
        nodes=["law-base", "amendment-1", "amendment-2"],
        edges=[
            ApplicabilityEdge(
                source_doc_id="amendment-2",
                target_doc_id="law-base",
                edge_type=ApplicabilityEdgeType.AMENDS,
                effective_date="1 January 2023",
            ),
            ApplicabilityEdge(
                source_doc_id="amendment-1",
                target_doc_id="law-base",
                edge_type=ApplicabilityEdgeType.AMENDS,
                effective_date="1 January 2020",
            ),
        ],
    )

    history = graph.get_amendment_history("law-base")

    assert [edge.source_doc_id for edge in history] == ["amendment-1", "amendment-2"]


def test_build_corpus_registry_and_pipeline_hook_persist_graph(tmp_path: Path) -> None:
    doc = _law_doc(
        doc_id="law-base",
        title="Base Law 2018",
        pages={1: "Base Law 2018 comes into force on 1 January 2018."},
    )
    registry = build_corpus_registry_from_parsed_documents([doc])
    assert "law-base" in registry.laws

    settings = SimpleNamespace(
        ingestion=SimpleNamespace(
            ingest_version="v-test",
            build_applicability_graph=True,
            applicability_graph_path=str(tmp_path / "applicability_graph.json"),
        )
    )
    with patch("shafi.ingestion.pipeline.get_settings", return_value=settings):
        from shafi.ingestion.pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            parser=SimpleNamespace(),
            chunker=SimpleNamespace(),
            sac=SimpleNamespace(),
            embedder=SimpleNamespace(),
            store=SimpleNamespace(),
        )
        pipeline._maybe_persist_applicability_graph(docs=[doc], manifest_path=tmp_path / "manifest.json")

    assert (tmp_path / "applicability_graph.json").exists()
