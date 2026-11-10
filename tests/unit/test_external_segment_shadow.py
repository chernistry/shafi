from __future__ import annotations

import json
from typing import TYPE_CHECKING

from shafi.eval.external_segment_shadow import (
    ExternalSegmentFamily,
    evaluate_external_segment_shadow,
    load_shadow_benchmark_cases,
    route_external_segment_family,
    run_external_segment_shadow_ablation,
)
from shafi.ingestion.external_segment_payload import load_external_segment_payload
from shafi.ingestion.rich_segment_text import SegmentTextMode

if TYPE_CHECKING:
    from pathlib import Path


def _write_payload(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "embedding_model": "test-model",
                "segments_path": "segments.jsonl",
                "output_cache_name": "cache",
                "segments": [
                    {
                        "segment_id": "doc-a:1:1",
                        "doc_id": "doc-a",
                        "page_number": 1,
                        "text": "Claimant: Fursa Consulting",
                        "title": "doc-a",
                        "structure_type": "paragraph",
                        "hierarchy": ["doc-a"],
                        "context_text": "Caption page for CFI 010/2024 Claimant: Fursa Consulting",
                        "embedding_text": "caption claimant",
                        "metadata": {
                            "case_refs": ["CFI 010/2024"],
                            "law_refs": [],
                            "token_count": 4,
                            "document_descriptor": "case | CFI 010/2024 | 2024",
                        },
                    },
                    {
                        "segment_id": "doc-b:4:1",
                        "doc_id": "doc-b",
                        "page_number": 4,
                        "text": "Article 16 requires the annual return.",
                        "title": "doc-b",
                        "structure_type": "paragraph",
                        "hierarchy": ["doc-b", "Article 16"],
                        "context_text": "Operating Law 2018 Article 16 requires the annual return.",
                        "embedding_text": "article 16 annual return",
                        "metadata": {
                            "case_refs": [],
                            "law_refs": ["Operating Law 2018", "Article 16"],
                            "token_count": 6,
                            "document_descriptor": "law/regulation | Operating Law 2018 | 2018",
                        },
                    },
                    {
                        "segment_id": "doc-c:2:1",
                        "doc_id": "doc-c",
                        "page_number": 2,
                        "text": "Legislative Authority This Law is made by the Ruler of Dubai.",
                        "title": "doc-c",
                        "structure_type": "paragraph",
                        "hierarchy": ["doc-c"],
                        "context_text": "General Partnership Law Legislative Authority This Law is made by the Ruler of Dubai.",
                        "embedding_text": "legislative authority ruler of dubai",
                        "metadata": {
                            "case_refs": [],
                            "law_refs": ["General Partnership Law 2004"],
                            "token_count": 9,
                            "document_descriptor": "law/regulation | General Partnership Law 2004 | 2004",
                        },
                    },
                ],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q-claimant",
                        "question": "Who were the claimants in case CFI 010/2024?",
                        "telemetry": {
                            "doc_refs": ["CFI 010/2024"],
                            "retrieved_page_ids": ["doc-a_2"],
                            "used_page_ids": ["doc-a_1"],
                        },
                    },
                    {
                        "question_id": "q-article",
                        "question": "According to Article 16 of the Operating Law 2018, what must be filed?",
                        "telemetry": {
                            "doc_refs": ["Operating Law 2018"],
                            "retrieved_page_ids": ["doc-b_3"],
                            "used_page_ids": ["doc-b_4"],
                        },
                    },
                    {
                        "question_id": "q-authority",
                        "question": "Which law names the Ruler of Dubai as the legislative authority?",
                        "telemetry": {
                            "doc_refs": ["General Partnership Law 2004"],
                            "retrieved_page_ids": ["doc-c_5"],
                            "used_page_ids": ["doc-c_2"],
                        },
                    },
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def test_load_external_segment_payload_and_shadow_cases(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    eval_path = tmp_path / "eval.json"
    _write_payload(payload_path)
    _write_eval(eval_path)

    payload = load_external_segment_payload(payload_path)
    cases = load_shadow_benchmark_cases(eval_path)

    assert payload.embedding_model == "test-model"
    assert payload.segments[0].page_id == "doc-a_1"
    assert [case.question_id for case in cases] == ["q-claimant", "q-article", "q-authority"]


def test_route_external_segment_family_covers_required_clusters() -> None:
    assert (
        route_external_segment_family("Who were the claimants in case CFI 010/2024?", doc_refs=["CFI 010/2024"])
        is ExternalSegmentFamily.TITLE_CAPTION_CLAIMANT
    )
    assert (
        route_external_segment_family(
            "According to Article 16 of the Operating Law 2018, what must be filed?",
            doc_refs=["Operating Law 2018"],
        )
        is ExternalSegmentFamily.EXACT_PROVISION
    )
    assert (
        route_external_segment_family(
            "Which law names the Ruler of Dubai as the legislative authority?",
            doc_refs=["General Partnership Law 2004"],
        )
        is ExternalSegmentFamily.AUTHORITY_DATE_LAW_NUMBER
    )


def test_evaluate_external_segment_shadow_projects_page_valid_hits(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    eval_path = tmp_path / "eval.json"
    _write_payload(payload_path)
    _write_eval(eval_path)

    summary = evaluate_external_segment_shadow(
        payload=load_external_segment_payload(payload_path),
        cases=load_shadow_benchmark_cases(eval_path),
        projected_top_k=1,
        candidate_pool_size=3,
        composer_mode=SegmentTextMode.RICH,
    )

    assert len(summary.cases) == 3
    assert all(case.projected_hit for case in summary.cases)
    assert all(case.projected_page_ids == [case.gold_page_ids[0]] for case in summary.cases)
    by_family = {metric.family: metric for metric in summary.family_metrics}
    assert by_family[ExternalSegmentFamily.TITLE_CAPTION_CLAIMANT].baseline_hit_rate == 0.0
    assert by_family[ExternalSegmentFamily.TITLE_CAPTION_CLAIMANT].projected_hit_rate == 1.0
    assert by_family[ExternalSegmentFamily.EXACT_PROVISION].projected_precision == 1.0


def test_run_external_segment_shadow_ablation_reports_plain_and_rich(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    eval_path = tmp_path / "eval.json"
    _write_payload(payload_path)
    _write_eval(eval_path)

    summary = run_external_segment_shadow_ablation(
        payload_path=payload_path,
        benchmark_path=eval_path,
        projected_top_k=1,
        candidate_pool_size=3,
    )

    assert summary.plain_summary.composer_mode is SegmentTextMode.PLAIN
    assert summary.rich_summary.composer_mode is SegmentTextMode.RICH
    assert set(summary.title_header_noise_rate) == {"plain", "rich"}
