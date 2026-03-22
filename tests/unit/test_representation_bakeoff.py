from __future__ import annotations

from pathlib import Path

from rag_challenge.eval.representation_bakeoff import (
    build_bakeoff_markdown,
    load_external_benchmark_csv,
    load_local_representation_metrics,
    summarize_representation_candidates,
)


def test_load_external_benchmark_csv_parses_markdown_links(tmp_path: Path) -> None:
    csv_path = tmp_path / "mteb.csv"
    csv_path.write_text(
        ",Model,Embedding Dimensions,Max Tokens,Retrieval\n"
        '0,"[voyage-law-2](https://example.com/voyage)",1024,16000,65.39\n',
        encoding="utf-8",
    )

    rows = load_external_benchmark_csv(csv_path)

    assert rows[0].model_name == "voyage-law-2"
    assert rows[0].source_url == "https://example.com/voyage"
    assert rows[0].retrieval_score == 65.39


def test_load_local_representation_metrics_supports_jsonl(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"model_name":"kanon","slice_name":"hot","doc_recall":0.7,"page_recall":0.6,"same_doc_localization":0.5,"latency_ms":100}\n'
        '{"model_name":"voyage-law-2","slice_name":"hot","doc_recall":0.8,"page_recall":0.7,"same_doc_localization":0.6,"latency_ms":120}\n',
        encoding="utf-8",
    )

    metrics = load_local_representation_metrics(metrics_path)

    assert [metric.model_name for metric in metrics] == ["kanon", "voyage-law-2"]


def test_summarize_representation_candidates_ranks_by_local_metrics() -> None:
    metrics = [
        {
            "model_name": "kanon",
            "slice_name": "hot",
            "doc_recall": 0.70,
            "page_recall": 0.60,
            "same_doc_localization": 0.50,
            "latency_ms": 100,
        },
        {
            "model_name": "voyage-law-2",
            "slice_name": "hot",
            "doc_recall": 0.82,
            "page_recall": 0.76,
            "same_doc_localization": 0.61,
            "latency_ms": 140,
        },
    ]
    summaries = summarize_representation_candidates(
        load_local_representation_metrics_from_list(metrics),
    )

    assert summaries[0].model_name == "voyage-law-2"


def test_build_bakeoff_markdown_requires_local_metrics_for_promotion() -> None:
    markdown = build_bakeoff_markdown(
        [],
        external_rows_used=[Path("/tmp/external.csv")],
        local_metric_files=[],
    )

    assert "not sufficient for promotion" in markdown


def load_local_representation_metrics_from_list(items: list[dict[str, object]]):
    from rag_challenge.eval.representation_bakeoff import LocalRepresentationMetric

    return [LocalRepresentationMetric.model_validate(item) for item in items]
