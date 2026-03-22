from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts import stress_test_ingestion as mod

from rag_challenge.ingestion.pipeline import IngestionStats

if TYPE_CHECKING:
    from pathlib import Path


def test_build_report_marks_external_uncertainty() -> None:
    stats = IngestionStats(
        docs_parsed=300,
        docs_failed=0,
        chunks_created=900,
        chunks_embedded=900,
        chunks_upserted=900,
        sac_summaries_generated=300,
        elapsed_s=12.0,
    )
    timings = mod.StageTimings(parse_s=4.0, chunk_s=3.0, sac_s=1.0, embed_s=2.0, upsert_s=0.5, delete_s=0.1)

    report = mod._build_report(
        target_docs=300,
        fixture_templates=3,
        stats=stats,
        stage_timings=timings,
        python_peak_alloc_mb=24.5,
    )

    assert report["docs_per_s"] == 25.0
    assert report["chunks_per_s"] == 75.0
    assert report["largest_measured_stage"] == "parse"
    assert report["blocking_bottleneck"] == "none_local_blocker_detected"
    assert report["largest_uncertainty"] == "external_services_mocked"
    assert report["readiness_verdict"] == "proxy_ready_with_external_uncertainty"


@pytest.mark.asyncio
async def test_run_proxy_executes_real_parser_and_chunker(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "sample_a.txt").write_text("Title A\n\nBody text for sample A.", encoding="utf-8")
    (fixture_dir / "sample_b.txt").write_text("Title B\n\nBody text for sample B.", encoding="utf-8")

    report = await mod._run_proxy(fixture_dir=fixture_dir, target_docs=4)

    assert report["target_docs"] == 4
    assert report["docs_parsed"] == 4
    assert report["docs_failed"] == 0
    assert report["chunks_created"] >= 4
    assert report["readiness_verdict"] == "proxy_ready_with_external_uncertainty"


def test_main_writes_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "sample_a.txt").write_text("Title A\n\nBody text for sample A.", encoding="utf-8")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "sys.argv",
        [
            "stress_test_ingestion.py",
            "--fixture-dir",
            str(fixture_dir),
            "--target-docs",
            "3",
            "--out-dir",
            str(out_dir),
        ],
    )

    mod.main()

    payload = json.loads((out_dir / "ingestion_stress_report.json").read_text(encoding="utf-8"))
    assert payload["target_docs"] == 3
    assert payload["docs_parsed"] == 3
    markdown = (out_dir / "ingestion_stress_report.md").read_text(encoding="utf-8")
    assert "Ingestion Stress Proxy" in markdown
    assert "external_services_mocked" in markdown
