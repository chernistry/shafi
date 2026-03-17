from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "scripts" / "scan_private_doc_anomalies.py"
    spec = importlib.util.spec_from_file_location("scan_private_doc_anomalies", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pdf(path: Path, pages: list[str]) -> None:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 520, 760), text, fontsize=12)
    pdf.save(path)
    pdf.close()


def _run_cli(
    *,
    input_dir: Path,
    output_dir: Path,
    coverage_priors_json: Path | None = None,
    mode: str = "raw-pdf-corpus",
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/scan_private_doc_anomalies.py",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--mode",
        mode,
    ]
    if coverage_priors_json is not None:
        command.extend(["--coverage-priors-json", str(coverage_priors_json)])
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _default_metadata(
    module,
    *,
    page_num: int = 1,
    image_count: int = 0,
    internal_link_count: int = 0,
    link_destinations: list[str] | None = None,
    external_urls: list[str] | None = None,
):
    return module.PdfPageMetadata(
        page_num=page_num,
        image_count=image_count,
        internal_link_count=internal_link_count,
        link_destinations=link_destinations if link_destinations is not None else (["page:2"] if internal_link_count else []),
        external_urls=external_urls if external_urls is not None else [],
    )


def test_scan_private_doc_anomalies_empty_directory_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    _run_cli(input_dir=tmp_path, output_dir=output_dir)

    assert (output_dir / "scan_results.jsonl").exists()
    assert (output_dir / "scan_results.jsonl").read_text(encoding="utf-8") == ""
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    top20 = (output_dir / "top20_report.md").read_text(encoding="utf-8")
    assert "Docs scanned: 0" in summary
    assert "Weighted suspicion scoring is heuristic" in summary
    assert "Top 20 Suspicious Documents" in top20


def test_scan_private_doc_anomalies_single_clean_pdf_emits_record(tmp_path: Path) -> None:
    pdf_path = tmp_path / "clean_doc.pdf"
    _write_pdf(pdf_path, ["This is a clean first page.", "This is a clean second page."])
    output_dir = tmp_path / "out"

    _run_cli(input_dir=tmp_path, output_dir=output_dir)

    rows = [
        json.loads(line)
        for line in (output_dir / "scan_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["doc_id"] == "clean_doc"
    assert row["filename"] == "clean_doc.pdf"
    assert row["mode"] == "raw-pdf-corpus"
    assert row["page_count"] == 2
    assert row["extracted_page_count"] == 2
    assert row["signals"]
    assert len(row["per_page"]) == 2
    assert isinstance(row["doc_family_tags"], list)
    assert row["internal_link_count"] == 0
    assert row["exact_duplicate_cluster_id"] is None
    assert "family_query_coverage_bucket" in row
    assert "toc_pointer_type" in row
    assert isinstance(row["suspicion_score"], int)
    assert isinstance(row["reason_tags"], list)


def test_scan_private_doc_anomalies_structural_smoke_detects_title_and_contents(tmp_path: Path) -> None:
    module = _load_module()
    pdf_path = tmp_path / "structure.pdf"
    _write_pdf(
        pdf_path,
        [
            "LAW NO. 5 OF 2024\nThis Law may be cited as the Example Law.",
            "CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4\nArticle 3 ... 6\nArticle 4 ... 8",
        ],
    )

    records = module.scan_pdf_corpus(input_dir=tmp_path, mode="raw-pdf-corpus", coverage_priors={})

    assert len(records) == 1
    per_page = records[0]["per_page"]
    assert per_page[0]["signals"]["title_page_signature"] is True
    assert per_page[1]["signals"]["contents_signature"] is True
    assert per_page[1]["signals"]["high_article_reference_density"] is True


def test_scan_private_doc_anomalies_ocr_text_quality_smoke() -> None:
    module = _load_module()

    low_text = module.analyze_page_signals(
        text="short",
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )
    punct_heavy = module.analyze_page_signals(
        text="!!!\n--\n[]\n@@\n;;",
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module, image_count=1),
        fallback_triggered=True,
    )
    broken_lines = module.analyze_page_signals(
        text="A\nB\nC\nD\nE\nF\nG",
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )

    assert low_text.page_record["signals"]["low_text_page"] is True
    assert punct_heavy.page_record["signals"]["layout_artifact_page"] is True
    assert punct_heavy.page_record["signals"]["ocr_fallback_likelihood"] > 0.0
    assert broken_lines.page_record["signals"]["broken_lines_short_burst"] is True
    assert broken_lines.page_record["signals"]["many_one_word_lines"] is True


def test_scan_private_doc_anomalies_unicode_smoke() -> None:
    module = _load_module()
    text = (
        "Alpha\u200bBeta\u00a0Gamma\ufffd "
        "\u2018quote\u2019 \u201cdouble\u201d "
        "\u2013 \u2014 \u0661\u0662 \u060c"
    )

    analysis = module.analyze_page_signals(
        text=text,
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )
    signals = analysis.page_record["signals"]

    assert signals["zero_width_count"] == 1
    assert signals["nbsp_count"] == 1
    assert signals["replacement_char_count"] == 1
    assert signals["smart_quote_count"] == 4
    assert signals["dash_variant_count"] == 2
    assert signals["eastern_arabic_digit_count"] == 2
    assert signals["arabic_punctuation_count"] == 1


def test_scan_private_doc_anomalies_scoring_ignores_single_smart_quote_noise() -> None:
    module = _load_module()

    score, reasons = module._score_record(
        {
            "signals": {"smart_quote_count": 1, "dash_variant_count": 0},
            "doc_family_tags": [],
            "per_page": [],
            "page_count": 1,
        }
    )

    assert isinstance(score, int)
    assert "weird_unicode_or_replacement" not in reasons


def test_scan_private_doc_anomalies_scoring_keeps_meaningful_unicode_disruption() -> None:
    module = _load_module()

    _, reasons = module._score_record(
        {
            "signals": {"smart_quote_count": 4, "dash_variant_count": 0},
            "doc_family_tags": [],
            "per_page": [],
            "page_count": 1,
        }
    )

    assert "weird_unicode_or_replacement" in reasons


def test_scan_private_doc_anomalies_detects_compacted_pipe_dense_table_page() -> None:
    module = _load_module()
    compact_table = (
        "Field | Value | Note Registrar | DIFC Registrar | active "
        "Law Number | 11/2004 | source Penalty | USD 1000 | schedule "
        "Reference | Article 4 | title Schedule | Schedule 1 | interpretation"
    )

    analysis = module.analyze_page_signals(
        text=compact_table,
        page_num=2,
        page_count=2,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )

    assert analysis.page_record["signals"]["table_heavy_signature"] is True


def test_scan_private_doc_anomalies_mixed_script_smoke() -> None:
    module = _load_module()
    lexical = "\u0647\u0630\u0627 \u0646\u0635 \u0639\u0631\u0628\u064a \u0643\u0627\u0645\u0644"
    mixed = "Heading \u0646\u0635 English"
    ornamental = "Seal\n\u062f\u0628\u064a\nApproved"
    digits_only = "\u0661\u0662\u0663\u0664"

    lexical_analysis = module.analyze_page_signals(
        text=lexical,
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )
    mixed_analysis = module.analyze_page_signals(
        text=mixed,
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )
    ornamental_analysis = module.analyze_page_signals(
        text=ornamental,
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )
    digits_analysis = module.analyze_page_signals(
        text=digits_only,
        page_num=1,
        page_count=1,
        metadata=_default_metadata(module),
        fallback_triggered=False,
    )

    assert lexical_analysis.page_record["dominant_arabic_category"] == "lexical_arabic"
    assert mixed_analysis.page_record["dominant_arabic_category"] == "mixed_latin_arabic"
    assert ornamental_analysis.page_record["dominant_arabic_category"] == "ornamental_arabic"
    assert digits_analysis.page_record["dominant_arabic_category"] == "arabic_digit_presence"


def test_scan_private_doc_anomalies_same_case_multi_artifact_hint(tmp_path: Path) -> None:
    module = _load_module()
    _write_pdf(
        tmp_path / "law_artifact.pdf",
        [
            "LAW NO. 5 OF 2024\nReference: CFI 057/2025\nThis Law may be cited as Example Law.",
            "Article 1 ...",
        ],
    )
    _write_pdf(
        tmp_path / "order_artifact.pdf",
        [
            "IN THE DIFC COURTS\nCFI 057/2025",
            "IT IS ORDERED THAT the claim is dismissed.\nNo order as to costs.",
        ],
    )

    records = module.scan_pdf_corpus(input_dir=tmp_path, mode="raw-pdf-corpus", coverage_priors={})
    records_by_id = {record["doc_id"]: record for record in records}

    assert records_by_id["law_artifact"]["signals"]["same_case_multi_artifact_hint"] is True
    assert records_by_id["order_artifact"]["signals"]["same_case_multi_artifact_hint"] is True


def test_scan_private_doc_anomalies_link_smoke() -> None:
    module = _load_module()
    page_analysis = module.analyze_page_signals(
        text="CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4\nArticle 3 ... 6",
        page_num=1,
        page_count=2,
        metadata=_default_metadata(
            module,
            internal_link_count=3,
            link_destinations=["page:2", "page:4", "page:6"],
            external_urls=["https://legislation.difc.ae/law"],
        ),
        fallback_triggered=False,
    )
    doc_analysis = module.analyze_doc_signals(
        path=Path("dummy.pdf"),
        page_count=2,
        extraction_page_count=2,
        page_records=[{**page_analysis.page_record, "_raw_text": "CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4"}],
        page_metadata=[
            _default_metadata(
                module,
                internal_link_count=3,
                link_destinations=["page:2", "page:4", "page:6"],
                external_urls=["https://legislation.difc.ae/law"],
            ),
            _default_metadata(module, page_num=2),
        ],
        full_text="CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4\nArticle 3 ... 6",
    )

    assert page_analysis.page_record["signals"]["contents_internal_link_density"] > 0.0
    assert page_analysis.page_record["page_internal_link_count"] == 3
    assert page_analysis.page_record["link_destinations"] == ["page:2", "page:4", "page:6"]
    assert doc_analysis.signals["pdf_internal_link_graph_present"] is True
    assert doc_analysis.signals["contents_internal_link_density"] > 0.0
    assert doc_analysis.extras["internal_link_count"] == 3
    assert doc_analysis.extras["contents_link_count"] == 3
    assert doc_analysis.extras["toc_target_pages"] == [2, 4, 6]
    assert doc_analysis.extras["external_link_targets"] == ["https://legislation.difc.ae/law"]


def test_scan_private_doc_anomalies_coverage_priors_round_trip(tmp_path: Path) -> None:
    pdf_path = tmp_path / "clean_doc.pdf"
    _write_pdf(pdf_path, ["This is a clean first page."])
    coverage_priors = {"underqueried_family": ["mixed_script"], "weight": 2}
    coverage_path = tmp_path / "coverage_priors.json"
    coverage_path.write_text(json.dumps(coverage_priors), encoding="utf-8")
    output_dir = tmp_path / "out"

    _run_cli(input_dir=tmp_path, output_dir=output_dir, coverage_priors_json=coverage_path)

    rows = [
        json.loads(line)
        for line in (output_dir / "scan_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["coverage_priors"] == coverage_priors


def test_scan_private_doc_anomalies_exact_duplicate_clusters(tmp_path: Path) -> None:
    module = _load_module()
    duplicate_pages = [
        "LAW NO. 7 OF 2024\nThis Law may be cited as the Duplicate Law.",
        "Article 1\nThis is duplicate text.",
    ]
    _write_pdf(tmp_path / "dup_a.pdf", duplicate_pages)
    _write_pdf(tmp_path / "dup_b.pdf", duplicate_pages)

    records = module.scan_pdf_corpus(input_dir=tmp_path, mode="raw-pdf-corpus", coverage_priors={})
    clusters = {record["doc_id"]: record["exact_duplicate_cluster_id"] for record in records}

    assert clusters["dup_a"] is not None
    assert clusters["dup_a"] == clusters["dup_b"]
    assert len(records[0]["exact_duplicate_cluster_members"]) == 2
    summary = module.build_summary_markdown(records)
    assert "## Exact Duplicate Clusters" in summary
    assert "Weighted suspicion scoring is heuristic" in summary
    cluster_summaries = module.build_exact_duplicate_cluster_summaries(records)
    assert cluster_summaries[0]["member_count"] == 2
    assert cluster_summaries[0]["member_doc_ids"] == ["dup_a", "dup_b"]
    assert len(cluster_summaries[0]["member_sha256s"]) == 2
    assert cluster_summaries[0]["member_filenames"] == ["dup_a.pdf", "dup_b.pdf"]


def test_scan_private_doc_anomalies_doc_family_and_tracked_changes_tags(tmp_path: Path) -> None:
    module = _load_module()
    _write_pdf(
        tmp_path / "law_doc.pdf",
        [
            "LAW NO. 8 OF 2024\nThis Law may be cited as the Example Law.",
            "CONTENTS\nArticle 1 ... 2\nArticle 2 ... 4",
            "Article 1\nThis Law applies in the DIFC.",
        ],
    )
    _write_pdf(
        tmp_path / "judgment_doc.pdf",
        [
            "IN THE DIFC COURTS\nJudgment",
            "Claimant v Defendant\nBackground\nDiscussion\nThe court considers the claim.",
        ],
    )
    _write_pdf(
        tmp_path / "tracked_amendment.pdf",
        [
            "LAW NO. 9 OF 2024\nAmendment Law",
            "This law shall be amended as follows.\nThe following shall be substituted.\nThe words are deleted.",
        ],
    )

    records = module.scan_pdf_corpus(input_dir=tmp_path, mode="raw-pdf-corpus", coverage_priors={})
    by_id = {record["doc_id"]: record for record in records}

    assert "consolidated_law" in by_id["law_doc"]["doc_family_tags"]
    assert "judgment" in by_id["judgment_doc"]["doc_family_tags"]
    assert by_id["tracked_amendment"]["tracked_changes_detected"] is True
    assert by_id["tracked_amendment"]["tracked_changes_page_count"] >= 1
    assert by_id["tracked_amendment"]["tracked_changes_confidence"] == "medium"
    assert by_id["tracked_amendment"]["tracked_changes_visual_semantics"] is True
    assert by_id["tracked_amendment"]["translation_caveat"] is False
    assert "tracked_changes_amendment_law" in by_id["tracked_amendment"]["doc_family_tags"]
    assert "visual-diff semantics" in by_id["tracked_amendment"]["risk_note"]
    assert by_id["tracked_amendment"]["suspicion_score"] > 0
    assert "tracked_changes_visual_semantics" in by_id["tracked_amendment"]["reason_tags"]


def test_scan_private_doc_anomalies_output_writes_duplicate_cluster_summary_file(tmp_path: Path) -> None:
    duplicate_pages = [
        "LAW NO. 7 OF 2024\nThis Law may be cited as the Duplicate Law.",
        "Article 1\nThis is duplicate text.",
    ]
    _write_pdf(tmp_path / "dup_a.pdf", duplicate_pages)
    _write_pdf(tmp_path / "dup_b.pdf", duplicate_pages)
    output_dir = tmp_path / "out"

    _run_cli(input_dir=tmp_path, output_dir=output_dir)

    cluster_payload = json.loads((output_dir / "exact_duplicate_clusters.json").read_text(encoding="utf-8"))
    assert len(cluster_payload) == 1
    assert cluster_payload[0]["cluster_id"] == "cluster_001"
    assert cluster_payload[0]["member_count"] == 2
    assert cluster_payload[0]["member_doc_ids"] == ["dup_a", "dup_b"]
    assert len(cluster_payload[0]["member_sha256s"]) == 2
    assert cluster_payload[0]["member_filenames"] == ["dup_a.pdf", "dup_b.pdf"]
    assert "consolidated_law" in cluster_payload[0]["doc_family_tags"]


def test_scan_private_doc_anomalies_builds_cluster_collapsed_and_family_reports() -> None:
    module = _load_module()
    records = [
        {
            "doc_id": "dup_a",
            "filename": "dup_a.pdf",
            "suspicion_score": 40,
            "reason_tags": ["exact_duplicate_cluster", "underqueried_family_gap"],
            "doc_family_tags": ["consolidated_law"],
            "exact_duplicate_cluster_id": "cluster_001",
            "collision_doc_ids": [],
            "duplicate_same_family_doc_ids": [],
            "normalized_title": "law no 7 of 2024",
            "family_query_coverage_bucket": "zero-hit",
            "coverage_priors": {"family_buckets": {"consolidated_law": "zero-hit"}},
        },
        {
            "doc_id": "dup_b",
            "filename": "dup_b.pdf",
            "suspicion_score": 39,
            "reason_tags": ["exact_duplicate_cluster"],
            "doc_family_tags": ["consolidated_law"],
            "exact_duplicate_cluster_id": "cluster_001",
            "collision_doc_ids": [],
            "duplicate_same_family_doc_ids": [],
            "normalized_title": "law no 7 of 2024",
            "family_query_coverage_bucket": "zero-hit",
            "coverage_priors": {"family_buckets": {"consolidated_law": "zero-hit"}},
        },
    ]

    collapsed = module.build_cluster_collapsed_review(records)
    family_groups = module.build_family_stratified_review(records, focus_families=("consolidated_law",))

    assert len(collapsed) == 1
    assert collapsed[0]["member_count"] == 2
    assert collapsed[0]["member_doc_ids"] == ["dup_a", "dup_b"]
    assert family_groups[0]["family_tag"] == "consolidated_law"
    assert family_groups[0]["coverage_bucket"] == "zero-hit"
    assert family_groups[0]["entries"][0]["member_count"] == 2


def test_scan_private_doc_anomalies_toc_and_top20_output(tmp_path: Path) -> None:
    pdf_path = tmp_path / "toc_doc.pdf"
    _write_pdf(
        pdf_path,
        [
            "LAW NO. 5 OF 2024\nCONTENTS\nArticle 1 ... 2\nArticle 2 ... 3",
            "Preface page",
            "Article 1\nSubstantive text",
            "Article 2\nSubstantive text",
        ],
    )
    output_dir = tmp_path / "out"

    _run_cli(input_dir=tmp_path, output_dir=output_dir)

    rows = [
        json.loads(line)
        for line in (output_dir / "scan_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["toc_pointer_type"] in {"internal_like", "linked", "pdf_like", "uncertain"}
    assert (output_dir / "top20_report.md").exists()
    assert (output_dir / "top20_cluster_collapsed_report.md").exists()
    assert (output_dir / "top_by_family_report.md").exists()
    assert "Top 20 Cluster-Collapsed Suspicious Documents" in (
        output_dir / "top20_cluster_collapsed_report.md"
    ).read_text(encoding="utf-8")
    assert "Family-Stratified Suspicious Documents" in (output_dir / "top_by_family_report.md").read_text(
        encoding="utf-8"
    )
