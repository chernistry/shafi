from __future__ import annotations

import json
from pathlib import Path

import scripts.private_doctor_preflight as doctor
from scripts.private_doctor_preflight import main


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_private_doctor_preflight_happy_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_check_required_runtime_modules", lambda: (True, "ok"))
    monkeypatch.setattr(doctor, "_check_docling_runtime", lambda: (True, "ok"))
    monkeypatch.setattr(doctor, "_check_sparse_runtime", lambda: (True, "indices=5"))
    monkeypatch.setattr(doctor, "_check_ocr_runtime", lambda: (True, "pages=2 section_lengths=100,100"))
    manifest_path = tmp_path / "run_manifest.json"
    fingerprint_path = tmp_path / "candidate_fingerprint.json"
    archive_path = tmp_path / "code_archive.zip"
    concurrency_path = tmp_path / "concurrency_drift_report.json"
    ocr_path = tmp_path / "ocr_audit.json"
    unsupported_path = Path(__file__).resolve().parents[1] / "fixtures" / "unsupported_synthetic_pack.json"
    scanner_path = tmp_path / "scan_results.jsonl"
    out_json = tmp_path / "doctor.json"
    out_md = tmp_path / "doctor.md"

    _write_json(
        manifest_path,
        {"run_manifest": {"fingerprint": "manifest-fp"}},
    )
    _write_json(
        fingerprint_path,
        {"candidate_fingerprint": {"label": "candidate-x", "fingerprint": "candidate-fp"}},
    )
    archive_path.write_bytes(b"zip")
    _write_json(
        concurrency_path,
        {
            "runtime_recommendation": "query_concurrency=1_stable_only",
            "answer_drift_count": 0,
            "page_drift_count": 0,
            "model_drift_count": 0,
        },
    )
    _write_json(
        ocr_path,
        {
            "fallback_risk_docs": 0,
            "blank_page_docs": 0,
            "noncontiguous_section_docs": 0,
            "chunk_page_mismatch_docs": 0,
        },
    )
    scanner_rows = [
        {
            "doc_id": "doc-a",
            "filename": "doc-a.pdf",
            "suspicion_score": 22,
            "reason_tags": ["tracked_changes_visual_semantics"],
            "tracked_changes_visual_semantics": True,
        },
        {
            "doc_id": "doc-b",
            "filename": "doc-b.pdf",
            "suspicion_score": 21,
            "reason_tags": ["same_family_duplicate"],
            "duplicate_same_family_doc_ids": ["doc-c"],
        },
        {
            "doc_id": "doc-c",
            "filename": "doc-c.pdf",
            "suspicion_score": 20,
            "reason_tags": [],
            "translation_caveat": True,
        },
    ]
    scanner_path.write_text(
        "\n".join(json.dumps(row) for row in scanner_rows) + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--manifest-json",
            str(manifest_path),
            "--candidate-fingerprint-json",
            str(fingerprint_path),
            "--code-archive",
            str(archive_path),
            "--concurrency-report-json",
            str(concurrency_path),
            "--ocr-audit-json",
            str(ocr_path),
            "--unsupported-pack-json",
            str(unsupported_path),
            "--scanner-jsonl",
            str(scanner_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["candidate_label"] == "candidate-x"
    assert payload["overall_ready"] is True
    assert payload["blocking_issues"] == []
    assert payload["checks"]["required_runtime_modules"] == {"ready": True, "detail": "ok"}
    assert payload["checks"]["docling_runtime"] == {"ready": True, "detail": "ok"}
    assert payload["checks"]["sparse_runtime"] == {"ready": True, "detail": "indices=5"}
    assert payload["checks"]["ocr_runtime"] == {"ready": True, "detail": "pages=2 section_lengths=100,100"}
    assert payload["checks"]["scanner_advisory"]["risk_level"] == "medium"
    assert payload["checks"]["scanner_advisory"]["risk_indicator"] == "yellow"
    assert payload["checks"]["scanner_advisory"]["clustered_families"] == [
        "duplicate_same_family",
        "tracked_changes_visual_semantics",
        "translation_caveat_arabic_prevails",
    ]
    warning_lines = payload["checks"]["scanner_advisory"]["warnings"]
    assert any("tracked_changes_visual_semantics" in line for line in warning_lines)
    assert any("duplicate_same_family" in line for line in warning_lines)
    assert any("translation_caveat_arabic_prevails" in line for line in warning_lines)
    markdown = out_md.read_text(encoding="utf-8")
    assert "overall_ready: `True`" in markdown
    assert "## Scanner Top Docs" in markdown
    assert "## Scanner Warnings" in markdown
    assert "[SCANNER] 1 docs cluster in tracked_changes_visual_semantics" in markdown


def test_private_doctor_preflight_reports_failures(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_check_required_runtime_modules", lambda: (False, "missing:fastembed"))
    monkeypatch.setattr(doctor, "_check_docling_runtime", lambda: (False, "missing"))
    monkeypatch.setattr(doctor, "_check_sparse_runtime", lambda: (False, "SparseEncoderError"))
    monkeypatch.setattr(doctor, "_check_ocr_runtime", lambda: (False, "page2_blank_after_ocr"))
    manifest_path = tmp_path / "run_manifest.json"
    fingerprint_path = tmp_path / "candidate_fingerprint.json"
    archive_path = tmp_path / "code_archive.zip"
    concurrency_path = tmp_path / "concurrency_drift_report.json"
    ocr_path = tmp_path / "ocr_audit.json"
    unsupported_path = Path(__file__).resolve().parents[1] / "fixtures" / "unsupported_synthetic_pack.json"
    scanner_path = tmp_path / "missing_scan_results.jsonl"
    out_json = tmp_path / "doctor.json"
    out_md = tmp_path / "doctor.md"

    _write_json(manifest_path, {"run_manifest": {}})
    _write_json(fingerprint_path, {"candidate_fingerprint": {"label": "candidate-y"}})
    archive_path.write_bytes(b"")
    _write_json(
        concurrency_path,
        {
            "runtime_recommendation": "query_concurrency=2",
            "answer_drift_count": 1,
            "page_drift_count": 0,
            "model_drift_count": 0,
        },
    )
    _write_json(
        ocr_path,
        {
            "fallback_risk_docs": 1,
            "blank_page_docs": 0,
            "noncontiguous_section_docs": 0,
            "chunk_page_mismatch_docs": 0,
        },
    )

    exit_code = main(
        [
            "--manifest-json",
            str(manifest_path),
            "--candidate-fingerprint-json",
            str(fingerprint_path),
            "--code-archive",
            str(archive_path),
            "--concurrency-report-json",
            str(concurrency_path),
            "--ocr-audit-json",
            str(ocr_path),
            "--unsupported-pack-json",
            str(unsupported_path),
            "--scanner-jsonl",
            str(scanner_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["overall_ready"] is False
    assert set(payload["blocking_issues"]) == {
        "manifest",
        "candidate_fingerprint",
        "code_archive",
        "stable_concurrency",
        "required_runtime_modules",
        "docling_runtime",
        "sparse_runtime",
        "ocr_runtime",
        "ocr_audit",
    }
    assert "manifest: ready=`False`" in out_md.read_text(encoding="utf-8")
    assert payload["checks"]["scanner_advisory"]["detail"] == "missing"


def test_check_scanner_results_counts_one_page_enactment_warning(tmp_path: Path) -> None:
    scanner_path = tmp_path / "scan_results.jsonl"
    scanner_path.write_text(
        json.dumps(
            {
                "doc_id": "doc-enactment",
                "filename": "doc-enactment.pdf",
                "suspicion_score": 25,
                "reason_tags": [],
                "one_page_enactment_notice": True,
                "doc_family_tags": ["enactment_notice_one_page"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = doctor._check_scanner_results(scanner_path)

    assert summary is not None
    assert summary["clustered_families"] == ["one_page_enactment_notice"]
    assert summary["warnings"] == [
        "[SCANNER] 1 docs cluster in one_page_enactment_notice - review before submission. "
        "These families are underexercised on the public question set."
    ]


def test_check_ocr_audit_accepts_list_shaped_fields(tmp_path: Path) -> None:
    audit_path = tmp_path / "ocr_audit.json"
    _write_json(
        audit_path,
        {
            "fallback_risk_docs": [],
            "blank_page_docs": [],
            "noncontiguous_section_docs": [],
            "chunk_page_mismatch_docs": [],
        },
    )

    assert doctor._check_ocr_audit(audit_path) == (True, "ok")
