# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from typing import Any, cast

from shafi.submission.common import SubmissionCase
from shafi.submission.generate import _project_submission_result
from shafi.submission.platform import PlatformCaseResult, _project_platform_answer, _result_anomaly_flags

JsonDict = dict[str, Any]
_REQUIRED_RUNTIME_MODULES = ("fitz", "docling", "qdrant_client", "fastembed", "rapidocr", "openai", "cohere")
_RISK_INDICATOR_BY_LEVEL = {"low": "green", "medium": "yellow", "high": "red"}


def _scanner_reason_tags(row: JsonDict) -> set[str]:
    return {str(tag).strip() for tag in cast("list[object]", row.get("reason_tags") or []) if str(tag).strip()}


def _scanner_doc_family_tags(row: JsonDict) -> set[str]:
    return {str(tag).strip() for tag in cast("list[object]", row.get("doc_family_tags") or []) if str(tag).strip()}


def _matches_scanner_warning_family(row: JsonDict, family: str) -> bool:
    reason_tags = _scanner_reason_tags(row)
    doc_family_tags = _scanner_doc_family_tags(row)
    if family == "tracked_changes_visual_semantics":
        return bool(row.get("tracked_changes_visual_semantics") or row.get("tracked_changes_detected")) or family in reason_tags
    if family == "one_page_enactment_notice":
        return bool(row.get("one_page_enactment_notice")) or "enactment_notice_one_page" in doc_family_tags
    if family == "duplicate_same_family":
        return bool(cast("list[object]", row.get("duplicate_same_family_doc_ids") or [])) or "same_family_duplicate" in reason_tags
    if family == "translation_caveat_arabic_prevails":
        return bool(row.get("translation_caveat") or row.get("translation_caveat_arabic_prevails")) or "translation_caveat_doc" in doc_family_tags
    return False


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", payload)


def _check_manifest(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    payload = _load_json(path)
    manifest = payload.get("run_manifest")
    if not isinstance(manifest, dict):
        return False, "missing_run_manifest"
    manifest_dict = cast("JsonDict", manifest)
    fingerprint = str(manifest_dict.get("fingerprint") or "").strip()
    return bool(fingerprint), "ok" if fingerprint else "missing_fingerprint"


def _check_candidate_fingerprint(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    payload = _load_json(path)
    candidate = payload.get("candidate_fingerprint")
    if not isinstance(candidate, dict):
        return False, "missing_candidate_fingerprint"
    candidate_dict = cast("JsonDict", candidate)
    fingerprint = str(candidate_dict.get("fingerprint") or "").strip()
    return bool(fingerprint), "ok" if fingerprint else "missing_fingerprint"


def _check_archive(path: Path) -> tuple[bool, str, int]:
    if not path.exists():
        return False, "missing", 0
    size_bytes = path.stat().st_size if path.is_file() else 0
    if size_bytes <= 0:
        return False, "empty", size_bytes
    return True, "ok", size_bytes


def _check_concurrency(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    payload = _load_json(path)
    recommendation = str(payload.get("runtime_recommendation") or "").strip()
    answer_drift_count = int(payload.get("answer_drift_count") or 0)
    page_drift_count = int(payload.get("page_drift_count") or 0)
    model_drift_count = int(payload.get("model_drift_count") or 0)
    if (
        recommendation == "query_concurrency=1_stable_only"
        and answer_drift_count == 0
        and page_drift_count == 0
        and model_drift_count == 0
    ):
        return True, "ok"
    return False, "not_stable_only"


def _check_ocr_audit(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    payload = _load_json(path)

    def _count(value: object) -> int:
        if isinstance(value, list):
            return len(cast("list[object]", value))
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip() or "0")
            except ValueError:
                return 1
        try:
            return int(cast("Any", value))
        except (TypeError, ValueError):
            return 1

    checks = (
        _count(payload.get("fallback_risk_docs")) == 0,
        _count(payload.get("blank_page_docs")) == 0,
        _count(payload.get("noncontiguous_section_docs")) == 0,
        _count(payload.get("chunk_page_mismatch_docs")) == 0,
    )
    return (True, "ok") if all(checks) else (False, "parser_or_page_boundary_risk")


def _check_docling_runtime() -> tuple[bool, str]:
    spec = importlib.util.find_spec("docling")
    if spec is None:
        return False, "missing"
    origin = str(spec.origin or "").strip()
    return True, origin or "ok"


def _check_required_runtime_modules() -> tuple[bool, str]:
    missing = [module_name for module_name in _REQUIRED_RUNTIME_MODULES if importlib.util.find_spec(module_name) is None]
    if missing:
        return False, f"missing:{','.join(missing)}"
    return True, "ok"


def _check_sparse_runtime() -> tuple[bool, str]:
    try:
        from shafi.core.sparse_bm25 import BM25SparseEncoder

        encoder = BM25SparseEncoder(model_name="Qdrant/bm25")
        vector = encoder.encode_query("employment law article 10 penalty notice")
        index_count = len(vector.indices)
        if index_count <= 0 or index_count != len(vector.values):
            return False, f"invalid_sparse_vector:{index_count}/{len(vector.values)}"
        return True, f"indices={index_count}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _normalize_runtime_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def _check_ocr_runtime() -> tuple[bool, str]:
    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont

        from shafi.ingestion.parser import DocumentParser

        page_texts = [
            (
                "LAW ON THE APPLICATION OF CIVIL AND COMMERCIAL LAWS IN THE DIFC\n"
                "DIFC LAW NO 3 OF 2004\n"
                "CONSOLIDATED VERSION NOVEMBER 2024\n"
                "AMENDMENT LAW DIFC LAW NO 8 OF 2024"
            ),
            (
                "EMPLOYMENT LAW\n"
                "DIFC LAW NO 2 OF 2019\n"
                "CONSOLIDATED VERSION NO 5 JULY 2025\n"
                "EMPLOYMENT LAW AMENDMENT LAW DIFC LAW NO 4 OF 2021"
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="ocr_runtime_smoke_") as tmp_dir:
            root = Path(tmp_dir)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 64)
            except Exception:
                font = ImageFont.load_default()

            image_paths: list[Path] = []
            for index, page_text in enumerate(page_texts, start=1):
                image = Image.new("L", (1800, 1400), 255)
                drawer = ImageDraw.Draw(image)
                drawer.multiline_text((120, 160), page_text, fill=0, font=font, spacing=42)
                image_path = root / f"page_{index}.jpg"
                image.save(image_path, format="JPEG", quality=32, optimize=True)
                image_paths.append(image_path)

            pdf_path = root / "ocr_runtime_smoke.pdf"
            fitz_any = cast("Any", fitz)
            pdf_doc = fitz_any.open()
            for image_path in image_paths:
                with Image.open(image_path) as image_obj:
                    width, height = image_obj.size
                page = pdf_doc.new_page(width=width, height=height)
                page.insert_image(fitz_any.Rect(0, 0, width, height), filename=str(image_path))
            pdf_doc.save(pdf_path)
            pdf_doc.close()

            parser = DocumentParser(pdf_text_min_chars=200, pdf_text_min_words=20)
            parsed = parser.parse_file(pdf_path)
            if len(parsed.sections) < 2:
                return False, f"section_count={len(parsed.sections)}"

            normalized_sections = [_normalize_runtime_text(section.text) for section in parsed.sections]
            if not normalized_sections[0] or "2004" not in normalized_sections[0]:
                return False, "page1_missing_2004_anchor"
            if len(normalized_sections) < 2 or not normalized_sections[1]:
                return False, "page2_blank_after_ocr"
            if "2019" not in normalized_sections[1] or "EMPLOYMENT" not in normalized_sections[1]:
                return False, "page2_missing_employment_2019_anchor"
            if len(parsed.full_text.strip()) < 120:
                return False, f"full_text_too_short:{len(parsed.full_text.strip())}"
            section_lengths = ",".join(str(len(section.text.strip())) for section in parsed.sections[:2])
            return True, f"pages={len(parsed.sections)} section_lengths={section_lengths}"
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _check_unsupported_pack(path: Path) -> tuple[bool, str, int]:
    if not path.exists():
        return False, "missing", 0
    payload = _load_json(path)
    cases_obj = payload.get("cases")
    if not isinstance(cases_obj, list):
        return False, "missing_cases", 0
    raw_cases = cast("list[object]", cases_obj)
    failures = 0
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            failures += 1
            continue
        case_payload = cast("JsonDict", raw_case)
        case = SubmissionCase(
            case_id=str(case_payload["case_id"]),
            question=str(case_payload["question"]),
            answer_type=str(case_payload["answer_type"]),
        )
        answer_text = str(case_payload["answer_text"])
        telemetry = cast("dict[str, object]", case_payload.get("telemetry") or {})

        submission_result = _project_submission_result(
            case_id=case.case_id,
            answer_type=case.answer_type,
            answer_text=answer_text,
            telemetry=telemetry,
        )
        platform_result = PlatformCaseResult(
            case=case,
            answer_text=answer_text,
            telemetry=telemetry,
            total_ms=0,
        )
        projected = _project_platform_answer(platform_result)
        projected_telemetry = cast("dict[str, object]", projected["telemetry"])
        projected_retrieval = cast("dict[str, object]", projected_telemetry["retrieval"])

        if submission_result["answer"] != case_payload["expected_submission_answer"]:
            failures += 1
            continue
        if submission_result["retrieved_chunk_ids"] != case_payload["expected_submission_pages"]:
            failures += 1
            continue
        if projected["answer"] != case_payload["expected_platform_answer"]:
            failures += 1
            continue
        if projected_retrieval["retrieved_chunk_pages"] != case_payload["expected_platform_pages"]:
            failures += 1
            continue
        if _result_anomaly_flags(platform_result) != case_payload["expected_anomaly_flags"]:
            failures += 1
            continue

    return (failures == 0, "ok" if failures == 0 else "contract_failures", failures)


def _check_scanner_results(path: Path | None) -> JsonDict | None:
    if path is None:
        return None
    if not path.exists():
        return {"ready": True, "detail": "missing", "advisory_only": True}
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(cast("JsonDict", payload))
    high_risk = [row for row in rows if int(row.get("suspicion_score") or 0) >= 20]
    top_docs = [
        {
            "doc_id": row.get("doc_id"),
            "filename": row.get("filename"),
            "score": row.get("suspicion_score"),
            "reason_tags": cast("list[str]", row.get("reason_tags") or [])[:3],
        }
        for row in sorted(rows, key=lambda item: (-int(item.get("suspicion_score") or 0), str(item.get("filename") or "")))[:3]
    ]
    warning_families = (
        "tracked_changes_visual_semantics",
        "one_page_enactment_notice",
        "duplicate_same_family",
        "translation_caveat_arabic_prevails",
    )
    clustered_family_counts = {
        family: sum(1 for row in high_risk if _matches_scanner_warning_family(row, family)) for family in warning_families
    }
    clustered_families = sorted(family for family, count in clustered_family_counts.items() if count > 0)
    warnings = [
        (
            f"[SCANNER] {count} docs cluster in {family} - review before submission. "
            "These families are underexercised on the public question set."
        )
        for family, count in clustered_family_counts.items()
        if count > 0
    ]
    risk_level = "low"
    if len(high_risk) >= 5:
        risk_level = "high"
    elif high_risk:
        risk_level = "medium"
    risk_indicator = _RISK_INDICATOR_BY_LEVEL[risk_level]
    detail = f"docs={len(rows)} high_risk={len(high_risk)} risk={risk_level} indicator={risk_indicator}"
    if clustered_families:
        detail += f" clustered={','.join(sorted(clustered_families))}"
    return {
        "ready": True,
        "detail": detail,
        "advisory_only": True,
        "docs_scanned": len(rows),
        "high_risk_doc_count": len(high_risk),
        "risk_level": risk_level,
        "risk_indicator": risk_indicator,
        "top_docs": top_docs,
        "clustered_families": sorted(clustered_families),
        "warnings": warnings,
    }


def _render_markdown(summary: JsonDict) -> str:
    checks = cast("JsonDict", summary["checks"])
    lines = [
        "# Private Doctor / Preflight",
        "",
        f"- candidate_label: `{summary['candidate_label']}`",
        f"- overall_ready: `{summary['overall_ready']}`",
        "",
        "## Checks",
        "",
    ]
    for key in (
        "manifest",
        "candidate_fingerprint",
        "code_archive",
        "stable_concurrency",
        "required_runtime_modules",
        "docling_runtime",
        "sparse_runtime",
        "ocr_runtime",
        "ocr_audit",
        "unsupported_pack",
        "scanner_advisory",
    ):
        if key not in checks:
            continue
        row = cast("JsonDict", checks[key])
        lines.append(f"- {key}: ready=`{row['ready']}` detail=`{row['detail']}`")
        if key == "scanner_advisory" and isinstance(row.get("top_docs"), list):
            top_docs = cast("list[JsonDict]", row.get("top_docs") or [])
            if top_docs:
                lines.append("")
                lines.append("## Scanner Top Docs")
                lines.append("")
                for top_doc in top_docs:
                    reason_tags = cast("list[str]", top_doc.get("reason_tags") or [])
                    lines.append(
                        "- "
                        f"{top_doc.get('filename')} "
                        f"(score={top_doc.get('score')}, reasons={','.join(reason_tags) or 'none'})"
                    )
        if key == "scanner_advisory" and isinstance(row.get("warnings"), list):
            warnings = cast("list[str]", row.get("warnings") or [])
            if warnings:
                lines.append("")
                lines.append("## Scanner Warnings")
                lines.append("")
                for warning in warnings:
                    lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Blocking Issues")
    lines.append("")
    blocking = cast("list[str]", summary["blocking_issues"])
    if blocking:
        for issue in blocking:
            lines.append(f"- {issue}")
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded private-day doctor/preflight bundle.")
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--candidate-fingerprint-json", type=Path, required=True)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--concurrency-report-json", type=Path, required=True)
    parser.add_argument("--ocr-audit-json", type=Path, required=True)
    parser.add_argument("--unsupported-pack-json", type=Path, required=True)
    parser.add_argument("--scanner-jsonl", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    manifest_ready, manifest_detail = _check_manifest(args.manifest_json)
    fingerprint_ready, fingerprint_detail = _check_candidate_fingerprint(args.candidate_fingerprint_json)
    archive_ready, archive_detail, archive_size = _check_archive(args.code_archive)
    concurrency_ready, concurrency_detail = _check_concurrency(args.concurrency_report_json)
    runtime_modules_ready, runtime_modules_detail = _check_required_runtime_modules()
    docling_ready, docling_detail = _check_docling_runtime()
    sparse_ready, sparse_detail = _check_sparse_runtime()
    ocr_runtime_ready, ocr_runtime_detail = _check_ocr_runtime()
    ocr_ready, ocr_detail = _check_ocr_audit(args.ocr_audit_json)
    unsupported_ready, unsupported_detail, unsupported_failures = _check_unsupported_pack(args.unsupported_pack_json)
    scanner_summary = _check_scanner_results(args.scanner_jsonl)

    fingerprint_payload = _load_json(args.candidate_fingerprint_json) if args.candidate_fingerprint_json.exists() else {}
    candidate_payload_obj: object = fingerprint_payload.get("candidate_fingerprint") or {}
    candidate_payload = cast("JsonDict", candidate_payload_obj if isinstance(candidate_payload_obj, dict) else {})
    candidate_label = str(candidate_payload.get("label") or "").strip() or "unknown"

    checks: JsonDict = {
        "manifest": {"ready": manifest_ready, "detail": manifest_detail},
        "candidate_fingerprint": {"ready": fingerprint_ready, "detail": fingerprint_detail},
        "code_archive": {"ready": archive_ready, "detail": archive_detail, "size_bytes": archive_size},
        "stable_concurrency": {"ready": concurrency_ready, "detail": concurrency_detail},
        "required_runtime_modules": {"ready": runtime_modules_ready, "detail": runtime_modules_detail},
        "docling_runtime": {"ready": docling_ready, "detail": docling_detail},
        "sparse_runtime": {"ready": sparse_ready, "detail": sparse_detail},
        "ocr_runtime": {"ready": ocr_runtime_ready, "detail": ocr_runtime_detail},
        "ocr_audit": {"ready": ocr_ready, "detail": ocr_detail},
        "unsupported_pack": {
            "ready": unsupported_ready,
            "detail": unsupported_detail,
            "failure_count": unsupported_failures,
        },
    }
    if scanner_summary is not None:
        checks["scanner_advisory"] = scanner_summary
    blocking_issues = [
        key
        for key, row in checks.items()
        if isinstance(row, dict)
        and not bool(cast("JsonDict", row).get("ready"))
        and not bool(cast("JsonDict", row).get("advisory_only"))
    ]
    summary: JsonDict = {
        "candidate_label": candidate_label,
        "overall_ready": not blocking_issues,
        "checks": checks,
        "blocking_issues": blocking_issues,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
