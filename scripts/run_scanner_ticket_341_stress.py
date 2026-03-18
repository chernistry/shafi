from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import fitz

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from scripts.scan_private_doc_anomalies import build_summary_markdown, build_top20_report_markdown, scan_pdf_corpus
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.scan_private_doc_anomalies import build_summary_markdown, build_top20_report_markdown, scan_pdf_corpus

try:
    from rag_challenge.ingestion.parser import DocumentParser
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rag_challenge.ingestion.parser import DocumentParser

JsonDict = dict[str, Any]

DEFAULT_SOURCE_FILENAMES = {
    "law": "01ab862cf9ef7012d76d06ebdafa1023ba91138961790824bbcdfd45d90adb86.pdf",
    "judgment": "1a255edc261961ec64870466a27ac4e25b5ebc2abe298e1b69f8dd2fc27288f6.pdf",
    "enactment": "0ee112f26f9b33c272a4fad301c52e7309c98236d627ad2451202ca6f5f0cdc7.pdf",
}
UNICODE_FONT_FILE = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
_ASCII_TO_EASTERN_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_SMART_PUNCT_TRANSLATION = str.maketrans(
    {
        '"': "\u201c",
        "'": "\u2019",
        "-": "\u2014",
    }
)


@dataclass(frozen=True, slots=True)
class StressSources:
    law: Path
    judgment: Path
    enactment: Path


@dataclass(frozen=True, slots=True)
class StressVariant:
    doc_id: str
    source_kind: str
    description: str
    build_pages: Callable[[dict[str, list[str]]], list[str]]


def _compact_page_text(text: str, *, limit: int = 2400) -> str:
    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()[:limit].strip()


def _load_source_pages(path: Path, *, max_pages: int = 3) -> list[str]:
    parser = DocumentParser()
    extraction = parser.extract_pdf_pages_for_scan(path)
    pages = [_compact_page_text(page) for page in extraction.pages if page.strip()]
    if not pages:
        raise ValueError(f"No readable pages extracted from {path}")
    return pages[:max_pages]


def _write_pdf(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open()
    text_kwargs: JsonDict = {"fontsize": 11}
    if UNICODE_FONT_FILE.exists():
        text_kwargs["fontfile"] = str(UNICODE_FONT_FILE)
        text_kwargs["fontname"] = "arial-unicode"
    for text in pages:
        page = pdf.new_page()
        page.insert_textbox(fitz.Rect(48, 48, 565, 790), text, **text_kwargs)
    pdf.save(path)
    pdf.close()


def _load_sources(
    *,
    docs_dir: Path,
    law_pdf: Path | None,
    judgment_pdf: Path | None,
    enactment_pdf: Path | None,
) -> StressSources:
    def _resolve(override: Path | None, key: str) -> Path:
        if override is not None:
            return override
        return docs_dir / DEFAULT_SOURCE_FILENAMES[key]

    sources = StressSources(
        law=_resolve(law_pdf, "law"),
        judgment=_resolve(judgment_pdf, "judgment"),
        enactment=_resolve(enactment_pdf, "enactment"),
    )
    missing = [str(path) for path in (sources.law, sources.judgment, sources.enactment) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing stress source PDFs: {', '.join(missing)}")
    return sources


def _default_variants() -> list[StressVariant]:
    def build_unicode_weirdness(source_pages: dict[str, list[str]]) -> list[str]:
        page = source_pages["law"][0]
        mutated = page.replace("LAW NO.", "LAW\u200b NO.\u00a0\ufffd", 1) if "LAW NO." in page else (
            "LAW\u200b NO.\u00a0\ufffd 11 OF 2004\n" + page
        )
        return [mutated, *source_pages["law"][1:2]]

    def build_arabic_header(source_pages: dict[str, list[str]]) -> list[str]:
        pages = list(source_pages["judgment"][:2])
        header = "\u062e\u062a\u0645 \u0627\u0644\u0645\u062d\u0643\u0645\u0629\n\u0628\u064a\u0627\u0646 \u0639\u0631\u0628\u064a \u0645\u062e\u062a\u0635\u0631\n"
        pages[0] = header + pages[0]
        if len(pages) > 1:
            pages[1] = "\u062f\u0628\u064a\n" + pages[1]
        return pages

    def build_eastern_arabic_digits(source_pages: dict[str, list[str]]) -> list[str]:
        pages = source_pages["law"][:2]
        return [page.translate(_ASCII_TO_EASTERN_ARABIC_DIGITS) for page in pages]

    def build_smart_punctuation(source_pages: dict[str, list[str]]) -> list[str]:
        page = source_pages["law"][0]
        appended = page + '\nThis Law may be cited as the "Stress-Test Law" - Interim Version - 2024.'
        return [appended.translate(_SMART_PUNCT_TRANSLATION), *source_pages["law"][1:2]]

    def build_low_text_page(source_pages: dict[str, list[str]]) -> list[str]:
        first_page = source_pages["judgment"][0]
        return [first_page, "x"]

    def build_table_heavy(source_pages: dict[str, list[str]]) -> list[str]:
        table_page = "\n".join(
            [
                "Field | Value | Note",
                "Registrar | DIFC Registrar | active",
                "Law Number | 11/2004 | source",
                "Penalty | USD 1000 | schedule",
                "Reference | Article 4 | title",
                "Schedule | Schedule 1 | interpretation",
            ]
        )
        return [source_pages["law"][0], table_page]

    def build_tracked_changes(source_pages: dict[str, list[str]]) -> list[str]:
        first_page = source_pages["enactment"][0]
        amendment_notice = (
            "\nDIFC Law Amendment Law\n"
            "Amendment note: the words ~~old text~~ are deleted and replaced by __new text__.\n"
            "The underlined words are inserted and the struck through words are deleted.\n"
            "Article 4 shall be amended as follows."
        )
        return [first_page + amendment_notice]

    return [
        StressVariant(
            doc_id="unicode_weirdness",
            source_kind="law",
            description="Inject zero-width, NBSP, and replacement characters into title/law-number lines.",
            build_pages=build_unicode_weirdness,
        ),
        StressVariant(
            doc_id="arabic_headers_stamps",
            source_kind="judgment",
            description="Prepend short Arabic header and stamp-like fragments to early pages.",
            build_pages=build_arabic_header,
        ),
        StressVariant(
            doc_id="eastern_arabic_digits",
            source_kind="law",
            description="Translate ASCII digits into Eastern Arabic digits in a non-production copy.",
            build_pages=build_eastern_arabic_digits,
        ),
        StressVariant(
            doc_id="smart_quotes_weird_dashes",
            source_kind="law",
            description="Replace straight quotes and hyphen-minus with curly quotes and em dashes.",
            build_pages=build_smart_punctuation,
        ),
        StressVariant(
            doc_id="low_text_page",
            source_kind="judgment",
            description="Strip one page down to minimal text.",
            build_pages=build_low_text_page,
        ),
        StressVariant(
            doc_id="table_heavy_page",
            source_kind="law",
            description="Inject a pipe-delimited grid to trigger table-heavy structure detection.",
            build_pages=build_table_heavy,
        ),
        StressVariant(
            doc_id="tracked_changes_notice",
            source_kind="enactment",
            description="Inject amendment-style inline notice with deleted/replaced wording.",
            build_pages=build_tracked_changes,
        ),
    ]


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _max_page_signal(record: JsonDict, signal_name: str) -> int | float:
    page_records = cast("list[JsonDict]", record.get("per_page") or [])
    values: list[int | float] = []
    for page_record in page_records:
        signals = cast("JsonDict", page_record.get("signals") or {})
        value = signals.get(signal_name)
        if isinstance(value, bool):
            values.append(int(value))
        elif isinstance(value, int | float):
            values.append(value)
    if not values:
        return 0
    return max(values)


def _any_page_signal(record: JsonDict, signal_name: str) -> bool:
    return bool(_max_page_signal(record, signal_name))


def _evaluate_variant(record: JsonDict) -> JsonDict:
    doc_id = str(record["doc_id"])
    signals = cast("JsonDict", record.get("signals") or {})
    reason_tags = set(cast("list[str]", record.get("reason_tags") or []))

    if doc_id == "unicode_weirdness":
        observed = {
            "zero_width_count": int(signals.get("zero_width_count") or 0),
            "nbsp_count": int(signals.get("nbsp_count") or 0),
            "replacement_char_count": int(signals.get("replacement_char_count") or 0),
            "non_ascii_ratio": float(signals.get("non_ascii_ratio") or 0.0),
        }
        passed = (
            (observed["zero_width_count"] > 0 or observed["replacement_char_count"] > 0)
            and observed["nbsp_count"] > 0
            and observed["non_ascii_ratio"] > 0.0
        )
        return {
            "expected_signals": ["zero_width_count_or_replacement_char_count", "nbsp_count", "non_ascii_ratio"],
            "observed": observed,
            "passed": passed,
        }

    if doc_id == "arabic_headers_stamps":
        observed = {
            "arabic_on_page_1": bool(signals.get("arabic_on_page_1")),
            "arabic_categories_present": cast("list[str]", signals.get("arabic_categories_present") or []),
            "likely_stamp_or_seal_overlay": bool(signals.get("likely_stamp_or_seal_overlay")),
        }
        passed = observed["arabic_on_page_1"] and (
            bool(observed["arabic_categories_present"]) or observed["likely_stamp_or_seal_overlay"]
        )
        return {
            "expected_signals": ["arabic_on_page_1", "arabic_categories_present_or_stamp_overlay"],
            "observed": observed,
            "passed": passed,
        }

    if doc_id == "eastern_arabic_digits":
        observed = {
            "eastern_arabic_digit_count": int(signals.get("eastern_arabic_digit_count") or 0),
            "arabic_indic_digit_count": int(signals.get("arabic_indic_digit_count") or 0),
        }
        passed = observed["eastern_arabic_digit_count"] > 0 and observed["arabic_indic_digit_count"] > 0
        return {"expected_signals": ["eastern_arabic_digit_count", "arabic_indic_digit_count"], "observed": observed, "passed": passed}

    if doc_id == "smart_quotes_weird_dashes":
        observed = {
            "smart_quote_count": int(signals.get("smart_quote_count") or 0),
            "dash_variant_count": int(signals.get("dash_variant_count") or 0),
        }
        observed["parser_normalized_ascii_punctuation"] = (
            observed["smart_quote_count"] == 0 and observed["dash_variant_count"] == 0
        )
        passed = (
            (
                observed["smart_quote_count"] > 0
                and observed["dash_variant_count"] > 0
            )
            or bool(observed["parser_normalized_ascii_punctuation"])
        )
        return {
            "expected_signals": ["smart_quote_count_and_dash_variant_count_or_parser_normalized_ascii_punctuation"],
            "observed": observed,
            "passed": passed,
        }

    if doc_id == "low_text_page":
        observed = {
            "low_text_page": _any_page_signal(record, "low_text_page"),
            "ocr_fallback_likelihood": float(_max_page_signal(record, "ocr_fallback_likelihood")),
        }
        passed = observed["low_text_page"] is True
        return {"expected_signals": ["low_text_page"], "observed": observed, "passed": passed}

    if doc_id == "table_heavy_page":
        observed = {
            "table_heavy_signature": _any_page_signal(record, "table_heavy_signature"),
            "table_or_numeric_density_reason": "table_or_numeric_density" in reason_tags,
        }
        passed = observed["table_heavy_signature"] is True
        return {"expected_signals": ["table_heavy_signature"], "observed": observed, "passed": passed}

    if doc_id == "tracked_changes_notice":
        observed = {
            "tracked_changes_detected": bool(record.get("tracked_changes_detected")),
            "tracked_changes_page_count": int(record.get("tracked_changes_page_count") or 0),
            "tracked_changes_visual_semantics_reason": "tracked_changes_visual_semantics" in reason_tags,
        }
        passed = (
            observed["tracked_changes_detected"] is True
            and observed["tracked_changes_page_count"] > 0
            and observed["tracked_changes_visual_semantics_reason"] is True
        )
        return {"expected_signals": ["tracked_changes_detected", "tracked_changes_page_count"], "observed": observed, "passed": passed}

    raise ValueError(f"Unexpected stress doc id: {doc_id}")


def _render_report_markdown(report: JsonDict) -> str:
    lines = [
        "# Ticket 341 Scanner Stress Validation",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Variants checked: `{report['variant_count']}`",
        "",
        "## Source PDFs",
        "",
    ]
    source_paths = cast("JsonDict", report["source_pdfs"])
    for key in ("law", "judgment", "enactment"):
        lines.append(f"- `{key}`: `{source_paths[key]}`")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| variant | passed | expected_signals | observed |",
            "| --- | --- | --- | --- |",
        ]
    )
    for variant in cast("list[JsonDict]", report["variants"]):
        observed = json.dumps(variant["observed"], ensure_ascii=False, sort_keys=True)
        expected = ", ".join(cast("list[str]", variant["expected_signals"]))
        lines.append(f"| {variant['doc_id']} | {variant['passed']} | {expected} | `{observed}` |")
    return "\n".join(lines) + "\n"


def run_ticket_341_stress_validation(
    *,
    docs_dir: Path,
    output_dir: Path,
    law_pdf: Path | None = None,
    judgment_pdf: Path | None = None,
    enactment_pdf: Path | None = None,
) -> JsonDict:
    sources = _load_sources(
        docs_dir=docs_dir,
        law_pdf=law_pdf,
        judgment_pdf=judgment_pdf,
        enactment_pdf=enactment_pdf,
    )
    source_pages = {
        "law": _load_source_pages(sources.law),
        "judgment": _load_source_pages(sources.judgment),
        "enactment": _load_source_pages(sources.enactment, max_pages=1),
    }

    fixtures_dir = output_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    variants = _default_variants()
    for variant in variants:
        pages = variant.build_pages(source_pages)
        _write_pdf(fixtures_dir / f"{variant.doc_id}.pdf", pages)

    records = scan_pdf_corpus(input_dir=fixtures_dir, mode="raw-pdf-corpus", coverage_priors={})
    _write_jsonl(output_dir / "scan_results.jsonl", records)
    (output_dir / "summary.md").write_text(build_summary_markdown(records), encoding="utf-8")
    (output_dir / "top20_report.md").write_text(build_top20_report_markdown(records), encoding="utf-8")

    records_by_id = {str(record["doc_id"]): record for record in records}
    results: list[JsonDict] = []
    for variant in variants:
        record = records_by_id.get(variant.doc_id)
        if record is None:
            results.append(
                {
                    "doc_id": variant.doc_id,
                    "description": variant.description,
                    "expected_signals": ["record_present"],
                    "observed": {"record_present": False},
                    "passed": False,
                }
            )
            continue
        evaluation = _evaluate_variant(record)
        results.append(
            {
                "doc_id": variant.doc_id,
                "description": variant.description,
                **evaluation,
            }
        )

    report: JsonDict = {
        "overall_pass": all(bool(result["passed"]) for result in results),
        "variant_count": len(results),
        "source_pdfs": {
            "law": str(sources.law),
            "judgment": str(sources.judgment),
            "enactment": str(sources.enactment),
        },
        "variants": results,
    }
    (output_dir / "stress_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "stress_validation_report.md").write_text(_render_report_markdown(report), encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Ticket 341 scanner synthetic perturbation stress validation.")
    parser.add_argument("--docs-dir", type=Path, default=Path("dataset/dataset_documents"))
    parser.add_argument("--out-dir", type=Path, default=Path("platform_runs/scanner_stress_validation/ticket_341"))
    parser.add_argument("--law-pdf", type=Path)
    parser.add_argument("--judgment-pdf", type=Path)
    parser.add_argument("--enactment-pdf", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = run_ticket_341_stress_validation(
        docs_dir=args.docs_dir,
        output_dir=args.out_dir,
        law_pdf=args.law_pdf,
        judgment_pdf=args.judgment_pdf,
        enactment_pdf=args.enactment_pdf,
    )
    return 0 if bool(report["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
