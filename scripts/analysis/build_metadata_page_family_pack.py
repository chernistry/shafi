# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

import fitz

JsonDict = dict[str, object]

_LAW_NUMBER_RE = re.compile(r"\b(?:DIFC\s+LAW\s+NO\.?\s+\d+\s+OF\s+\d{4}|LAW\s+NO\.?\s+\d+\s+OF\s+\d{4})\b", re.IGNORECASE)
_CITATION_TITLE_RE = re.compile(r"\bThis\s+Law\s+may\s+be\s+cited\s+as\b", re.IGNORECASE)
_LEGISLATIVE_AUTHORITY_RE = re.compile(r"\b(?:Legislative Authority|made by the Ruler of Dubai|made by the)\b", re.IGNORECASE)
_ENACTMENT_RE = re.compile(r"\b(?:date of enactment|enacted on|Enactment Notice)\b", re.IGNORECASE)
_COMMENCEMENT_RE = re.compile(r"\b(?:Commencement|come(?:s)? into force|effective date)\b", re.IGNORECASE)
_ADMINISTRATION_RE = re.compile(r"\b(?:Administration of the Law|administered by)\b", re.IGNORECASE)
_NO_WAIVER_RE = re.compile(r"\bNo waiver\b", re.IGNORECASE)
_SCHEDULE_RE = re.compile(r"\bSCHEDULE\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"\bMaximum Fine\b", re.IGNORECASE)


def _extract_pdf_pages(path: Path, *, doc_key: str, doc_label: str) -> list[JsonDict]:
    doc = fitz.open(str(path))
    pages: list[JsonDict] = []
    for idx in range(doc.page_count):
        page_number = idx + 1
        text = " ".join(doc.load_page(idx).get_text("text").split())
        lowered = text.lower()
        pages.append(
            {
                "page_id": f"{doc_key}#{page_number}",
                "page_number": page_number,
                "doc_key": doc_key,
                "doc_label": doc_label,
                "text_preview": text[:400],
                "flags": {
                    "is_cover_page": page_number == 1,
                    "is_contents_page": "contents" in lowered,
                    "has_law_number": bool(_LAW_NUMBER_RE.search(text)),
                    "has_citation_title": bool(_CITATION_TITLE_RE.search(text)),
                    "has_legislative_authority": bool(_LEGISLATIVE_AUTHORITY_RE.search(text)),
                    "has_enactment": bool(_ENACTMENT_RE.search(text)),
                    "has_commencement": bool(_COMMENCEMENT_RE.search(text)),
                    "has_administration": bool(_ADMINISTRATION_RE.search(text)),
                    "has_no_waiver": bool(_NO_WAIVER_RE.search(text)),
                    "has_schedule": bool(_SCHEDULE_RE.search(text)),
                    "has_table": bool(_TABLE_RE.search(text)),
                },
            }
        )
    return pages


def _case_specs() -> list[JsonDict]:
    return [
        {
            "case_id": "law-cover-number",
            "doc_key": "law_cover",
            "question": "What is the official law number shown on the cover page?",
            "family": "official_law_number",
            "gold_pages": ["law_cover#1"],
        },
        {
            "case_id": "law-citation-title",
            "doc_key": "law_cover",
            "question": "What is the citation title of this Law?",
            "family": "citation_title",
            "gold_pages": ["law_cover#1", "law_cover#3"],
        },
        {
            "case_id": "law-who-made",
            "doc_key": "law_cover",
            "question": "Who made this Law?",
            "family": "who_made",
            "gold_pages": ["law_cover#1", "law_cover#3"],
        },
        {
            "case_id": "law-enactment",
            "doc_key": "law_cover",
            "question": "On what date was this Law enacted?",
            "family": "enactment",
            "gold_pages": ["law_cover#1", "law_cover#3"],
        },
        {
            "case_id": "law-commencement",
            "doc_key": "law_cover",
            "question": "What is the commencement rule for this Law?",
            "family": "commencement",
            "gold_pages": ["law_cover#1", "law_cover#3"],
        },
        {
            "case_id": "law-cover-title",
            "doc_key": "law_cover",
            "question": "What title appears on the cover page for this Law?",
            "family": "title_page",
            "gold_pages": ["law_cover#1"],
        },
        {
            "case_id": "employment-cover-number",
            "doc_key": "employment",
            "question": "What is the official law number shown on the cover page?",
            "family": "official_law_number",
            "gold_pages": ["employment#1"],
        },
        {
            "case_id": "employment-citation-title",
            "doc_key": "employment",
            "question": "What is the citation title of this Law?",
            "family": "citation_title",
            "gold_pages": ["employment#1", "employment#4"],
        },
        {
            "case_id": "employment-enactment",
            "doc_key": "employment",
            "question": "On what date was this Law enacted?",
            "family": "enactment",
            "gold_pages": ["employment#1", "employment#5"],
        },
        {
            "case_id": "employment-commencement",
            "doc_key": "employment",
            "question": "What is the commencement rule for this Law?",
            "family": "commencement",
            "gold_pages": ["employment#1", "employment#5"],
        },
        {
            "case_id": "employment-administered-by",
            "doc_key": "employment",
            "question": "Who administers this Law?",
            "family": "administration",
            "gold_pages": ["employment#1", "employment#6"],
        },
        {
            "case_id": "employment-no-waiver",
            "doc_key": "employment",
            "question": "Can an employee waive rights under this Law?",
            "family": "no_waiver",
            "gold_pages": ["employment#1", "employment#7"],
        },
    ]


def build_pack(*, law_cover_pdf: Path, employment_pdf: Path) -> JsonDict:
    docs = {
        "law_cover": _extract_pdf_pages(law_cover_pdf, doc_key="law_cover", doc_label="17_title_page_reference_doc.pdf"),
        "employment": _extract_pdf_pages(employment_pdf, doc_key="employment", doc_label="18_employment_law_reference_doc.pdf"),
    }
    cases: list[JsonDict] = []
    for spec in _case_specs():
        doc_key = str(spec["doc_key"])
        cases.append(
            {
                "case_id": spec["case_id"],
                "question": spec["question"],
                "family": spec["family"],
                "doc_key": doc_key,
                "doc_label": docs[doc_key][0]["doc_label"],
                "gold_pages": spec["gold_pages"],
                "pages": docs[doc_key],
            }
        )
    return {
        "summary": {
            "case_count": len(cases),
            "doc_count": len(docs),
            "family_counts": {
                family: sum(1 for case in cases if str(case["family"]) == family)
                for family in sorted({str(case["family"]) for case in cases})
            },
        },
        "cases": cases,
    }


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload["summary"])
    lines = [
        "# Metadata Page-Family Pack",
        "",
        f"- case_count: `{summary['case_count']}`",
        f"- doc_count: `{summary['doc_count']}`",
        "",
        "## Family Counts",
        "",
    ]
    for key, value in sorted(cast("dict[str, int]", summary["family_counts"]).items()):
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the synthetic metadata page-family pack.")
    parser.add_argument("--law-cover-pdf", type=Path, required=True)
    parser.add_argument("--employment-pdf", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_pack(law_cover_pdf=args.law_cover_pdf, employment_pdf=args.employment_pdf)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
