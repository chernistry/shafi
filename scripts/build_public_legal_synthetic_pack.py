from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import cast

_PACK_CASES: list[dict[str, object]] = [
    {
        "case_id": "exact-operating-art16",
        "question": "According to Article 16(1) of the Operating Law, what must be filed with the Registrar?",
        "answer_type": "free_text",
        "family": "exact_reference",
        "subtype": "article_provision",
        "source_fixtures": ["tests/fixtures/legalbenchrag_mini_bootstrap/corpus/operating_law.txt"],
        "doc_refs": ["Operating Law"],
        "provision_refs": ["Article 16(1)"],
        "expected_behavior": "retrieve_exact_ref",
    },
    {
        "case_id": "exact-employment-art11b",
        "question": "Does Article 11(2)(b) of the Employment Law prohibit dismissing an employee without notice?",
        "answer_type": "boolean",
        "family": "exact_reference",
        "subtype": "article_provision",
        "source_fixtures": ["tests/fixtures/legalbenchrag_mini_bootstrap/corpus/employment_law.txt"],
        "doc_refs": ["Employment Law"],
        "provision_refs": ["Article 11(2)(b)"],
        "expected_behavior": "retrieve_exact_ref",
    },
    {
        "case_id": "limitation-section4",
        "question": "Under Section 4 of the Limitation Act 2020, how many years is the prescribed period for actions in tort?",
        "answer_type": "number",
        "family": "explicit_provision",
        "subtype": "section_lookup",
        "source_fixtures": ["tests/fixtures/docs/limitation_act.txt"],
        "doc_refs": ["Limitation Act 2020"],
        "provision_refs": ["Section 4"],
        "expected_behavior": "retrieve_exact_ref",
    },
    {
        "case_id": "limitation-short-title",
        "question": "According to Section 1 of the Limitation Act 2020, what is the short title of the Act?",
        "answer_type": "name",
        "family": "explicit_provision",
        "subtype": "title_lookup",
        "source_fixtures": ["tests/fixtures/docs/limitation_act.txt"],
        "doc_refs": ["Limitation Act 2020"],
        "provision_refs": ["Section 1"],
        "expected_behavior": "retrieve_exact_ref",
    },
    {
        "case_id": "notice-compare",
        "question": "Which has the longer notice period: the service agreement in Smith v. Jones or the sample contract?",
        "answer_type": "name",
        "family": "compare",
        "subtype": "same_fact_compare",
        "source_fixtures": [
            "tests/fixtures/docs/smith_v_jones.txt",
            "tests/fixtures/docs/sample_contract.txt",
        ],
        "doc_refs": ["Smith v. Jones [2021] UKSC 45", "Service Agreement"],
        "provision_refs": [],
        "expected_behavior": "multi_doc_compare",
    },
    {
        "case_id": "unsupported-free-text",
        "question": "Who administers the Foundations Law?",
        "answer_type": "free_text",
        "family": "unsupported",
        "subtype": "free_text_no_support",
        "source_fixtures": ["tests/fixtures/unsupported_synthetic_pack.json"],
        "doc_refs": ["Foundations Law 2018"],
        "provision_refs": [],
        "expected_behavior": "unsupported_free_text",
    },
    {
        "case_id": "unsupported-strict-null",
        "question": "Was Article 8(1) violated?",
        "answer_type": "boolean",
        "family": "unsupported",
        "subtype": "strict_null",
        "source_fixtures": ["tests/fixtures/unsupported_synthetic_pack.json"],
        "doc_refs": [],
        "provision_refs": ["Article 8(1)"],
        "expected_behavior": "unsupported_strict_null",
    },
    {
        "case_id": "ocr-risk-short-scan",
        "question": "According to page 2 of the scanned short multi-page PDF, what clause text is shown there?",
        "answer_type": "free_text",
        "family": "ocr_risk",
        "subtype": "page_specific_scan",
        "source_fixtures": ["tests/fixtures/ocr_fallback_synthetic_pack.json"],
        "doc_refs": ["scan-short-multipage.pdf"],
        "provision_refs": [],
        "expected_behavior": "parser_ocr_watchpoint",
    },
    {
        "case_id": "ocr-risk-text-rich-control",
        "question": "According to page 2 of the text-rich multi-page PDF, what additional extracted text appears on page two?",
        "answer_type": "free_text",
        "family": "ocr_risk",
        "subtype": "page_specific_control",
        "source_fixtures": ["tests/fixtures/ocr_fallback_synthetic_pack.json"],
        "doc_refs": ["text-rich-multipage.pdf"],
        "provision_refs": [],
        "expected_behavior": "parser_page_identity_control",
    },
]

_REQUIRED_FAMILIES = {"exact_reference", "explicit_provision", "unsupported", "compare", "ocr_risk"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public-only legal synthetic stress pack.")
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def build_pack() -> dict[str, object]:
    family_counts = Counter(str(case["family"]) for case in _PACK_CASES)
    source_fixture_counts = Counter(
        fixture
        for case in _PACK_CASES
        for fixture in cast("list[object]", case.get("source_fixtures", []))
        if isinstance(fixture, str) and fixture.strip()
    )
    missing = sorted(_REQUIRED_FAMILIES.difference(family_counts))
    if missing:
        raise ValueError(f"Missing required public-pack families: {', '.join(missing)}")

    case_ids = [str(case["case_id"]) for case in _PACK_CASES]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Public legal synthetic pack contains duplicate case IDs")

    return {
        "ticket": 80,
        "created_at": "2026-03-13",
        "required_families": sorted(_REQUIRED_FAMILIES),
        "summary": {
            "case_count": len(_PACK_CASES),
            "family_counts": dict(sorted(family_counts.items())),
            "source_fixture_counts": dict(sorted(source_fixture_counts.items())),
        },
        "cases": _PACK_CASES,
    }


def main() -> None:
    args = _parse_args()
    payload = build_pack()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
