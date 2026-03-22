from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]

_EXPLICIT_ANCHOR_RE = re.compile(
    r"\b(page 2|second page|page 1|first page|title page|title pages|cover page|title/cover page)\b",
    re.IGNORECASE,
)

MANUAL_CASE_OVERRIDES: dict[str, JsonDict] = {
    "bd8d0befc731315ee2a477221feb950b44e68d9596823a90c47f78fc04870870": {
        "question_id": "bd8d0befc731315ee2a477221feb950b44e68d9596823a90c47f78fc04870870",
        "trust_tier": "trusted",
        "gold_origin": "manual_override",
        "audit_note": "Manual high-risk boolean title-year benchmark case.",
        "gold_page_ids": [
            "9f3ba7bfbe6197f1142e2cbfc0ed440f95a65794b5b95937bb807ef8c2ddb9b1_4",
            "7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869_4",
        ],
        "gold_items": [],
        "items": [
            {
                "id": "item_1",
                "text": "",
                "gold_page_ids": [
                    "9f3ba7bfbe6197f1142e2cbfc0ed440f95a65794b5b95937bb807ef8c2ddb9b1_4",
                    "7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869_4",
                ],
                "slots": [
                    {
                        "name": "employment_law_year",
                        "gold_page_ids": ["9f3ba7bfbe6197f1142e2cbfc0ed440f95a65794b5b95937bb807ef8c2ddb9b1_4"],
                        "evidence_markers": [],
                    },
                    {
                        "name": "intellectual_property_law_year",
                        "gold_page_ids": ["7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869_4"],
                        "evidence_markers": [],
                    },
                ],
            }
        ],
        "wrong_document_risk": False,
    },
    "d9c088343bcf9b1b7a17a4a92b394925494a8ae2a2f86b09d10d267179eb01bb": {
        "question_id": "d9c088343bcf9b1b7a17a4a92b394925494a8ae2a2f86b09d10d267179eb01bb",
        "trust_tier": "trusted",
        "gold_origin": "manual_override",
        "audit_note": "Manual multi-slot enacted-plus-amended benchmark case.",
        "gold_page_ids": [
            "fe81efddb24f01f4455fc6b16fc0867615e840b0d36e7abfd1441e5c2eaed92d_1",
            "d0f7fdbb0fb81d646b912a216d61979f658188d8d49473d9ed7138580ec0e533_1",
            "22442c5ee999e2519c68de908be511875a84f2b810ed540c2dcfcbcc65031434_1",
        ],
        "gold_items": [],
        "items": [
            {
                "id": "item_1",
                "text": "",
                "gold_page_ids": [
                    "fe81efddb24f01f4455fc6b16fc0867615e840b0d36e7abfd1441e5c2eaed92d_1",
                    "d0f7fdbb0fb81d646b912a216d61979f658188d8d49473d9ed7138580ec0e533_1",
                    "22442c5ee999e2519c68de908be511875a84f2b810ed540c2dcfcbcc65031434_1",
                ],
                "slots": [
                    {
                        "name": "enactment_date",
                        "gold_page_ids": ["fe81efddb24f01f4455fc6b16fc0867615e840b0d36e7abfd1441e5c2eaed92d_1"],
                        "evidence_markers": [],
                    },
                    {
                        "name": "amended_limited_partnership_law",
                        "gold_page_ids": ["d0f7fdbb0fb81d646b912a216d61979f658188d8d49473d9ed7138580ec0e533_1"],
                        "evidence_markers": [],
                    },
                    {
                        "name": "amended_foundations_law",
                        "gold_page_ids": ["22442c5ee999e2519c68de908be511875a84f2b810ed540c2dcfcbcc65031434_1"],
                        "evidence_markers": [],
                    },
                ],
            }
        ],
        "wrong_document_risk": False,
    },
    "4ce050c0d6261bf3ee2eafa9c7d5fc7273e390a4a1c09ab6e26f691c68199d1b": {
        "question_id": "4ce050c0d6261bf3ee2eafa9c7d5fc7273e390a4a1c09ab6e26f691c68199d1b",
        "trust_tier": "trusted",
        "gold_origin": "manual_override",
        "audit_note": (
            "Direct source audit confirms Leasing Law page 4 and canonical Trust Law page 5 both contain "
            "the administration clause; consolidated Trust family pages are surrogate context, not gold."
        ),
        "gold_page_ids": [
            "8c6d34f8833e88c664c99576d875f0d1bcab6bd6360e9c6fe6dec3f50f9bde01_4",
            "9ad89c8f1ca0b257e77e8bbc354e7aff4ec6bcce10d28f172b71e7918908c221_5",
        ],
        "gold_items": [],
        "items": [
            {
                "id": "item_1",
                "text": "",
                "gold_page_ids": [
                    "8c6d34f8833e88c664c99576d875f0d1bcab6bd6360e9c6fe6dec3f50f9bde01_4",
                    "9ad89c8f1ca0b257e77e8bbc354e7aff4ec6bcce10d28f172b71e7918908c221_5",
                ],
                "slots": [
                    {
                        "name": "leasing_law_administration",
                        "gold_page_ids": ["8c6d34f8833e88c664c99576d875f0d1bcab6bd6360e9c6fe6dec3f50f9bde01_4"],
                        "evidence_markers": [],
                    },
                    {
                        "name": "trust_law_administration",
                        "gold_page_ids": ["9ad89c8f1ca0b257e77e8bbc354e7aff4ec6bcce10d28f172b71e7918908c221_5"],
                        "evidence_markers": [],
                    },
                ],
            }
        ],
        "wrong_document_risk": True,
    },
    "96bccc8b15e2795578584484ea3533e71d6e044d13420cf77a32393b7502fc1c": {
        "question_id": "96bccc8b15e2795578584484ea3533e71d6e044d13420cf77a32393b7502fc1c",
        "trust_tier": "trusted",
        "gold_origin": "manual_override",
        "audit_note": (
            "Direct source audit confirms canonical Trust Law page 5 contains 'Administration of this Law'; "
            "predicted consolidated Trust page 4 is a same-law surrogate and should not replace the gold page."
        ),
        "gold_page_ids": [
            "7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869_4",
            "9ad89c8f1ca0b257e77e8bbc354e7aff4ec6bcce10d28f172b71e7918908c221_5",
        ],
        "gold_items": [],
        "items": [
            {
                "id": "item_1",
                "text": "",
                "gold_page_ids": [
                    "7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869_4",
                    "9ad89c8f1ca0b257e77e8bbc354e7aff4ec6bcce10d28f172b71e7918908c221_5",
                ],
                "slots": [
                    {
                        "name": "intellectual_property_law_administration",
                        "gold_page_ids": ["7d2514bd549b3771c085b0d0b74c6e4353feb3895811046061142738ccfc6869_4"],
                        "evidence_markers": [],
                    },
                    {
                        "name": "trust_law_administration",
                        "gold_page_ids": ["9ad89c8f1ca0b257e77e8bbc354e7aff4ec6bcce10d28f172b71e7918908c221_5"],
                        "evidence_markers": [],
                    },
                ],
            }
        ],
        "wrong_document_risk": True,
    },
    "7700103c51940db23ba51a0efefbef679201af5b0a60935853d10bf81a260466": {
        "question_id": "7700103c51940db23ba51a0efefbef679201af5b0a60935853d10bf81a260466",
        "trust_tier": "trusted",
        "gold_origin": "reviewed_correction",
        "audit_note": (
            "Corrected after direct source audit: Employment Law Amendment Law enactment notice page "
            "explicitly states DIFC Law No. 4 of 2021; previous Trust Law gold was mismatched."
        ),
        "gold_page_ids": [
            "bac066005f21591dbcff19a56cdef279bd4f32482e3758925a38c478e75b81a8_1",
        ],
        "gold_items": [],
        "items": [],
        "wrong_document_risk": False,
    },
}


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            items.append(text)
    return items


def _load_qid_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    out: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        out.add(text)
    return out


def _load_scaffold_records(path: Path) -> list[JsonDict]:
    payload = _load_json(path)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold JSON must contain a top-level 'records' list: {path}")
    out: list[JsonDict] = []
    for raw_item in cast("list[object]", records_obj):
        if isinstance(raw_item, dict):
            out.append(cast("JsonDict", raw_item))
    return out


def _default_trust_tier(case: JsonDict) -> str:
    raw = str(case.get("trust_tier") or "").strip().lower()
    if raw in {"trusted", "suspect"}:
        return raw
    has_manual_structure = bool(_coerce_str_list(case.get("gold_items"))) or bool(case.get("items"))
    return "trusted" if has_manual_structure else "suspect"


def _default_gold_origin(case: JsonDict, *, trust_tier: str) -> str:
    raw = str(case.get("gold_origin") or "").strip().lower()
    if raw in {"manual_override", "seeded_eval", "reviewed_correction"}:
        return raw
    return "manual_override" if trust_tier == "trusted" else "seeded_eval"


def _normalize_case_metadata(case: JsonDict) -> JsonDict:
    normalized = cast("JsonDict", json.loads(json.dumps(case)))
    trust_tier = _default_trust_tier(normalized)
    normalized["trust_tier"] = trust_tier
    gold_origin = _default_gold_origin(normalized, trust_tier=trust_tier)
    normalized["gold_origin"] = gold_origin
    audit_note = str(normalized.get("audit_note") or "").strip()
    if not audit_note:
        if gold_origin == "seeded_eval":
            audit_note = "Inherited seed gold from accepted eval; review before using as hard gate."
        elif gold_origin == "reviewed_correction":
            audit_note = "Gold corrected after direct source audit."
        elif trust_tier == "trusted":
            audit_note = "Manually curated hidden-G benchmark case."
    normalized["audit_note"] = audit_note
    return normalized


def _seed_case_from_eval(eval_case: JsonDict) -> JsonDict:
    telemetry = cast("JsonDict", eval_case.get("telemetry")) if isinstance(eval_case.get("telemetry"), dict) else {}
    question_id = str(eval_case.get("question_id") or eval_case.get("case_id") or "").strip()
    used_page_ids = _coerce_str_list(telemetry.get("used_page_ids"))
    cited_page_ids = _coerce_str_list(telemetry.get("cited_page_ids"))
    return {
        "question_id": question_id,
        "trust_tier": "suspect",
        "gold_origin": "seeded_eval",
        "audit_note": "Auto-seeded from accepted eval used_page_ids; review before using as hard gate.",
        "gold_page_ids": used_page_ids or cited_page_ids,
        "gold_items": [],
        "items": [],
        "wrong_document_risk": False,
    }


def _is_explicit_anchor(question: str) -> bool:
    return bool(_EXPLICIT_ANCHOR_RE.search(question))


def _is_scaffold_case_eligible(record: JsonDict) -> bool:
    manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
    if manual_verdict != "correct":
        return False
    if not str(record.get("question_id") or "").strip():
        return False
    gold_page_ids = _coerce_str_list(record.get("minimal_required_support_pages"))
    if not gold_page_ids:
        return False
    question = str(record.get("question") or "").strip()
    failure_class = str(record.get("failure_class") or "").strip().lower()
    return failure_class == "support_undercoverage" or _is_explicit_anchor(question)


def _scaffold_family_tags(record: JsonDict) -> list[str]:
    question = str(record.get("question") or "").strip().lower()
    tags: list[str] = []
    if "title page" in question or "title pages" in question or "cover page" in question or "title/cover page" in question:
        tags.append("title_page")
    if "page 2" in question or "second page" in question:
        tags.append("page_2")
    if "page " in question or "title page" in question or "cover page" in question or "title/cover page" in question:
        tags.append("explicit_page")
    if "caption" in question or "header" in question:
        tags.append("caption_header")
    if "article " in question:
        tags.append("article_anchor")
    return list(dict.fromkeys(tags))


def _seed_case_from_scaffold(record: JsonDict) -> JsonDict:
    gold_page_ids = _coerce_str_list(record.get("minimal_required_support_pages"))
    question = str(record.get("question") or "").strip()
    failure_class = str(record.get("failure_class") or "").strip().lower()
    family_tags = _scaffold_family_tags(record)
    note_parts = [
        "Trusted scaffold-backed page-id case from truth_audit_scaffold minimal_required_support_pages.",
    ]
    if failure_class == "support_undercoverage":
        note_parts.append("Manual verdict stayed correct while support under-covered the gold page set.")
    elif _is_explicit_anchor(question):
        note_parts.append("Question contains an explicit page/title/cover anchor and the scaffold provides reviewed gold pages.")
    if family_tags:
        note_parts.append(f"Families: {', '.join(family_tags)}.")
    return {
        "question_id": str(record.get("question_id") or "").strip(),
        "trust_tier": "trusted",
        "gold_origin": "manual_override",
        "audit_note": " ".join(note_parts),
        "gold_page_ids": gold_page_ids,
        "gold_items": [],
        "items": [],
        "wrong_document_risk": False,
    }


def _should_promote_existing_case(existing_case: JsonDict) -> bool:
    trust_tier = str(existing_case.get("trust_tier") or "").strip().lower()
    gold_origin = str(existing_case.get("gold_origin") or "").strip().lower()
    return trust_tier != "trusted" or gold_origin == "seeded_eval"


def _apply_manual_overrides(case: JsonDict) -> JsonDict:
    question_id = str(case.get("question_id") or "").strip()
    override = MANUAL_CASE_OVERRIDES.get(question_id)
    if override is None:
        return _normalize_case_metadata(case)
    return cast("JsonDict", json.loads(json.dumps(override)))


def _merge_cases(
    *,
    benchmark_payload: JsonDict,
    eval_payload: JsonDict | None,
    scaffold_records: list[JsonDict] | None,
    source_eval_name: str | None,
    qid_filter: set[str] | None,
) -> JsonDict:
    existing_cases_obj = benchmark_payload.get("cases")
    if not isinstance(existing_cases_obj, list):
        raise ValueError("Benchmark JSON must contain a top-level 'cases' list")

    existing_by_question_id: dict[str, JsonDict] = {}
    ordered_existing_ids: list[str] = []
    for raw_case in cast("list[object]", existing_cases_obj):
        if not isinstance(raw_case, dict):
            continue
        case = cast("JsonDict", raw_case)
        question_id = str(case.get("question_id") or "").strip()
        if not question_id or question_id in existing_by_question_id:
            continue
        existing_by_question_id[question_id] = _apply_manual_overrides(_normalize_case_metadata(case))
        ordered_existing_ids.append(question_id)

    if scaffold_records is not None:
        for record in scaffold_records:
            question_id = str(record.get("question_id") or "").strip()
            if not question_id:
                continue
            if qid_filter is not None and question_id not in qid_filter:
                continue
            if not _is_scaffold_case_eligible(record):
                continue
            existing_case = existing_by_question_id.get(question_id)
            if existing_case is None or not _should_promote_existing_case(existing_case):
                continue
            existing_by_question_id[question_id] = _apply_manual_overrides(_seed_case_from_scaffold(record))

    merged_cases: list[JsonDict] = [existing_by_question_id[question_id] for question_id in ordered_existing_ids]
    seen_question_ids = set(ordered_existing_ids)

    if scaffold_records is not None:
        for record in scaffold_records:
            question_id = str(record.get("question_id") or "").strip()
            if not question_id or question_id in seen_question_ids:
                continue
            if qid_filter is not None and question_id not in qid_filter:
                continue
            if not _is_scaffold_case_eligible(record):
                continue
            merged_cases.append(_apply_manual_overrides(_seed_case_from_scaffold(record)))
            seen_question_ids.add(question_id)

    if eval_payload is not None:
        eval_cases_obj = eval_payload.get("cases")
        if not isinstance(eval_cases_obj, list):
            raise ValueError("Eval JSON must contain a top-level 'cases' list")
        for raw_case in cast("list[object]", eval_cases_obj):
            if not isinstance(raw_case, dict):
                continue
            eval_case = cast("JsonDict", raw_case)
            question_id = str(eval_case.get("question_id") or eval_case.get("case_id") or "").strip()
            if not question_id or question_id in seen_question_ids:
                continue
            merged_cases.append(_normalize_case_metadata(_seed_case_from_eval(eval_case)))
            seen_question_ids.add(question_id)

    benchmark_source_eval = str(benchmark_payload.get("source_eval") or "").strip()
    description_parts = [
        "Tracked seed benchmark derived from the accepted local eval plus source-backed manual scaffold additions.",
        "Manually curated high-risk cases keep item/slot gold pages, while accepted eval spillover remains regression-only suspect gold.",
    ]
    if scaffold_records is not None:
        description_parts.append(
            "Scaffold-backed additions are limited to reviewed support-undercoverage or explicit-page-anchor cases with minimal_required_support_pages."
        )
    return {
        "name": "hidden_g_benchmark_seed_v2",
        "source_eval": source_eval_name or benchmark_source_eval or "manual_scaffold_seed",
        "description": " ".join(description_parts),
        "cases": merged_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the hidden-G benchmark seed with accepted eval output and reviewed scaffold cases.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Existing benchmark JSON to extend.")
    parser.add_argument("--eval", type=Path, default=None, help="Accepted eval JSON used as the suspect seed source.")
    parser.add_argument("--scaffold", type=Path, default=None, help="Truth-audit scaffold JSON used for trusted manual additions.")
    parser.add_argument("--qids-file", type=Path, default=None, help="Optional filter limiting scaffold additions to listed question ids.")
    parser.add_argument("--out", type=Path, default=None, help="Output path. Defaults to overwriting --benchmark.")
    args = parser.parse_args()

    if args.eval is None and args.scaffold is None:
        parser.error("Provide at least one of --eval or --scaffold.")

    benchmark_payload = _load_json(args.benchmark)
    eval_payload = _load_json(args.eval) if args.eval is not None else None
    scaffold_records = _load_scaffold_records(args.scaffold) if args.scaffold is not None else None
    merged = _merge_cases(
        benchmark_payload=benchmark_payload,
        eval_payload=eval_payload,
        scaffold_records=scaffold_records,
        source_eval_name=args.eval.name if args.eval is not None else None,
        qid_filter=_load_qid_filter(args.qids_file),
    )

    out_path = args.out or args.benchmark
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
