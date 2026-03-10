from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]

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


def _apply_manual_overrides(case: JsonDict) -> JsonDict:
    question_id = str(case.get("question_id") or "").strip()
    override = MANUAL_CASE_OVERRIDES.get(question_id)
    if override is None:
        return _normalize_case_metadata(case)
    return cast("JsonDict", json.loads(json.dumps(override)))


def _merge_cases(*, benchmark_payload: JsonDict, eval_payload: JsonDict, source_eval_name: str) -> JsonDict:
    existing_cases_obj = benchmark_payload.get("cases")
    if not isinstance(existing_cases_obj, list):
        raise ValueError("Benchmark JSON must contain a top-level 'cases' list")

    eval_cases_obj = eval_payload.get("cases")
    if not isinstance(eval_cases_obj, list):
        raise ValueError("Eval JSON must contain a top-level 'cases' list")

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

    merged_cases: list[JsonDict] = [existing_by_question_id[question_id] for question_id in ordered_existing_ids]
    seen_question_ids = set(ordered_existing_ids)
    for raw_case in cast("list[object]", eval_cases_obj):
        if not isinstance(raw_case, dict):
            continue
        eval_case = cast("JsonDict", raw_case)
        question_id = str(eval_case.get("question_id") or eval_case.get("case_id") or "").strip()
        if not question_id or question_id in seen_question_ids:
            continue
        merged_cases.append(_normalize_case_metadata(_seed_case_from_eval(eval_case)))
        seen_question_ids.add(question_id)

    return {
        "name": "hidden_g_benchmark_seed_v2",
        "source_eval": source_eval_name,
        "description": (
            "Tracked seed benchmark derived from the latest accepted local Docker eval. "
            "Manually curated high-risk cases keep item/slot gold pages, while uncovered public cases "
            "inherit seed page ids from the accepted eval as a regression-only guardrail."
        ),
        "cases": merged_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the hidden-G benchmark seed with an accepted eval JSON.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Existing benchmark JSON to extend.")
    parser.add_argument("--eval", type=Path, required=True, help="Accepted eval JSON used as the sync source.")
    parser.add_argument("--out", type=Path, default=None, help="Output path. Defaults to overwriting --benchmark.")
    args = parser.parse_args()

    benchmark_payload = _load_json(args.benchmark)
    eval_payload = _load_json(args.eval)
    merged = _merge_cases(
        benchmark_payload=benchmark_payload,
        eval_payload=eval_payload,
        source_eval_name=args.eval.name,
    )

    out_path = args.out or args.benchmark
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
