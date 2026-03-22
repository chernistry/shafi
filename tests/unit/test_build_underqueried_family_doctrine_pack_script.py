from __future__ import annotations

from scripts import build_underqueried_family_doctrine_pack as mod


def test_build_underqueried_family_doctrine_pack_groups_underqueried_families() -> None:
    gap_report = {
        "family_coverage_report": [
            {
                "family_tag": "consolidated_law",
                "doc_count": 2,
                "question_count": 0,
                "coverage_ratio": 0.0,
                "coverage_bucket": "zero-hit",
            },
            {
                "family_tag": "schedule_annex_heavy",
                "doc_count": 1,
                "question_count": 1,
                "coverage_ratio": 1.0,
                "coverage_bucket": "one-hit",
            },
            {
                "family_tag": "regulations",
                "doc_count": 5,
                "question_count": 3,
                "coverage_ratio": 0.6,
                "coverage_bucket": "exercised",
            },
        ]
    }
    scan_records = [
        {
            "doc_id": "doc_a",
            "filename": "doc_a.pdf",
            "suspicion_score": 30,
            "reason_tags": ["underqueried_family_gap"],
            "doc_family_tags": ["consolidated_law"],
            "exact_duplicate_cluster_id": "cluster_001",
            "collision_doc_ids": [],
            "duplicate_same_family_doc_ids": [],
            "normalized_title": "law no 1 of 2024",
            "family_query_coverage_bucket": "zero-hit",
            "coverage_priors": {"family_buckets": {"consolidated_law": "zero-hit"}},
        },
        {
            "doc_id": "doc_b",
            "filename": "doc_b.pdf",
            "suspicion_score": 29,
            "reason_tags": ["exact_duplicate_cluster"],
            "doc_family_tags": ["consolidated_law"],
            "exact_duplicate_cluster_id": "cluster_001",
            "collision_doc_ids": [],
            "duplicate_same_family_doc_ids": [],
            "normalized_title": "law no 1 of 2024",
            "family_query_coverage_bucket": "zero-hit",
            "coverage_priors": {"family_buckets": {"consolidated_law": "zero-hit"}},
        },
        {
            "doc_id": "doc_c",
            "filename": "doc_c.pdf",
            "suspicion_score": 12,
            "reason_tags": ["underqueried_family_gap"],
            "doc_family_tags": ["schedule_annex_heavy"],
            "exact_duplicate_cluster_id": None,
            "collision_doc_ids": [],
            "duplicate_same_family_doc_ids": [],
            "normalized_title": "schedule example",
            "family_query_coverage_bucket": "one-hit",
            "coverage_priors": {"family_buckets": {"schedule_annex_heavy": "one-hit"}},
        },
    ]
    variants = [
        {
            "id": "q1__unicode_variant",
            "question": "What does the consolidated law say?",
            "variant_type": "unicode_variant",
            "expected_gold_doc_ids": ["doc_a"],
        }
    ]

    pack = mod.build_underqueried_family_doctrine_pack(
        gap_report=gap_report,
        scan_records=scan_records,
        variants=variants,
    )

    assert pack["measurement_only"] is True
    assert pack["underqueried_family_count"] == 2
    assert pack["zero_hit_families"] == ["consolidated_law"]
    first_entry = pack["entries"][0]
    assert first_entry["family_tag"] == "consolidated_law"
    assert first_entry["representative_docs"][0]["member_count"] == 2
    assert first_entry["anti_overfit_variants"][0]["id"] == "q1__unicode_variant"
