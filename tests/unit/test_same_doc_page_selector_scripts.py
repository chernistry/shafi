import json
from pathlib import Path

from scripts import export_same_doc_page_selector_features as export_mod
from scripts import run_same_doc_page_selector_falsifier as falsifier_mod


def test_extract_query_flags_detects_explicit_page_signals() -> None:
    title_flags = export_mod.extract_query_flags(
        "According to the title page of the DIFC Law on the Application of Civil and Commercial Laws, what is its official law number?"
    )
    assert title_flags["query_has_title_page"] is True
    assert title_flags["query_has_law_number"] is True
    assert title_flags["requested_page"] is None

    page_flags = export_mod.extract_query_flags(
        "According to page 2 of the judgment, from which specific claim number did the appeal originate?"
    )
    assert page_flags["query_has_second_page"] is True
    assert page_flags["query_has_case_ref"] is True
    assert page_flags["requested_page"] == 2


def test_build_rows_exports_observed_and_gold_only_candidates(tmp_path: Path) -> None:
    miss_pack = tmp_path / "miss_pack.json"
    baseline = tmp_path / "baseline.json"
    questions = tmp_path / "questions.json"
    miss_pack.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "qid": "q1",
                        "question_family": "single_doc_title_cover",
                        "miss_family": "title_page",
                        "route": "strict",
                        "trust_tier": "trusted",
                        "target_doc_ids": ["docA"],
                        "gold_pages": ["docA_1", "docA_9"],
                        "used_pages": ["docA_7", "docA_3", "docA_1"],
                        "false_positive_pages": ["docA_7", "docA_3"],
                        "ocr_risk": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "predicted_page_ids": ["docA_7", "docA_3", "docA_1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    questions.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "According to the title page of the DIFC Law, what is its official law number?",
                    "answer_type": "number",
                }
            ]
        ),
        encoding="utf-8",
    )

    rows, summary = export_mod.build_rows(
        miss_pack_path=miss_pack,
        baseline_predictions_path=baseline,
        questions_path=questions,
    )

    assert len(rows) == 4
    page_one = next(row for row in rows if row["page_id"] == "docA_1")
    assert page_one["candidate_observed"] is True
    assert page_one["is_gold"] is True
    assert page_one["baseline_rank"] == 3
    gold_only = next(row for row in rows if row["page_id"] == "docA_9")
    assert gold_only["candidate_observed"] is False
    assert gold_only["candidate_gold_only"] is True
    assert summary["summary"]["cases_with_observed_gold_candidate"] == 1


def test_same_doc_falsifier_finds_bounded_precision_win() -> None:
    rows = [
        {
            "qid": "title_q",
            "page_id": "docA_7",
            "page_number": 7,
            "baseline_rank": 1,
            "is_gold": False,
            "is_baseline_predicted": True,
            "candidate_observed": True,
            "requested_page_match": False,
            "is_page_one": False,
            "is_page_two": False,
            "query_has_title_page": True,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": False,
            "query_has_section": False,
        },
        {
            "qid": "title_q",
            "page_id": "docA_3",
            "page_number": 3,
            "baseline_rank": 2,
            "is_gold": False,
            "is_baseline_predicted": True,
            "candidate_observed": True,
            "requested_page_match": False,
            "is_page_one": False,
            "is_page_two": False,
            "query_has_title_page": True,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": False,
            "query_has_section": False,
        },
        {
            "qid": "title_q",
            "page_id": "docA_1",
            "page_number": 1,
            "baseline_rank": 3,
            "is_gold": True,
            "is_baseline_predicted": True,
            "candidate_observed": True,
            "requested_page_match": False,
            "is_page_one": True,
            "is_page_two": False,
            "query_has_title_page": True,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": False,
            "query_has_section": False,
        },
        {
            "qid": "article_q",
            "page_id": "docB_13",
            "page_number": 13,
            "baseline_rank": 1,
            "is_gold": True,
            "is_baseline_predicted": True,
            "candidate_observed": True,
            "requested_page_match": False,
            "is_page_one": False,
            "is_page_two": False,
            "query_has_title_page": False,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": True,
            "query_has_section": False,
        },
        {
            "qid": "article_q",
            "page_id": "docB_12",
            "page_number": 12,
            "baseline_rank": 2,
            "is_gold": False,
            "is_baseline_predicted": True,
            "candidate_observed": True,
            "requested_page_match": False,
            "is_page_one": False,
            "is_page_two": False,
            "query_has_title_page": False,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": True,
            "query_has_section": False,
        },
        {
            "qid": "wrong_q",
            "page_id": "docC_6",
            "page_number": 6,
            "baseline_rank": 1,
            "is_gold": False,
            "is_baseline_predicted": True,
            "candidate_observed": True,
            "requested_page_match": False,
            "is_page_one": False,
            "is_page_two": False,
            "query_has_title_page": False,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": True,
            "query_has_section": False,
        },
        {
            "qid": "wrong_q",
            "page_id": "docZ_4",
            "page_number": 4,
            "baseline_rank": None,
            "is_gold": True,
            "is_baseline_predicted": False,
            "candidate_observed": False,
            "requested_page_match": False,
            "is_page_one": False,
            "is_page_two": False,
            "query_has_title_page": False,
            "query_has_cover_page": False,
            "query_has_first_page": False,
            "query_has_second_page": False,
            "query_has_article": True,
            "query_has_section": False,
        },
    ]

    payload = falsifier_mod.run_falsifier(rows)

    assert payload["verdict"] == "win"
    assert payload["delta_f_beta"] > 0.0
    assert payload["delta_avg_pages_per_case"] <= 0.0
    assert set(payload["improved_qids"]) == {"title_q", "article_q"}
