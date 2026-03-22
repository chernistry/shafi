from pathlib import Path

from scripts import audit_structured_document_stress_pack as audit_mod
from scripts import build_structured_document_stress_pack as build_mod


def test_classify_subtypes_detects_structural_queries() -> None:
    assert build_mod.classify_subtypes(
        "According to the title page of the Common Reporting Standard Law, what is its official DIFC Law number?"
    ) == ["title_page"]
    assert build_mod.classify_subtypes(
        "According to page 2 of the judgment, from which specific claim number did the appeal originate?"
    ) == ["page_specific"]
    assert build_mod.classify_subtypes(
        "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate in the DIFC?"
    ) == ["article_provision"]


def test_build_pack_marks_zero_count_subtypes(tmp_path: Path) -> None:
    source = tmp_path / "debug.json"
    source.write_text(
        """
        {
          "cases": [
            {
              "question_id": "q1",
              "question": "Under Article 8(1) of the Operating Law 2018, is a person permitted to operate in the DIFC?",
              "answer_type": "boolean",
              "answer": "null",
              "used_pages": [],
              "telemetry": {}
            },
            {
              "question_id": "q2",
              "question": "According to the title page of the Common Reporting Standard Law, what is its official DIFC Law number?",
              "answer_type": "number",
              "answer": "7",
              "used_pages": ["doc_1"],
              "telemetry": {}
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    pack = build_mod.build_pack(source)

    assert pack["summary"]["total_cases"] == 2
    assert pack["summary"]["subtype_counts"]["article_provision"] == 1
    assert pack["summary"]["subtype_counts"]["title_page"] == 1
    assert "schedule" in pack["summary"]["zero_count_subtypes"]


def test_audit_pack_flags_right_doc_wrong_page() -> None:
    report = audit_mod.audit_pack(
        {
            "cases": [
                {
                    "qid": "q1",
                    "question": "According to page 2 of the judgment, from which specific claim number did the appeal originate?",
                    "answer_type": "name",
                    "current_answer": "ENF 316/2023",
                    "current_used_pages": ["doc_1"],
                    "subtypes": ["page_specific"],
                    "null_answer": False,
                    "empty_used_pages": False,
                }
            ]
        }
    )

    assert report["summary"]["overall_stage_breakdown"]["right_doc_wrong_page"] == 1
    assert report["subtype_stats"]["page_specific"]["hit_rate"] == 0.0
