from __future__ import annotations

from typing import TYPE_CHECKING

from rag_challenge.ingestion.document_interrogation import (
    DocumentInterrogationInput,
    DocumentInterrogationPageInput,
    build_compact_shadow_text,
    build_document_interrogation_system_prompt,
    build_document_interrogation_user_prompt,
    load_document_interrogation_inputs,
    parse_document_interrogation_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_document_interrogation_prompts_include_schema_and_pages() -> None:
    doc = DocumentInterrogationInput(
        doc_id="law-1",
        doc_title="DIFC Employment Law",
        pages=[DocumentInterrogationPageInput(page_num=1, text="Title page")],
    )

    system_prompt = build_document_interrogation_system_prompt()
    user_prompt = build_document_interrogation_user_prompt(doc)

    assert "Return JSON only" in system_prompt
    assert '"doc_id": "law-1"' in user_prompt
    assert '"page_num": 1' in user_prompt
    assert '"canonical_law_title": ""' in user_prompt
    assert '"amendment_targets": []' in user_prompt


def test_parse_document_interrogation_json_builds_compact_shadow_text() -> None:
    raw_json = """
    {
      "doc_id": "law-1",
      "doc_title": "  the DIFC Employment Law  ",
      "document_type": "law",
      "issuing_authority": "issued by the DIFC Authority",
      "law_title": "DIFC Employment Law",
      "law_number": "No. 2",
      "law_year": "2019",
      "key_parties": ["Alpha Ltd"],
      "amendment_relationships": [
        "amends DIFC Employment Law",
        "as amended by the DIFC Employment Law",
        "repealed by DIFC Employment Law",
        "replaces DIFC Employment Law",
        "replaced by DIFC Employment Law",
        "consolidated with amendments up to DIFC Employment Law",
        "amends DIFC Employment Law"
      ],
      "authoritative_sections": ["Article 16"],
      "likely_answer_page_families": ["title_cover", "article_body"],
      "page_signals": [
        {
          "page_num": 1,
          "heading_summary": "Title page",
          "page_template_family": "title_cover",
          "field_labels_present": ["Date of Issue"],
          "primary_evidence": true
        }
      ]
    }
    """

    record = parse_document_interrogation_json(raw_json)

    assert record.doc_id == "law-1"
    assert record.canonical_law_title == "DIFC Employment Law"
    assert record.canonical_issuing_authority == "DIFC Authority"
    assert record.normalized_amendment_relationships == [
        "amends DIFC Employment Law",
        "amended by DIFC Employment Law",
        "repealed by DIFC Employment Law",
        "replaces DIFC Employment Law",
        "replaced by DIFC Employment Law",
        "consolidated with amendments up to DIFC Employment Law",
    ]
    assert record.amendment_targets == ["DIFC Employment Law"]
    assert "DIFC Employment Law" in record.compact_shadow_text
    assert "Article 16" in record.compact_shadow_text
    assert "amends DIFC Employment Law" in record.compact_shadow_text
    assert "amended by DIFC Employment Law" in record.compact_shadow_text
    assert "repealed by DIFC Employment Law" in record.compact_shadow_text
    assert "replaced by DIFC Employment Law" in record.compact_shadow_text
    assert "DIFC Authority" in record.compact_shadow_text


def test_build_compact_shadow_text_is_stable_and_compact() -> None:
    record = parse_document_interrogation_json(
        """
        {
          "doc_id": "law-2",
          "doc_title": "Fee Schedule",
          "document_type": "schedule",
          "issuing_authority": "",
          "law_title": "",
          "law_number": "",
          "law_year": "",
          "key_parties": [],
          "amendment_relationships": ["superseded by the Fee Schedule"],
          "authoritative_sections": ["Schedule A"],
          "likely_answer_page_families": ["schedule_table"],
          "page_signals": []
        }
        """
    )

    compact = build_compact_shadow_text(record)

    assert "Fee Schedule" in compact
    assert "superseded by Fee Schedule" in compact
    assert "\n" not in compact


def test_load_document_interrogation_inputs_supports_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "docs.jsonl"
    input_path.write_text(
        '{"doc_id":"a","doc_title":"A","pages":[{"page_num":1,"text":"x"}]}\n'
        '{"doc_id":"b","doc_title":"B","pages":[{"page_num":2,"text":"y"}]}\n',
        encoding="utf-8",
    )

    docs = load_document_interrogation_inputs(input_path)

    assert [doc.doc_id for doc in docs] == ["a", "b"]
