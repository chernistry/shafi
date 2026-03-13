import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag_challenge.models import QueryComplexity


@pytest.fixture
def mock_settings():
    settings = SimpleNamespace(
        llm=SimpleNamespace(
            complex_min_length=150,
            complex_keywords=[
                "compare",
                "difference",
                "exception",
                "notwithstanding",
                "distinguish",
                "analyze",
                "evaluate",
                "contrast",
                "jurisdiction",
                "precedent",
            ],
            complex_min_entities=2,
            simple_model="gpt-4o-mini",
            complex_model="gpt-4o",
            simple_max_tokens=300,
            complex_max_tokens=500,
        )
    )
    with patch("rag_challenge.core.classifier.get_settings", return_value=settings):
        yield settings


def test_short_simple_query(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    assert classifier.classify("What is the statute of limitations?") == QueryComplexity.SIMPLE


def test_long_query_is_complex(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    assert classifier.classify("x" * 200) == QueryComplexity.COMPLEX


def test_multiple_keywords_is_complex(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    query = "Compare the difference between these two jurisdictions"
    assert classifier.classify(query) == QueryComplexity.COMPLEX


def test_single_keyword_is_simple(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    query = "What is the exception to this rule?"
    assert classifier.classify(query) == QueryComplexity.SIMPLE


def test_multiple_legal_entities_is_complex(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    query = "Under § 501 and Section 12, what applies?"
    assert classifier.classify(query) == QueryComplexity.COMPLEX


def test_multi_part_question_is_complex(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    query = "What is the rule; and also what are the exceptions?"
    assert classifier.classify(query) == QueryComplexity.COMPLEX


def test_case_citation_counts_as_entity(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    query = "How does Smith v. Jones relate to Article 5?"
    assert classifier.classify(query) == QueryComplexity.COMPLEX


def test_model_and_max_token_selection(mock_settings):
    from rag_challenge.core.classifier import QueryClassifier

    classifier = QueryClassifier()
    assert classifier.select_model(QueryComplexity.SIMPLE) == "gpt-4o-mini"
    assert classifier.select_model(QueryComplexity.COMPLEX) == "gpt-4o"
    assert classifier.select_max_tokens(QueryComplexity.SIMPLE) == 300
    assert classifier.select_max_tokens(QueryComplexity.COMPLEX) == 500


def test_normalize_usc_citation():
    from rag_challenge.core.classifier import QueryClassifier

    assert QueryClassifier.normalize_query("42 USC 1983") == "42 U.S.C. § 1983"
    assert QueryClassifier.normalize_query("42 U.S.C. §1983") == "42 U.S.C. § 1983"
    assert QueryClassifier.normalize_query("17  USC   107") == "17 U.S.C. § 107"


def test_normalize_collapses_whitespace():
    from rag_challenge.core.classifier import QueryClassifier

    assert QueryClassifier.normalize_query("  hello   world  ") == "hello world"


def test_normalize_difc_references():
    from rag_challenge.core.classifier import QueryClassifier

    assert QueryClassifier.normalize_query("law no 12 of 2004") == "Law No. 12 of 2004"
    assert QueryClassifier.normalize_query("cfi 10/2024") == "CFI 010/2024"
    assert QueryClassifier.normalize_query("article 5 ( 2 )") == "Article 5(2)"


def test_extract_doc_refs_normalizes_and_dedupes():
    from rag_challenge.core.classifier import QueryClassifier

    refs = QueryClassifier.extract_doc_refs("In CFI 10/2024 under law no 12 of 2004, see CFI 010/2024")
    assert "Law No. 12 of 2004" in refs
    assert "CFI 010/2024" in refs
    assert refs.count("CFI 010/2024") == 1


def test_extract_query_refs_includes_law_titles():
    from rag_challenge.core.classifier import QueryClassifier

    refs = QueryClassifier.extract_query_refs("Compare Schedule 1 of the Trust Law 2018 under Article 5(2).")
    assert "Trust Law 2018" in refs


def test_extract_exact_legal_refs_returns_statute_style_refs_only():
    from rag_challenge.core.classifier import QueryClassifier

    refs = QueryClassifier.extract_exact_legal_refs(
        "According to Article 16 ( 1 ) of the Operating Law 2018 in CFI 010/2024, what applies under law no 7 of 2018?"
    )

    assert "Article 16(1)" in refs
    assert "Operating Law 2018" in refs
    assert "Law No. 7 of 2018" in refs
    assert "CFI 010/2024" not in refs


def test_extract_explicit_page_reference_detects_title_page():
    from rag_challenge.core.classifier import QueryClassifier

    ref = QueryClassifier.extract_explicit_page_reference("Who is listed on the title page of CFI 010/2024?")
    assert ref is not None
    assert ref.kind == "title_page"
    assert ref.requested_page == 1


def test_extract_explicit_page_reference_detects_second_page():
    from rag_challenge.core.classifier import QueryClassifier

    ref = QueryClassifier.extract_explicit_page_reference("What appears on the second page of the judgment?")
    assert ref is not None
    assert ref.kind == "second_page"
    assert ref.requested_page == 2


def test_extract_explicit_page_reference_detects_numeric_page():
    from rag_challenge.core.classifier import QueryClassifier

    ref = QueryClassifier.extract_explicit_page_reference("Which date appears on page 7?")
    assert ref is not None
    assert ref.kind == "numeric_page"
    assert ref.requested_page == 7


def test_extract_explicit_page_reference_detects_caption_header():
    from rag_challenge.core.classifier import QueryClassifier

    ref = QueryClassifier.extract_explicit_page_reference("What claimant name appears in the caption header?")
    assert ref is not None
    assert ref.kind == "caption_header"
    assert ref.requested_page == 1


def test_explicit_page_reference_audit_groups_phrase_types(tmp_path):
    from scripts.audit_explicit_page_reference_candidates import build_audit

    ledger_path = tmp_path / "page_trace_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "qid": "q1",
                        "question": "Who is listed on the title page?",
                        "gold_pages": ["docA_1"],
                        "used_pages": ["docA_2"],
                        "false_positive_pages": ["docA_2"],
                        "failure_stage": "wrong_page_used_same_doc",
                        "trust_tier": "trusted",
                        "gold_in_used": False,
                    },
                    {
                        "qid": "q2",
                        "question": "What is on the second page?",
                        "gold_pages": ["docB_2"],
                        "used_pages": ["docB_2"],
                        "false_positive_pages": [],
                        "failure_stage": "retained_to_used",
                        "trust_tier": "trusted",
                        "gold_in_used": True,
                    },
                    {
                        "qid": "q3",
                        "question": "What amount appears on page 9?",
                        "gold_pages": ["docC_9"],
                        "used_pages": ["docC_8"],
                        "false_positive_pages": ["docC_8"],
                        "failure_stage": "wrong_page_used_same_doc",
                        "trust_tier": "suspect",
                        "gold_in_used": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = build_audit(page_trace_ledger_path=ledger_path, min_meaningful_qids=3)

    assert payload["summary"]["meaningful_qid_count"] == 3
    assert payload["summary"]["phrase_type_counts"] == {
        "numeric_page": 1,
        "title_page": 1,
        "second_page": 1,
        "caption_header": 0,
    }
    assert payload["summary"]["verdict"] == "continue_to_ticket_14"
