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
