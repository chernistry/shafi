from rag_challenge.eval.sources import select_used_pages


def test_select_used_pages_prefers_cited_page_ids() -> None:
    payload = {
        "used_page_ids": ["abc123_2"],
        "cited_page_ids": ["abc123_3"],
        "context_page_ids": ["abc123_3", "abc123_7"],
    }
    assert select_used_pages(payload, max_pages=6) == ["abc123_2"]


def test_select_used_pages_falls_back_to_cited_only_when_used_pages_missing() -> None:
    # Ticket 2004: context_page_ids no longer included — mirrors submission-side logic
    payload = {
        "cited_page_ids": ["abc123_3"],
        "context_page_ids": ["abc123_3", "abc123_7"],
    }
    assert select_used_pages(payload, max_pages=6) == ["abc123_3"]


def test_select_used_pages_returns_empty_when_only_context_pages() -> None:
    # Ticket 2004: context_page_ids alone no longer qualify as used pages
    payload = {"context_page_ids": ["abc123_2", "abc123_5"]}
    assert select_used_pages(payload, max_pages=6) == []


def test_select_used_pages_falls_back_to_chunk_to_page_mapping() -> None:
    payload = {"cited_chunk_ids": ["abc123:0:0:deadbeef", "abc123:1:0:cafebabe"]}
    assert select_used_pages(payload, max_pages=6) == ["abc123_1", "abc123_2"]


def test_select_used_pages_applies_cap() -> None:
    payload = {"cited_page_ids": ["a_1", "a_2", "a_3"]}
    assert select_used_pages(payload, max_pages=2) == ["a_1", "a_2"]
