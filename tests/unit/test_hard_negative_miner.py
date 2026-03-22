from __future__ import annotations

from shafi.ml.hard_negative_miner import HardNegativeMiner, NegativeStrategy


def _miner() -> HardNegativeMiner:
    return HardNegativeMiner(
        page_texts={
            "law-a_1": "Employment Law 2020 issued by DIFC Authority.",
            "law-a_2": "Article 5 requires notice before termination.",
            "law-b_1": "Employment Amendment Law 2021 transitional obligations.",
            "case-a_1": "Alice Smith v Bob Jones before Justice Smith.",
        },
        page_doc_ids={
            "law-a_1": "law-a",
            "law-a_2": "law-a",
            "law-b_1": "law-b",
            "case-a_1": "case-a",
        },
        aliases_by_doc_id={
            "law-a": ["Employment Law"],
            "law-b": ["Employment Amendment Law"],
            "case-a": ["Alice Smith"],
        },
    )


def test_mine_same_doc_excludes_gold_pages() -> None:
    candidates = _miner().mine_same_doc(
        query="What does Article 5 of Employment Law provide?",
        gold_doc_ids=["law-a"],
        gold_page_ids=["law-a_2"],
    )

    assert [candidate.page_id for candidate in candidates] == ["law-a_1"]
    assert all(candidate.strategy is NegativeStrategy.SAME_DOC for candidate in candidates)


def test_mine_alias_confusable_uses_alias_hits() -> None:
    candidates = _miner().mine_alias_confusable(
        query="Did Alice Smith appear before Justice Smith?",
        gold_page_ids=["case-a_1"],
    )

    assert candidates == []


def test_mine_cross_doc_and_lexical_near_miss_rank_overlap() -> None:
    miner = _miner()
    cross_doc = miner.mine_cross_doc(
        query="Which employment law was issued by DIFC Authority?",
        gold_doc_ids=["law-a"],
        gold_page_ids=["law-a_1"],
    )
    lexical = miner.mine_lexical_near_miss(
        query="Which employment law was issued by DIFC Authority?",
        gold_page_ids=["law-a_1"],
    )

    assert cross_doc[0].page_id == "law-b_1"
    assert lexical[0].page_id in {"law-a_2", "law-b_1"}


def test_deduplicate_keeps_strongest_candidate_per_page() -> None:
    miner = _miner()
    candidates = miner.mine(
        query="What does Employment Law require?",
        gold_page_ids=["law-a_2"],
        gold_doc_ids=["law-a"],
        top_k=5,
    )

    assert len({candidate.page_id for candidate in candidates}) == len(candidates)
    assert "law-a_2" not in {candidate.page_id for candidate in candidates}
