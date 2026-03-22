from shafi.core.premise_guard import check_query_premise
from shafi.models import DocType, RankedChunk


def _chunk(text: str) -> RankedChunk:
    return RankedChunk(
        chunk_id="c0",
        doc_id="d0",
        doc_title="Doc",
        doc_type=DocType.CASE_LAW,
        text=text,
        retrieval_score=0.8,
        rerank_score=0.8,
    )


def test_guard_triggers_when_term_missing_from_context() -> None:
    decision = check_query_premise(
        "What did the jury decide in ENF 053/2025?",
        [_chunk("The court granted the enforcement application.")],
        ["jury", "miranda", "parole", "plea bargain", "plea"],
    )
    assert decision.triggered is True
    assert decision.term == "jury"


def test_guard_does_not_trigger_when_term_present_in_context() -> None:
    decision = check_query_premise(
        "What did the jury decide in ENF 053/2025?",
        [_chunk("The jury recommendation is recorded in this extracted source.")],
        ["jury"],
    )
    assert decision.triggered is False


def test_guard_word_boundary_avoids_false_match() -> None:
    decision = check_query_premise(
        "Please provide the outcome.",
        [_chunk("Court outcome details are available.")],
        ["plea"],
    )
    assert decision.triggered is False


def test_guard_triggers_when_term_only_negated_in_context() -> None:
    decision = check_query_premise(
        "What did the jury decide in ENF 053/2025?",
        [_chunk("The DIFC process has no jury and no plea bargain mechanism.")],
        ["jury", "plea bargain"],
    )
    assert decision.triggered is True
    assert decision.term == "jury"
