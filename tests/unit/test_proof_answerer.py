from __future__ import annotations

from rag_challenge.core.claim_graph import Claim, ClaimGraph, SupportType
from rag_challenge.core.proof_answerer import ProofCarryingCompiler, ProofCompilerConfig
from rag_challenge.core.query_contract import QueryContract
from rag_challenge.core.span_grounder import EvidenceSpan


def _claim(*, claim_id: str, text: str, chunk_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text=text,
        support_type=SupportType.DIRECTLY_STATED,
        evidence_spans=[
            EvidenceSpan(
                chunk_id=chunk_id,
                page_id="doc1_1",
                char_start=0,
                char_end=len(text),
                span_text=text,
                match_method="exact",
            )
        ],
        confidence=0.95,
    )


def test_compile_boolean_preserves_supported_yes_answer() -> None:
    compiler = ProofCarryingCompiler(ProofCompilerConfig(min_support_coverage=0.5))
    graph = ClaimGraph(
        claims=[_claim(claim_id="claim_1", text="Yes, it applies.", chunk_id="c0")],
        answer_text="Yes, it applies.",
        support_coverage=1.0,
    )
    contract = QueryContract(query_text="Does it apply?", answer_type="boolean")

    proof = compiler.compile(graph, contract)

    assert proof.answer_text.lower().startswith("yes")
    assert proof.is_fully_supported is True


def test_compile_free_text_removes_unsupported_claims_and_keeps_citations() -> None:
    compiler = ProofCarryingCompiler(ProofCompilerConfig(min_support_coverage=0.5))
    graph = ClaimGraph(
        claims=[
            _claim(claim_id="claim_1", text="The law applies.", chunk_id="c0"),
            Claim(
                claim_id="claim_2",
                claim_text="The penalty is 10 days.",
                support_type=SupportType.UNSUPPORTED,
                evidence_spans=[],
                confidence=0.1,
            ),
        ],
        answer_text="The law applies. The penalty is 10 days.",
        unsupported_claims=["claim_2"],
        support_coverage=0.5,
    )
    contract = QueryContract(query_text="What does the law say?", answer_type="free_text")

    proof = compiler.compile(graph, contract)

    assert "The law applies." in proof.answer_text
    assert "(cite: c0)" in proof.answer_text
    assert "The penalty is 10 days." not in proof.answer_text
    assert proof.provenance_chain[0] == ["claim_1"]
    assert proof.dropped_claims[0].claim_id == "claim_2"


def test_compile_returns_fallback_when_graph_sparse() -> None:
    compiler = ProofCarryingCompiler(ProofCompilerConfig(min_support_coverage=0.9, allow_partial_answers=False))
    graph = ClaimGraph(
        claims=[
            Claim(
                claim_id="claim_1",
                claim_text="Unsupported claim.",
                support_type=SupportType.UNSUPPORTED,
                evidence_spans=[],
                confidence=0.1,
            )
        ],
        answer_text="Unsupported claim.",
        unsupported_claims=["claim_1"],
        support_coverage=0.0,
    )
    contract = QueryContract(query_text="What is the answer?", answer_type="free_text")

    proof = compiler.compile(graph, contract)

    assert proof.answer_text == ""
    assert proof.fallback_reason == "no_verified_claims"


def test_attach_provenance_maps_sentence_to_claims() -> None:
    compiler = ProofCarryingCompiler()
    proof = compiler.attach_provenance(
        "Sentence one. Sentence two.",
        [
            _claim(claim_id="claim_1", text="Sentence one.", chunk_id="c0"),
            _claim(claim_id="claim_2", text="Sentence two.", chunk_id="c1"),
        ],
    )

    assert proof[0] == ["claim_1"]
    assert proof[1] == ["claim_2"]


def test_ensure_fluency_collapses_whitespace() -> None:
    compiler = ProofCarryingCompiler()
    assert compiler.ensure_fluency("A   B") == "A B"
