from __future__ import annotations

import json
import pickle

from rag_challenge.ml.adversarial_grounding_rows import (
    AdversarialNegativeFamily,
    build_adversarial_grounding_rows,
    load_legal_ner_taxonomy_labels,
)
from rag_challenge.ml.grounding_dataset import (
    DocCandidateRecord,
    GroundingMlRow,
    PageCandidateRecord,
    PageRetrievalFeatureRecord,
    SupportFactFeatureRecord,
)
from rag_challenge.ml.training_scaffold import split_rows_by_holdout_doc_family


def _row(
    *,
    question_id: str,
    question: str,
    holdout_doc_family_key: str,
    doc_ref_signatures: list[str],
    label_page_ids: list[str],
    page_candidates: list[PageCandidateRecord],
    scope_mode: str = "single_field_single_doc",
) -> GroundingMlRow:
    return GroundingMlRow(
        question_id=question_id,
        question=question,
        answer_type="names",
        label_page_ids=label_page_ids,
        label_source="reviewed",
        scope_mode=scope_mode,
        doc_ref_signatures=doc_ref_signatures,
        holdout_doc_family_key=holdout_doc_family_key,
        doc_candidates=[
            DocCandidateRecord(
                doc_id=candidate.doc_id,
                page_candidate_count=1,
                candidate_sources=["legacy_context"],
            )
            for candidate in page_candidates
        ],
        page_candidates=page_candidates,
        support_fact_features=SupportFactFeatureRecord(doc_ref_count=1),
        page_retrieval_features=PageRetrievalFeatureRecord(legacy_retrieved_page_count=len(page_candidates)),
        source_paths={"legacy_raw_results": "/tmp/legacy.json"},
    )


def test_build_adversarial_grounding_rows_emits_expected_negative_families(tmp_path) -> None:
    legal_ner_dir = tmp_path / "legal_ner"
    data_dir = legal_ner_dir / "data"
    data_dir.mkdir(parents=True)
    with (data_dir / "combined_class_labels.pkl").open("wb") as handle:
        pickle.dump(["PETITIONER", "DATE", "PROVISION", "STATUTE"], handle)

    row_a = _row(
        question_id="q1",
        question="Who were the claimants under the Operating Law 2018?",
        holdout_doc_family_key="operating law",
        doc_ref_signatures=["operating law"],
        label_page_ids=["doc-a_2"],
        page_candidates=[
            PageCandidateRecord(page_id="doc-a_2", doc_id="doc-a", page_num=2),
            PageCandidateRecord(page_id="doc-a_3", doc_id="doc-a", page_num=3),
            PageCandidateRecord(page_id="doc-b_1", doc_id="doc-b", page_num=1),
        ],
    )
    row_b = _row(
        question_id="q2",
        question="Which authority issued the Operating Regulation 2019?",
        holdout_doc_family_key="operating law",
        doc_ref_signatures=["operating regulation"],
        label_page_ids=["doc-c_1"],
        page_candidates=[
            PageCandidateRecord(page_id="doc-c_1", doc_id="doc-c", page_num=1),
            PageCandidateRecord(page_id="doc-c_2", doc_id="doc-c", page_num=2),
        ],
    )
    row_c = _row(
        question_id="q3",
        question="Which pages support this unavailable claim?",
        holdout_doc_family_key="unsupported",
        doc_ref_signatures=["unsupported"],
        label_page_ids=[],
        page_candidates=[PageCandidateRecord(page_id="doc-u_1", doc_id="doc-u", page_num=1)],
        scope_mode="negative_unanswerable",
    )

    rows = build_adversarial_grounding_rows(
        [row_a, row_b, row_c],
        legal_ner_taxonomy_path=legal_ner_dir,
    )

    by_family = {row.negative_family for row in rows if row.question_id == "q1"}
    assert AdversarialNegativeFamily.SAME_DOC_NEARBY_PAGE in by_family
    assert AdversarialNegativeFamily.SAME_FAMILY_WRONG_LAW in by_family
    assert AdversarialNegativeFamily.TITLE_AUTHORITY_CONFUSER in by_family
    unsupported = [row for row in rows if row.question_id == "q3"]
    assert any(row.negative_family is AdversarialNegativeFamily.CONTRADICTION_UNSUPPORTED for row in unsupported)
    assert all(row.lineage.source_paths["legacy_raw_results"] == "/tmp/legacy.json" for row in rows)
    assert any("PETITIONER" in row.taxonomy_labels for row in rows if row.question_id == "q1")


def test_split_rows_by_holdout_doc_family_keeps_groups_intact() -> None:
    rows = [
        _row(
            question_id="q1",
            question="Question 1",
            holdout_doc_family_key="family-a",
            doc_ref_signatures=["family-a"],
            label_page_ids=["doc-a_1"],
            page_candidates=[PageCandidateRecord(page_id="doc-a_1", doc_id="doc-a", page_num=1)],
        ),
        _row(
            question_id="q2",
            question="Question 2",
            holdout_doc_family_key="family-a",
            doc_ref_signatures=["family-a"],
            label_page_ids=["doc-a_2"],
            page_candidates=[PageCandidateRecord(page_id="doc-a_2", doc_id="doc-a", page_num=2)],
        ),
        _row(
            question_id="q3",
            question="Question 3",
            holdout_doc_family_key="family-b",
            doc_ref_signatures=["family-b"],
            label_page_ids=["doc-b_1"],
            page_candidates=[PageCandidateRecord(page_id="doc-b_1", doc_id="doc-b", page_num=1)],
        ),
    ]

    train_rows, dev_rows = split_rows_by_holdout_doc_family(rows, dev_ratio=0.5, seed=1062)

    train_keys = {row.holdout_doc_family_key for row in train_rows}
    dev_keys = {row.holdout_doc_family_key for row in dev_rows}
    assert train_keys.isdisjoint(dev_keys)
    assert train_keys | dev_keys == {"family-a", "family-b"}


def test_export_grounding_dataset_populates_holdout_keys(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.json"
    sidecar_path = tmp_path / "sidecar.json"
    golden_path = tmp_path / "golden.json"
    benchmark_path = tmp_path / "benchmark.json"
    output_dir = tmp_path / "out"

    raw_payload = [
        {
            "answer_text": "annual return",
            "case": {
                "case_id": "qid-1",
                "question": "According to Article 16 of the Operating Law 2018, what must be filed?",
                "answer_type": "name",
            },
            "telemetry": {
                "doc_refs": ["Operating Law 2018"],
                "retrieved_page_ids": ["law_16"],
                "context_page_ids": ["law_16"],
                "cited_page_ids": ["law_16"],
                "used_page_ids": ["law_16"],
                "retrieved_chunk_ids": ["law:15:0:a"],
                "context_chunk_ids": ["law:15:0:a"],
                "cited_chunk_ids": ["law:15:0:a"],
                "chunk_snippets": {"law:15:0:a": "Article 16 requires the filing of the annual return."},
            },
        }
    ]
    legacy_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    sidecar_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    golden_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "qid-1",
                    "question": "According to Article 16 of the Operating Law 2018, what must be filed?",
                    "answer_type": "name",
                    "golden_answer": "annual return",
                    "golden_page_ids": ["law_16"],
                }
            ]
        ),
        encoding="utf-8",
    )
    benchmark_path.write_text(
        json.dumps({"cases": [{"question_id": "qid-1", "gold_page_ids": ["law_16"], "trust_tier": "trusted"}]}),
        encoding="utf-8",
    )

    from rag_challenge.ml.grounding_dataset import export_grounding_ml_dataset

    export_grounding_ml_dataset(
        legacy_raw_results_path=legacy_path,
        sidecar_raw_results_path=sidecar_path,
        golden_labels_path=golden_path,
        page_benchmark_path=benchmark_path,
        suspect_labels_path=None,
        output_dir=output_dir,
    )

    train_line = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = GroundingMlRow.model_validate_json(train_line)
    assert row.doc_ref_signatures == ["operating law"]
    assert row.holdout_doc_family_key == "operating law"


def test_load_legal_ner_taxonomy_labels_reads_all_pickles(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    for name, payload in {
        "combined_class_labels.pkl": ["DATE", "ORG"],
        "judgement_class_labels.pkl": ["COURT"],
    }.items():
        with (data_dir / name).open("wb") as handle:
            pickle.dump(payload, handle)

    labels = load_legal_ner_taxonomy_labels(tmp_path)

    assert labels == ["COURT", "DATE", "ORG"]
