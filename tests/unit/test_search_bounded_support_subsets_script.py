from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _submission_record(*, qid: str, answer: object, page_id: str) -> dict[str, object]:
    doc_id, page_num = page_id.rsplit("_", 1)
    return {
        "question_id": qid,
        "answer": answer,
        "telemetry": {
            "model_name": "baseline-model",
            "retrieval": {
                "retrieved_chunk_pages": [
                    {"doc_id": doc_id, "page_numbers": [int(page_num)]},
                ]
            },
        },
    }


def _raw_record(*, qid: str, answer_text: str, used_page_id: str, context_page_id: str) -> dict[str, object]:
    return {
        "case": {"case_id": qid, "question": qid, "answer_type": "boolean"},
        "answer_text": answer_text,
        "telemetry": {
            "used_page_ids": [used_page_id],
            "context_page_ids": [context_page_id],
        },
        "total_ms": 10,
    }


def _preflight(*, p95: int) -> dict[str, object]:
    return {
        "phase": "warmup",
        "page_count_distribution": {"min": 1, "p50": 1, "p95": p95, "max": p95},
        "code_archive_sha256": "code",
        "questions_sha256": "questions",
        "documents_zip_sha256": "docs",
        "pdf_count": 1,
        "phase_collection_name": "collection",
        "qdrant_point_count": 1,
        "truth_audit_workbook_path": "workbook.md",
    }


def _scaffold(*, qid: str, gold_page_id: str, candidate_page_id: str | None = None) -> dict[str, object]:
    gold_doc, gold_page = gold_page_id.rsplit("_", 1)
    previews = [
        {"doc_id": gold_doc, "page": int(gold_page), "doc_title": "Case Title"},
    ]
    if candidate_page_id is not None:
        cand_doc, cand_page = candidate_page_id.rsplit("_", 1)
        previews.append({"doc_id": cand_doc, "page": int(cand_page), "doc_title": "Case Title"})
    return {
        "records": [
            {
                "question_id": qid,
                "minimal_required_support_pages": [gold_page_id],
                "support_page_previews": previews,
                "resolved_doc_titles": {gold_doc: "Case Title"},
            }
        ]
    }


def test_search_bounded_support_subsets_picks_gold_page_subset(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    baseline_preflight = tmp_path / "baseline_preflight.json"
    baseline_scaffold = tmp_path / "baseline_scaffold.json"
    page_submission = tmp_path / "page_submission.json"
    page_raw = tmp_path / "page_raw.json"
    page_preflight = tmp_path / "page_preflight.json"
    candidate_scaffold = tmp_path / "candidate_scaffold.json"
    benchmark = tmp_path / "benchmark.json"
    anchor_slice = tmp_path / "anchor_slice.json"
    seed_qids = tmp_path / "seed_qids.txt"
    out_md = tmp_path / "search.md"
    out_json = tmp_path / "search.json"
    best_dir = tmp_path / "best"

    qid = "q1"
    baseline_submission.write_text(
        json.dumps(
            {"architecture_summary": {}, "answers": [_submission_record(qid=qid, answer=False, page_id="doc_bad_1")]}
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps([_raw_record(qid=qid, answer_text="False", used_page_id="doc_bad_1", context_page_id="doc_bad_1")]),
        encoding="utf-8",
    )
    baseline_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")
    baseline_scaffold.write_text(json.dumps(_scaffold(qid=qid, gold_page_id="doc_good_2")), encoding="utf-8")

    page_submission.write_text(
        json.dumps(
            {"architecture_summary": {}, "answers": [_submission_record(qid=qid, answer=False, page_id="doc_good_2")]}
        ),
        encoding="utf-8",
    )
    page_raw.write_text(
        json.dumps(
            [_raw_record(qid=qid, answer_text="False", used_page_id="doc_good_2", context_page_id="doc_good_2")]
        ),
        encoding="utf-8",
    )
    page_preflight.write_text(json.dumps(_preflight(p95=1)), encoding="utf-8")
    candidate_scaffold.write_text(
        json.dumps(_scaffold(qid=qid, gold_page_id="doc_good_2", candidate_page_id="doc_good_2")),
        encoding="utf-8",
    )

    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": qid,
                        "gold_page_ids": ["doc_good_2"],
                        "gold_items": [],
                        "items": [],
                        "wrong_document_risk": False,
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    anchor_slice.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "question_id": qid,
                        "status": "support_improved",
                        "answer_changed": False,
                        "candidate_used_hit": True,
                        "candidate_used_equivalent_hit": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seed_qids.write_text("q1\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/search_bounded_support_subsets.py",
            "--baseline-label",
            "baseline_x",
            "--page-source-label",
            "page_source_y",
            "--anchor-slice-json",
            str(anchor_slice),
            "--include-status",
            "support_improved",
            "--require-no-answer-change",
            "--require-used-support",
            "--seed-qids-file",
            str(seed_qids),
            "--benchmark",
            str(benchmark),
            "--baseline-submission",
            str(baseline_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--baseline-preflight",
            str(baseline_preflight),
            "--baseline-scaffold",
            str(baseline_scaffold),
            "--page-source-submission",
            str(page_submission),
            "--page-source-raw-results",
            str(page_raw),
            "--page-source-preflight",
            str(page_preflight),
            "--candidate-scaffold",
            str(candidate_scaffold),
            "--out",
            str(out_md),
            "--json-out",
            str(out_json),
            "--best-label",
            "best_candidate",
            "--best-out-dir",
            str(best_dir),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/shafi",
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["results"][0]["qids"] == ["q1"]
    assert payload["results"][0]["recommendation"] == "PROMISING"
    assert payload["results"][0]["benchmark_trusted_candidate"] == 1.0
    assert payload["submission_policy"] == "NO_SUBMIT_WITHOUT_USER_APPROVAL"

    best_submission = json.loads((best_dir / "submission_best_candidate.json").read_text(encoding="utf-8"))
    best_answer = best_submission["answers"][0]
    assert best_answer["answer"] is False
    assert best_answer["telemetry"]["retrieval"]["retrieved_chunk_pages"][0]["doc_id"] == "doc_good"
