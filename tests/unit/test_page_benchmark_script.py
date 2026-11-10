from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_pdf(path: Path, pages: list[str]) -> None:
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_score_page_benchmark_script_reports_f_beta_and_orphans(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    eval_path = tmp_path / "eval.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc_1", "doc_2"],
                        "gold_items": ["Alpha", "Beta"],
                        "wrong_document_risk": True,
                        "items": [
                            {
                                "id": "item-1",
                                "text": "Alpha only.",
                                "gold_page_ids": ["doc_1", "doc_2"],
                                "slots": [
                                    {"name": "title", "gold_page_ids": ["doc_1"]},
                                    {"name": "updated", "gold_page_ids": ["doc_2"]},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "answer": "Alpha only.",
                        "telemetry": {"used_page_ids": ["doc_1", "doc_3"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_page_benchmark.py",
            "--eval",
            str(eval_path),
            "--benchmark",
            str(benchmark_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "## All Cases" in result.stdout
    assert "## Trusted Tier" in result.stdout
    assert "Page-level F_beta(2.5)" in result.stdout
    assert "orphan=['doc_3']" in result.stdout
    assert "trust_tier=trusted" in result.stdout
    assert "gold_origin=manual_override" in result.stdout
    assert "item_coverage=0.5000" in result.stdout
    assert "Mean slot recall: 0.5000" in result.stdout
    assert "Evidence-family full-coverage rate: 0.0000" in result.stdout
    assert "Overprune violations: 1" in result.stdout
    assert "Wrong-document tagged case rate: 1.0000" in result.stdout


def test_build_page_trace_ledger_can_export_bounded_miss_pack(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    scaffold_path = tmp_path / "scaffold.json"
    raw_results_path = tmp_path / "raw_results.json"
    ledger_json = tmp_path / "ledger.json"
    ledger_md = tmp_path / "ledger.md"
    miss_pack_json = tmp_path / "miss_pack.json"
    miss_pack_md = tmp_path / "miss_pack.md"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q_explicit", "gold_page_ids": ["docA_2"], "trust_tier": "trusted"},
                    {"question_id": "q_title", "gold_page_ids": ["docB_1"], "trust_tier": "trusted"},
                    {"question_id": "q_same", "gold_page_ids": ["docC_3"], "trust_tier": "trusted"},
                    {
                        "question_id": "q_multi",
                        "gold_page_ids": ["docD_1", "docE_2"],
                        "trust_tier": "trusted",
                        "wrong_document_risk": True,
                    },
                    {"question_id": "q_ocr", "gold_page_ids": ["docF_1"], "trust_tier": "trusted", "ocr_risk": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    scaffold_path.write_text(
        json.dumps(
            {
                "records": [
                    {"question_id": "q_explicit", "question": "What is on page 2?", "route_family": "page_sensitive"},
                    {"question_id": "q_title", "question": "What is on the title page?", "route_family": "metadata"},
                    {"question_id": "q_same", "question": "Who is the claimant?", "route_family": "metadata"},
                    {
                        "question_id": "q_multi",
                        "question": "Compare the parties in both cases.",
                        "route_family": "comparison",
                    },
                    {
                        "question_id": "q_ocr",
                        "question": "What is the scanned cover page?",
                        "route_family": "ocr",
                        "ocr_risk": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_results_path.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q_explicit"},
                    "telemetry": {
                        "retrieved_page_ids": ["docA_1"],
                        "context_page_ids": ["docA_1"],
                        "used_page_ids": ["docA_1"],
                    },
                },
                {
                    "case": {"case_id": "q_title"},
                    "telemetry": {
                        "retrieved_page_ids": ["docB_2"],
                        "context_page_ids": ["docB_2"],
                        "used_page_ids": ["docB_2"],
                    },
                },
                {
                    "case": {"case_id": "q_same"},
                    "telemetry": {
                        "retrieved_page_ids": ["docC_2"],
                        "context_page_ids": ["docC_2"],
                        "used_page_ids": ["docC_2"],
                    },
                },
                {
                    "case": {"case_id": "q_multi"},
                    "telemetry": {
                        "retrieved_page_ids": ["docD_3"],
                        "context_page_ids": ["docD_3"],
                        "used_page_ids": ["docD_3"],
                    },
                },
                {
                    "case": {"case_id": "q_ocr"},
                    "telemetry": {
                        "retrieved_page_ids": ["docF_2"],
                        "context_page_ids": ["docF_2"],
                        "used_page_ids": ["docF_2"],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_page_trace_ledger.py",
            "--raw-results",
            str(raw_results_path),
            "--benchmark",
            str(benchmark_path),
            "--scaffold",
            str(scaffold_path),
            "--out-json",
            str(ledger_json),
            "--out-md",
            str(ledger_md),
            "--miss-pack-json",
            str(miss_pack_json),
            "--miss-pack-md",
            str(miss_pack_md),
            "--miss-pack-max-per-family",
            "1",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    miss_pack = json.loads(miss_pack_json.read_text(encoding="utf-8"))
    families = {case["miss_family"] for case in miss_pack["cases"]}
    assert families == {"explicit_page", "title_page", "same_doc", "mixed_doc", "ocr_risk"}
    assert miss_pack["summary"]["selected_case_count"] == 5


def test_score_page_benchmark_respects_include_qids_file(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    eval_path = tmp_path / "eval.json"
    include_qids = tmp_path / "include_qids.txt"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "gold_page_ids": ["doc_1"]},
                    {"question_id": "q2", "gold_page_ids": ["doc_2"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    eval_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "answer": "A", "telemetry": {"used_page_ids": ["doc_1"]}},
                    {"question_id": "q2", "answer": "B", "telemetry": {"used_page_ids": ["doc_9"]}},
                ]
            }
        ),
        encoding="utf-8",
    )
    include_qids.write_text("q1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_page_benchmark.py",
            "--eval",
            str(eval_path),
            "--benchmark",
            str(benchmark_path),
            "--include-qids-file",
            str(include_qids),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "- Cases: 1" in result.stdout
    assert "q2" not in result.stdout


def test_score_page_benchmark_accepts_predictions_json(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    predictions_path = tmp_path / "predictions.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "gold_page_ids": ["doc_2"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"question_id": "q1", "predicted_page_ids": ["doc_2"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_page_benchmark.py",
            "--predictions-json",
            str(predictions_path),
            "--benchmark",
            str(benchmark_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "## All Cases" in result.stdout
    assert "Page-level F_beta(2.5): 1.0000" in result.stdout


def test_cohere_page_falsifier_uses_stub_scores_on_cached_candidates(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    _write_pdf(docs_dir / "docA.pdf", ["cover page", "gold page"])

    benchmark_path = tmp_path / "benchmark.json"
    miss_pack_path = tmp_path / "miss_pack.json"
    ledger_path = tmp_path / "page_trace_ledger.json"
    stub_scores_path = tmp_path / "stub_scores.json"
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    out_baseline = tmp_path / "baseline_predictions.json"
    out_cohere = tmp_path / "cohere_predictions.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_page_ids": ["docA_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    miss_pack_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "qid": "q1",
                        "miss_family": "explicit_page",
                        "gold_pages": ["docA_2"],
                        "used_pages": ["docA_1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "qid": "q1",
                        "question": "What is on page 2?",
                        "used_pages": ["docA_1"],
                        "context_pages": ["docA_2"],
                        "retrieved_pages": ["docA_1", "docA_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    stub_scores_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "scores": {
                            "docA_1": 0.1,
                            "docA_2": 0.9,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/cohere_page_falsifier.py",
            "--miss-pack",
            str(miss_pack_path),
            "--page-trace-ledger",
            str(ledger_path),
            "--benchmark",
            str(benchmark_path),
            "--dataset-documents",
            str(docs_dir),
            "--stub-scores-json",
            str(stub_scores_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--out-baseline-predictions",
            str(out_baseline),
            "--out-cohere-predictions",
            str(out_cohere),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["summary"]["verdict"] == "continue_local_page_reranker"
    assert payload["cases"][0]["cohere_page_ids"] == ["docA_2"]
    assert payload["summary"]["delta_f_beta"] > 0.0


def test_cohere_page_falsifier_marks_missing_api_key_as_blocked(tmp_path: Path) -> None:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    _write_pdf(docs_dir / "docA.pdf", ["cover page", "gold page"])

    benchmark_path = tmp_path / "benchmark.json"
    miss_pack_path = tmp_path / "miss_pack.json"
    ledger_path = tmp_path / "page_trace_ledger.json"
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    benchmark_path.write_text(
        json.dumps({"cases": [{"question_id": "q1", "trust_tier": "trusted", "gold_page_ids": ["docA_2"]}]}),
        encoding="utf-8",
    )
    miss_pack_path.write_text(
        json.dumps({"cases": [{"qid": "q1", "gold_pages": ["docA_2"], "used_pages": ["docA_1"]}]}),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "qid": "q1",
                        "question": "What is on page 2?",
                        "used_pages": ["docA_1"],
                        "context_pages": ["docA_2"],
                        "retrieved_pages": ["docA_1", "docA_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/cohere_page_falsifier.py",
            "--miss-pack",
            str(miss_pack_path),
            "--page-trace-ledger",
            str(ledger_path),
            "--benchmark",
            str(benchmark_path),
            "--dataset-documents",
            str(docs_dir),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={key: value for key, value in os.environ.items() if key not in {"COHERE_API_KEY", "CO_API_KEY"}},
    )

    assert result.returncode == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["summary"]["verdict"] == "blocked_missing_api_key"
