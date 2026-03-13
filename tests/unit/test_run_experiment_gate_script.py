from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_run_experiment_gate_script_reports_and_appends_ledger(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    benchmark = tmp_path / "benchmark.json"
    scaffold = tmp_path / "scaffold.json"
    baseline_preflight = tmp_path / "baseline_preflight.json"
    candidate_preflight = tmp_path / "candidate_preflight.json"
    report_path = tmp_path / "report.md"
    ledger_path = tmp_path / "ledger.json"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [1]}]}},
                    },
                    {
                        "question_id": "q2",
                        "answer": "Alpha",
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [3]}]}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [2]}]}},
                    },
                    {
                        "question_id": "q2",
                        "answer": "Alpha",
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [3]}]}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc_2"],
                        "gold_items": [],
                        "wrong_document_risk": False,
                        "items": [],
                    },
                    {
                        "question_id": "q2",
                        "trust_tier": "suspect",
                        "gold_origin": "seeded_eval",
                        "gold_page_ids": ["doc_3"],
                        "gold_items": [],
                        "wrong_document_risk": False,
                        "items": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "minimal_required_support_pages": ["doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_1"], "context_page_ids": ["doc_1"]},
                    "total_ms": 10,
                },
                {
                    "case": {"case_id": "q2"},
                    "answer_text": "Alpha",
                    "telemetry": {"used_page_ids": ["doc_3"], "context_page_ids": ["doc_3"]},
                    "total_ms": 10,
                },
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_2"], "context_page_ids": ["doc_2"]},
                    "total_ms": 10,
                },
                {
                    "case": {"case_id": "q2"},
                    "answer_text": "Alpha",
                    "telemetry": {"used_page_ids": ["doc_3"], "context_page_ids": ["doc_3"]},
                    "total_ms": 10,
                },
            ]
        ),
        encoding="utf-8",
    )
    baseline_preflight.write_text(json.dumps({"page_count_distribution": {"p95": 3}}), encoding="utf-8")
    candidate_preflight.write_text(json.dumps({"page_count_distribution": {"p95": 3}}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_gate.py",
            "--label",
            "candidate-a",
            "--baseline-label",
            "baseline-a",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--candidate-raw-results",
            str(candidate_raw),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(scaffold),
            "--baseline-preflight",
            str(baseline_preflight),
            "--candidate-preflight",
            str(candidate_preflight),
            "--seed-qid",
            "q1",
            "--out",
            str(report_path),
            "--ledger-json",
            str(ledger_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Experiment Gate Report" in report
    assert "- Recommendation: `PROMISING`" in report
    assert "- Answer changes vs baseline: `0`" in report
    assert "- Retrieval-page projection changes vs baseline: `1`" in report
    assert "- Baseline trusted F_beta(2.5): `0.0000`" in report
    assert "- Candidate trusted F_beta(2.5): `1.0000`" in report
    assert "Improved IDs: `q1`" in report

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    experiments = ledger["experiments"]
    assert len(experiments) == 1
    assert experiments[0]["label"] == "candidate-a"
    assert experiments[0]["recommendation"] == "PROMISING"
    assert experiments[0]["improved_seed_cases"] == ["q1"]


def test_run_experiment_gate_script_treats_same_title_same_page_as_equivalent(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    benchmark = tmp_path / "benchmark.json"
    baseline_scaffold = tmp_path / "baseline_scaffold.json"
    candidate_scaffold = tmp_path / "candidate_scaffold.json"
    report_path = tmp_path / "report.md"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q43",
                        "answer": ["Alpha LLC"],
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-old", "page_numbers": [1]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q43",
                        "answer": ["Alpha LLC"],
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-new", "page_numbers": [1]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q43",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc-old_1"],
                        "gold_items": [],
                        "wrong_document_risk": False,
                        "items": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q43",
                        "minimal_required_support_pages": ["doc-old_1"],
                        "support_page_previews": [
                            {
                                "doc_id": "doc-old",
                                "page": 1,
                                "doc_title": "Case A Alpha v Beta",
                            }
                        ],
                        "resolved_doc_titles": {"doc-old": "Case A Alpha v Beta"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q43",
                        "support_page_previews": [
                            {
                                "doc_id": "doc-new",
                                "page": 1,
                                "doc_title": "Case A Alpha v Beta",
                            }
                        ],
                        "resolved_doc_titles": {"doc-new": "Case A Alpha v Beta"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q43"},
                    "answer_text": '["Alpha LLC"]',
                    "telemetry": {"used_page_ids": ["doc-old_1"], "context_page_ids": ["doc-old_1"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q43"},
                    "answer_text": '["Alpha LLC"]',
                    "telemetry": {"used_page_ids": ["doc-new_1"], "context_page_ids": ["doc-new_1"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_gate.py",
            "--label",
            "candidate-equiv",
            "--baseline-label",
            "baseline-a",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--candidate-raw-results",
            str(candidate_raw),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(baseline_scaffold),
            "--candidate-scaffold",
            str(candidate_scaffold),
            "--seed-qid",
            "q43",
            "--out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Equivalent seed cases: `1`" in report
    assert "Equivalent IDs: `q43`" in report
    assert "- Regressed seed cases: `0`" in report
    assert "candidate_used_equivalent=True" in report


def test_run_experiment_gate_script_treats_stale_gold_doc_hash_as_equivalent_by_resolved_title(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    benchmark = tmp_path / "benchmark.json"
    baseline_scaffold = tmp_path / "baseline_scaffold.json"
    candidate_scaffold = tmp_path / "candidate_scaffold.json"
    report_path = tmp_path / "report.md"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q398",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-old", "page_numbers": [3]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q398",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc-live", "page_numbers": [1]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q398",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc-stale_1"],
                        "gold_items": [],
                        "wrong_document_risk": False,
                        "items": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q398",
                        "minimal_required_support_pages": ["doc-stale_1"],
                        "resolved_doc_titles": {"doc-live": "Olexa v Odon [2025] DIFC SCT 295"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q398",
                        "support_page_previews": [
                            {
                                "doc_id": "doc-live",
                                "page": 1,
                                "doc_title": "Olexa v Odon [2025] DIFC SCT 295",
                                "snippet": "Title page",
                            }
                        ],
                        "resolved_doc_titles": {"doc-live": "Olexa v Odon [2025] DIFC SCT 295"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q398"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc-old_3"], "context_page_ids": ["doc-old_3"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q398"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc-live_1"], "context_page_ids": ["doc-live_1"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_gate.py",
            "--label",
            "candidate-stale-equiv",
            "--baseline-label",
            "baseline-stale-equiv",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--candidate-raw-results",
            str(candidate_raw),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(baseline_scaffold),
            "--candidate-scaffold",
            str(candidate_scaffold),
            "--seed-qid",
            "q398",
            "--out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- Equivalent seed cases: `1`" in report
    assert "Equivalent IDs: `q398`" in report
    assert "candidate_used_equivalent=True" in report


def test_run_experiment_gate_script_accepts_seed_qid_file(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    benchmark = tmp_path / "benchmark.json"
    scaffold = tmp_path / "scaffold.json"
    seed_qids = tmp_path / "seed_qids.txt"
    report_path = tmp_path / "report.md"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [1]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [2]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc_2"],
                        "gold_items": [],
                        "wrong_document_risk": False,
                        "items": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "minimal_required_support_pages": ["doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_1"], "context_page_ids": ["doc_1"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_2"], "context_page_ids": ["doc_2"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    seed_qids.write_text("# comment\nq1\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_gate.py",
            "--label",
            "candidate-file-seed",
            "--baseline-label",
            "baseline-file-seed",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--candidate-raw-results",
            str(candidate_raw),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(scaffold),
            "--seed-qid-file",
            str(seed_qids),
            "--out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Improved IDs: `q1`" in report


def test_run_experiment_gate_uses_default_seed_qids_file_next_to_benchmark(tmp_path: Path) -> None:
    baseline_submission = tmp_path / "baseline_submission.json"
    candidate_submission = tmp_path / "candidate_submission.json"
    baseline_raw = tmp_path / "baseline_raw.json"
    candidate_raw = tmp_path / "candidate_raw.json"
    benchmark = tmp_path / "benchmark.json"
    benchmark_seed_qids = tmp_path / "benchmark_qids.txt"
    scaffold = tmp_path / "scaffold.json"
    report_path = tmp_path / "report.md"

    baseline_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [1]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_submission.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "question_id": "q1",
                        "answer": False,
                        "telemetry": {"retrieval": {"retrieved_chunk_pages": [{"doc_id": "doc", "page_numbers": [2]}]}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question_id": "q1",
                        "trust_tier": "trusted",
                        "gold_origin": "manual_override",
                        "gold_page_ids": ["doc_2"],
                        "gold_items": [],
                        "wrong_document_risk": False,
                        "items": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    benchmark_seed_qids.write_text("q1\n", encoding="utf-8")
    scaffold.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "question_id": "q1",
                        "minimal_required_support_pages": ["doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_1"], "context_page_ids": ["doc_1"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    candidate_raw.write_text(
        json.dumps(
            [
                {
                    "case": {"case_id": "q1"},
                    "answer_text": "False",
                    "telemetry": {"used_page_ids": ["doc_2"], "context_page_ids": ["doc_2"]},
                    "total_ms": 10,
                }
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_gate.py",
            "--label",
            "candidate-default-seed",
            "--baseline-label",
            "baseline-default-seed",
            "--baseline-submission",
            str(baseline_submission),
            "--candidate-submission",
            str(candidate_submission),
            "--baseline-raw-results",
            str(baseline_raw),
            "--candidate-raw-results",
            str(candidate_raw),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(scaffold),
            "--out",
            str(report_path),
        ],
        cwd="/Users/sasha/IdeaProjects/personal_projects/rag_challenge",
        capture_output=True,
        text=True,
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Improved IDs: `q1`" in report
