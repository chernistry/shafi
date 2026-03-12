from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.select_anchor_slice_qids import load_anchor_slice_rows, select_qids
except ModuleNotFoundError:  # pragma: no cover - direct script execution uses a different import root
    from select_anchor_slice_qids import load_anchor_slice_rows, select_qids

JsonDict = dict[str, Any]


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _qids_from_args(args: argparse.Namespace) -> set[str]:
    out: set[str] = set()
    for raw in args.extra_qid:
        text = str(raw).strip()
        if text:
            out.add(text)
    if args.extra_qids_file is not None:
        for line in args.extra_qids_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                out.add(text)
    return out


def _write_qids(path: Path, qids: list[str]) -> None:
    path.write_text("\n".join(qids) + ("\n" if qids else ""), encoding="utf-8")


def _write_summary(
    *,
    path: Path,
    label: str,
    baseline_label: str,
    selected_qids: list[str],
    selection_json: Path,
    submission_path: Path,
    raw_results_path: Path,
    preflight_path: Path,
    gate_report_path: Path,
    anchor_slice_path: Path,
) -> None:
    payload: JsonDict = {
        "label": label,
        "baseline_label": baseline_label,
        "selected_qids": selected_qids,
        "selection_json": str(selection_json),
        "candidate_submission": str(submission_path),
        "candidate_raw_results": str(raw_results_path),
        "candidate_preflight": str(preflight_path),
        "gate_report": str(gate_report_path),
        "anchor_slice_report": str(anchor_slice_path),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a bounded support-only counterfactual candidate from an anchor-slice report.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--anchor-slice-json", type=Path, required=True)
    parser.add_argument("--include-status", action="append", default=[])
    parser.add_argument("--exclude-status", action="append", default=[])
    parser.add_argument("--exclude-qid", action="append", default=[])
    parser.add_argument("--extra-qid", action="append", default=[])
    parser.add_argument("--extra-qids-file", type=Path, default=None)
    parser.add_argument("--require-no-answer-change", action="store_true")
    parser.add_argument("--require-used-support", action="store_true")
    parser.add_argument("--answer-source-submission", type=Path, required=True)
    parser.add_argument("--answer-source-raw-results", type=Path, required=True)
    parser.add_argument("--answer-source-preflight", type=Path, required=True)
    parser.add_argument("--page-source-submission", type=Path, required=True)
    parser.add_argument("--page-source-raw-results", type=Path, required=True)
    parser.add_argument("--page-source-preflight", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--seed-qids-file", type=Path, required=True)
    parser.add_argument("--research-dir", type=Path, required=True)
    parser.add_argument("--ledger-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    research_dir = args.research_dir
    research_dir.mkdir(parents=True, exist_ok=True)

    rows = load_anchor_slice_rows(args.anchor_slice_json)
    selected_qids, selection_report = select_qids(
        rows=rows,
        include_statuses={str(status).strip() for status in args.include_status if str(status).strip()},
        exclude_statuses={str(status).strip() for status in args.exclude_status if str(status).strip()},
        require_no_answer_change=bool(args.require_no_answer_change),
        require_used_support=bool(args.require_used_support),
        excluded_qids={str(qid).strip() for qid in args.exclude_qid if str(qid).strip()},
    )
    extra_qids = _qids_from_args(args)
    final_qids = list(dict.fromkeys([*selected_qids, *sorted(extra_qids)]))

    qids_path = research_dir / f"anchor_qids_{args.label}.txt"
    selection_json = research_dir / f"anchor_qids_{args.label}.json"
    _write_qids(qids_path, final_qids)
    selection_payload = {
        **selection_report,
        "extra_qids": sorted(extra_qids),
        "final_qids": final_qids,
    }
    selection_json.write_text(json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    submission_path = root / "platform_runs" / "warmup" / f"submission_{args.label}.json"
    raw_results_path = root / "platform_runs" / "warmup" / f"raw_results_{args.label}.json"
    preflight_path = root / "platform_runs" / "warmup" / f"preflight_summary_{args.label}.json"
    counterfactual_report = research_dir / f"counterfactual_{args.label}_report.json"
    gate_report = research_dir / f"experiment_gate_{args.label}_vs_{args.baseline_label}.md"
    anchor_slice_report = research_dir / f"anchor_slice_{args.label}_vs_{args.baseline_label}.md"
    anchor_slice_json = research_dir / f"anchor_slice_{args.label}_vs_{args.baseline_label}.json"
    summary_path = research_dir / f"bounded_counterfactual_cycle_{args.label}.json"

    build_cmd = [
        sys.executable,
        "scripts/build_counterfactual_candidate.py",
        "--answer-source-submission",
        str(args.answer_source_submission),
        "--answer-source-raw-results",
        str(args.answer_source_raw_results),
        "--answer-source-preflight",
        str(args.answer_source_preflight),
        "--page-source-submission",
        str(args.page_source_submission),
        "--page-source-raw-results",
        str(args.page_source_raw_results),
        "--page-source-preflight",
        str(args.page_source_preflight),
        "--page-source-page-qids-file",
        str(qids_path),
        "--out-submission",
        str(submission_path),
        "--out-raw-results",
        str(raw_results_path),
        "--out-preflight",
        str(preflight_path),
        "--out-report",
        str(counterfactual_report),
    ]
    _run(build_cmd, cwd=root)

    gate_cmd = [
        sys.executable,
        "scripts/run_experiment_gate.py",
        "--label",
        args.label,
        "--baseline-label",
        args.baseline_label,
        "--baseline-submission",
        str(args.answer_source_submission),
        "--candidate-submission",
        str(submission_path),
        "--baseline-raw-results",
        str(args.answer_source_raw_results),
        "--candidate-raw-results",
        str(raw_results_path),
        "--benchmark",
        str(args.benchmark),
        "--scaffold",
        str(args.scaffold),
        "--candidate-scaffold",
        str(args.scaffold),
        "--baseline-preflight",
        str(args.answer_source_preflight),
        "--candidate-preflight",
        str(preflight_path),
        "--out",
        str(gate_report),
        "--ledger-json",
        str(args.ledger_json),
    ]
    for qid in args.seed_qids_file.read_text(encoding="utf-8").splitlines():
        text = qid.strip()
        if text and not text.startswith("#"):
            gate_cmd.extend(["--seed-qid", text])
    _run(gate_cmd, cwd=root)

    slice_cmd = [
        sys.executable,
        "scripts/build_anchor_slice_report.py",
        "--baseline-label",
        args.baseline_label,
        "--candidate-label",
        args.label,
        "--baseline-submission",
        str(args.answer_source_submission),
        "--candidate-submission",
        str(submission_path),
        "--baseline-raw-results",
        str(args.answer_source_raw_results),
        "--candidate-raw-results",
        str(raw_results_path),
        "--baseline-scaffold",
        str(args.scaffold),
        "--candidate-scaffold",
        str(args.scaffold),
        "--qids-file",
        str(args.seed_qids_file),
        "--out",
        str(anchor_slice_report),
        "--json-out",
        str(anchor_slice_json),
    ]
    _run(slice_cmd, cwd=root)

    _write_summary(
        path=summary_path,
        label=args.label,
        baseline_label=args.baseline_label,
        selected_qids=final_qids,
        selection_json=selection_json,
        submission_path=submission_path,
        raw_results_path=raw_results_path,
        preflight_path=preflight_path,
        gate_report_path=gate_report,
        anchor_slice_path=anchor_slice_report,
    )


if __name__ == "__main__":
    main()
