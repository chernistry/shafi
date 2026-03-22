from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _resolve(root: Path, raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _answers_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    answers_obj = payload.get("answers")
    if not isinstance(answers_obj, list):
        raise ValueError(f"Submission at {path} is missing answers[]")
    answers = cast("list[object]", answers_obj)
    out: dict[str, JsonDict] = {}
    for raw in answers:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _answer_value(record: JsonDict) -> str:
    return json.dumps(record.get("answer"), ensure_ascii=False, sort_keys=True)


def _candidate_qids(
    *,
    baseline_submission: Path,
    rider_source_submission: Path,
    scaffold: Path,
) -> list[str]:
    baseline = _answers_by_id(baseline_submission)
    rider_source = _answers_by_id(rider_source_submission)
    scaffold_obj = _load_json(scaffold)
    records_obj = scaffold_obj.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {scaffold} is missing records[]")
    records = cast("list[object]", records_obj)
    incorrect_qids = {
        str(cast("JsonDict", raw).get("question_id") or "").strip()
        for raw in records
        if isinstance(raw, dict) and str(cast("JsonDict", raw).get("manual_verdict") or "").strip() == "incorrect"
    }
    out: list[str] = []
    for qid, baseline_record in baseline.items():
        rider_record = rider_source.get(qid)
        if rider_record is None or qid not in incorrect_qids:
            continue
        if _answer_value(baseline_record) != _answer_value(rider_record):
            out.append(qid)
    return sorted(out)


def _write_qids(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_optional(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> bool:
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            timeout=max(1, timeout_seconds),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return True


def _subset_label(base_label: str, qids: list[str]) -> str:
    suffix = "_".join(qid[:6] for qid in qids)
    return f"{base_label}_plus_{suffix}"


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _recommendation_rank(value: object) -> int:
    recommendation = str(value or "").strip().upper()
    order = {
        "PROMISING": 3,
        "SAFE": 2,
        "EXPERIMENTAL_NO_SUBMIT": 1,
        "REJECT": 0,
    }
    return order.get(recommendation, -1)


def _combined_score(row: JsonDict) -> tuple[int, int, float, float, int, float, float, int, int, str]:
    return (
        1 if bool(row.get("lineage_ok")) else 0,
        _recommendation_rank(row.get("recommendation")),
        _coerce_float(row.get("hidden_g_trusted_delta")),
        _coerce_float(row.get("hidden_g_all_delta")),
        _coerce_int(row.get("resolved_incorrect_count")),
        _coerce_float(row.get("judge_pass_delta")),
        _coerce_float(row.get("judge_grounding_delta")),
        -_coerce_int(row.get("answer_drift")),
        -_coerce_int(row.get("page_drift")),
        str(row.get("label") or ""),
    )


def _render_md(*, rows: list[JsonDict], source_qids: list[str]) -> str:
    lines = [
        "# Exactness Rider Subset Search",
        "",
        f"- `candidate_qids`: `{source_qids}`",
        f"- `subsets`: `{len(rows)}`",
        "- `submission_policy`: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | Label | QIDs | Gate | Lineage | Hidden-G Trusted Δ | Hidden-G All Δ | Resolved Incorrect | Judge Pass Δ | Judge Grounding Δ | Answer Drift | Page Drift |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(sorted(rows, key=_combined_score, reverse=True), start=1):
        qids = cast("list[str]", row.get("qids") or [])
        judge_status = str(row.get("judge_status") or "not_run")
        lines.append(
            "| "
            f"{index} | `{row['label']}` | `{','.join(qids)}` | "
            f"`{row.get('recommendation', 'unknown')}` | `{row['lineage_ok']}` | "
            f"{_coerce_float(row.get('hidden_g_trusted_delta')):+.4f} | "
            f"{_coerce_float(row.get('hidden_g_all_delta')):+.4f} | "
            f"{_coerce_int(row.get('resolved_incorrect_count'))} | "
            f"{_coerce_float(row.get('judge_pass_delta')):+.4f} ({judge_status}) | "
            f"{_coerce_float(row.get('judge_grounding_delta')):+.4f} ({judge_status}) | "
            f"{_coerce_int(row.get('answer_drift'))} | {_coerce_int(row.get('page_drift'))} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search page-stable exactness rider subsets on top of a support candidate.")
    parser.add_argument("--build-baseline-label", required=True)
    parser.add_argument("--build-baseline-submission", required=True)
    parser.add_argument("--build-baseline-raw-results", required=True)
    parser.add_argument("--build-baseline-preflight", required=True)
    parser.add_argument("--compare-baseline-label", default=None)
    parser.add_argument("--compare-baseline-submission", default=None)
    parser.add_argument("--compare-baseline-raw-results", default=None)
    parser.add_argument("--compare-baseline-preflight", default=None)
    parser.add_argument("--rider-source-submission", required=True)
    parser.add_argument("--rider-source-raw-results", required=True)
    parser.add_argument("--rider-source-preflight", required=True)
    parser.add_argument("--truth-audit-scaffold", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-qid", action="append", default=[])
    parser.add_argument("--candidate-qids-file", type=Path, default=None)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--judge-top-k", type=int, default=0)
    parser.add_argument("--judge-timeout-seconds", type=int, default=30)
    return parser.parse_args()


def _delta(candidate: JsonDict, baseline: JsonDict, field: str) -> float:
    return _coerce_float(candidate.get(field)) - _coerce_float(baseline.get(field))


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    build_baseline_label = str(args.build_baseline_label)
    build_baseline_submission = _resolve(root, args.build_baseline_submission)
    build_baseline_raw_results = _resolve(root, args.build_baseline_raw_results)
    build_baseline_preflight = _resolve(root, args.build_baseline_preflight)

    compare_baseline_label = str(args.compare_baseline_label or build_baseline_label)
    compare_baseline_submission = _resolve(root, args.compare_baseline_submission or args.build_baseline_submission)
    compare_baseline_raw_results = _resolve(root, args.compare_baseline_raw_results or args.build_baseline_raw_results)
    compare_baseline_preflight = _resolve(root, args.compare_baseline_preflight or args.build_baseline_preflight)

    rider_source_submission = _resolve(root, args.rider_source_submission)
    rider_source_raw_results = _resolve(root, args.rider_source_raw_results)
    rider_source_preflight = _resolve(root, args.rider_source_preflight)
    truth_audit_scaffold = _resolve(root, args.truth_audit_scaffold)
    benchmark = _resolve(root, args.benchmark)
    questions = _resolve(root, args.questions)
    docs_dir = _resolve(root, args.docs_dir)
    out_dir = _resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_qids = {text for raw in args.candidate_qid if (text := str(raw).strip())}
    if args.candidate_qids_file is not None:
        for line in args.candidate_qids_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                explicit_qids.add(text)
    source_qids = sorted(explicit_qids) if explicit_qids else _candidate_qids(
        baseline_submission=build_baseline_submission,
        rider_source_submission=rider_source_submission,
        scaffold=truth_audit_scaffold,
    )
    if not source_qids:
        raise ValueError("No candidate exactness qids available")

    baseline_gate_json = out_dir / "compare_baseline_gate.json"
    baseline_gate_md = out_dir / "compare_baseline_gate.md"
    _run(
        [
            sys.executable,
            "scripts/run_experiment_gate.py",
            "--label",
            build_baseline_label,
            "--baseline-label",
            compare_baseline_label,
            "--baseline-submission",
            str(compare_baseline_submission),
            "--candidate-submission",
            str(build_baseline_submission),
            "--baseline-raw-results",
            str(compare_baseline_raw_results),
            "--candidate-raw-results",
            str(build_baseline_raw_results),
            "--benchmark",
            str(benchmark),
            "--scaffold",
            str(truth_audit_scaffold),
            "--baseline-preflight",
            str(compare_baseline_preflight),
            "--candidate-preflight",
            str(build_baseline_preflight),
            "--out",
            str(baseline_gate_md),
            "--out-json",
            str(baseline_gate_json),
        ],
        cwd=root,
    )
    build_baseline_gate = _load_json(baseline_gate_json)

    rows: list[JsonDict] = []
    max_subset_size = max(1, int(args.max_subset_size))
    for size in range(1, min(max_subset_size, len(source_qids)) + 1):
        for subset in itertools.combinations(source_qids, size):
            qids = list(subset)
            label = _subset_label(build_baseline_label, qids)
            candidate_dir = out_dir / label
            candidate_dir.mkdir(parents=True, exist_ok=True)
            qids_file = candidate_dir / "answer_qids.txt"
            _write_qids(qids_file, qids)

            submission = candidate_dir / "submission.json"
            raw_results = candidate_dir / "raw_results.json"
            preflight = candidate_dir / "preflight.json"
            report = candidate_dir / "counterfactual_report.json"
            _run(
                [
                    sys.executable,
                    "scripts/build_counterfactual_candidate.py",
                    "--answer-source-submission",
                    str(build_baseline_submission),
                    "--answer-source-raw-results",
                    str(build_baseline_raw_results),
                    "--answer-source-preflight",
                    str(build_baseline_preflight),
                    "--page-source-submission",
                    str(rider_source_submission),
                    "--page-source-raw-results",
                    str(rider_source_raw_results),
                    "--page-source-preflight",
                    str(rider_source_preflight),
                    "--page-source-answer-qids-file",
                    str(qids_file),
                    "--page-source-pages-default",
                    "none",
                    "--out-submission",
                    str(submission),
                    "--out-raw-results",
                    str(raw_results),
                    "--out-preflight",
                    str(preflight),
                    "--out-report",
                    str(report),
                ],
                cwd=root,
            )

            lineage_json = candidate_dir / "lineage.json"
            lineage_md = candidate_dir / "lineage.md"
            _run(
                [
                    sys.executable,
                    "scripts/verify_candidate_lineage.py",
                    "--baseline-submission",
                    str(build_baseline_submission),
                    "--candidate-submission",
                    str(submission),
                    "--allowed-answer-qids-file",
                    str(qids_file),
                    "--out-json",
                    str(lineage_json),
                    "--out-md",
                    str(lineage_md),
                ],
                cwd=root,
            )

            gate_json = candidate_dir / "gate.json"
            gate_md = candidate_dir / "gate.md"
            _run(
                [
                    sys.executable,
                    "scripts/run_experiment_gate.py",
                    "--label",
                    label,
                    "--baseline-label",
                    compare_baseline_label,
                    "--baseline-submission",
                    str(compare_baseline_submission),
                    "--candidate-submission",
                    str(submission),
                    "--baseline-raw-results",
                    str(compare_baseline_raw_results),
                    "--candidate-raw-results",
                    str(raw_results),
                    "--benchmark",
                    str(benchmark),
                    "--scaffold",
                    str(truth_audit_scaffold),
                    "--baseline-preflight",
                    str(compare_baseline_preflight),
                    "--candidate-preflight",
                    str(preflight),
                    "--out",
                    str(gate_md),
                    "--out-json",
                    str(gate_json),
                ],
                cwd=root,
            )

            exactness_json = candidate_dir / "exactness.json"
            exactness_md = candidate_dir / "exactness.md"
            _run(
                [
                    sys.executable,
                    "scripts/audit_exactness_candidate.py",
                    "--baseline-label",
                    compare_baseline_label,
                    "--baseline-submission",
                    str(compare_baseline_submission),
                    "--candidate-label",
                    label,
                    "--candidate-submission",
                    str(submission),
                    "--truth-audit-scaffold",
                    str(truth_audit_scaffold),
                    "--out-json",
                    str(exactness_json),
                    "--out-md",
                    str(exactness_md),
                    "--judge-scope",
                    "none",
                ],
                cwd=root,
            )

            lineage_payload = _load_json(lineage_json)
            gate_payload = _load_json(gate_json)
            exactness_payload = _load_json(exactness_json)
            rows.append(
                {
                    "label": label,
                    "qids": qids,
                    "submission": str(submission),
                    "raw_results": str(raw_results),
                    "preflight": str(preflight),
                    "lineage_ok": lineage_payload.get("lineage_ok"),
                    "answer_drift": gate_payload.get("answer_changed_count", lineage_payload.get("answer_changed_count")),
                    "page_drift": gate_payload.get("page_changed_count", lineage_payload.get("page_changed_count")),
                    "recommendation": gate_payload.get("recommendation"),
                    "hidden_g_trusted_delta": _delta(gate_payload, build_baseline_gate, "candidate_hidden_g_trusted"),
                    "hidden_g_all_delta": _delta(gate_payload, build_baseline_gate, "candidate_hidden_g_all"),
                    "resolved_incorrect_count": len(cast("list[object]", exactness_payload.get("resolved_incorrect_qids") or [])),
                    "still_mismatched_incorrect_count": len(
                        cast("list[object]", exactness_payload.get("still_mismatched_incorrect_qids") or [])
                    ),
                    "judge_pass_delta": 0.0,
                    "judge_grounding_delta": 0.0,
                    "judge_status": "not_run",
                    "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
                }
            )

    ranked = sorted(rows, key=_combined_score, reverse=True)
    top_k = min(max(0, int(args.judge_top_k)), len(ranked))
    for row in ranked[:top_k]:
        label = str(row["label"])
        candidate_dir = out_dir / label / "judge"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        out_json = candidate_dir / "exactness_judge.json"
        out_md = candidate_dir / "family_debug_rank.md"
        ok = _run_optional(
            [
                sys.executable,
                "scripts/audit_exactness_candidate.py",
                "--baseline-label",
                compare_baseline_label,
                "--baseline-submission",
                str(compare_baseline_submission),
                "--candidate-label",
                label,
                "--candidate-submission",
                str(row["submission"]),
                "--truth-audit-scaffold",
                str(truth_audit_scaffold),
                "--baseline-raw-results",
                str(compare_baseline_raw_results),
                "--candidate-raw-results",
                str(row["raw_results"]),
                "--questions",
                str(questions),
                "--docs-dir",
                str(docs_dir),
                "--judge-scope",
                "all",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
                "--out-dir",
                str(candidate_dir),
            ],
            cwd=root,
            timeout_seconds=int(args.judge_timeout_seconds),
        )
        if not ok:
            row["judge_status"] = "timeout_or_error"
            continue
        debug_payload = _load_json(out_json)
        debug_eval_path = debug_payload.get("debug_eval_path")
        if isinstance(debug_eval_path, str) and debug_eval_path.strip():
            debug_eval = _load_json(Path(debug_eval_path))
            summary_obj = debug_eval.get("summary")
            summary = cast("JsonDict", summary_obj) if isinstance(summary_obj, dict) else {}
            judge_obj = summary.get("judge")
            judge = cast("JsonDict", judge_obj) if isinstance(judge_obj, dict) else {}
            row["judge_pass_delta"] = _coerce_float(judge.get("pass_rate"))
            row["judge_grounding_delta"] = _coerce_float(judge.get("avg_grounding"))
            row["judge_status"] = "ok"

    ranked = sorted(rows, key=_combined_score, reverse=True)
    payload = {
        "build_baseline_label": build_baseline_label,
        "compare_baseline_label": compare_baseline_label,
        "candidate_qids": source_qids,
        "build_baseline_gate": build_baseline_gate,
        "ranked_candidates": ranked,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    (out_dir / "exactness_rider_subset_search.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "exactness_rider_subset_search.md").write_text(
        _render_md(rows=ranked, source_qids=source_qids),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
