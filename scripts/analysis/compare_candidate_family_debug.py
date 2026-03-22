from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    raw_results: Path


def _run(cmd: list[str], *, cwd: Path, timeout_seconds: float | None = None) -> bool:
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
        return True
    except subprocess.TimeoutExpired:
        return False


def _resolve(root: Path, raw: str | Path) -> Path:
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _parse_candidate(raw: str, *, root: Path) -> CandidateSpec:
    if "=" not in raw:
        raise ValueError(f"Expected candidate as label=/absolute/or/relative/path.json, got: {raw}")
    label, raw_path = raw.split("=", 1)
    label_text = label.strip()
    if not label_text:
        raise ValueError(f"Empty candidate label in: {raw}")
    return CandidateSpec(label=label_text, raw_results=_resolve(root, raw_path.strip()))


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


def _load_summary(eval_path: Path) -> JsonDict:
    obj = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {eval_path}")
    summary_obj = cast("JsonDict", obj).get("summary")
    if not isinstance(summary_obj, dict):
        raise ValueError(f"Missing summary in {eval_path}")
    return cast("JsonDict", summary_obj)


def _judge(summary: JsonDict) -> JsonDict:
    judge_obj = summary.get("judge")
    return cast("JsonDict", judge_obj) if isinstance(judge_obj, dict) else {}


def _candidate_sort_key(row: JsonDict) -> tuple[float, float, float, float, float, float, str]:
    return (
        0.0 if bool(row.get("judge_timeout")) else 1.0,
        _coerce_float(row.get("judge_pass_delta")),
        _coerce_float(row.get("judge_grounding_delta")),
        _coerce_float(row.get("judge_accuracy_delta")),
        _coerce_float(row.get("citation_delta")),
        -_coerce_float(row.get("ttft_p50_delta_ms")),
        str(row.get("label") or ""),
    )


def _render_markdown(*, family_label: str, include_qids_file: Path, rows: list[JsonDict]) -> str:
    lines = [
        "# Candidate Family Debug Comparison",
        "",
        f"- family_label: `{family_label}`",
        f"- include_qids_file: `{include_qids_file}`",
        f"- candidates: `{len(rows)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | Candidate | Judge Timeout | Judge Pass | Judge Pass Δ | Judge Grounding | Judge Grounding Δ | Judge Accuracy Δ | Citation Δ | Format Δ | TTFT p50 Δ ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(rows, key=_candidate_sort_key, reverse=True)
    for index, row in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{index} | `{row['label']}` | "
            f"`{row.get('judge_timeout', False)}` | "
            f"{row['judge_pass_rate']:.4f} | {row['judge_pass_delta']:+.4f} | "
            f"{row['judge_grounding']:.4f} | {row['judge_grounding_delta']:+.4f} | "
            f"{row['judge_accuracy_delta']:+.4f} | {row['citation_delta']:+.4f} | "
            f"{row['format_delta']:+.4f} | {row['ttft_p50_delta_ms']:+.1f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run debug-judge family comparisons across multiple no-submit candidates.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-raw-results", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--include-qids-file", required=True)
    parser.add_argument("--family-label", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="label=path/to/raw_results.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-scope", choices=("all", "free_text", "none"), default="all")
    parser.add_argument("--judge-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--case-scope", choices=("changed", "all"), default="changed")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_raw_results = _resolve(root, args.baseline_raw_results)
    questions = _resolve(root, args.questions)
    docs_dir = _resolve(root, args.docs_dir)
    include_qids_file = _resolve(root, args.include_qids_file)
    out_dir = _resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [_parse_candidate(raw, root=root) for raw in args.candidate]
    if not candidates:
        raise ValueError("At least one --candidate is required")

    baseline_eval_path = out_dir / f"eval_candidate_debug_{args.baseline_label}.json"
    baseline_summary: JsonDict | None = None
    rows: list[JsonDict] = []

    for candidate in candidates:
        cmd = [
            sys.executable,
            "scripts/evaluate_candidate_debug_signal.py",
            "--baseline-label",
            str(args.baseline_label),
            "--baseline-raw-results",
            str(baseline_raw_results),
            "--candidate-label",
            candidate.label,
            "--candidate-raw-results",
            str(candidate.raw_results),
            "--questions",
            str(questions),
            "--docs-dir",
            str(docs_dir),
            "--out-dir",
            str(out_dir),
            "--case-scope",
            str(args.case_scope),
            "--judge-scope",
            str(args.judge_scope),
            "--include-qids-file",
            str(include_qids_file),
        ]
        judge_timeout = False
        completed = _run(
            cmd,
            cwd=root,
            timeout_seconds=float(args.judge_timeout_seconds),
        )
        if not completed and str(args.judge_scope) != "none":
            judge_timeout = True
            fallback_cmd = list(cmd)
            judge_scope_index = fallback_cmd.index("--judge-scope")
            fallback_cmd[judge_scope_index + 1] = "none"
            _run(fallback_cmd, cwd=root)

        if baseline_summary is None:
            baseline_summary = _load_summary(baseline_eval_path)
        candidate_summary = _load_summary(out_dir / f"eval_candidate_debug_{candidate.label}.json")

        baseline_judge = _judge(baseline_summary)
        candidate_judge = _judge(candidate_summary)
        row = {
            "label": candidate.label,
            "candidate_raw_results": str(candidate.raw_results),
            "judge_pass_rate": _coerce_float(candidate_judge.get("pass_rate")),
            "judge_pass_delta": _coerce_float(candidate_judge.get("pass_rate")) - _coerce_float(baseline_judge.get("pass_rate")),
            "judge_grounding": _coerce_float(candidate_judge.get("avg_grounding")),
            "judge_grounding_delta": _coerce_float(candidate_judge.get("avg_grounding")) - _coerce_float(baseline_judge.get("avg_grounding")),
            "judge_accuracy_delta": _coerce_float(candidate_judge.get("avg_accuracy")) - _coerce_float(baseline_judge.get("avg_accuracy")),
            "citation_delta": _coerce_float(candidate_summary.get("citation_coverage")) - _coerce_float(baseline_summary.get("citation_coverage")),
            "format_delta": _coerce_float(candidate_summary.get("answer_type_format_compliance")) - _coerce_float(baseline_summary.get("answer_type_format_compliance")),
            "ttft_p50_delta_ms": _coerce_float(candidate_summary.get("ttft_p50_ms")) - _coerce_float(baseline_summary.get("ttft_p50_ms")),
            "judge_timeout": judge_timeout,
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        }
        rows.append(row)

    ranked = sorted(rows, key=_candidate_sort_key, reverse=True)
    payload = {
        "family_label": args.family_label,
        "include_qids_file": str(include_qids_file),
        "baseline_label": args.baseline_label,
        "case_scope": args.case_scope,
        "judge_scope": args.judge_scope,
        "judge_timeout_seconds": float(args.judge_timeout_seconds),
        "ranked_candidates": ranked,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    out_json = _resolve(root, args.out_json)
    out_md = _resolve(root, args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        _render_markdown(
            family_label=str(args.family_label),
            include_qids_file=include_qids_file,
            rows=rows,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
