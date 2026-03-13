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
    submission: Path
    raw_results: Path
    preflight: Path | None
    candidate_scaffold: Path | None
    allowed_answer_qids: list[str]
    allowed_page_qids: list[str]


def _resolve(root: Path, raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (root / value).resolve()


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


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


def _candidate_paths(*, out_dir: Path, label: str) -> dict[str, Path]:
    candidate_dir = out_dir / label
    return {
        "dir": candidate_dir,
        "allowed_answer_qids": candidate_dir / "allowed_answer_qids.txt",
        "allowed_page_qids": candidate_dir / "allowed_page_qids.txt",
        "lineage_json": candidate_dir / "lineage.json",
        "lineage_md": candidate_dir / "lineage.md",
        "gate_json": candidate_dir / "gate.json",
        "gate_md": candidate_dir / "gate.md",
        "exactness_json": candidate_dir / "exactness.json",
        "exactness_md": candidate_dir / "exactness.md",
    }


def _load_manifest(path: Path, *, root: Path) -> list[CandidateSpec]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    candidates_obj = cast("JsonDict", obj).get("candidates")
    if not isinstance(candidates_obj, list):
        raise ValueError(f"Manifest at {path} is missing candidates[]")
    out: list[CandidateSpec] = []
    seen: set[str] = set()
    candidates = cast("list[object]", candidates_obj)
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        label = str(row.get("label") or "").strip()
        if not label:
            raise ValueError("Candidate manifest entry missing label")
        if label in seen:
            raise ValueError(f"Duplicate candidate label in manifest: {label}")
        seen.add(label)
        submission = _resolve(root, row.get("submission"))
        raw_results = _resolve(root, row.get("raw_results"))
        if submission is None or raw_results is None:
            raise ValueError(f"Candidate {label} is missing submission/raw_results")
        out.append(
            CandidateSpec(
                label=label,
                submission=submission,
                raw_results=raw_results,
                preflight=_resolve(root, row.get("preflight")),
                candidate_scaffold=_resolve(root, row.get("candidate_scaffold")),
                allowed_answer_qids=_coerce_str_list(row.get("allowed_answer_qids")),
                allowed_page_qids=_coerce_str_list(row.get("allowed_page_qids")),
            )
        )
    if not out:
        raise ValueError(f"No valid candidates in manifest: {path}")
    return out


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast("list[object]", value)
    out: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _write_qids(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")


def _verify_lineage(
    *,
    root: Path,
    baseline_submission: Path,
    candidate: CandidateSpec,
    paths: dict[str, Path],
) -> JsonDict:
    _write_qids(paths["allowed_answer_qids"], candidate.allowed_answer_qids)
    _write_qids(paths["allowed_page_qids"], candidate.allowed_page_qids)
    cmd = [
        sys.executable,
        "scripts/verify_candidate_lineage.py",
        "--baseline-submission",
        str(baseline_submission),
        "--candidate-submission",
        str(candidate.submission),
        "--allowed-answer-qids-file",
        str(paths["allowed_answer_qids"]),
        "--allowed-page-qids-file",
        str(paths["allowed_page_qids"]),
        "--out-json",
        str(paths["lineage_json"]),
        "--out-md",
        str(paths["lineage_md"]),
    ]
    _run(cmd, cwd=root)
    return _load_json(paths["lineage_json"])


def _run_gate(
    *,
    root: Path,
    baseline_label: str,
    baseline_submission: Path,
    baseline_raw_results: Path,
    baseline_preflight: Path | None,
    benchmark: Path,
    scaffold: Path,
    candidate: CandidateSpec,
    paths: dict[str, Path],
) -> JsonDict:
    cmd = [
        sys.executable,
        "scripts/run_experiment_gate.py",
        "--label",
        candidate.label,
        "--baseline-label",
        baseline_label,
        "--baseline-submission",
        str(baseline_submission),
        "--candidate-submission",
        str(candidate.submission),
        "--baseline-raw-results",
        str(baseline_raw_results),
        "--candidate-raw-results",
        str(candidate.raw_results),
        "--benchmark",
        str(benchmark),
        "--scaffold",
        str(scaffold),
        "--out",
        str(paths["gate_md"]),
        "--out-json",
        str(paths["gate_json"]),
    ]
    if candidate.candidate_scaffold is not None:
        cmd.extend(["--candidate-scaffold", str(candidate.candidate_scaffold)])
    if baseline_preflight is not None:
        cmd.extend(["--baseline-preflight", str(baseline_preflight)])
    if candidate.preflight is not None:
        cmd.extend(["--candidate-preflight", str(candidate.preflight)])
    _run(cmd, cwd=root)
    return _load_json(paths["gate_json"])


def _run_exactness(
    *,
    root: Path,
    baseline_label: str,
    baseline_submission: Path,
    scaffold: Path,
    candidate: CandidateSpec,
    paths: dict[str, Path],
) -> JsonDict:
    cmd = [
        sys.executable,
        "scripts/audit_exactness_candidate.py",
        "--baseline-label",
        baseline_label,
        "--baseline-submission",
        str(baseline_submission),
        "--candidate-label",
        candidate.label,
        "--candidate-submission",
        str(candidate.submission),
        "--truth-audit-scaffold",
        str(scaffold),
        "--out-json",
        str(paths["exactness_json"]),
        "--out-md",
        str(paths["exactness_md"]),
        "--judge-scope",
        "none",
    ]
    _run(cmd, cwd=root)
    return _load_json(paths["exactness_json"])


def _run_family_debug(
    *,
    root: Path,
    baseline_label: str,
    baseline_raw_results: Path,
    questions: Path,
    docs_dir: Path,
    include_qids_file: Path,
    candidates: list[CandidateSpec],
    out_dir: Path,
    judge_scope: str,
) -> JsonDict:
    out_json = out_dir / "family_debug_rank.json"
    out_md = out_dir / "family_debug_rank.md"
    cmd = [
        sys.executable,
        "scripts/compare_candidate_family_debug.py",
        "--baseline-label",
        baseline_label,
        "--baseline-raw-results",
        str(baseline_raw_results),
        "--questions",
        str(questions),
        "--docs-dir",
        str(docs_dir),
        "--include-qids-file",
        str(include_qids_file),
        "--family-label",
        "candidate_ceiling_portfolio",
        "--out-dir",
        str(out_dir),
        "--judge-scope",
        judge_scope,
        "--case-scope",
        "all",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    for candidate in candidates:
        cmd.extend(["--candidate", f"{candidate.label}={candidate.raw_results}"])
    _run(cmd, cwd=root)
    return _load_json(out_json)


def _combined_score(row: JsonDict) -> tuple[float, float, float, float, float, float, int, int, str]:
    lineage_ok = bool(row.get("lineage_ok"))
    recommendation = str(row.get("recommendation") or "")
    recommendation_bonus = {"PROMISING": 2.0, "EXPERIMENTAL_NO_SUBMIT": 1.0}.get(recommendation.upper(), 0.0)
    return (
        1.0 if lineage_ok else 0.0,
        recommendation_bonus,
        _coerce_float(row.get("hidden_g_trusted_delta")),
        _coerce_float(row.get("hidden_g_all_delta")),
        _coerce_float(row.get("judge_pass_delta")),
        _coerce_float(row.get("judge_grounding_delta")) + (_coerce_int(row.get("resolved_incorrect_count")) * 0.5),
        -_coerce_int(row.get("page_drift")),
        -_coerce_int(row.get("answer_drift")),
        str(row.get("label") or ""),
    )


def _render_markdown(*, rows: list[JsonDict], baseline_label: str, include_qids_file: Path) -> str:
    lines = [
        "# Candidate Ceiling Cycle",
        "",
        f"- baseline_label: `{baseline_label}`",
        f"- include_qids_file: `{include_qids_file}`",
        f"- candidates: `{len(rows)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | Label | Recommendation | Lineage | Hidden-G Trusted Δ | Hidden-G All Δ | Judge Pass Δ | Judge Grounding Δ | Resolved Exactness | Answer Drift | Page Drift | Page p95 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(sorted(rows, key=_combined_score, reverse=True), start=1):
        lines.append(
            "| "
            f"{index} | `{row['label']}` | `{row['recommendation']}` | `{row['lineage_ok']}` | "
            f"{_coerce_float(row.get('hidden_g_trusted_delta')):.4f} | {_coerce_float(row.get('hidden_g_all_delta')):.4f} | "
            f"{_coerce_float(row.get('judge_pass_delta')):+.4f} | {_coerce_float(row.get('judge_grounding_delta')):+.4f} | "
            f"{_coerce_int(row.get('resolved_incorrect_count'))} | {_coerce_int(row.get('answer_drift'))} | "
            f"{_coerce_int(row.get('page_drift'))} | {_coerce_int(row.get('page_p95'))} |"
        )
    return "\n".join(lines) + "\n"


def _family_rows_by_label(payload: JsonDict) -> dict[str, JsonDict]:
    ranked_obj = payload.get("ranked_candidates")
    if not isinstance(ranked_obj, list):
        return {}
    rows = cast("list[object]", ranked_obj)
    out: dict[str, JsonDict] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        label = str(row.get("label") or "").strip()
        if label:
            out[label] = row
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lineage + gate + exactness + family-debug ranking for a candidate ceiling manifest.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--baseline-raw-results", required=True)
    parser.add_argument("--baseline-preflight", default=None)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--scaffold", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--judge-scope", choices=("all", "free_text", "none"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline_submission = cast("Path", _resolve(root, args.baseline_submission))
    baseline_raw_results = cast("Path", _resolve(root, args.baseline_raw_results))
    baseline_preflight = _resolve(root, args.baseline_preflight)
    benchmark = cast("Path", _resolve(root, args.benchmark))
    scaffold = cast("Path", _resolve(root, args.scaffold))
    questions = cast("Path", _resolve(root, args.questions))
    docs_dir = cast("Path", _resolve(root, args.docs_dir))
    manifest_json = cast("Path", _resolve(root, args.manifest_json))
    out_dir = cast("Path", _resolve(root, args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_manifest(manifest_json, root=root)
    include_qids = sorted({qid for candidate in candidates for qid in [*candidate.allowed_answer_qids, *candidate.allowed_page_qids]})
    include_qids_file = out_dir / "include_qids.txt"
    _write_qids(include_qids_file, include_qids)

    rows: list[JsonDict] = []
    for candidate in candidates:
        paths = _candidate_paths(out_dir=out_dir, label=candidate.label)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        lineage = _verify_lineage(
            root=root,
            baseline_submission=baseline_submission,
            candidate=candidate,
            paths=paths,
        )
        gate = _run_gate(
            root=root,
            baseline_label=str(args.baseline_label),
            baseline_submission=baseline_submission,
            baseline_raw_results=baseline_raw_results,
            baseline_preflight=baseline_preflight,
            benchmark=benchmark,
            scaffold=scaffold,
            candidate=candidate,
            paths=paths,
        )
        exactness = _run_exactness(
            root=root,
            baseline_label=str(args.baseline_label),
            baseline_submission=baseline_submission,
            scaffold=scaffold,
            candidate=candidate,
            paths=paths,
        )
        rows.append(
            {
                "label": candidate.label,
                "submission": str(candidate.submission),
                "raw_results": str(candidate.raw_results),
                "preflight": None if candidate.preflight is None else str(candidate.preflight),
                "recommendation": gate.get("recommendation"),
                "lineage_ok": lineage.get("lineage_ok"),
                "answer_drift": lineage.get("answer_changed_count"),
                "page_drift": lineage.get("page_changed_count"),
                "page_p95": gate.get("candidate_page_p95"),
                "hidden_g_trusted_delta": _coerce_float(gate.get("benchmark_trusted_candidate")) - _coerce_float(gate.get("benchmark_trusted_baseline")),
                "hidden_g_all_delta": _coerce_float(gate.get("benchmark_all_candidate")) - _coerce_float(gate.get("benchmark_all_baseline")),
                "resolved_incorrect_count": len(cast("list[object]", exactness.get("resolved_incorrect_qids") or [])),
                "still_mismatched_incorrect_count": len(cast("list[object]", exactness.get("still_mismatched_incorrect_qids") or [])),
                "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
            }
        )

    family_debug = _run_family_debug(
        root=root,
        baseline_label=str(args.baseline_label),
        baseline_raw_results=baseline_raw_results,
        questions=questions,
        docs_dir=docs_dir,
        include_qids_file=include_qids_file,
        candidates=candidates,
        out_dir=out_dir / "family_debug",
        judge_scope=str(args.judge_scope),
    )
    family_rows = _family_rows_by_label(family_debug)
    for row in rows:
        family = family_rows.get(str(row["label"]), {})
        row["judge_pass_delta"] = family.get("judge_pass_delta", 0.0)
        row["judge_grounding_delta"] = family.get("judge_grounding_delta", 0.0)
        row["judge_accuracy_delta"] = family.get("judge_accuracy_delta", 0.0)
        row["citation_delta"] = family.get("citation_delta", 0.0)
        row["format_delta"] = family.get("format_delta", 0.0)

    ranked = sorted(rows, key=_combined_score, reverse=True)
    payload = {
        "baseline_label": args.baseline_label,
        "include_qids_file": str(include_qids_file),
        "manifest_json": str(manifest_json),
        "family_debug_json": str((out_dir / "family_debug" / "family_debug_rank.json").resolve()),
        "ranked_candidates": ranked,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    out_json = out_dir / "candidate_ceiling_cycle.json"
    out_md = out_dir / "candidate_ceiling_cycle.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_markdown(rows=rows, baseline_label=str(args.baseline_label), include_qids_file=include_qids_file), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
