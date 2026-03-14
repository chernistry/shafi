#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WARMUP_DIR = ROOT / "platform_runs" / "warmup"
DEFAULT_QUESTIONS = WARMUP_DIR / "questions.json"
DEFAULT_TRUTH_AUDIT = WARMUP_DIR / "truth_audit_scaffold.json"

for candidate in (SRC, ROOT):
    candidate_text = str(candidate)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)


def _as_json_dict(value: object) -> JsonDict | None:
    if isinstance(value, dict):
        return cast("JsonDict", value)
    return None


def _rewrite_host_qdrant_url(url: str | None) -> tuple[str | None, str | None]:
    normalized = str(url or "").strip()
    if not normalized:
        return None, None
    parsed = urlsplit(normalized)
    if parsed.hostname != "qdrant":
        return normalized, None
    replacement = parsed._replace(netloc=f"localhost:{parsed.port or 6333}")
    return urlunsplit(replacement), "rewrote_qdrant_hostname_for_host_shell"


def _build_env(base_env: Mapping[str, str], qdrant_url: str | None) -> tuple[dict[str, str], list[str]]:
    env = dict(base_env)
    notes: list[str] = []
    requested_qdrant = str(qdrant_url or env.get("QDRANT_URL") or "").strip()
    if not requested_qdrant:
        requested_qdrant = "http://localhost:6333"
        notes.append("defaulted_qdrant_url_to_localhost_for_host_rehearsal")
    effective_qdrant, qdrant_note = _rewrite_host_qdrant_url(requested_qdrant)
    if effective_qdrant:
        env["QDRANT_URL"] = effective_qdrant
    if qdrant_note:
        notes.append(qdrant_note)
    if not str(env.get("EVAL_API_KEY") or "").strip():
        env["EVAL_API_KEY"] = "local-rehearsal-placeholder"
        notes.append("set_placeholder_eval_api_key_for_local_rehearsal")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env, notes


def _artifact_paths(suffix: str) -> dict[str, Path]:
    return {
        "raw_results": WARMUP_DIR / f"raw_results_{suffix}.json",
        "submission": WARMUP_DIR / f"submission_{suffix}.json",
        "preflight_summary": WARMUP_DIR / f"preflight_summary_{suffix}.json",
        "truth_audit": WARMUP_DIR / f"truth_audit_scaffold_{suffix}.json",
        "truth_audit_workbook": WARMUP_DIR / f"truth_audit_workbook_{suffix}.md",
        "code_archive": WARMUP_DIR / f"code_archive_{suffix}.zip",
        "code_archive_audit": WARMUP_DIR / f"code_archive_audit_{suffix}.json",
    }


def _question_id(row: JsonDict) -> str:
    case = _as_json_dict(row.get("case"))
    if case is not None:
        for key in ("case_id", "question_id", "id"):
            value = str(case.get(key) or "").strip()
            if value:
                return value
    telemetry = _as_json_dict(row.get("telemetry"))
    if telemetry is not None:
        value = str(telemetry.get("question_id") or "").strip()
        if value:
            return value
    return str(row.get("question_id") or row.get("id") or "").strip()


def _answer_text(row: JsonDict) -> str:
    value = row.get("answer_text")
    return str(value if value is not None else "null").strip()


def _used_pages(row: JsonDict) -> list[str]:
    from rag_challenge.submission.common import select_submission_used_pages

    telemetry = _as_json_dict(row.get("telemetry"))
    if telemetry is not None:
        return select_submission_used_pages(cast("dict[str, object]", telemetry))
    return []


def _model_name(row: JsonDict) -> str:
    telemetry = _as_json_dict(row.get("telemetry"))
    if telemetry is None:
        return ""
    return str(telemetry.get("model_llm") or telemetry.get("model_name") or "").strip()


def _load_json_list(path: Path) -> list[JsonDict]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {path}")
    payload_rows = cast("list[object]", payload)
    rows: list[JsonDict] = []
    for row_obj in payload_rows:
        row = _as_json_dict(row_obj)
        if row is not None:
            rows.append(row)
    return rows


def _build_equivalence_canary(*, baseline_rows: list[JsonDict], candidate_rows: list[JsonDict]) -> JsonDict:
    baseline_by_id = {_question_id(row): row for row in baseline_rows if _question_id(row)}
    candidate_by_id = {_question_id(row): row for row in candidate_rows if _question_id(row)}
    answer_drift_case_ids: list[str] = []
    model_drift_case_ids: list[str] = []
    page_drift_case_ids: list[str] = []
    missing_case_ids: list[str] = []

    for qid, baseline_row in baseline_by_id.items():
        candidate_row = candidate_by_id.get(qid)
        if candidate_row is None:
            missing_case_ids.append(qid)
            continue
        if _answer_text(baseline_row) != _answer_text(candidate_row):
            answer_drift_case_ids.append(qid)
        if _model_name(baseline_row) != _model_name(candidate_row):
            model_drift_case_ids.append(qid)
        if _used_pages(baseline_row) != _used_pages(candidate_row):
            page_drift_case_ids.append(qid)

    return {
        "baseline_concurrency": 1,
        "candidate_concurrency": 1,
        "total_cases": len(baseline_rows),
        "answer_drift_case_ids": answer_drift_case_ids,
        "model_drift_case_ids": model_drift_case_ids,
        "page_drift_case_ids": page_drift_case_ids,
        "missing_case_ids": missing_case_ids,
        "answer_drift_count": len(answer_drift_case_ids),
        "model_drift_count": len(model_drift_case_ids),
        "page_drift_count": len(page_drift_case_ids),
        "stable": not any((answer_drift_case_ids, model_drift_case_ids, page_drift_case_ids, missing_case_ids)),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_artifacts(*, suffix: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name, src in _artifact_paths(suffix).items():
        if not src.exists():
            raise FileNotFoundError(f"Expected artifact missing after rehearsal run: {src}")
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    return copied


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical private-run rehearsal twice at query_concurrency=1.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--truth-audit", type=Path, default=DEFAULT_TRUTH_AUDIT)
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--timeout-s", type=float, default=2400.0)
    parser.add_argument("--artifact-prefix", default="ticket64_private_rehearsal")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.runner_session_pool import RunnerSessionPool

    args = _parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.questions.resolve()
    truth_audit_path = args.truth_audit.resolve() if args.truth_audit else None
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    env, env_notes = _build_env(os.environ, str(args.qdrant_url or ""))
    pool = RunnerSessionPool(session_name="ticket64-private-run-rehearsal")
    suffixes = [f"{args.artifact_prefix}_run_a", f"{args.artifact_prefix}_run_b"]
    command_base = [
        "uv",
        "run",
        "python",
        "-m",
        "rag_challenge.submission.platform",
        "--skip-ingest",
        "--query-concurrency",
        "1",
    ]
    if bool(args.fail_fast):
        command_base.append("--fail-fast")

    try:
        for lease, suffix in zip(("run_a", "run_b"), suffixes, strict=True):
            pool.run(
                [*command_base, "--artifact-suffix", suffix],
                cwd=ROOT,
                lease=lease,
                env=env,
                timeout_s=float(args.timeout_s),
            )
    finally:
        pool.close()

    pool.write_summary(out_dir / "runner_session_pool.json")
    copied_runs = [_copy_artifacts(suffix=suffix, out_dir=out_dir) for suffix in suffixes]
    canary = _build_equivalence_canary(
        baseline_rows=_load_json_list(Path(copied_runs[0]["raw_results"])),
        candidate_rows=_load_json_list(Path(copied_runs[1]["raw_results"])),
    )
    canary_path = out_dir / "equivalence_canary.json"
    _write_json(canary_path, canary)
    analysis_truth_audit_path = truth_audit_path if truth_audit_path is not None and truth_audit_path.exists() else Path(copied_runs[0]["truth_audit"])

    analysis_cmd = [
        "uv",
        "run",
        "python",
        "scripts/analyze_answer_drift.py",
        "--canary",
        str(canary_path),
        "--questions",
        str(questions_path),
        "--out-dir",
        str(out_dir),
    ]
    dashboard_cmd = [
        "uv",
        "run",
        "python",
        "scripts/private_phase_dashboard.py",
        "--run-a",
        copied_runs[0]["raw_results"],
        "--run-b",
        copied_runs[1]["raw_results"],
        "--label-a",
        suffixes[0],
        "--label-b",
        suffixes[1],
        "--questions",
        str(questions_path),
        "--out-json",
        str(out_dir / "telemetry_dashboard.json"),
        "--out-md",
        str(out_dir / "telemetry_dashboard.md"),
    ]
    if analysis_truth_audit_path.exists():
        analysis_cmd.extend(["--truth-audit", str(analysis_truth_audit_path)])
        dashboard_cmd.extend(["--truth-audit", str(analysis_truth_audit_path)])

    pool = RunnerSessionPool(session_name="ticket64-private-run-postprocess")
    try:
        pool.run(analysis_cmd, cwd=ROOT, lease="drift-report", env=env, timeout_s=300.0)
        pool.run(dashboard_cmd, cwd=ROOT, lease="telemetry-dashboard", env=env, timeout_s=300.0)
    finally:
        pool.close()
    pool.write_summary(out_dir / "postprocess_session_pool.json")

    _write_json(
        out_dir / "rehearsal_manifest.json",
        {
            "questions_path": str(questions_path),
            "truth_audit_path": str(truth_audit_path) if truth_audit_path else None,
            "run_suffixes": suffixes,
            "copied_artifacts": copied_runs,
            "effective_qdrant_url": env.get("QDRANT_URL"),
            "analysis_truth_audit_path": str(analysis_truth_audit_path),
            "env_notes": env_notes,
            "timeout_s": float(args.timeout_s),
            "stable": bool(canary["stable"]),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
