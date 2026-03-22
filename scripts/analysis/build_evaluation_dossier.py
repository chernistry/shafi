# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CRITICAL_CASES = (
    "89fd4fbcdcf5c17ba256395ee64378a3f2125b081394a9568964defb28fdef75",
    "4aa0f4e28b151c9fb03acbccd37d724bb0f9fae137c8fdc268c6c03cd6c7c7ac",
    "6e8d0c41f3e5b8a5383db8964a64254de33aec88f0c7abea793c37ecf4c4db43",
    "acd3200d75f4507d2cfbbcb1c568d7adf8da409063bee2e2e0b7832c4894a5a9",
)


@dataclass(frozen=True)
class FailureReplay:
    case_id: str
    question: str
    taxonomy: str
    answer: str
    recommended_fix: str
    unsupported_claims: list[str]
    used_pages: list[str]
    context_page_ids: list[str]
    model_llm: str
    ttft_ms: float | None


@dataclass(frozen=True)
class RunSummary:
    eval_path: Path
    judge_path: Path | None
    pass_rate: float | None
    judge_cases: int | None
    avg_accuracy: float | None
    avg_grounding: float | None
    avg_clarity: float | None
    citation_coverage: float | None
    format_compliance: float | None
    ttft_p50_ms: float | None
    avg_ttft_ms: float | None
    failures: int
    critical_status: dict[str, str]
    failed_cases: list[FailureReplay] = field(default_factory=list)

    @property
    def pass_ratio(self) -> str:
        if self.pass_rate is None or self.judge_cases is None:
            return "n/a"
        passed = round(self.pass_rate * self.judge_cases)
        return f"{passed}/{self.judge_cases}"

    @property
    def clean_judge_run(self) -> bool:
        return (
            self.pass_rate == 1.0
            and self.judge_cases == 30
            and self.citation_coverage == 1.0
            and self.format_compliance == 1.0
            and self.failures == 0
        )


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _infer_judge_path(eval_path: Path) -> Path | None:
    candidate = eval_path.with_name(eval_path.name.replace("eval_", "judge_", 1).replace(".json", ".jsonl"))
    return candidate if candidate.exists() else None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def _avg_ttft_ms(cases: list[dict[str, Any]]) -> float | None:
    values = [_coerce_float(case.get("ttft_ms")) for case in cases]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _failure_taxonomy(judge_result: dict[str, Any]) -> str:
    unsupported = _coerce_str_list(judge_result.get("unsupported_claims"))
    recommended_fix = str(judge_result.get("recommended_fix") or "").strip().lower()
    if unsupported:
        return "unsupported_claim"
    if "redundant" in recommended_fix or "concise" in recommended_fix or "repetitive" in recommended_fix:
        return "clarity_redundancy"
    if "ignored" in recommended_fix or "omit" in recommended_fix or "omitted" in recommended_fix:
        return "omission"
    if "abrupt" in recommended_fix or "incomplete" in recommended_fix:
        return "truncation"
    return "other"


def _judge_rows_by_case(judge_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows_by_case: dict[str, dict[str, Any]] = {}
    for row in judge_rows:
        case_id = str(row.get("case_id") or row.get("question_id") or "").strip()
        if case_id:
            rows_by_case[case_id] = row
    return rows_by_case


def _critical_case_status(
    cases: list[dict[str, Any]],
    judge_rows: dict[str, dict[str, Any]],
) -> dict[str, str]:
    status = {case_id: "missing" for case_id in CRITICAL_CASES}
    for case in cases:
        case_id = str(case.get("case_id") or case.get("question_id") or "").strip()
        if case_id not in status:
            continue
        if case.get("failure"):
            status[case_id] = "error"
            continue
        judge_row = judge_rows.get(case_id)
        judge_result = judge_row.get("judge_result") if isinstance(judge_row, dict) else None
        if isinstance(judge_result, dict):
            verdict = str(judge_result.get("verdict") or "").strip().upper()
            if verdict in {"PASS", "FAIL"}:
                status[case_id] = verdict
                continue
        status[case_id] = "seen"
    return status


def _failed_cases(
    *,
    cases: list[dict[str, Any]],
    judge_rows: dict[str, dict[str, Any]],
) -> list[FailureReplay]:
    failed: list[FailureReplay] = []
    for case in cases:
        case_id = str(case.get("case_id") or case.get("question_id") or "").strip()
        if not case_id:
            continue
        judge_row = judge_rows.get(case_id)
        if not isinstance(judge_row, dict):
            continue
        judge_result = judge_row.get("judge_result")
        if not isinstance(judge_result, dict):
            continue
        verdict = str(judge_result.get("verdict") or "").strip().upper()
        if verdict != "FAIL":
            continue

        telemetry = case.get("telemetry")
        telemetry_dict = telemetry if isinstance(telemetry, dict) else {}
        failed.append(
            FailureReplay(
                case_id=case_id,
                question=str(case.get("question") or judge_row.get("question") or "").strip(),
                taxonomy=_failure_taxonomy(judge_result),
                answer=str(case.get("answer") or judge_row.get("answer") or "").strip(),
                recommended_fix=str(judge_result.get("recommended_fix") or "").strip(),
                unsupported_claims=_coerce_str_list(judge_result.get("unsupported_claims")),
                used_pages=_coerce_str_list(judge_row.get("used_pages")) or _coerce_str_list(telemetry_dict.get("used_page_ids")),
                context_page_ids=_coerce_str_list(telemetry_dict.get("context_page_ids")),
                model_llm=str(telemetry_dict.get("model_llm") or "").strip(),
                ttft_ms=_coerce_float(case.get("ttft_ms")) or _coerce_float(telemetry_dict.get("ttft_ms")),
            )
        )
    return failed


def _collect_run(eval_path: Path) -> RunSummary:
    payload = _load_json(eval_path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary in {eval_path}")
    judge_summary = summary.get("judge")
    if not isinstance(judge_summary, dict):
        judge_summary = {}

    cases_obj = payload.get("cases")
    cases = [case for case in cases_obj if isinstance(case, dict)] if isinstance(cases_obj, list) else []
    failures_obj = payload.get("failures")
    failures = len(failures_obj) if isinstance(failures_obj, list) else int(summary.get("failures") or 0)

    judge_path = _infer_judge_path(eval_path)
    judge_rows_list = _load_jsonl(judge_path) if judge_path is not None else []
    judge_rows = _judge_rows_by_case(judge_rows_list)

    judge_cases_raw = judge_summary.get("cases")
    judge_cases = int(judge_cases_raw) if isinstance(judge_cases_raw, (int, float, str)) and str(judge_cases_raw).strip() else None
    if judge_cases is None and judge_rows_list:
        judge_cases = len(judge_rows_list)

    return RunSummary(
        eval_path=eval_path,
        judge_path=judge_path,
        pass_rate=_coerce_float(judge_summary.get("pass_rate")),
        judge_cases=judge_cases,
        avg_accuracy=_coerce_float(judge_summary.get("avg_accuracy")),
        avg_grounding=_coerce_float(judge_summary.get("avg_grounding")),
        avg_clarity=_coerce_float(judge_summary.get("avg_clarity")),
        citation_coverage=_coerce_float(summary.get("citation_coverage")),
        format_compliance=_coerce_float(summary.get("answer_type_format_compliance")),
        ttft_p50_ms=_coerce_float(summary.get("ttft_p50_ms")),
        avg_ttft_ms=_avg_ttft_ms(cases),
        failures=failures,
        critical_status=_critical_case_status(cases, judge_rows),
        failed_cases=_failed_cases(cases=cases, judge_rows=judge_rows),
    )


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} ms"


def _longest_clean_suffix(runs: list[RunSummary]) -> int:
    streak = 0
    for run in reversed(runs):
        if not run.clean_judge_run:
            break
        streak += 1
    return streak


def _taxonomy_counts(runs: list[RunSummary]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        for failure in run.failed_cases:
            counts[failure.taxonomy] = counts.get(failure.taxonomy, 0) + 1
    return counts


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"


def _build_markdown(runs: list[RunSummary]) -> str:
    clean_suffix = _longest_clean_suffix(runs)
    lines: list[str] = []
    lines.append("# Evaluation Dossier")
    lines.append("")
    lines.append("Generated from recorded eval artifacts. All metrics below are sourced from files in `data/`.")
    lines.append("")
    lines.append("## Stability Summary")
    lines.append("")
    lines.append(f"- Runs included: {len(runs)}")
    lines.append(f"- Longest clean suffix (`30/30`, `citation_coverage=1.0`, `format=1.0`, no harness failures): {clean_suffix}")
    lines.append(
        f"- Non-clean runs present: {'yes' if any(not run.clean_judge_run for run in runs) else 'no'}"
    )
    lines.append("")
    lines.append("## Run Table")
    lines.append("")
    lines.append("| Eval Artifact | Judge | Pass Rate | Accuracy | Grounding | Clarity | Citation | Format | TTFT p50 | Avg TTFT | Failures |")
    lines.append("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for run in runs:
        lines.append(
            "| "
            f"`{run.eval_path.name}` | "
            f"`{run.judge_path.name if run.judge_path else 'n/a'}` | "
            f"{run.pass_ratio} | "
            f"{_fmt_float(run.avg_accuracy)} | "
            f"{_fmt_float(run.avg_grounding)} | "
            f"{_fmt_float(run.avg_clarity)} | "
            f"{_fmt_float(run.citation_coverage)} | "
            f"{_fmt_float(run.format_compliance)} | "
            f"{_fmt_ms(run.ttft_p50_ms)} | "
            f"{_fmt_ms(run.avg_ttft_ms)} | "
            f"{run.failures} |"
        )
    lines.append("")
    lines.append("## Critical Cases")
    lines.append("")
    lines.append("| Run | 89fd4fbc | 4aa0f4e2 | 6e8d0c41 | acd3200d |")
    lines.append("|:--|:--:|:--:|:--:|:--:|")
    for run in runs:
        lines.append(
            f"| `{run.eval_path.name}` | "
            f"{run.critical_status[CRITICAL_CASES[0]]} | "
            f"{run.critical_status[CRITICAL_CASES[1]]} | "
            f"{run.critical_status[CRITICAL_CASES[2]]} | "
            f"{run.critical_status[CRITICAL_CASES[3]]} |"
        )
    lines.append("")
    lines.append("## Failure Taxonomy")
    lines.append("")
    counts = _taxonomy_counts(runs)
    if not counts:
        lines.append("- No judge FAIL cases across the included runs.")
    else:
        for taxonomy, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{taxonomy}`: {count}")
    lines.append("")
    lines.append("## Failure Replay")
    lines.append("")
    if not any(run.failed_cases for run in runs):
        lines.append("- No judge FAIL cases across the included runs.")
    else:
        for run in runs:
            if not run.failed_cases:
                continue
            lines.append(f"### `{run.eval_path.name}`")
            lines.append("")
            for failure in run.failed_cases:
                lines.append(f"- `question_id`: `{failure.case_id}`")
                lines.append(f"  `taxonomy`: `{failure.taxonomy}`")
                lines.append(f"  `ttft_ms`: `{_fmt_ms(failure.ttft_ms)}` | `model_llm`: `{failure.model_llm or 'n/a'}`")
                lines.append(f"  `used_pages`: `{failure.used_pages}`")
                lines.append(f"  `context_page_ids`: `{failure.context_page_ids}`")
                if failure.unsupported_claims:
                    lines.append(f"  `unsupported_claims`: `{failure.unsupported_claims}`")
                lines.append(f"  `recommended_fix`: {_truncate(failure.recommended_fix, limit=260) or 'n/a'}")
                lines.append(f"  `answer`: {_truncate(failure.answer, limit=320) or 'n/a'}")
                lines.append(f"  `question`: {_truncate(failure.question, limit=260) or 'n/a'}")
            lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `pass_rate` is the LLM-as-judge free-text pass rate from the eval summary.")
    lines.append("- `retrieved_chunk_ids` in competition submission must remain aligned to `used_page_ids`, not full context.")
    lines.append("- Local doc-ref hit-rate may be unavailable outside the Docker/Qdrant network; this does not affect judge scoring.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown evaluation dossier from eval artifacts.")
    parser.add_argument("--eval", dest="eval_paths", nargs="+", required=True, help="Eval JSON artifact paths")
    parser.add_argument("--out", required=True, help="Output markdown path")
    args = parser.parse_args()

    runs = [_collect_run(Path(raw).resolve()) for raw in args.eval_paths]
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_markdown(runs), encoding="utf-8")


if __name__ == "__main__":
    main()
