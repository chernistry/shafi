from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class FrontierCandidate:
    qids: list[str]
    labels: list[str]
    recommendation: str
    answer_changed_count: int
    retrieval_page_projection_changed_count: int
    benchmark_all_delta: float
    benchmark_trusted_delta: float
    judge_pass_delta: float
    judge_grounding_delta: float
    candidate_page_p95: int


def _load_candidates(path: Path) -> list[FrontierCandidate]:
    obj_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj_raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    obj = cast("JsonDict", obj_raw)
    results_obj = obj.get("results")
    if not isinstance(results_obj, list):
        raise ValueError(f"Expected results array in {path}")
    results = cast("list[object]", results_obj)

    rows: list[FrontierCandidate] = []
    for raw in results:
        if not isinstance(raw, dict):
            continue
        row = cast("JsonDict", raw)
        baseline_all = float(row.get("benchmark_all_baseline") or 0.0)
        candidate_all = float(row.get("benchmark_all_candidate") or 0.0)
        baseline_trusted = float(row.get("benchmark_trusted_baseline") or 0.0)
        candidate_trusted = float(row.get("benchmark_trusted_candidate") or 0.0)
        judge_pass_baseline = float(row.get("judge_pass_rate_baseline") or 0.0)
        judge_pass_candidate = float(row.get("judge_pass_rate_candidate") or 0.0)
        judge_grounding_baseline = float(row.get("judge_grounding_baseline") or 0.0)
        judge_grounding_candidate = float(row.get("judge_grounding_candidate") or 0.0)

        rows.append(
            FrontierCandidate(
                qids=[str(item) for item in cast("list[object]", row.get("qids") or []) if str(item).strip()],
                labels=[str(item) for item in cast("list[object]", row.get("labels") or []) if str(item).strip()],
                recommendation=str(row.get("recommendation") or "").strip(),
                answer_changed_count=int(row.get("answer_changed_count") or 0),
                retrieval_page_projection_changed_count=int(row.get("retrieval_page_projection_changed_count") or 0),
                benchmark_all_delta=candidate_all - baseline_all,
                benchmark_trusted_delta=candidate_trusted - baseline_trusted,
                judge_pass_delta=judge_pass_candidate - judge_pass_baseline,
                judge_grounding_delta=judge_grounding_candidate - judge_grounding_baseline,
                candidate_page_p95=int(row.get("candidate_page_p95") or 0),
            )
        )
    return rows


def _eligible(rows: list[FrontierCandidate]) -> list[FrontierCandidate]:
    return [
        row
        for row in rows
        if row.recommendation == "PROMISING" and row.answer_changed_count == 0
    ]


def _select_conservative(rows: list[FrontierCandidate]) -> FrontierCandidate | None:
    eligible = _eligible(rows)
    if not eligible:
        return None
    max_trusted = max(row.benchmark_trusted_delta for row in eligible)
    trusted_shortlist = [row for row in eligible if row.benchmark_trusted_delta >= max_trusted - 1e-9]
    max_all = max(row.benchmark_all_delta for row in trusted_shortlist)
    shortlist = [row for row in trusted_shortlist if row.benchmark_all_delta >= max_all - 1e-9]
    return min(
        shortlist,
        key=lambda row: (
            row.retrieval_page_projection_changed_count,
            row.candidate_page_p95,
            -row.judge_pass_delta,
            -row.judge_grounding_delta,
        ),
    )


def _select_aggressive(rows: list[FrontierCandidate]) -> FrontierCandidate | None:
    eligible = _eligible(rows)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row.benchmark_all_delta,
            row.benchmark_trusted_delta,
            row.judge_pass_delta,
            row.judge_grounding_delta,
            -row.retrieval_page_projection_changed_count,
        ),
    )


def _dominates(lhs: FrontierCandidate, rhs: FrontierCandidate) -> bool:
    return (
        lhs.answer_changed_count <= rhs.answer_changed_count
        and lhs.retrieval_page_projection_changed_count <= rhs.retrieval_page_projection_changed_count
        and lhs.candidate_page_p95 <= rhs.candidate_page_p95
        and lhs.benchmark_trusted_delta >= rhs.benchmark_trusted_delta
        and lhs.benchmark_all_delta >= rhs.benchmark_all_delta
        and lhs.judge_pass_delta >= rhs.judge_pass_delta
        and lhs.judge_grounding_delta >= rhs.judge_grounding_delta
        and (
            lhs.answer_changed_count < rhs.answer_changed_count
            or lhs.retrieval_page_projection_changed_count < rhs.retrieval_page_projection_changed_count
            or lhs.candidate_page_p95 < rhs.candidate_page_p95
            or lhs.benchmark_trusted_delta > rhs.benchmark_trusted_delta
            or lhs.benchmark_all_delta > rhs.benchmark_all_delta
            or lhs.judge_pass_delta > rhs.judge_pass_delta
            or lhs.judge_grounding_delta > rhs.judge_grounding_delta
        )
    )


def _pareto_frontier(rows: list[FrontierCandidate]) -> list[FrontierCandidate]:
    eligible = _eligible(rows)
    frontier: list[FrontierCandidate] = []
    for candidate in eligible:
        if any(_dominates(other, candidate) for other in eligible if other is not candidate):
            continue
        frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (
            -row.benchmark_trusted_delta,
            -row.benchmark_all_delta,
            row.retrieval_page_projection_changed_count,
            row.candidate_page_p95,
        ),
    )


def _label(row: FrontierCandidate | None) -> str:
    if row is None:
        return ""
    return " + ".join(qid[:6] for qid in row.qids)


def _render_markdown(
    *,
    source_path: Path,
    conservative: FrontierCandidate | None,
    aggressive: FrontierCandidate | None,
    frontier: list[FrontierCandidate],
) -> str:
    lines = [
        "# Support Frontier Ranking",
        "",
        f"- source: `{source_path}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        f"- conservative_pick: `{_label(conservative) or 'none'}`",
        f"- aggressive_pick: `{_label(aggressive) or 'none'}`",
        f"- frontier_size: `{len(frontier)}`",
        "",
        "| Rank | QIDs | Page Drift | Hidden-G Trusted Δ | Hidden-G All Δ | Judge Pass Δ | Judge Grounding Δ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(frontier, start=1):
        lines.append(
            f"| {index} | {' + '.join(qid[:6] for qid in row.qids)} | "
            f"{row.retrieval_page_projection_changed_count} | "
            f"{row.benchmark_trusted_delta:.4f} | {row.benchmark_all_delta:.4f} | "
            f"{row.judge_pass_delta:.4f} | {row.judge_grounding_delta:.4f} |"
        )
    lines.append("")
    if conservative is not None:
        lines.extend(
            [
                "## Conservative Pick",
                "",
                f"- qids: `{conservative.qids}`",
                f"- page_drift: `{conservative.retrieval_page_projection_changed_count}`",
                f"- hidden_g_trusted_delta: `{conservative.benchmark_trusted_delta:.4f}`",
                f"- hidden_g_all_delta: `{conservative.benchmark_all_delta:.4f}`",
                "",
            ]
        )
    if aggressive is not None:
        lines.extend(
            [
                "## Aggressive Pick",
                "",
                f"- qids: `{aggressive.qids}`",
                f"- page_drift: `{aggressive.retrieval_page_projection_changed_count}`",
                f"- hidden_g_trusted_delta: `{aggressive.benchmark_trusted_delta:.4f}`",
                f"- hidden_g_all_delta: `{aggressive.benchmark_all_delta:.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank bounded support frontier candidates from combo-search output.")
    parser.add_argument("--combo-json", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_candidates(args.combo_json.resolve())
    conservative = _select_conservative(rows)
    aggressive = _select_aggressive(rows)
    frontier = _pareto_frontier(rows)

    payload = {
        "source": str(args.combo_json.resolve()),
        "conservative_pick": None if conservative is None else asdict(conservative),
        "aggressive_pick": None if aggressive is None else asdict(aggressive),
        "frontier": [asdict(row) for row in frontier],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(
        _render_markdown(
            source_path=args.combo_json.resolve(),
            conservative=conservative,
            aggressive=aggressive,
            frontier=frontier,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
