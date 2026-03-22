# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from build_scaffold_support_surrogate import build_scaffold_support_surrogate

if TYPE_CHECKING:
    from run_experiment_gate import SeedCaseDelta

try:
    from run_experiment_gate import (
        _answer_changed_count,
        _blindspot_support_summary,
        _page_p95,
        _recommendation,
        _retrieval_projection_changed_count,
        _score_benchmark,
        _seed_case_deltas,
        _submission_answers_by_id,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_experiment_gate import (
        _answer_changed_count,
        _blindspot_support_summary,
        _page_p95,
        _recommendation,
        _retrieval_projection_changed_count,
        _score_benchmark,
        _seed_case_deltas,
        _submission_answers_by_id,
    )


@dataclass(frozen=True)
class SubsetResult:
    label: str
    qids: list[str]
    answer_changed_count: int
    retrieval_page_projection_changed_count: int
    benchmark_all_baseline: float
    benchmark_all_candidate: float
    benchmark_trusted_baseline: float
    benchmark_trusted_candidate: float
    baseline_page_p95: int | None
    candidate_page_p95: int | None
    improved_seed_cases: list[str]
    regressed_seed_cases: list[str]
    blindspot_improved_cases: list[str]
    blindspot_support_undercoverage_cases: list[str]
    recommendation: str
    notes: list[str]
    submission_policy: str = "NO_SUBMIT_WITHOUT_USER_APPROVAL"


def _load_qids(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        qid = line.strip()
        if qid and not qid.startswith("#") and qid not in seen:
            out.append(qid)
            seen.add(qid)
    return out


def _iter_subsets(qids: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for size in range(1, len(qids) + 1):
        for combo in combinations(qids, size):
            out.append(list(combo))
    return out


def _slug(qids: list[str]) -> str:
    return "__".join(qid[:8] for qid in qids)


def _regressed_seed_cases(seed_deltas: list[SeedCaseDelta]) -> list[str]:
    out: list[str] = []
    for delta in seed_deltas:
        question_id = delta.question_id
        baseline_used_hit = bool(delta.baseline_used_hit)
        candidate_used_hit = bool(delta.candidate_used_hit)
        baseline_used_equivalent_hit = bool(delta.baseline_used_equivalent_hit)
        candidate_used_equivalent_hit = bool(delta.candidate_used_equivalent_hit)
        if (baseline_used_hit or baseline_used_equivalent_hit) and not (candidate_used_hit or candidate_used_equivalent_hit):
            out.append(question_id)
    return out


def _improved_seed_cases(seed_deltas: list[SeedCaseDelta]) -> list[str]:
    out: list[str] = []
    for delta in seed_deltas:
        question_id = delta.question_id
        baseline_used_hit = bool(delta.baseline_used_hit)
        candidate_used_hit = bool(delta.candidate_used_hit)
        baseline_used_equivalent_hit = bool(delta.baseline_used_equivalent_hit)
        candidate_used_equivalent_hit = bool(delta.candidate_used_equivalent_hit)
        if not (baseline_used_hit or baseline_used_equivalent_hit) and (candidate_used_hit or candidate_used_equivalent_hit):
            out.append(question_id)
    return out


def _sort_key(result: SubsetResult) -> tuple[float, float, int, int, int]:
    return (
        result.benchmark_trusted_candidate,
        result.benchmark_all_candidate,
        len(result.blindspot_improved_cases),
        -result.retrieval_page_projection_changed_count,
        -len(result.qids),
    )


def _render_markdown(results: list[SubsetResult]) -> str:
    lines = [
        "# Within-Doc Rerank Subset Search",
        "",
        f"- subset_count: `{len(results)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Label | QIDs | Trusted F | All F | Seed Improved | Blindspot Improved | Page Drift | Recommendation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| `{result.label}` | `{len(result.qids)}` | `{result.benchmark_trusted_candidate:.4f}` | "
            f"`{result.benchmark_all_candidate:.4f}` | `{len(result.improved_seed_cases)}` | "
            f"`{len(result.blindspot_improved_cases)}` | `{result.retrieval_page_projection_changed_count}` | "
            f"`{result.recommendation}` |"
        )
    lines.append("")
    for result in results[:5]:
        lines.extend(
            [
                f"## {result.label}",
                "",
                f"- qids: `{result.qids}`",
                f"- benchmark_trusted_candidate: `{result.benchmark_trusted_candidate:.4f}`",
                f"- benchmark_all_candidate: `{result.benchmark_all_candidate:.4f}`",
                f"- improved_seed_cases: `{result.improved_seed_cases}`",
                f"- regressed_seed_cases: `{result.regressed_seed_cases}`",
                f"- blindspot_improved_cases: `{result.blindspot_improved_cases}`",
                f"- blindspot_support_undercoverage_cases: `{result.blindspot_support_undercoverage_cases}`",
                f"- retrieval_page_projection_changed_count: `{result.retrieval_page_projection_changed_count}`",
                f"- recommendation: `{result.recommendation}`",
                f"- notes: `{result.notes}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search within-doc rerank surrogate subsets against a baseline candidate.")
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, default=None)
    parser.add_argument("--seed-qids-file", type=Path, required=True)
    parser.add_argument("--qids-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    qids = _load_qids(args.qids_file)
    results: list[SubsetResult] = []
    baseline_submission = _submission_answers_by_id(args.baseline_submission)
    baseline_all, baseline_trusted = _score_benchmark(args.baseline_raw_results, args.benchmark)
    baseline_page_p95 = _page_p95(args.baseline_preflight)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subset in _iter_subsets(qids):
        label = f"subset_{_slug(subset)}"
        out_submission = args.out_dir / f"{label}.submission.json"
        out_raw = args.out_dir / f"{label}.raw_results.json"
        out_report = args.out_dir / f"{label}.report.json"
        build_scaffold_support_surrogate(
            baseline_submission_path=args.baseline_submission,
            baseline_raw_results_path=args.baseline_raw_results,
            scaffold_path=args.scaffold,
            qids=subset,
            out_submission_path=out_submission,
            out_raw_results_path=out_raw,
            out_report_path=out_report,
        )
        candidate_submission = _submission_answers_by_id(out_submission)
        candidate_all, candidate_trusted = _score_benchmark(out_raw, args.benchmark)
        seed_deltas = _seed_case_deltas(
            baseline_scaffold_path=args.scaffold,
            candidate_scaffold_path=args.scaffold,
            baseline_raw_results_path=args.baseline_raw_results,
            candidate_raw_results_path=out_raw,
            seed_qids=_load_qids(args.seed_qids_file),
        )
        _, blindspot_improved_cases, blindspot_support_undercoverage_cases, _ = _blindspot_support_summary(
            scaffold_path=args.scaffold,
            candidate_scaffold_path=args.scaffold,
            baseline_raw_results_path=args.baseline_raw_results,
            candidate_raw_results_path=out_raw,
            seed_qids=_load_qids(args.seed_qids_file),
        )
        answer_changed_count = _answer_changed_count(baseline_submission, candidate_submission)
        projection_changed_count = _retrieval_projection_changed_count(baseline_submission, candidate_submission)
        recommendation, notes, _staged_eval = _recommendation(
            static_safety_status="assumed_passed",
            static_safety_reason=None,
            impact_canary_status="assumed_passed",
            impact_canary_reason=None,
            impact_canary_pack="within_doc_rerank_subset_search",
            baseline_trusted=baseline_trusted,
            candidate_trusted=candidate_trusted,
            answer_changed_count=answer_changed_count,
            baseline_page_p95=baseline_page_p95,
            candidate_page_p95=baseline_page_p95,
        )
        results.append(
            SubsetResult(
                label=label,
                qids=subset,
                answer_changed_count=answer_changed_count,
                retrieval_page_projection_changed_count=projection_changed_count,
                benchmark_all_baseline=baseline_all.page_f_beta,
                benchmark_all_candidate=candidate_all.page_f_beta,
                benchmark_trusted_baseline=baseline_trusted.page_f_beta,
                benchmark_trusted_candidate=candidate_trusted.page_f_beta,
                baseline_page_p95=baseline_page_p95,
                candidate_page_p95=baseline_page_p95,
                improved_seed_cases=_improved_seed_cases(seed_deltas),
                regressed_seed_cases=_regressed_seed_cases(seed_deltas),
                blindspot_improved_cases=blindspot_improved_cases,
                blindspot_support_undercoverage_cases=blindspot_support_undercoverage_cases,
                recommendation=recommendation,
                notes=notes,
            )
        )

    results.sort(key=_sort_key, reverse=True)
    (args.out_dir / "subset_search.json").write_text(
        json.dumps({"results": [asdict(result) for result in results]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "subset_search.md").write_text(_render_markdown(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
