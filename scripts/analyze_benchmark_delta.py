"""Compare per-case hidden-G benchmark deltas between two raw_results files."""
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

try:
    from score_page_benchmark import _load_benchmark, _score_case
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest/module import
    from scripts.score_page_benchmark import _load_benchmark, _score_case

JsonDict = dict[str, object]


@dataclass(frozen=True)
class CaseDelta:
    question_id: str
    trust_tier: str
    gold_origin: str
    audit_note: str
    gold_pages: list[str]
    baseline_f_beta: float
    candidate_f_beta: float
    delta_f_beta: float
    baseline_used_pages: list[str]
    candidate_used_pages: list[str]
    baseline_context_pages: list[str]
    candidate_context_pages: list[str]
    baseline_orphan_pages: list[str]
    candidate_orphan_pages: list[str]
    baseline_missing_pages: list[str]
    candidate_missing_pages: list[str]
    baseline_slot_recall: float | None
    candidate_slot_recall: float | None
    baseline_evidence_family_complete: bool | None
    candidate_evidence_family_complete: bool | None
    baseline_overprune_violations: int
    candidate_overprune_violations: int
    baseline_wrong_document_issue: bool
    candidate_wrong_document_issue: bool


def _load_json_list(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [cast("JsonDict", raw) for raw in cast("list[object]", obj) if isinstance(raw, dict)]


def _raw_results_by_id(path: Path) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    for raw in _load_json_list(path):
        case_obj = raw.get("case")
        case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def _eval_case_from_raw(raw_case: JsonDict | None) -> JsonDict | None:
    if raw_case is None:
        return None
    case_obj = raw_case.get("case")
    case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
    return {
        "question_id": str(case.get("case_id") or case.get("question_id") or "").strip(),
        "answer": raw_case.get("answer_text"),
        "telemetry": raw_case.get("telemetry") if isinstance(raw_case.get("telemetry"), dict) else {},
    }


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for raw in cast("list[object]", value) if (text := str(raw).strip())]


def _telemetry_pages(raw_case: JsonDict | None, key: str) -> list[str]:
    if raw_case is None:
        return []
    telemetry_obj = raw_case.get("telemetry")
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    return _coerce_str_list(telemetry.get(key))


def _build_case_deltas(
    *,
    baseline_raw_results: dict[str, JsonDict],
    candidate_raw_results: dict[str, JsonDict],
    benchmark_path: Path,
    beta: float,
) -> list[CaseDelta]:
    deltas: list[CaseDelta] = []
    for benchmark_case in _load_benchmark(benchmark_path):
        baseline_raw = baseline_raw_results.get(benchmark_case.question_id)
        candidate_raw = candidate_raw_results.get(benchmark_case.question_id)
        baseline_score = _score_case(benchmark_case, _eval_case_from_raw(baseline_raw), beta=beta)
        candidate_score = _score_case(benchmark_case, _eval_case_from_raw(candidate_raw), beta=beta)
        deltas.append(
            CaseDelta(
                question_id=benchmark_case.question_id,
                trust_tier=benchmark_case.trust_tier,
                gold_origin=benchmark_case.gold_origin,
                audit_note=benchmark_case.audit_note,
                gold_pages=list(benchmark_case.gold_page_ids),
                baseline_f_beta=baseline_score.f_beta,
                candidate_f_beta=candidate_score.f_beta,
                delta_f_beta=candidate_score.f_beta - baseline_score.f_beta,
                baseline_used_pages=_telemetry_pages(baseline_raw, "used_page_ids"),
                candidate_used_pages=_telemetry_pages(candidate_raw, "used_page_ids"),
                baseline_context_pages=_telemetry_pages(baseline_raw, "context_page_ids"),
                candidate_context_pages=_telemetry_pages(candidate_raw, "context_page_ids"),
                baseline_orphan_pages=list(baseline_score.orphan_pages),
                candidate_orphan_pages=list(candidate_score.orphan_pages),
                baseline_missing_pages=list(baseline_score.missing_pages),
                candidate_missing_pages=list(candidate_score.missing_pages),
                baseline_slot_recall=baseline_score.slot_recall,
                candidate_slot_recall=candidate_score.slot_recall,
                baseline_evidence_family_complete=baseline_score.evidence_family_complete,
                candidate_evidence_family_complete=candidate_score.evidence_family_complete,
                baseline_overprune_violations=baseline_score.overprune_violations,
                candidate_overprune_violations=candidate_score.overprune_violations,
                baseline_wrong_document_issue=baseline_score.wrong_document_issue,
                candidate_wrong_document_issue=candidate_score.wrong_document_issue,
            )
        )
    deltas.sort(
        key=lambda row: (
            0 if row.trust_tier == "trusted" else 1,
            -row.delta_f_beta,
            row.question_id,
        )
    )
    return deltas


def _mean_score(deltas: list[CaseDelta], attr: str) -> float:
    if not deltas:
        return 0.0
    return sum(float(getattr(delta, attr)) for delta in deltas) / len(deltas)


def _summaries(deltas: list[CaseDelta]) -> dict[str, dict[str, float | int]]:
    def summarize(rows: list[CaseDelta]) -> dict[str, float | int]:
        return {
            "cases": len(rows),
            "baseline_f_beta": round(_mean_score(rows, "baseline_f_beta"), 6),
            "candidate_f_beta": round(_mean_score(rows, "candidate_f_beta"), 6),
            "delta_f_beta": round(_mean_score(rows, "candidate_f_beta") - _mean_score(rows, "baseline_f_beta"), 6),
            "improved": sum(1 for row in rows if row.delta_f_beta > 1e-9),
            "regressed": sum(1 for row in rows if row.delta_f_beta < -1e-9),
            "unchanged": sum(1 for row in rows if abs(row.delta_f_beta) <= 1e-9),
        }

    trusted = [row for row in deltas if row.trust_tier == "trusted"]
    suspect = [row for row in deltas if row.trust_tier == "suspect"]
    return {
        "all": summarize(deltas),
        "trusted": summarize(trusted),
        "suspect": summarize(suspect),
    }


def _section(title: str, rows: list[CaseDelta]) -> list[str]:
    lines = ["", title, ""]
    if not rows:
        lines.append("- None")
        return lines
    for row in rows:
        lines.extend(
            [
                f"- `{row.question_id}`: delta={row.delta_f_beta:+.4f}, baseline={row.baseline_f_beta:.4f}, candidate={row.candidate_f_beta:.4f}",
                f"  trust_tier={row.trust_tier}, gold_origin={row.gold_origin}",
                f"  gold={row.gold_pages}",
                f"  baseline_used={row.baseline_used_pages}",
                f"  candidate_used={row.candidate_used_pages}",
                f"  baseline_context={row.baseline_context_pages}",
                f"  candidate_context={row.candidate_context_pages}",
            ]
        )
        if row.baseline_missing_pages or row.candidate_missing_pages:
            lines.append(
                f"  missing baseline={row.baseline_missing_pages} candidate={row.candidate_missing_pages}"
            )
        if row.baseline_orphan_pages or row.candidate_orphan_pages:
            lines.append(
                f"  orphan baseline={row.baseline_orphan_pages} candidate={row.candidate_orphan_pages}"
            )
        if row.baseline_slot_recall is not None or row.candidate_slot_recall is not None:
            lines.append(
                f"  slot_recall baseline={row.baseline_slot_recall} candidate={row.candidate_slot_recall}"
            )
        if (
            row.baseline_evidence_family_complete is not None
            or row.candidate_evidence_family_complete is not None
        ):
            lines.append(
                "  evidence_family_complete "
                f"baseline={row.baseline_evidence_family_complete} candidate={row.candidate_evidence_family_complete}"
            )
        if row.baseline_overprune_violations or row.candidate_overprune_violations:
            lines.append(
                "  overprune_violations "
                f"baseline={row.baseline_overprune_violations} candidate={row.candidate_overprune_violations}"
            )
        if row.baseline_wrong_document_issue or row.candidate_wrong_document_issue:
            lines.append(
                "  wrong_document_issue "
                f"baseline={row.baseline_wrong_document_issue} candidate={row.candidate_wrong_document_issue}"
            )
        if row.audit_note:
            lines.append(f"  audit_note={row.audit_note}")
    return lines


def render_report(*, baseline_label: str, candidate_label: str, deltas: list[CaseDelta]) -> str:
    summaries = _summaries(deltas)
    improved_trusted = sorted(
        [row for row in deltas if row.trust_tier == "trusted" and row.delta_f_beta > 1e-9],
        key=lambda row: (row.delta_f_beta, row.question_id),
        reverse=True,
    )
    regressed_trusted = sorted(
        [row for row in deltas if row.trust_tier == "trusted" and row.delta_f_beta < -1e-9],
        key=lambda row: (row.delta_f_beta, row.question_id),
    )
    improved_all = sorted(
        [row for row in deltas if row.delta_f_beta > 1e-9],
        key=lambda row: (row.delta_f_beta, row.question_id),
        reverse=True,
    )[:15]
    regressed_all = sorted(
        [row for row in deltas if row.delta_f_beta < -1e-9],
        key=lambda row: (row.delta_f_beta, row.question_id),
    )[:15]

    lines = [
        "# Benchmark Delta Report",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Candidate: `{candidate_label}`",
        "",
        "## Summary",
        "",
    ]
    for label in ("all", "trusted", "suspect"):
        summary = summaries[label]
        lines.append(
            f"- {label}: cases=`{summary['cases']}`, baseline=`{summary['baseline_f_beta']:.4f}`, "
            f"candidate=`{summary['candidate_f_beta']:.4f}`, delta=`{summary['delta_f_beta']:+.4f}`, "
            f"improved=`{summary['improved']}`, regressed=`{summary['regressed']}`, unchanged=`{summary['unchanged']}`"
        )

    lines.extend(_section("## Trusted Improvements", improved_trusted))
    lines.extend(_section("## Trusted Regressions", regressed_trusted))
    lines.extend(_section("## Top Improvements", improved_all))
    lines.extend(_section("## Top Regressions", regressed_all))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare per-case hidden-G benchmark deltas between two raw_results files.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--baseline-raw-results", type=Path, required=True)
    parser.add_argument("--candidate-raw-results", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=2.5)
    args = parser.parse_args()

    deltas = _build_case_deltas(
        baseline_raw_results=_raw_results_by_id(args.baseline_raw_results),
        candidate_raw_results=_raw_results_by_id(args.candidate_raw_results),
        benchmark_path=args.benchmark,
        beta=args.beta,
    )
    args.out.write_text(
        render_report(
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            deltas=deltas,
        )
        + "\n",
        encoding="utf-8",
    )
    args.json_out.write_text(
        json.dumps(
            {
                "baseline_label": args.baseline_label,
                "candidate_label": args.candidate_label,
                "beta": args.beta,
                "summary": _summaries(deltas),
                "deltas": [asdict(row) for row in deltas],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
