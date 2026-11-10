# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from analyze_leaderboard import build_summary as build_leaderboard_summary
from analyze_leaderboard import load_rows as load_leaderboard_rows


@dataclass(frozen=True)
class HistoryRow:
    version: str
    strategy: str
    det: float
    asst: float
    g: float
    total: float
    result: str


def _load_json(path: Path) -> dict[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("dict[str, object]", obj)


def _as_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _as_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _infer_quantization_denominator(
    values: list[float],
    *,
    min_denominator: int = 2,
    max_denominator: int = 2000,
    tolerance: float = 5e-4,
) -> int | None:
    if not values:
        return None
    for denominator in range(min_denominator, max_denominator + 1):
        max_error = max(abs((value * denominator) - round(value * denominator)) for value in values)
        if max_error <= tolerance:
            return denominator
    return None


def _parse_history(path: Path | None) -> list[HistoryRow]:
    if path is None or not path.exists():
        return []
    rows: list[HistoryRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("|"):
            continue
        cells = [part.strip() for part in text.strip("|").split("|")]
        if len(cells) != 7:
            continue
        if cells[0].lower() == "version" or set(cells[0]) == {"-"}:
            continue
        try:
            rows.append(
                HistoryRow(
                    version=cells[0].replace("**", "").strip(),
                    strategy=cells[1].replace("**", "").strip(),
                    det=float(cells[2].replace("**", "").strip()),
                    asst=float(cells[3].replace("**", "").strip()),
                    g=float(cells[4].replace("**", "").strip()),
                    total=float(cells[5].replace("**", "").strip()),
                    result=cells[6].replace("**", "").strip(),
                )
            )
        except ValueError:
            continue
    return rows


def _classify_history_strategy(strategy: str) -> str:
    lowered = strategy.lower()
    if any(token in lowered for token in ("context pages", "reranked pages", "page swap", "page mutation", "page-only")):
        return "page_mutation"
    if any(token in lowered for token in ("suffix", "exactness", "onora", "casing")):
        return "answer_only"
    if "baseline" in lowered:
        return "baseline"
    return "other"


def _history_summary(rows: list[HistoryRow]) -> dict[str, object]:
    if not rows:
        return {}
    champion = max(rows, key=lambda row: row.total)
    page_rows = [row for row in rows if _classify_history_strategy(row.strategy) == "page_mutation"]
    answer_rows = [row for row in rows if _classify_history_strategy(row.strategy) == "answer_only"]

    def _mean_delta_total(items: list[HistoryRow]) -> float | None:
        if not items:
            return None
        return sum(item.total - champion.total for item in items) / len(items)

    def _mean_delta_g(items: list[HistoryRow]) -> float | None:
        if not items:
            return None
        return sum(item.g - champion.g for item in items) / len(items)

    return {
        "champion_version": champion.version,
        "champion_total": champion.total,
        "champion_g": champion.g,
        "page_mutation_versions": [row.version for row in page_rows],
        "answer_only_versions": [row.version for row in answer_rows],
        "page_mutation_mean_delta_total_vs_champion": _mean_delta_total(page_rows),
        "page_mutation_mean_delta_g_vs_champion": _mean_delta_g(page_rows),
        "answer_only_mean_delta_total_vs_champion": _mean_delta_total(answer_rows),
        "answer_only_mean_delta_g_vs_champion": _mean_delta_g(answer_rows),
    }


def build_summary(
    *,
    leaderboard_path: Path,
    team_name: str,
    history_path: Path | None = None,
    exactness_report_path: Path | None = None,
) -> dict[str, object]:
    rows = load_leaderboard_rows(leaderboard_path)
    leaderboard_summary = build_leaderboard_summary(rows, team_name=team_name)
    det_values = [row.det for row in rows]
    asst_values = [row.asst for row in rows]
    det_denominator = _infer_quantization_denominator(det_values, max_denominator=2000)
    asst_denominator = _infer_quantization_denominator(asst_values, max_denominator=1000)

    gtf = _as_float(leaderboard_summary["g"]) * _as_float(leaderboard_summary["t"]) * _as_float(leaderboard_summary["f"])
    delta_total_per_full_deterministic_answer = 0.01 * gtf
    delta_total_per_full_free_text_answer = 0.01 * gtf
    delta_total_per_free_text_step = (0.3 * (0.2 / 30.0)) * gtf
    delta_total_per_det_partial_tick = ((0.7 / float(det_denominator)) * gtf) if det_denominator else 0.0

    exactness_summary: dict[str, object] = {}
    if exactness_report_path is not None and exactness_report_path.exists():
        report = _load_json(exactness_report_path)
        answer_changed = _as_int(report.get("answer_changed_count"))
        page_changed = _as_int(report.get("page_changed_count"))
        nested_benchmark = report.get("hidden_g_page_benchmark")
        nested_page_metrics_identical = False
        if isinstance(nested_benchmark, dict):
            nested_page_metrics_identical = bool(cast("dict[str, object]", nested_benchmark).get("page_metrics_identical"))
        page_metrics_identical = bool(report.get("page_metrics_identical")) or nested_page_metrics_identical
        current_total = _as_float(leaderboard_summary["total"])
        exactness_summary = {
            "answer_changed_count": answer_changed,
            "page_changed_count": page_changed,
            "page_metrics_identical": page_metrics_identical,
            "strict_upper_bound_total_if_all_answer_changes_are_real": (
                current_total + (answer_changed * delta_total_per_full_deterministic_answer)
                if answer_changed > 0 and page_changed == 0
                else None
            ),
        }

    history_rows = _parse_history(history_path)
    return {
        "leaderboard_summary": leaderboard_summary,
        "det_lattice_denominator": det_denominator,
        "asst_lattice_denominator": asst_denominator,
        "det_interpretation": (
            "Det appears to lie on a 1/420 lattice, consistent with 70 deterministic items plus partial-credit ticks."
            if det_denominator == 420
            else (
                f"Det appears quantized on a 1/{det_denominator} lattice."
                if det_denominator is not None
                else "Det lattice could not be inferred cleanly."
            )
        ),
        "asst_interpretation": (
            "Asst appears to lie on a 1/150 lattice, consistent with 30 free-text items scored in 0.2 steps."
            if asst_denominator == 150
            else (
                f"Asst appears quantized on a 1/{asst_denominator} lattice."
                if asst_denominator is not None
                else "Asst lattice could not be inferred cleanly."
            )
        ),
        "delta_total_per_full_deterministic_answer": delta_total_per_full_deterministic_answer,
        "delta_total_per_full_free_text_answer": delta_total_per_full_free_text_answer,
        "delta_total_per_free_text_step": delta_total_per_free_text_step,
        "delta_total_per_det_partial_tick": delta_total_per_det_partial_tick,
        "history": _history_summary(history_rows),
        "exactness_estimate": exactness_summary,
    }


def build_report(summary: dict[str, object]) -> str:
    leaderboard_summary = cast("dict[str, object]", summary["leaderboard_summary"])
    history_summary = cast("dict[str, object]", summary.get("history") or {})
    exactness_summary = cast("dict[str, object]", summary.get("exactness_estimate") or {})
    lines = [
        "# Platform Scoring Reverse-Engineering Report",
        "",
        "## Confirmed Formula",
        "",
        "- `S = 0.7 * Det + 0.3 * Asst`",
        "- `Total = S * G * T * F`",
        "",
        "## Inferred Lattices",
        "",
        f"- Det lattice denominator: `{summary.get('det_lattice_denominator')}`",
        f"- Det interpretation: {summary.get('det_interpretation')}",
        f"- Asst lattice denominator: `{summary.get('asst_lattice_denominator')}`",
        f"- Asst interpretation: {summary.get('asst_interpretation')}",
        "",
        "## Current Team Geometry",
        "",
        f"- Team: `{leaderboard_summary.get('team_name')}`",
        f"- Rank: `{leaderboard_summary.get('rank')}`",
        f"- Total: `{_as_float(leaderboard_summary.get('total')):.6f}`",
        f"- S: `{_as_float(leaderboard_summary.get('s')):.6f}`",
        f"- G: `{_as_float(leaderboard_summary.get('g')):.6f}`",
        f"- `+1` full deterministic answer upper bound: `+{_as_float(summary.get('delta_total_per_full_deterministic_answer')):.6f}` total",
        f"- `+1` full free-text answer upper bound: `+{_as_float(summary.get('delta_total_per_full_free_text_answer')):.6f}` total",
        f"- `+0.2` judge step on one free-text answer: `+{_as_float(summary.get('delta_total_per_free_text_step')):.6f}` total",
        f"- `+1` Det partial-credit tick: `+{_as_float(summary.get('delta_total_per_det_partial_tick')):.6f}` total",
        "",
    ]

    if history_summary:
        lines.extend(
            [
                "## Historical Strategy Signal",
                "",
                f"- Champion version in history: `{history_summary.get('champion_version')}`",
                f"- Page-mutation versions: `{', '.join(cast('list[str]', history_summary.get('page_mutation_versions') or [])) or '(none)'}`",
                f"- Answer-only versions: `{', '.join(cast('list[str]', history_summary.get('answer_only_versions') or [])) or '(none)'}`",
                f"- Mean page-mutation `ΔTotal` vs champion: `{_as_float(history_summary.get('page_mutation_mean_delta_total_vs_champion')):+.6f}`",
                f"- Mean page-mutation `ΔG` vs champion: `{_as_float(history_summary.get('page_mutation_mean_delta_g_vs_champion')):+.6f}`",
                f"- Mean answer-only `ΔTotal` vs champion: `{_as_float(history_summary.get('answer_only_mean_delta_total_vs_champion')):+.6f}`",
                f"- Mean answer-only `ΔG` vs champion: `{_as_float(history_summary.get('answer_only_mean_delta_g_vs_champion')):+.6f}`",
                "",
            ]
        )

    if exactness_summary:
        upper_bound = exactness_summary.get("strict_upper_bound_total_if_all_answer_changes_are_real")
        lines.extend(
            [
                "## Exactness-Only Strict Estimate",
                "",
                f"- answer_changed_count: `{exactness_summary.get('answer_changed_count')}`",
                f"- page_changed_count: `{exactness_summary.get('page_changed_count')}`",
                f"- page_metrics_identical: `{exactness_summary.get('page_metrics_identical')}`",
                f"- strict upper-bound total if every answer delta is real: `{_as_float(upper_bound):.6f}`" if upper_bound is not None else "- strict upper-bound total if every answer delta is real: `(not applicable)`",
                "",
            ]
        )

    lines.extend(
        [
            "## Strict Interpretation",
            "",
            "- Use this report as a veto layer, not as permission to submit.",
            "- If a branch mutates pages broadly, historical evidence says its local estimate is untrustworthy even before platform submission.",
            "- If a branch is answer-only and page-stable, its upside can be bounded tightly without burning a submission.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse-engineer platform scoring lattice and strict local estimate.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--history-md", type=Path, default=None)
    parser.add_argument("--exactness-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    summary = build_summary(
        leaderboard_path=args.leaderboard,
        team_name=args.team,
        history_path=args.history_md,
        exactness_report_path=args.exactness_report,
    )
    report = build_report(summary)
    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
