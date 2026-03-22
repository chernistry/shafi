# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", payload)


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in value if isinstance(item, dict)]


def _coerce_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _f_beta(predicted: set[str], gold: set[str], beta: float = 2.5) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    if precision <= 0 and recall <= 0:
        return 0.0
    beta_sq = beta * beta
    return ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)


def _page_score(page: JsonDict, *, family: str) -> float:
    flags = cast("JsonDict", page["flags"])
    page_number = _coerce_int(page.get("page_number"), default=0)
    score = 0.0
    if bool(flags.get("is_cover_page")):
        score += 8.0
    if family in {"official_law_number", "title_page"} and bool(flags.get("has_law_number")):
        score += 10.0
    if family == "citation_title" and bool(flags.get("has_citation_title")):
        score += 12.0
    if family == "who_made" and bool(flags.get("has_legislative_authority")):
        score += 14.0
    if family == "enactment" and bool(flags.get("has_enactment")):
        score += 14.0
    if family == "commencement" and bool(flags.get("has_commencement")):
        score += 14.0
    if family == "administration" and bool(flags.get("has_administration")):
        score += 14.0
    if family == "no_waiver" and bool(flags.get("has_no_waiver")):
        score += 14.0
    score += max(0.0, 3.0 - (page_number * 0.1))
    return score


def _select_baseline(case: JsonDict) -> list[str]:
    pages = _coerce_dict_list(case.get("pages"))
    family = str(case.get("family") or "")
    ranked = sorted(
        pages,
        key=lambda page: (-_page_score(page, family=family), _coerce_int(page.get("page_number"), default=9999), str(page.get("page_id") or "")),
    )
    return [str(ranked[0]["page_id"])] if ranked else []


def _select_page_family(case: JsonDict) -> list[str]:
    pages = _coerce_dict_list(case.get("pages"))
    family = str(case.get("family") or "")
    if not pages:
        return []

    cover_page = next((str(page["page_id"]) for page in pages if bool(cast("JsonDict", page["flags"]).get("is_cover_page"))), "")
    ranked = sorted(
        pages,
        key=lambda page: (-_page_score(page, family=family), _coerce_int(page.get("page_number"), default=9999), str(page.get("page_id") or "")),
    )
    selected: list[str] = []
    if cover_page and family in {
        "citation_title",
        "who_made",
        "enactment",
        "commencement",
        "administration",
        "no_waiver",
    }:
        selected.append(cover_page)
    for page in ranked:
        page_id = str(page["page_id"])
        if page_id in selected:
            continue
        selected.append(page_id)
        if len(selected) >= 2:
            break
    return selected[:2]


def run_falsifier(pack: JsonDict) -> JsonDict:
    cases = _coerce_dict_list(pack.get("cases"))
    results: list[JsonDict] = []
    improved = 0

    for case in cases:
        gold_pages = {str(page_id) for page_id in cast("list[object]", case.get("gold_pages") or [])}
        baseline = _select_baseline(case)
        candidate = _select_page_family(case)
        baseline_score = _f_beta(set(baseline), gold_pages)
        candidate_score = _f_beta(set(candidate), gold_pages)
        if candidate_score > baseline_score + 1e-9:
            improved += 1
        results.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "question": case["question"],
                "gold_pages": sorted(gold_pages),
                "baseline_pages": baseline,
                "candidate_pages": candidate,
                "baseline_f_beta": baseline_score,
                "candidate_f_beta": candidate_score,
                "improved": candidate_score > baseline_score + 1e-9,
            }
        )

    mean_baseline = sum(cast("float", row["baseline_f_beta"]) for row in results) / len(results)
    mean_candidate = sum(cast("float", row["candidate_f_beta"]) for row in results) / len(results)
    return {
        "summary": {
            "case_count": len(results),
            "improved_case_count": improved,
            "baseline_mean_f_beta": mean_baseline,
            "candidate_mean_f_beta": mean_candidate,
            "delta_mean_f_beta": mean_candidate - mean_baseline,
            "verdict": "win" if improved >= 3 and mean_candidate > mean_baseline else "fail_closed",
        },
        "cases": results,
    }


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload["summary"])
    lines = [
        "# Metadata Page-Family Falsifier",
        "",
        f"- case_count: `{summary['case_count']}`",
        f"- improved_case_count: `{summary['improved_case_count']}`",
        f"- baseline_mean_f_beta: `{summary['baseline_mean_f_beta']:.6f}`",
        f"- candidate_mean_f_beta: `{summary['candidate_mean_f_beta']:.6f}`",
        f"- delta_mean_f_beta: `{summary['delta_mean_f_beta']:.6f}`",
        f"- verdict: `{summary['verdict']}`",
        "",
        "## Improved Cases",
        "",
    ]
    improved_rows = [row for row in cast("list[JsonDict]", payload["cases"]) if bool(row["improved"])]
    if not improved_rows:
        lines.append("- none")
        return "\n".join(lines)
    for row in improved_rows:
        lines.extend(
            [
                f"### {row['case_id']}",
                f"- family: `{row['family']}`",
                f"- question: {row['question']}",
                f"- gold_pages: `{row['gold_pages']}`",
                f"- baseline_pages: `{row['baseline_pages']}`",
                f"- candidate_pages: `{row['candidate_pages']}`",
                f"- baseline_f_beta: `{row['baseline_f_beta']:.6f}`",
                f"- candidate_f_beta: `{row['candidate_f_beta']:.6f}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the top-2 metadata page-family falsifier.")
    parser.add_argument("--pack-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_falsifier(_load_json(args.pack_json))
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
