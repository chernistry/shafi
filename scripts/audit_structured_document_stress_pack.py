# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

_PAGE_REF_RE = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


def _load_pack(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at {path}")
    return cast("dict[str, Any]", raw)


def _requested_page(question: str) -> int | None:
    lowered = str(question or "").strip().lower()
    if "second page" in lowered:
        return 2
    if "first page" in lowered or "title page" in lowered or "cover page" in lowered:
        return 1
    match = _PAGE_REF_RE.search(lowered)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def classify_stage(case: dict[str, Any]) -> str:
    question = str(case.get("question") or "")
    answer_type = str(case.get("answer_type") or "free_text").strip().lower()
    answer = str(case.get("current_answer") or "").strip()
    used_pages = [
        str(page).strip()
        for page in cast("list[object]", case.get("current_used_pages") or [])
        if str(page).strip()
    ]
    subtypes = {
        str(subtype).strip()
        for subtype in cast("list[object]", case.get("subtypes") or [])
        if str(subtype).strip()
    }

    if answer_type != "free_text" and answer.lower() == "null":
        return "unsupported_null" if not used_pages else "answer_support_mismatch"
    if answer_type == "free_text" and answer.lower().startswith("there is no information"):
        return "unsupported_free_text"
    if not used_pages:
        return "context_or_retrieval_miss"

    requested_page = _requested_page(question)
    if (
        requested_page is not None
        and {"title_page", "page_specific"} & subtypes
        and not any(page.endswith(f"_{requested_page}") for page in used_pages)
    ):
        return "right_doc_wrong_page"

    if "article_provision" in subtypes and not answer:
        return "answerer_miss"

    return "nonfailure"


def audit_pack(pack: dict[str, Any]) -> dict[str, Any]:
    cases = [cast("dict[str, Any]", case) for case in pack.get("cases", []) if isinstance(case, dict)]
    annotated_cases: list[dict[str, Any]] = []
    by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overall_stage_counts: Counter[str] = Counter()

    for case in cases:
        stage = classify_stage(case)
        annotation = {**case, "observed_stage": stage}
        annotated_cases.append(annotation)
        overall_stage_counts[stage] += 1
        for subtype in cast("list[str]", case.get("subtypes") or []):
            by_subtype[subtype].append(annotation)

    subtype_stats: dict[str, Any] = {}
    for subtype, subtype_cases in sorted(by_subtype.items()):
        total = len(subtype_cases)
        null_count = sum(1 for case in subtype_cases if bool(case.get("null_answer")))
        empty_used_count = sum(1 for case in subtype_cases if bool(case.get("empty_used_pages")))
        stage_counts = Counter(str(case.get("observed_stage") or "") for case in subtype_cases)
        nonfailure_count = stage_counts.get("nonfailure", 0)
        subtype_stats[subtype] = {
            "total_cases": total,
            "hit_rate": round(nonfailure_count / total, 4) if total else 0.0,
            "null_rate": round(null_count / total, 4) if total else 0.0,
            "empty_used_page_rate": round(empty_used_count / total, 4) if total else 0.0,
            "stage_breakdown": dict(sorted(stage_counts.items())),
        }

    repeatable_failures = {
        subtype: stats["stage_breakdown"]
        for subtype, stats in subtype_stats.items()
        if sum(count for stage, count in stats["stage_breakdown"].items() if stage != "nonfailure") >= 2
    }

    return {
        "ticket": 70,
        "source_pack": pack.get("source_artifact") or "",
        "summary": {
            "total_cases": len(cases),
            "overall_stage_breakdown": dict(sorted(overall_stage_counts.items())),
            "repeatable_failure_clusters": repeatable_failures,
            "decision": (
                "no_structure_patch"
                if not repeatable_failures or set(repeatable_failures).issubset({"title_page", "page_specific", "article_provision"})
                else "structure_patch_candidate"
            ),
        },
        "subtype_stats": subtype_stats,
        "cases": annotated_cases,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Structured Retrieval Gap Audit",
        "",
        f"- total cases: `{report['summary']['total_cases']}`",
        f"- decision: `{report['summary']['decision']}`",
        "",
        "## Overall Stage Breakdown",
        "",
    ]
    for stage, count in cast("dict[str, int]", report["summary"]["overall_stage_breakdown"]).items():
        lines.append(f"- `{stage}`: `{count}`")
    lines.extend(["", "## By Subtype", ""])
    for subtype, stats in cast("dict[str, Any]", report["subtype_stats"]).items():
        lines.append(f"### {subtype}")
        lines.append(f"- total: `{stats['total_cases']}`")
        lines.append(f"- hit_rate: `{stats['hit_rate']}`")
        lines.append(f"- null_rate: `{stats['null_rate']}`")
        lines.append(f"- empty_used_page_rate: `{stats['empty_used_page_rate']}`")
        lines.append(f"- stage_breakdown: `{stats['stage_breakdown']}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the champion structural-stress pack.")
    parser.add_argument("--pack", required=True, help="Path to structured_document_stress_pack.json")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-md", required=True, help="Output Markdown path")
    args = parser.parse_args(argv)

    pack = _load_pack(Path(args.pack))
    report = audit_pack(pack)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
