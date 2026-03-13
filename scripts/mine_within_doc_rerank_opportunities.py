from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|SCT|ENF|DEC|TCD|ARB)\s+\d{1,4}/\d{4}\b", re.IGNORECASE)
_PARTY_TERMS = (
    "claimant",
    "claimants",
    "defendant",
    "defendants",
    "respondent",
    "respondents",
    "appellant",
    "appellants",
    "applicant",
    "applicants",
    "party",
    "parties",
)
_TITLE_TERMS = ("title page", "cover page", "first page", "caption", "header")
_PAGE2_TERMS = ("page 2", "second page")
_ARTICLE_TERMS = ("article", "schedule", "definitions", "law number", "operating law", "trust law")


@dataclass(frozen=True)
class OpportunityRow:
    question_id: str
    family: str
    question: str
    failure_class: str
    route_family: str
    support_shape_class: str
    gold_pages: list[str]
    retrieved_pages: list[str]
    context_pages: list[str]
    used_pages: list[str]
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    context_doc_ids: list[str]
    used_doc_ids: list[str]
    gold_in_retrieved: bool
    gold_in_context: bool
    gold_in_used: bool
    context_page_budget: int
    max_retrieved_pages_per_doc: int
    max_context_pages_per_doc: int
    same_doc_chunk_spam: bool
    within_doc_rerank_opportunity: bool
    doc_family_collapse_opportunity: bool
    collapsed_context_doc_ids: list[str]


@dataclass(frozen=True)
class FamilySummary:
    family: str
    case_count: int
    gold_in_retrieved_count: int
    gold_in_context_count: int
    gold_in_used_count: int
    same_doc_chunk_spam_count: int
    opportunity_count: int
    collapse_opportunity_count: int
    likely_actionable: bool
    suggested_context_page_budget: int
    suggested_max_pages_per_doc: int
    question_id_examples: list[str]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_raw_results(path: Path) -> dict[str, JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected list in {path}")
    out: dict[str, JsonDict] = {}
    for item in cast("list[object]", obj):
        if not isinstance(item, dict):
            continue
        item_dict = cast("JsonDict", item)
        case_obj = item_dict.get("case")
        if not isinstance(case_obj, dict):
            continue
        case_dict = cast("JsonDict", case_obj)
        qid = str(case_dict.get("case_id") or "").strip()
        if not qid:
            continue
        out[qid] = item_dict
    return out


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in cast("list[object]", value) if str(item).strip()]


def _page_doc(page_id: str) -> str:
    text = str(page_id).strip()
    if "_" not in text:
        return text
    return text.rsplit("_", 1)[0]


def _doc_sequence(page_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for page_id in page_ids:
        doc_id = _page_doc(page_id)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _page_counts_by_doc(page_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page_id in page_ids:
        doc_id = _page_doc(page_id)
        if not doc_id:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
    return counts


def _first_page_per_doc(page_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for page_id in page_ids:
        doc_id = _page_doc(page_id)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(page_id)
    return out


def _percentile_int(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    return int(ordered[index])


def _infer_family(*, question: str, failure_class: str, support_shape_class: str) -> str:
    q = re.sub(r"\s+", " ", question).strip().lower()
    ref_count = len(_CASE_REF_RE.findall(question))
    if any(term in q for term in _PAGE2_TERMS):
        return "explicit_page_two"
    if any(term in q for term in _TITLE_TERMS):
        if ref_count >= 2 and any(term in q for term in _PARTY_TERMS):
            return "comparison_title_party"
        return "single_doc_title_cover"
    if ref_count >= 2 and any(term in q for term in _PARTY_TERMS):
        return "comparison_party_metadata"
    if support_shape_class == "named_metadata":
        return "named_metadata_single_doc"
    if any(term in q for term in _ARTICLE_TERMS):
        return "statute_article_metadata"
    if failure_class == "support_undercoverage":
        return "generic_support_undercoverage"
    return "other"


def mine_within_doc_rerank_opportunities(
    *,
    scaffold_path: Path,
    raw_results_path: Path,
    benchmark_path: Path | None = None,
    manual_verdict: str = "correct",
    failure_class: str = "support_undercoverage",
) -> tuple[list[OpportunityRow], list[FamilySummary]]:
    scaffold = _load_json(scaffold_path)
    records = _coerce_dict_list(scaffold.get("records"))
    raw_results = _load_raw_results(raw_results_path)
    benchmark_cases_by_qid: dict[str, JsonDict] = {}
    if benchmark_path is not None:
        benchmark = _load_json(benchmark_path)
        benchmark_cases_by_qid = {
            str(case.get("question_id") or "").strip(): case
            for case in _coerce_dict_list(benchmark.get("cases"))
            if str(case.get("question_id") or "").strip()
        }

    rows: list[OpportunityRow] = []
    manual_verdict_filter = manual_verdict.strip().lower()
    failure_class_filter = failure_class.strip()
    for record in records:
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        if manual_verdict_filter != "any" and str(record.get("manual_verdict") or "").strip().lower() != manual_verdict_filter:
            continue
        if failure_class_filter.lower() != "any" and str(record.get("failure_class") or "").strip() != failure_class_filter:
            continue

        gold_pages = _coerce_str_list(record.get("minimal_required_support_pages"))
        if not gold_pages:
            benchmark_case = benchmark_cases_by_qid.get(qid)
            if benchmark_case is not None:
                gold_pages = _coerce_str_list(benchmark_case.get("gold_page_ids"))
        if not gold_pages:
            continue
        raw = raw_results.get(qid)
        if raw is None:
            continue
        telemetry = raw.get("telemetry")
        if not isinstance(telemetry, dict):
            continue
        telemetry_dict = cast("JsonDict", telemetry)

        retrieved_pages = _coerce_str_list(telemetry_dict.get("retrieved_page_ids"))
        context_pages = _coerce_str_list(telemetry_dict.get("context_page_ids"))
        used_pages = _coerce_str_list(telemetry_dict.get("used_page_ids"))
        gold_set = set(gold_pages)
        gold_doc_ids = _doc_sequence(gold_pages)
        retrieved_doc_ids = _doc_sequence(retrieved_pages)
        context_doc_ids = _doc_sequence(context_pages)
        used_doc_ids = _doc_sequence(used_pages)
        gold_in_retrieved = bool(gold_set.intersection(retrieved_pages))
        gold_in_context = bool(gold_set.intersection(context_pages))
        gold_in_used = bool(gold_set.intersection(used_pages))
        opportunity = (gold_in_retrieved or gold_in_context) and not gold_in_used
        context_page_budget = len(context_pages)
        retrieved_counts = _page_counts_by_doc(retrieved_pages)
        context_counts = _page_counts_by_doc(context_pages)
        max_retrieved_pages_per_doc = max(retrieved_counts.values(), default=0)
        max_context_pages_per_doc = max(context_counts.values(), default=0)
        missing_gold_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id not in context_doc_ids]
        same_doc_chunk_spam = bool(missing_gold_doc_ids) and max_context_pages_per_doc >= 2
        collapsed_context_doc_ids = _doc_sequence(_first_page_per_doc(retrieved_pages)[:context_page_budget])
        doc_family_collapse_opportunity = bool(missing_gold_doc_ids) and any(
            doc_id in collapsed_context_doc_ids for doc_id in gold_doc_ids
        )

        rows.append(
            OpportunityRow(
                question_id=qid,
                family=_infer_family(
                    question=str(record.get("question") or ""),
                    failure_class=str(record.get("failure_class") or ""),
                    support_shape_class=str(record.get("support_shape_class") or ""),
                ),
                question=str(record.get("question") or ""),
                failure_class=str(record.get("failure_class") or ""),
                route_family=str(record.get("route_family") or ""),
                support_shape_class=str(record.get("support_shape_class") or ""),
                gold_pages=gold_pages,
                retrieved_pages=retrieved_pages,
                context_pages=context_pages,
                used_pages=used_pages,
                gold_doc_ids=gold_doc_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                context_doc_ids=context_doc_ids,
                used_doc_ids=used_doc_ids,
                gold_in_retrieved=gold_in_retrieved,
                gold_in_context=gold_in_context,
                gold_in_used=gold_in_used,
                context_page_budget=context_page_budget,
                max_retrieved_pages_per_doc=max_retrieved_pages_per_doc,
                max_context_pages_per_doc=max_context_pages_per_doc,
                same_doc_chunk_spam=same_doc_chunk_spam,
                within_doc_rerank_opportunity=opportunity,
                doc_family_collapse_opportunity=doc_family_collapse_opportunity,
                collapsed_context_doc_ids=collapsed_context_doc_ids,
            )
        )

    grouped: dict[str, list[OpportunityRow]] = {}
    for row in rows:
        grouped.setdefault(row.family, []).append(row)

    summaries: list[FamilySummary] = []
    for family, family_rows in grouped.items():
        opportunity_count = sum(1 for row in family_rows if row.within_doc_rerank_opportunity)
        collapse_opportunity_count = sum(1 for row in family_rows if row.doc_family_collapse_opportunity)
        same_doc_chunk_spam_count = sum(1 for row in family_rows if row.same_doc_chunk_spam)
        actionable_rows = [row for row in family_rows if row.doc_family_collapse_opportunity]
        suggested_context_page_budget = _percentile_int(
            [row.context_page_budget for row in actionable_rows],
            0.75,
        )
        summaries.append(
            FamilySummary(
                family=family,
                case_count=len(family_rows),
                gold_in_retrieved_count=sum(1 for row in family_rows if row.gold_in_retrieved),
                gold_in_context_count=sum(1 for row in family_rows if row.gold_in_context),
                gold_in_used_count=sum(1 for row in family_rows if row.gold_in_used),
                same_doc_chunk_spam_count=same_doc_chunk_spam_count,
                opportunity_count=opportunity_count,
                collapse_opportunity_count=collapse_opportunity_count,
                likely_actionable=collapse_opportunity_count > 0,
                suggested_context_page_budget=suggested_context_page_budget,
                suggested_max_pages_per_doc=1 if collapse_opportunity_count > 0 else 0,
                question_id_examples=[row.question_id for row in family_rows[:5]],
            )
        )
    summaries.sort(
        key=lambda item: (
            -int(item.likely_actionable),
            -item.collapse_opportunity_count,
            -item.opportunity_count,
            item.family,
        )
    )
    return rows, summaries


def _render_markdown(*, rows: list[OpportunityRow], summaries: list[FamilySummary]) -> str:
    lines = [
        "# Within-Doc Rerank Opportunities",
        "",
        f"- cases: `{len(rows)}`",
        f"- families: `{len(summaries)}`",
        f"- opportunities: `{sum(1 for row in rows if row.within_doc_rerank_opportunity)}`",
        f"- collapse_opportunities: `{sum(1 for row in rows if row.doc_family_collapse_opportunity)}`",
        f"- same_doc_chunk_spam_cases: `{sum(1 for row in rows if row.same_doc_chunk_spam)}`",
        f"- ticket20_verdict: `{'go' if any(row.doc_family_collapse_opportunity for row in rows) else 'no_go'}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Family | Cases | Gold In Retrieved | Gold In Context | Gold In Used | Same-Doc Spam | Opportunities | Collapse Opportunities | Actionable | Page Budget | Max Pages / Doc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item.family}` | `{item.case_count}` | `{item.gold_in_retrieved_count}` | `{item.gold_in_context_count}` | "
            f"`{item.gold_in_used_count}` | `{item.same_doc_chunk_spam_count}` | `{item.opportunity_count}` | "
            f"`{item.collapse_opportunity_count}` | `{item.likely_actionable}` | `{item.suggested_context_page_budget}` | "
            f"`{item.suggested_max_pages_per_doc}` |"
        )
    lines.append("")
    for row in rows:
        lines.extend(
            [
                f"## {row.question_id}",
                "",
                f"- family: `{row.family}`",
                f"- question: {row.question}",
                f"- route_family: `{row.route_family}`",
                f"- support_shape_class: `{row.support_shape_class}`",
                f"- gold_pages: `{row.gold_pages}`",
                f"- gold_doc_ids: `{row.gold_doc_ids}`",
                f"- gold_in_retrieved: `{row.gold_in_retrieved}`",
                f"- gold_in_context: `{row.gold_in_context}`",
                f"- gold_in_used: `{row.gold_in_used}`",
                f"- context_page_budget: `{row.context_page_budget}`",
                f"- max_retrieved_pages_per_doc: `{row.max_retrieved_pages_per_doc}`",
                f"- max_context_pages_per_doc: `{row.max_context_pages_per_doc}`",
                f"- same_doc_chunk_spam: `{row.same_doc_chunk_spam}`",
                f"- within_doc_rerank_opportunity: `{row.within_doc_rerank_opportunity}`",
                f"- doc_family_collapse_opportunity: `{row.doc_family_collapse_opportunity}`",
                f"- collapsed_context_doc_ids: `{row.collapsed_context_doc_ids}`",
                f"- retrieved_pages: `{row.retrieved_pages}`",
                f"- context_pages: `{row.context_pages}`",
                f"- used_pages: `{row.used_pages}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine within-doc page rerank opportunities from scaffold + raw results.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--manual-verdict", default="correct")
    parser.add_argument("--failure-class", default="support_undercoverage")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summaries = mine_within_doc_rerank_opportunities(
        scaffold_path=args.scaffold,
        raw_results_path=args.raw_results,
        benchmark_path=args.benchmark,
        manual_verdict=args.manual_verdict,
        failure_class=args.failure_class,
    )
    payload = {
        "case_count": len(rows),
        "family_count": len(summaries),
        "opportunity_count": sum(1 for row in rows if row.within_doc_rerank_opportunity),
        "collapse_opportunity_count": sum(1 for row in rows if row.doc_family_collapse_opportunity),
        "same_doc_chunk_spam_count": sum(1 for row in rows if row.same_doc_chunk_spam),
        "ticket20_verdict": "go" if any(row.doc_family_collapse_opportunity for row in rows) else "no_go",
        "suggested_context_page_budget": _percentile_int(
            [row.context_page_budget for row in rows if row.doc_family_collapse_opportunity],
            0.75,
        ),
        "suggested_max_pages_per_doc": 1 if any(row.doc_family_collapse_opportunity for row in rows) else 0,
        "summaries": [asdict(item) for item in summaries],
        "rows": [asdict(row) for row in rows],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(rows=rows, summaries=summaries), encoding="utf-8")


if __name__ == "__main__":
    main()
