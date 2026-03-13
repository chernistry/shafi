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
    gold_in_retrieved: bool
    gold_in_context: bool
    gold_in_used: bool
    within_doc_rerank_opportunity: bool


@dataclass(frozen=True)
class FamilySummary:
    family: str
    case_count: int
    gold_in_retrieved_count: int
    gold_in_context_count: int
    gold_in_used_count: int
    opportunity_count: int
    likely_actionable: bool
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
    manual_verdict: str = "correct",
    failure_class: str = "support_undercoverage",
) -> tuple[list[OpportunityRow], list[FamilySummary]]:
    scaffold = _load_json(scaffold_path)
    records = _coerce_dict_list(scaffold.get("records"))
    raw_results = _load_raw_results(raw_results_path)

    rows: list[OpportunityRow] = []
    for record in records:
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        if str(record.get("manual_verdict") or "").strip().lower() != manual_verdict.lower():
            continue
        if str(record.get("failure_class") or "").strip() != failure_class:
            continue

        gold_pages = _coerce_str_list(record.get("minimal_required_support_pages"))
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
        gold_in_retrieved = bool(gold_set.intersection(retrieved_pages))
        gold_in_context = bool(gold_set.intersection(context_pages))
        gold_in_used = bool(gold_set.intersection(used_pages))
        opportunity = (gold_in_retrieved or gold_in_context) and not gold_in_used

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
                gold_in_retrieved=gold_in_retrieved,
                gold_in_context=gold_in_context,
                gold_in_used=gold_in_used,
                within_doc_rerank_opportunity=opportunity,
            )
        )

    grouped: dict[str, list[OpportunityRow]] = {}
    for row in rows:
        grouped.setdefault(row.family, []).append(row)

    summaries: list[FamilySummary] = []
    for family, family_rows in grouped.items():
        opportunity_count = sum(1 for row in family_rows if row.within_doc_rerank_opportunity)
        summaries.append(
            FamilySummary(
                family=family,
                case_count=len(family_rows),
                gold_in_retrieved_count=sum(1 for row in family_rows if row.gold_in_retrieved),
                gold_in_context_count=sum(1 for row in family_rows if row.gold_in_context),
                gold_in_used_count=sum(1 for row in family_rows if row.gold_in_used),
                opportunity_count=opportunity_count,
                likely_actionable=opportunity_count >= 2,
                question_id_examples=[row.question_id for row in family_rows[:5]],
            )
        )
    summaries.sort(key=lambda item: (-int(item.likely_actionable), -item.opportunity_count, item.family))
    return rows, summaries


def _render_markdown(*, rows: list[OpportunityRow], summaries: list[FamilySummary]) -> str:
    lines = [
        "# Within-Doc Rerank Opportunities",
        "",
        f"- cases: `{len(rows)}`",
        f"- families: `{len(summaries)}`",
        f"- opportunities: `{sum(1 for row in rows if row.within_doc_rerank_opportunity)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Family | Cases | Gold In Retrieved | Gold In Context | Gold In Used | Opportunities | Actionable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item.family}` | `{item.case_count}` | `{item.gold_in_retrieved_count}` | `{item.gold_in_context_count}` | "
            f"`{item.gold_in_used_count}` | `{item.opportunity_count}` | `{item.likely_actionable}` |"
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
                f"- gold_in_retrieved: `{row.gold_in_retrieved}`",
                f"- gold_in_context: `{row.gold_in_context}`",
                f"- gold_in_used: `{row.gold_in_used}`",
                f"- within_doc_rerank_opportunity: `{row.within_doc_rerank_opportunity}`",
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
        manual_verdict=args.manual_verdict,
        failure_class=args.failure_class,
    )
    payload = {
        "case_count": len(rows),
        "family_count": len(summaries),
        "opportunity_count": sum(1 for row in rows if row.within_doc_rerank_opportunity),
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
