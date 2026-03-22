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
_MONEY_TERMS = ("how much", "amount", "sum", "aed", "usd", "claim")


@dataclass(frozen=True)
class SignalFamilySummary:
    family: str
    scaffold_case_count: int
    scaffold_qid_count: int
    current_frontier_qid_count: int
    uncovered_qid_count: int
    oracle_opportunity_count: int
    oracle_unique_qid_count: int
    oracle_new_qid_count: int
    oracle_exact_gold_recovery_count: int
    likely_actionable: bool
    question_id_examples: list[str]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _load_known_qids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


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
    if any(term in q for term in _MONEY_TERMS):
        return "monetary_claim_projection"
    if failure_class == "support_undercoverage":
        return "generic_support_undercoverage"
    return "other"


def summarize_remaining_signal_classes(
    *,
    scaffold_path: Path,
    oracle_opportunities_path: Path,
    known_qids_path: Path | None,
) -> list[SignalFamilySummary]:
    known_qids = _load_known_qids(known_qids_path)
    scaffold = _load_json(scaffold_path)
    records = _coerce_dict_list(scaffold.get("records"))
    oracle = _load_json(oracle_opportunities_path)
    opportunities = _coerce_dict_list(oracle.get("opportunities"))

    scaffold_family_qids: dict[str, set[str]] = {}
    for record in records:
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
        failure_class = str(record.get("failure_class") or "").strip()
        question = str(record.get("question") or "")
        support_shape_class = str(record.get("support_shape_class") or "").strip()
        minimal_pages = record.get("minimal_required_support_pages")
        if manual_verdict != "correct":
            continue
        if not isinstance(minimal_pages, list) or not minimal_pages:
            continue
        if failure_class != "support_undercoverage" and not any(term in question.lower() for term in (*_TITLE_TERMS, *_PAGE2_TERMS)):
            continue
        family = _infer_family(question=question, failure_class=failure_class, support_shape_class=support_shape_class)
        scaffold_family_qids.setdefault(family, set()).add(qid)

    oracle_family_rows: dict[str, list[JsonDict]] = {}
    for row in opportunities:
        qid = str(row.get("question_id") or "").strip()
        if not qid:
            continue
        family = _infer_family(
            question=str(row.get("question") or ""),
            failure_class=str(row.get("failure_class") or ""),
            support_shape_class="",
        )
        oracle_family_rows.setdefault(family, []).append(row)

    families = sorted(set(scaffold_family_qids) | set(oracle_family_rows))
    out: list[SignalFamilySummary] = []
    for family in families:
        scaffold_qids = sorted(scaffold_family_qids.get(family, set()))
        frontier_qids = [qid for qid in scaffold_qids if qid in known_qids]
        uncovered_qids = [qid for qid in scaffold_qids if qid not in known_qids]
        oracle_rows = oracle_family_rows.get(family, [])
        oracle_unique_qids = sorted({str(row.get("question_id") or "").strip() for row in oracle_rows if str(row.get("question_id") or "").strip()})
        oracle_new_qids = [qid for qid in oracle_unique_qids if qid not in known_qids]
        exact_gold_recoveries = sum(1 for row in oracle_rows if bool(row.get("exact_gold_recovered")))
        likely_actionable = len(oracle_new_qids) >= 2 or (len(oracle_new_qids) >= 1 and exact_gold_recoveries >= 2 and len(uncovered_qids) >= 1)
        out.append(
            SignalFamilySummary(
                family=family,
                scaffold_case_count=len(scaffold_qids),
                scaffold_qid_count=len(scaffold_qids),
                current_frontier_qid_count=len(frontier_qids),
                uncovered_qid_count=len(uncovered_qids),
                oracle_opportunity_count=len(oracle_rows),
                oracle_unique_qid_count=len(oracle_unique_qids),
                oracle_new_qid_count=len(oracle_new_qids),
                oracle_exact_gold_recovery_count=exact_gold_recoveries,
                likely_actionable=likely_actionable,
                question_id_examples=scaffold_qids[:5],
            )
        )
    out.sort(
        key=lambda item: (
            -int(item.likely_actionable),
            -item.oracle_new_qid_count,
            -item.uncovered_qid_count,
            -item.oracle_exact_gold_recovery_count,
            item.family,
        )
    )
    return out


def _render_markdown(*, summaries: list[SignalFamilySummary], known_qid_count: int) -> str:
    lines = [
        "# Remaining Signal Classes",
        "",
        f"- families: `{len(summaries)}`",
        f"- frontier_known_qids: `{known_qid_count}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Family | Actionable | Scaffold QIDs | Uncovered QIDs | Oracle New QIDs | Oracle Exact Recoveries |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item.family}` | `{item.likely_actionable}` | `{item.scaffold_qid_count}` | `{item.uncovered_qid_count}` | "
            f"`{item.oracle_new_qid_count}` | `{item.oracle_exact_gold_recovery_count}` |"
        )
    lines.append("")
    for item in summaries:
        lines.extend(
            [
                f"## {item.family}",
                "",
                f"- likely_actionable: `{item.likely_actionable}`",
                f"- scaffold_qid_count: `{item.scaffold_qid_count}`",
                f"- current_frontier_qid_count: `{item.current_frontier_qid_count}`",
                f"- uncovered_qid_count: `{item.uncovered_qid_count}`",
                f"- oracle_opportunity_count: `{item.oracle_opportunity_count}`",
                f"- oracle_unique_qid_count: `{item.oracle_unique_qid_count}`",
                f"- oracle_new_qid_count: `{item.oracle_new_qid_count}`",
                f"- oracle_exact_gold_recovery_count: `{item.oracle_exact_gold_recovery_count}`",
                f"- question_id_examples: `{item.question_id_examples}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize remaining scaffold+oracle signal classes beyond the current frontier.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--oracle-opportunities-json", type=Path, required=True)
    parser.add_argument("--known-qids-file", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = summarize_remaining_signal_classes(
        scaffold_path=args.scaffold,
        oracle_opportunities_path=args.oracle_opportunities_json,
        known_qids_path=args.known_qids_file,
    )
    known_qids = _load_known_qids(args.known_qids_file)
    payload = {
        "family_count": len(summaries),
        "frontier_known_qid_count": len(known_qids),
        "summaries": [asdict(item) for item in summaries],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(summaries=summaries, known_qid_count=len(known_qids)), encoding="utf-8")


if __name__ == "__main__":
    main()
