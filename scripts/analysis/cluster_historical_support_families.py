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
class FamilySummary:
    family: str
    opportunity_count: int
    unique_qid_count: int
    new_qid_count: int
    exact_gold_recovery_count: int
    best_source_submission_count: int
    source_submission_examples: list[str]
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


def _infer_family(*, question: str, failure_class: str) -> str:
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
    if any(term in q for term in _ARTICLE_TERMS):
        return "statute_article_metadata"
    if any(term in q for term in _MONEY_TERMS):
        return "monetary_claim_projection"
    if failure_class == "support_undercoverage":
        return "generic_support_undercoverage"
    return "other"


def summarize_historical_support_families(
    *,
    opportunities_json: Path,
    known_qids: set[str],
) -> list[FamilySummary]:
    payload = _load_json(opportunities_json)
    rows = _coerce_dict_list(payload.get("opportunities"))
    grouped: dict[str, list[JsonDict]] = {}
    for row in rows:
        family = _infer_family(
            question=str(row.get("question") or ""),
            failure_class=str(row.get("failure_class") or ""),
        )
        grouped.setdefault(family, []).append(row)

    out: list[FamilySummary] = []
    for family, items in grouped.items():
        unique_qids = sorted({str(item.get("question_id") or "").strip() for item in items if str(item.get("question_id") or "").strip()})
        new_qids = [qid for qid in unique_qids if qid not in known_qids]
        source_submissions = sorted({str(item.get("source_submission") or "").strip() for item in items if str(item.get("source_submission") or "").strip()})
        exact_gold_recovery_count = sum(1 for item in items if bool(item.get("exact_gold_recovered")))
        out.append(
            FamilySummary(
                family=family,
                opportunity_count=len(items),
                unique_qid_count=len(unique_qids),
                new_qid_count=len(new_qids),
                exact_gold_recovery_count=exact_gold_recovery_count,
                best_source_submission_count=len(source_submissions),
                source_submission_examples=source_submissions[:3],
                question_id_examples=unique_qids[:5],
            )
        )
    out.sort(
        key=lambda item: (
            -item.new_qid_count,
            -item.unique_qid_count,
            -item.exact_gold_recovery_count,
            item.family,
        )
    )
    return out


def _render_markdown(*, summaries: list[FamilySummary], known_qid_count: int) -> str:
    lines = [
        "# Historical Support Family Summary",
        "",
        f"- families: `{len(summaries)}`",
        f"- known_qids: `{known_qid_count}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Family | Opportunities | Unique QIDs | New QIDs | Exact Gold Recoveries | Source Submissions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item.family}` | `{item.opportunity_count}` | `{item.unique_qid_count}` | `{item.new_qid_count}` | "
            f"`{item.exact_gold_recovery_count}` | `{item.best_source_submission_count}` |"
        )
    lines.append("")
    for item in summaries:
        lines.extend(
            [
                f"## {item.family}",
                "",
                f"- opportunities: `{item.opportunity_count}`",
                f"- unique_qids: `{item.unique_qid_count}`",
                f"- new_qids: `{item.new_qid_count}`",
                f"- exact_gold_recoveries: `{item.exact_gold_recovery_count}`",
                f"- source_submission_examples: `{item.source_submission_examples}`",
                f"- question_id_examples: `{item.question_id_examples}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster historical support opportunities into family-level signals.")
    parser.add_argument("--opportunities-json", type=Path, required=True)
    parser.add_argument("--known-qids-file", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    known_qids = _load_known_qids(args.known_qids_file)
    summaries = summarize_historical_support_families(
        opportunities_json=args.opportunities_json,
        known_qids=known_qids,
    )
    payload = {
        "family_count": len(summaries),
        "known_qid_count": len(known_qids),
        "summaries": [asdict(item) for item in summaries],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(summaries=summaries, known_qid_count=len(known_qids)), encoding="utf-8")


if __name__ == "__main__":
    main()
