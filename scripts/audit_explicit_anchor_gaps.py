from __future__ import annotations

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]

_EXPLICIT_ANCHOR_RE = re.compile(
    r"\b(page 2|second page|page 1|first page|title page|cover page)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplicitAnchorGap:
    question_id: str
    question: str
    manual_verdict: str
    failure_class: str
    gold_page_ids: list[str]
    current_page_ids: list[str]
    current_has_gold: bool
    missing_gold_page_ids: list[str]
    current_hits_anchor_family: bool


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _load_scaffold_records(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {path} is missing 'records'")
    return [cast("JsonDict", item) for item in records_obj if isinstance(item, dict)]


def _load_raw_results_pages(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    out: dict[str, list[str]] = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        case_obj = raw.get("case")
        case = cast("JsonDict", case_obj) if isinstance(case_obj, dict) else {}
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if not qid:
            continue
        telemetry_obj = raw.get("telemetry")
        telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        pages: list[str] = []
        for key in ("used_page_ids", "context_page_ids", "retrieved_page_ids"):
            for page_id in _coerce_str_list(telemetry.get(key)):
                if page_id not in pages:
                    pages.append(page_id)
        out[qid] = pages
    return out


def _anchor_family_hit(*, gold_page_ids: list[str], current_page_ids: list[str]) -> bool:
    gold_doc_pages = {page_id.rsplit("_", 1)[0]: page_id.rsplit("_", 1)[1] for page_id in gold_page_ids if "_" in page_id}
    for page_id in current_page_ids:
        if "_" not in page_id:
            continue
        doc_id, page_num = page_id.rsplit("_", 1)
        if gold_doc_pages.get(doc_id) == page_num:
            return True
    return False


def build_gaps(
    *,
    scaffold_path: Path,
    current_raw_results_path: Path,
    manual_verdicts: set[str] | None = None,
    failure_classes: set[str] | None = None,
) -> list[ExplicitAnchorGap]:
    records = _load_scaffold_records(scaffold_path)
    current_pages_by_qid = _load_raw_results_pages(current_raw_results_path)
    gaps: list[ExplicitAnchorGap] = []
    for record in records:
        question = str(record.get("question") or "").strip()
        if not question or not _EXPLICIT_ANCHOR_RE.search(question):
            continue
        manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
        if manual_verdicts is not None and manual_verdict not in manual_verdicts:
            continue
        failure_class = str(record.get("failure_class") or "").strip()
        if failure_classes is not None and failure_class not in failure_classes:
            continue
        gold_page_ids = _coerce_str_list(record.get("minimal_required_support_pages"))
        if not gold_page_ids:
            gold_page_ids = _coerce_str_list(record.get("gold_page_ids"))
        if not gold_page_ids:
            continue
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        current_page_ids = current_pages_by_qid.get(qid, [])
        gold_set = set(gold_page_ids)
        current_has_gold = bool(gold_set.intersection(current_page_ids))
        if current_has_gold:
            continue
        gaps.append(
            ExplicitAnchorGap(
                question_id=qid,
                question=question,
                manual_verdict=manual_verdict,
                failure_class=failure_class,
                gold_page_ids=gold_page_ids,
                current_page_ids=current_page_ids,
                current_has_gold=current_has_gold,
                missing_gold_page_ids=[page_id for page_id in gold_page_ids if page_id not in current_page_ids],
                current_hits_anchor_family=_anchor_family_hit(
                    gold_page_ids=gold_page_ids,
                    current_page_ids=current_page_ids,
                ),
            )
        )
    gaps.sort(key=lambda item: (item.current_hits_anchor_family, item.question_id))
    return gaps


def _render_markdown(
    *,
    scaffold_path: Path,
    current_raw_results_path: Path,
    gaps: list[ExplicitAnchorGap],
) -> str:
    lines = [
        "# Explicit Anchor Gap Audit",
        "",
        f"- scaffold: `{scaffold_path}`",
        f"- current_raw_results: `{current_raw_results_path}`",
        f"- gaps: `{len(gaps)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| QID | Manual Verdict | Failure Class | Current Hits Anchor Family | Missing Gold Pages |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for gap in gaps:
        lines.append(
            f"| `{gap.question_id}` | `{gap.manual_verdict}` | `{gap.failure_class}` | "
            f"`{gap.current_hits_anchor_family}` | `{gap.missing_gold_page_ids}` |"
        )
    lines.append("")
    for gap in gaps:
        lines.extend(
            [
                f"## {gap.question_id}",
                "",
                f"- question: {gap.question}",
                f"- manual_verdict: `{gap.manual_verdict}`",
                f"- failure_class: `{gap.failure_class}`",
                f"- gold_page_ids: `{gap.gold_page_ids}`",
                f"- current_page_ids: `{gap.current_page_ids}`",
                f"- missing_gold_page_ids: `{gap.missing_gold_page_ids}`",
                f"- current_hits_anchor_family: `{gap.current_hits_anchor_family}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current candidate for missed explicit-anchor gold pages.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--current-raw-results", type=Path, required=True)
    parser.add_argument("--manual-verdict", action="append", default=None)
    parser.add_argument("--failure-class", action="append", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_verdicts = {text.strip().lower() for text in args.manual_verdict} if args.manual_verdict else None
    failure_classes = {text.strip() for text in args.failure_class} if args.failure_class else None
    gaps = build_gaps(
        scaffold_path=args.scaffold,
        current_raw_results_path=args.current_raw_results,
        manual_verdicts=manual_verdicts,
        failure_classes=failure_classes,
    )
    payload = {
        "gap_count": len(gaps),
        "gaps": [asdict(item) for item in gaps],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(
        _render_markdown(
            scaffold_path=args.scaffold,
            current_raw_results_path=args.current_raw_results,
            gaps=gaps,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
