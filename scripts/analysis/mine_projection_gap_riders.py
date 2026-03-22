from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]

_EXPLICIT_ANCHOR_RE = re.compile(
    r"\b(page 2|second page|page 1|first page|title page|cover page|caption|header)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProjectionGapOpportunity:
    question_id: str
    question: str
    source_submission: str
    target_page_numbers: list[int]
    baseline_page_ids: list[str]
    source_page_ids: list[str]
    baseline_target_hits: int
    source_target_hits: int
    target_hit_gain: int
    source_page_count: int
    exact_gold_hit: bool
    explicit_anchor: bool


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _load_answers(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        payload_dict = cast("JsonDict", payload)
        answers = _coerce_dict_list(payload_dict.get("answers"))
    elif isinstance(payload, list):
        answers = _coerce_dict_list(cast("object", payload))
    else:
        raise ValueError(f"Expected JSON dict/list in {path}")
    out: dict[str, JsonDict] = {}
    for row in answers:
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_scaffold(path: Path) -> list[JsonDict]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    payload_dict = cast("JsonDict", payload)
    return _coerce_dict_list(payload_dict.get("records"))


def _page_ids_from_submission_row(row: JsonDict) -> list[str]:
    telemetry = _coerce_dict(row.get("telemetry"))
    retrieval = _coerce_dict(telemetry.get("retrieval"))
    chunks = _coerce_dict_list(retrieval.get("retrieved_chunk_pages"))
    out: list[str] = []
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "").strip()
        page_numbers = chunk.get("page_numbers")
        if not doc_id or not isinstance(page_numbers, list):
            continue
        for raw in cast("list[object]", page_numbers):
            if not isinstance(raw, int | float):
                continue
            page_id = f"{doc_id}_{int(raw)}"
            if page_id not in out:
                out.append(page_id)
    return out


def _page_numbers(page_ids: list[str]) -> list[int]:
    out: list[int] = []
    for page_id in page_ids:
        if "_" not in page_id:
            continue
        suffix = page_id.rsplit("_", 1)[1]
        if suffix.isdigit():
            out.append(int(suffix))
    return out


def _target_page_numbers(record: JsonDict) -> list[int]:
    gold_pages = record.get("minimal_required_support_pages")
    if not isinstance(gold_pages, list):
        return []
    out: list[int] = []
    for raw in cast("list[object]", gold_pages):
        text = str(raw).strip()
        if "_" not in text:
            continue
        suffix = text.rsplit("_", 1)[1]
        if suffix.isdigit():
            page_no = int(suffix)
            if page_no not in out:
                out.append(page_no)
    return out


def _count_target_hits(page_ids: list[str], target_page_numbers: list[int]) -> int:
    numbers = _page_numbers(page_ids)
    return sum(1 for num in numbers if num in target_page_numbers)


def build_projection_gap_opportunities(
    *,
    scaffold_path: Path,
    baseline_submission_path: Path,
    submissions_dir: Path,
) -> list[ProjectionGapOpportunity]:
    baseline_answers = _load_answers(baseline_submission_path)
    scaffold_records = _load_scaffold(scaffold_path)
    submission_paths = sorted(path for path in submissions_dir.glob("submission_v*.json") if path != baseline_submission_path)

    opportunities: list[ProjectionGapOpportunity] = []
    for record in scaffold_records:
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
        if manual_verdict != "correct":
            continue
        failure_class = str(record.get("failure_class") or "").strip()
        explicit_anchor = bool(_EXPLICIT_ANCHOR_RE.search(str(record.get("question") or "")))
        if failure_class != "support_undercoverage" and not explicit_anchor:
            continue
        target_page_numbers = _target_page_numbers(record)
        if not target_page_numbers:
            continue
        baseline_row = baseline_answers.get(qid)
        if baseline_row is None:
            continue
        baseline_page_ids = _page_ids_from_submission_row(baseline_row)
        baseline_target_hits = _count_target_hits(baseline_page_ids, target_page_numbers)

        for submission_path in submission_paths:
            source_answers = _load_answers(submission_path)
            source_row = source_answers.get(qid)
            if source_row is None:
                continue
            source_page_ids = _page_ids_from_submission_row(source_row)
            if not source_page_ids or source_page_ids == baseline_page_ids:
                continue
            source_target_hits = _count_target_hits(source_page_ids, target_page_numbers)
            target_hit_gain = source_target_hits - baseline_target_hits
            if target_hit_gain <= 0:
                continue
            gold_page_ids = [str(raw).strip() for raw in cast("list[object]", record.get("minimal_required_support_pages") or []) if str(raw).strip()]
            exact_gold_hit = any(page_id in source_page_ids for page_id in gold_page_ids)
            opportunities.append(
                ProjectionGapOpportunity(
                    question_id=qid,
                    question=str(record.get("question") or "").strip(),
                    source_submission=submission_path.name,
                    target_page_numbers=target_page_numbers,
                    baseline_page_ids=baseline_page_ids,
                    source_page_ids=source_page_ids,
                    baseline_target_hits=baseline_target_hits,
                    source_target_hits=source_target_hits,
                    target_hit_gain=target_hit_gain,
                    source_page_count=len(source_page_ids),
                    exact_gold_hit=exact_gold_hit,
                    explicit_anchor=explicit_anchor,
                )
            )

    opportunities.sort(
        key=lambda item: (
            -item.target_hit_gain,
            -int(item.exact_gold_hit),
            item.source_page_count,
            item.question_id,
            item.source_submission,
        )
    )
    return opportunities


def _render_markdown(*, opportunities: list[ProjectionGapOpportunity]) -> str:
    lines = [
        "# Projection Gap Rider Opportunities",
        "",
        f"- opportunities: `{len(opportunities)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| QID | Source | Target Pages | Baseline Hits | Source Hits | Gain | Exact Gold Hit | Source Page Count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in opportunities:
        lines.append(
            f"| `{item.question_id}` | `{item.source_submission}` | `{item.target_page_numbers}` | "
            f"`{item.baseline_target_hits}` | `{item.source_target_hits}` | `{item.target_hit_gain}` | "
            f"`{item.exact_gold_hit}` | `{item.source_page_count}` |"
        )
    lines.append("")
    for item in opportunities:
        lines.extend(
            [
                f"## {item.question_id} :: {item.source_submission}",
                "",
                f"- question: {item.question}",
                f"- target_page_numbers: `{item.target_page_numbers}`",
                f"- baseline_page_ids: `{item.baseline_page_ids}`",
                f"- source_page_ids: `{item.source_page_ids}`",
                f"- baseline_target_hits: `{item.baseline_target_hits}`",
                f"- source_target_hits: `{item.source_target_hits}`",
                f"- target_hit_gain: `{item.target_hit_gain}`",
                f"- exact_gold_hit: `{item.exact_gold_hit}`",
                f"- explicit_anchor: `{item.explicit_anchor}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine historical submission artifacts for projection-gap support riders.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--submissions-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    opportunities = build_projection_gap_opportunities(
        scaffold_path=args.scaffold,
        baseline_submission_path=args.baseline_submission,
        submissions_dir=args.submissions_dir,
    )
    payload = {
        "opportunity_count": len(opportunities),
        "opportunities": [asdict(item) for item in opportunities],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(opportunities=opportunities), encoding="utf-8")


if __name__ == "__main__":
    main()
