from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class HistoricalSupportOpportunity:
    question_id: str
    question: str
    manual_verdict: str
    failure_class: str
    source_submission: str
    answer_same_as_baseline: bool
    gold_page_ids: list[str]
    baseline_page_ids: list[str]
    source_page_ids: list[str]
    baseline_gold_hits: int
    source_gold_hits: int
    gold_hit_gain: int
    exact_gold_recovered: bool
    baseline_page_count: int
    source_page_count: int


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_dict(value: object) -> JsonDict:
    return cast("JsonDict", value) if isinstance(value, dict) else {}


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for raw in cast("list[object]", value):
            text = str(raw).strip()
            if text and text not in out:
                out.append(text)
        return out
    text = str(value or "").strip()
    return [text] if text else []


def _normalize_answer(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return " | ".join(_coerce_str_list(cast("list[object]", value))) or "null"
    text = str(value).strip()
    return text if text else "null"


def _load_answers(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        rows = _coerce_dict_list(cast("JsonDict", payload).get("answers"))
    elif isinstance(payload, list):
        rows = _coerce_dict_list(cast("list[object]", payload))
    else:
        raise ValueError(f"Expected JSON dict/list in {path}")
    out: dict[str, JsonDict] = {}
    for row in rows:
        qid = str(row.get("question_id") or "").strip()
        if qid:
            out[qid] = row
    return out


def _load_scaffold_records(path: Path) -> list[JsonDict]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return _coerce_dict_list(cast("JsonDict", payload).get("records"))


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
            if not isinstance(raw, (int, float)):
                continue
            page_id = f"{doc_id}_{int(raw)}"
            if page_id not in out:
                out.append(page_id)
    return out


def build_historical_support_oracle(
    *,
    scaffold_path: Path,
    baseline_submission_path: Path,
    submissions_dir: Path,
    manual_verdicts: set[str] | None,
    failure_classes: set[str] | None,
) -> list[HistoricalSupportOpportunity]:
    scaffold_records = _load_scaffold_records(scaffold_path)
    baseline_answers = _load_answers(baseline_submission_path)
    submission_paths = sorted(
        path
        for path in submissions_dir.glob("submission_v*.json")
        if path != baseline_submission_path
    )

    opportunities: list[HistoricalSupportOpportunity] = []
    for record in scaffold_records:
        qid = str(record.get("question_id") or "").strip()
        if not qid:
            continue
        manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
        failure_class = str(record.get("failure_class") or "").strip()
        if manual_verdicts is not None and manual_verdict not in manual_verdicts:
            continue
        if failure_classes is not None and failure_class not in failure_classes:
            continue
        gold_page_ids = _coerce_str_list(record.get("minimal_required_support_pages"))
        if not gold_page_ids:
            continue
        baseline_row = baseline_answers.get(qid)
        if baseline_row is None:
            continue
        baseline_page_ids = _page_ids_from_submission_row(baseline_row)
        baseline_gold_hits = sum(1 for page_id in gold_page_ids if page_id in baseline_page_ids)
        baseline_answer = _normalize_answer(baseline_row.get("answer"))

        for submission_path in submission_paths:
            source_answers = _load_answers(submission_path)
            source_row = source_answers.get(qid)
            if source_row is None:
                continue
            source_page_ids = _page_ids_from_submission_row(source_row)
            if not source_page_ids or source_page_ids == baseline_page_ids:
                continue
            source_gold_hits = sum(1 for page_id in gold_page_ids if page_id in source_page_ids)
            gold_hit_gain = source_gold_hits - baseline_gold_hits
            if gold_hit_gain <= 0:
                continue
            exact_gold_recovered = baseline_gold_hits < len(gold_page_ids) and source_gold_hits == len(gold_page_ids)
            opportunities.append(
                HistoricalSupportOpportunity(
                    question_id=qid,
                    question=str(record.get("question") or "").strip(),
                    manual_verdict=manual_verdict,
                    failure_class=failure_class,
                    source_submission=submission_path.name,
                    answer_same_as_baseline=_normalize_answer(source_row.get("answer")) == baseline_answer,
                    gold_page_ids=gold_page_ids,
                    baseline_page_ids=baseline_page_ids,
                    source_page_ids=source_page_ids,
                    baseline_gold_hits=baseline_gold_hits,
                    source_gold_hits=source_gold_hits,
                    gold_hit_gain=gold_hit_gain,
                    exact_gold_recovered=exact_gold_recovered,
                    baseline_page_count=len(baseline_page_ids),
                    source_page_count=len(source_page_ids),
                )
            )
    opportunities.sort(
        key=lambda item: (
            -item.gold_hit_gain,
            -int(item.exact_gold_recovered),
            -int(item.answer_same_as_baseline),
            item.source_page_count,
            item.question_id,
            item.source_submission,
        )
    )
    return opportunities


def _render_markdown(*, opportunities: list[HistoricalSupportOpportunity]) -> str:
    lines = [
        "# Historical Support Oracle Opportunities",
        "",
        f"- opportunities: `{len(opportunities)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| Rank | QID | Source | Verdict | Failure | Answer Same | Gold Gain | Exact Gold Recovered | Baseline Pages | Source Pages |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(opportunities, start=1):
        lines.append(
            f"| {index} | `{item.question_id}` | `{item.source_submission}` | "
            f"`{item.manual_verdict}` | `{item.failure_class or 'n/a'}` | "
            f"`{item.answer_same_as_baseline}` | `{item.gold_hit_gain}` | `{item.exact_gold_recovered}` | "
            f"`{item.baseline_page_count}` | `{item.source_page_count}` |"
        )
    lines.append("")
    for item in opportunities:
        lines.extend(
            [
                f"## {item.question_id} :: {item.source_submission}",
                "",
                f"- question: {item.question}",
                f"- manual_verdict: `{item.manual_verdict}`",
                f"- failure_class: `{item.failure_class or 'n/a'}`",
                f"- answer_same_as_baseline: `{item.answer_same_as_baseline}`",
                f"- gold_page_ids: `{item.gold_page_ids}`",
                f"- baseline_page_ids: `{item.baseline_page_ids}`",
                f"- source_page_ids: `{item.source_page_ids}`",
                f"- baseline_gold_hits: `{item.baseline_gold_hits}`",
                f"- source_gold_hits: `{item.source_gold_hits}`",
                f"- gold_hit_gain: `{item.gold_hit_gain}`",
                f"- exact_gold_recovered: `{item.exact_gold_recovered}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine historical warmup submissions for bounded support donors.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--baseline-submission", type=Path, required=True)
    parser.add_argument("--submissions-dir", type=Path, required=True)
    parser.add_argument("--manual-verdict", action="append", default=[])
    parser.add_argument("--failure-class", action="append", default=[])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_verdicts = {text.strip().lower() for text in args.manual_verdict if str(text).strip()} or None
    failure_classes = {str(text).strip() for text in args.failure_class if str(text).strip()} or None
    opportunities = build_historical_support_oracle(
        scaffold_path=args.scaffold,
        baseline_submission_path=args.baseline_submission,
        submissions_dir=args.submissions_dir,
        manual_verdicts=manual_verdicts,
        failure_classes=failure_classes,
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
