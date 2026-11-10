from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from shafi.core.classifier import QueryClassifier

JsonDict = dict[str, object]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _phrase_type_counts(records: list[JsonDict]) -> dict[str, int]:
    counts = {
        "numeric_page": 0,
        "title_page": 0,
        "second_page": 0,
        "caption_header": 0,
    }
    for record in records:
        kind = str(record.get("phrase_type") or "").strip()
        if kind in counts:
            counts[kind] += 1
    return counts


def _failure_stage_counts(records: list[JsonDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        stage = str(record.get("failure_stage") or "").strip() or "unknown"
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def build_audit(*, page_trace_ledger_path: Path, min_meaningful_qids: int = 3) -> JsonDict:
    ledger = _load_json(page_trace_ledger_path)
    ledger_records = _coerce_dict_list(ledger.get("records"))

    records: list[JsonDict] = []
    for record in ledger_records:
        question = str(record.get("question") or "").strip()
        explicit_ref = QueryClassifier.extract_explicit_page_reference(question)
        if explicit_ref is None:
            continue

        used_pages = _coerce_str_list(record.get("used_pages"))
        gold_pages = _coerce_str_list(record.get("gold_pages"))
        records.append(
            {
                "qid": str(record.get("qid") or ""),
                "question": question,
                "phrase_type": explicit_ref.kind,
                "phrase": explicit_ref.phrase,
                "requested_page": explicit_ref.requested_page,
                "failure_stage": str(record.get("failure_stage") or ""),
                "route": str(record.get("route") or ""),
                "trust_tier": str(record.get("trust_tier") or ""),
                "gold_in_used": bool(record.get("gold_in_used")),
                "gold_pages": gold_pages,
                "used_pages": used_pages,
                "false_positive_pages": _coerce_str_list(record.get("false_positive_pages")),
                "page_budget_overrun": _coerce_int(record.get("page_budget_overrun")),
            }
        )

    phrase_counts = _phrase_type_counts(records)
    failure_stage_counts = _failure_stage_counts(records)
    meaningful_qid_count = len(records)
    trusted_count = sum(1 for record in records if str(record.get("trust_tier") or "") == "trusted")
    gold_in_used_rate = (
        sum(1 for record in records if bool(record.get("gold_in_used"))) / meaningful_qid_count if meaningful_qid_count else 0.0
    )
    verdict = "kill_small_family" if meaningful_qid_count < min_meaningful_qids else "continue_to_ticket_14"
    return {
        "source_page_trace_ledger": str(page_trace_ledger_path),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "summary": {
            "meaningful_qid_count": meaningful_qid_count,
            "trusted_qid_count": trusted_count,
            "phrase_type_counts": phrase_counts,
            "failure_stage_counts": failure_stage_counts,
            "gold_in_used_rate": gold_in_used_rate,
            "verdict": verdict,
            "stop_threshold": min_meaningful_qids,
        },
        "records": records,
    }


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload.get("summary") or {})
    phrase_counts = cast("dict[str, object]", summary.get("phrase_type_counts") or {})
    failure_stage_counts = cast("dict[str, object]", summary.get("failure_stage_counts") or {})
    records = _coerce_dict_list(payload.get("records"))
    lines = [
        "# Explicit Page Reference Audit",
        "",
        f"- source_page_trace_ledger: `{payload.get('source_page_trace_ledger')}`",
        f"- submission_policy: `{payload.get('submission_policy')}`",
        "",
        "## Summary",
        "",
        f"- meaningful_qid_count: `{summary.get('meaningful_qid_count')}`",
        f"- trusted_qid_count: `{summary.get('trusted_qid_count')}`",
        f"- gold_in_used_rate: `{summary.get('gold_in_used_rate')}`",
        f"- verdict: `{summary.get('verdict')}`",
        "",
        "## Phrase Types",
        "",
    ]
    for key in ("numeric_page", "title_page", "second_page", "caption_header"):
        lines.append(f"- {key}: `{phrase_counts.get(key, 0)}`")
    lines.extend(["", "## Failure Stages", ""])
    for stage in sorted(failure_stage_counts):
        lines.append(f"- {stage}: `{failure_stage_counts[stage]}`")
    lines.extend(
        [
            "",
            "## Records",
            "",
            "| qid | phrase_type | requested_page | trust | failure_stage | gold_in_used |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record.get("qid") or "")[:12],
                    str(record.get("phrase_type") or ""),
                    str(record.get("requested_page") or ""),
                    str(record.get("trust_tier") or ""),
                    str(record.get("failure_stage") or ""),
                    str(record.get("gold_in_used") or False),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit explicit page-reference questions from a page-trace ledger.")
    parser.add_argument("--page-trace-ledger", type=Path, required=True)
    parser.add_argument("--min-meaningful-qids", type=int, default=3)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_audit(
        page_trace_ledger_path=args.page_trace_ledger,
        min_meaningful_qids=max(1, int(args.min_meaningful_qids)),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
