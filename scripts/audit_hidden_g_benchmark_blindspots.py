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
class BenchmarkBlindspot:
    question_id: str
    question: str
    manual_verdict: str
    failure_class: str
    explicit_anchor: bool
    in_benchmark: bool
    gold_page_ids: list[str]
    current_page_ids: list[str]
    current_has_gold: bool
    missing_gold_page_ids: list[str]


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


def _load_benchmark_qids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    cases_obj = payload.get("cases")
    if not isinstance(cases_obj, list):
        raise ValueError(f"Benchmark at {path} is missing 'cases'")
    out: set[str] = set()
    for case in cases_obj:
        if not isinstance(case, dict):
            continue
        qid = str(case.get("question_id") or "").strip()
        if qid:
            out.add(qid)
    return out


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


def build_blindspots(
    *,
    scaffold_path: Path,
    benchmark_path: Path,
    current_raw_results_path: Path,
) -> list[BenchmarkBlindspot]:
    records = _load_scaffold_records(scaffold_path)
    benchmark_qids = _load_benchmark_qids(benchmark_path)
    current_pages_by_qid = _load_raw_results_pages(current_raw_results_path)
    out: list[BenchmarkBlindspot] = []
    for record in records:
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
        gold_page_ids = _coerce_str_list(record.get("minimal_required_support_pages"))
        if not gold_page_ids:
            continue
        current_page_ids = current_pages_by_qid.get(qid, [])
        gold_set = set(gold_page_ids)
        current_has_gold = bool(gold_set.intersection(current_page_ids))
        out.append(
            BenchmarkBlindspot(
                question_id=qid,
                question=str(record.get("question") or "").strip(),
                manual_verdict=manual_verdict,
                failure_class=failure_class,
                explicit_anchor=explicit_anchor,
                in_benchmark=qid in benchmark_qids,
                gold_page_ids=gold_page_ids,
                current_page_ids=current_page_ids,
                current_has_gold=current_has_gold,
                missing_gold_page_ids=[page_id for page_id in gold_page_ids if page_id not in current_page_ids],
            )
        )
    out.sort(
        key=lambda item: (
            item.in_benchmark,
            item.current_has_gold,
            0 if item.explicit_anchor else 1,
            item.question_id,
        )
    )
    return out


def _render_markdown(*, blindspots: list[BenchmarkBlindspot]) -> str:
    lines = [
        "# Hidden-G Benchmark Blindspot Audit",
        "",
        f"- blindspots: `{len(blindspots)}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
        "| QID | In Benchmark | Current Has Gold | Explicit Anchor | Failure Class | Missing Gold Pages |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in blindspots:
        lines.append(
            f"| `{item.question_id}` | `{item.in_benchmark}` | `{item.current_has_gold}` | `{item.explicit_anchor}` | "
            f"`{item.failure_class}` | `{item.missing_gold_page_ids}` |"
        )
    lines.append("")
    for item in blindspots:
        lines.extend(
            [
                f"## {item.question_id}",
                "",
                f"- question: {item.question}",
                f"- in_benchmark: `{item.in_benchmark}`",
                f"- current_has_gold: `{item.current_has_gold}`",
                f"- explicit_anchor: `{item.explicit_anchor}`",
                f"- failure_class: `{item.failure_class}`",
                f"- gold_page_ids: `{item.gold_page_ids}`",
                f"- current_page_ids: `{item.current_page_ids}`",
                f"- missing_gold_page_ids: `{item.missing_gold_page_ids}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manually reviewed anchor/support cases missing from the hidden-G benchmark.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--current-raw-results", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blindspots = build_blindspots(
        scaffold_path=args.scaffold,
        benchmark_path=args.benchmark,
        current_raw_results_path=args.current_raw_results,
    )
    payload = {
        "blindspot_count": len(blindspots),
        "blindspots": [asdict(item) for item in blindspots],
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(blindspots=blindspots), encoding="utf-8")


if __name__ == "__main__":
    main()
