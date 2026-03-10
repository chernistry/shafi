from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in cast("list[object]", value):
        text = str(raw).strip()
        if text:
            items.append(text)
    return items


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _f_beta(*, predicted: set[str], gold: set[str], beta: float) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    true_positive = len(predicted.intersection(gold))
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / denom


@dataclass(frozen=True)
class BenchmarkSlot:
    name: str
    gold_page_ids: list[str]
    evidence_markers: list[str]


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    text: str
    gold_page_ids: list[str]
    slots: list[BenchmarkSlot]


@dataclass(frozen=True)
class BenchmarkCase:
    question_id: str
    gold_page_ids: list[str]
    gold_items: list[str]
    items: list[BenchmarkItem]
    wrong_document_risk: bool
    trust_tier: str
    gold_origin: str
    audit_note: str


@dataclass(frozen=True)
class CaseScore:
    question_id: str
    predicted_pages: list[str]
    gold_pages: list[str]
    f_beta: float
    orphan_pages: list[str]
    missing_pages: list[str]
    item_coverage: float | None
    slot_recall: float | None
    evidence_family_complete: bool | None
    overprune_violations: int
    wrong_document_issue: bool
    trust_tier: str
    gold_origin: str
    audit_note: str


def _default_trust_tier(case_dict: JsonDict) -> str:
    raw = str(case_dict.get("trust_tier") or "").strip().lower()
    if raw in {"trusted", "suspect"}:
        return raw
    has_manual_structure = bool(_coerce_str_list(case_dict.get("gold_items"))) or bool(case_dict.get("items"))
    return "trusted" if has_manual_structure else "suspect"


def _default_gold_origin(case_dict: JsonDict, *, trust_tier: str) -> str:
    raw = str(case_dict.get("gold_origin") or "").strip().lower()
    if raw in {"manual_override", "seeded_eval", "reviewed_correction"}:
        return raw
    return "manual_override" if trust_tier == "trusted" else "seeded_eval"


def _load_benchmark(path: Path) -> list[BenchmarkCase]:
    payload = _load_json(path)
    cases_obj = payload.get("cases")
    if not isinstance(cases_obj, list):
        raise ValueError("Benchmark JSON must contain a top-level 'cases' list")

    cases: list[BenchmarkCase] = []
    for raw_case in cast("list[object]", cases_obj):
        if not isinstance(raw_case, dict):
            continue
        case_dict = cast("JsonDict", raw_case)
        question_id = str(case_dict.get("question_id") or "").strip()
        if not question_id:
            continue
        items_obj = case_dict.get("items")
        item_specs: list[BenchmarkItem] = []
        if isinstance(items_obj, list):
            for raw_item in cast("list[object]", items_obj):
                if not isinstance(raw_item, dict):
                    continue
                item_dict = cast("JsonDict", raw_item)
                slots_obj = item_dict.get("slots")
                slot_specs: list[BenchmarkSlot] = []
                if isinstance(slots_obj, list):
                    for raw_slot in cast("list[object]", slots_obj):
                        if not isinstance(raw_slot, dict):
                            continue
                        slot_dict = cast("JsonDict", raw_slot)
                        slot_name = str(slot_dict.get("name") or "").strip()
                        if not slot_name:
                            continue
                        slot_specs.append(
                            BenchmarkSlot(
                                name=slot_name,
                                gold_page_ids=_coerce_str_list(slot_dict.get("gold_page_ids")),
                                evidence_markers=_coerce_str_list(slot_dict.get("evidence_markers")),
                            )
                        )
                item_id = str(item_dict.get("id") or item_dict.get("item_id") or "").strip() or f"item_{len(item_specs) + 1}"
                item_specs.append(
                    BenchmarkItem(
                        item_id=item_id,
                        text=str(item_dict.get("text") or "").strip(),
                        gold_page_ids=_coerce_str_list(item_dict.get("gold_page_ids")),
                        slots=slot_specs,
                    )
                )
        cases.append(
            BenchmarkCase(
                question_id=question_id,
                gold_page_ids=_coerce_str_list(case_dict.get("gold_page_ids")),
                gold_items=_coerce_str_list(case_dict.get("gold_items")),
                items=item_specs,
                wrong_document_risk=bool(case_dict.get("wrong_document_risk", False)),
                trust_tier=_default_trust_tier(case_dict),
                gold_origin=_default_gold_origin(case_dict, trust_tier=_default_trust_tier(case_dict)),
                audit_note=str(case_dict.get("audit_note") or "").strip(),
            )
        )
    return cases


def _eval_cases_by_question_id(eval_path: Path) -> dict[str, JsonDict]:
    payload = _load_json(eval_path)
    cases_obj = payload.get("cases")
    if not isinstance(cases_obj, list):
        raise ValueError("Eval JSON must contain a top-level 'cases' list")
    out: dict[str, JsonDict] = {}
    for raw_case in cast("list[object]", cases_obj):
        if not isinstance(raw_case, dict):
            continue
        case_dict = cast("JsonDict", raw_case)
        question_id = str(case_dict.get("question_id") or case_dict.get("case_id") or "").strip()
        if question_id:
            out[question_id] = case_dict
    return out


def _item_coverage(*, answer: str, gold_items: list[str]) -> float | None:
    if not gold_items:
        return None
    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return 0.0
    covered = 0
    for item in gold_items:
        item_text = _normalize_text(item)
        if item_text and item_text in normalized_answer:
            covered += 1
    return covered / len(gold_items)


def _slot_recall(*, predicted_pages: set[str], items: list[BenchmarkItem]) -> float | None:
    slots = [slot for item in items for slot in item.slots if slot.gold_page_ids]
    if not slots:
        return None
    covered = 0
    for slot in slots:
        if predicted_pages.intersection(slot.gold_page_ids):
            covered += 1
    return covered / len(slots)


def _evidence_family_complete(*, predicted_pages: set[str], items: list[BenchmarkItem]) -> bool | None:
    slots = [slot for item in items for slot in item.slots if slot.gold_page_ids]
    if not slots:
        return None
    return all(bool(predicted_pages.intersection(slot.gold_page_ids)) for slot in slots)


def _overprune_violations(*, predicted_pages: set[str], items: list[BenchmarkItem]) -> int:
    violations = 0
    for item in items:
        slot_page_sets = [set(slot.gold_page_ids) for slot in item.slots if slot.gold_page_ids]
        if len(slot_page_sets) < 2:
            continue
        union_pages: set[str] = set()
        for slot_pages in slot_page_sets:
            union_pages.update(slot_pages)
        if len(union_pages) < 2:
            continue
        predicted_overlap = predicted_pages.intersection(union_pages)
        if not predicted_overlap:
            continue
        if any(not predicted_pages.intersection(slot_pages) for slot_pages in slot_page_sets):
            violations += 1
    return violations


def _score_case(case: BenchmarkCase, eval_case: JsonDict | None, *, beta: float) -> CaseScore:
    telemetry_obj: object = eval_case.get("telemetry") if eval_case is not None else {}
    telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
    predicted_pages = _coerce_str_list(telemetry.get("used_page_ids"))
    if not predicted_pages:
        predicted_pages = _coerce_str_list(telemetry.get("cited_page_ids"))
    predicted_set = set(predicted_pages)
    gold_set = set(case.gold_page_ids)
    answer = str(eval_case.get("answer") or "").strip() if eval_case is not None else ""
    return CaseScore(
        question_id=case.question_id,
        predicted_pages=predicted_pages,
        gold_pages=list(case.gold_page_ids),
        f_beta=_f_beta(predicted=predicted_set, gold=gold_set, beta=beta),
        orphan_pages=sorted(predicted_set.difference(gold_set)),
        missing_pages=sorted(gold_set.difference(predicted_set)),
        item_coverage=_item_coverage(answer=answer, gold_items=case.gold_items),
        slot_recall=_slot_recall(predicted_pages=predicted_set, items=case.items),
        evidence_family_complete=_evidence_family_complete(predicted_pages=predicted_set, items=case.items),
        overprune_violations=_overprune_violations(predicted_pages=predicted_set, items=case.items),
        wrong_document_issue=bool(case.wrong_document_risk and (predicted_set.difference(gold_set) or gold_set.difference(predicted_set))),
        trust_tier=case.trust_tier,
        gold_origin=case.gold_origin,
        audit_note=case.audit_note,
    )


def _summary_lines(*, scores: list[CaseScore], beta: float) -> list[str]:
    if not scores:
        return ["- Cases: 0", f"- Page-level F_beta({beta:.1f}): 0.0000", "- Orphan-page case rate: 0.0000"]

    avg_f_beta = sum(score.f_beta for score in scores) / len(scores)
    orphan_case_rate = sum(1 for score in scores if score.orphan_pages) / len(scores)
    mean_item_coverage_values = [score.item_coverage for score in scores if score.item_coverage is not None]
    mean_item_coverage = (
        sum(mean_item_coverage_values) / len(mean_item_coverage_values) if mean_item_coverage_values else None
    )
    slot_recall_values = [score.slot_recall for score in scores if score.slot_recall is not None]
    mean_slot_recall = sum(slot_recall_values) / len(slot_recall_values) if slot_recall_values else None
    family_complete_values = [score.evidence_family_complete for score in scores if score.evidence_family_complete is not None]
    evidence_family_complete_rate = (
        sum(1 for covered in family_complete_values if covered) / len(family_complete_values)
        if family_complete_values
        else None
    )
    overprune_violations = sum(score.overprune_violations for score in scores)
    wrong_document_case_rate = sum(1 for score in scores if score.wrong_document_issue) / len(scores)

    lines = [
        f"- Cases: {len(scores)}",
        f"- Page-level F_beta({beta:.1f}): {avg_f_beta:.4f}",
        f"- Orphan-page case rate: {orphan_case_rate:.4f}",
    ]
    if mean_item_coverage is not None:
        lines.append(f"- Mean item coverage: {mean_item_coverage:.4f}")
    if mean_slot_recall is not None:
        lines.append(f"- Mean slot recall: {mean_slot_recall:.4f}")
    if evidence_family_complete_rate is not None:
        lines.append(f"- Evidence-family full-coverage rate: {evidence_family_complete_rate:.4f}")
    lines.append(f"- Overprune violations: {overprune_violations}")
    lines.append(f"- Wrong-document tagged case rate: {wrong_document_case_rate:.4f}")
    return lines


def _worst_scores(scores: list[CaseScore]) -> list[CaseScore]:
    return sorted(scores, key=lambda score: (score.f_beta, -len(score.missing_pages), -len(score.orphan_pages)))[:10]


def _append_worst_case_section(lines: list[str], *, title: str, scores: list[CaseScore]) -> None:
    lines.extend(["", title, ""])
    if not scores:
        lines.append("- None")
        return

    for score in _worst_scores(scores):
        lines.append(f"- `{score.question_id}`: F_beta={score.f_beta:.4f}")
        lines.append(f"  gold={score.gold_pages}")
        lines.append(f"  predicted={score.predicted_pages}")
        lines.append(f"  trust_tier={score.trust_tier}")
        lines.append(f"  gold_origin={score.gold_origin}")
        if score.missing_pages:
            lines.append(f"  missing={score.missing_pages}")
        if score.orphan_pages:
            lines.append(f"  orphan={score.orphan_pages}")
        if score.item_coverage is not None:
            lines.append(f"  item_coverage={score.item_coverage:.4f}")
        if score.slot_recall is not None:
            lines.append(f"  slot_recall={score.slot_recall:.4f}")
        if score.evidence_family_complete is not None:
            lines.append(f"  evidence_family_complete={'true' if score.evidence_family_complete else 'false'}")
        if score.overprune_violations:
            lines.append(f"  overprune_violations={score.overprune_violations}")
        if score.wrong_document_issue:
            lines.append("  wrong_document_issue=true")
        if score.audit_note:
            lines.append(f"  audit_note={score.audit_note}")


def _build_report(*, scores: list[CaseScore], beta: float) -> str:
    if not scores:
        return "# Manual Page Benchmark\n\nNo cases found.\n"

    trusted_scores = [score for score in scores if score.trust_tier == "trusted"]
    suspect_scores = [score for score in scores if score.trust_tier == "suspect"]

    lines = ["# Manual Page Benchmark", "", "## All Cases", ""]
    lines.extend(_summary_lines(scores=scores, beta=beta))
    lines.extend(["", "## Trusted Tier", ""])
    lines.extend(_summary_lines(scores=trusted_scores, beta=beta))
    if suspect_scores:
        lines.extend(["", "## Suspect Tier", ""])
        lines.extend(_summary_lines(scores=suspect_scores, beta=beta))

    _append_worst_case_section(lines, title="## Worst Trusted Cases", scores=trusted_scores)
    if suspect_scores:
        _append_worst_case_section(lines, title="## Worst Suspect Cases", scores=suspect_scores)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a manual page-level grounding benchmark against an eval JSON.")
    parser.add_argument("--eval", type=Path, required=True, help="Path to eval_*.json produced by the harness.")
    parser.add_argument("--benchmark", type=Path, required=True, help="Path to manual benchmark JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Optional markdown output path.")
    parser.add_argument("--beta", type=float, default=2.5, help="F-beta parameter (default: 2.5).")
    args = parser.parse_args()

    cases = _load_benchmark(args.benchmark)
    eval_cases = _eval_cases_by_question_id(args.eval)
    scores = [_score_case(case, eval_cases.get(case.question_id), beta=args.beta) for case in cases]
    report = _build_report(scores=scores, beta=args.beta)

    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
