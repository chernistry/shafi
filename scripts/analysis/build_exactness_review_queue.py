# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


@dataclass(frozen=True)
class ReviewCandidate:
    question_id: str
    score: int
    reasons: list[str]
    answer_type: str
    route_family: str
    question: str
    current_answer_text: str
    expected_answer: str
    manual_verdict: str
    failure_class: str
    manual_exactness_labels: list[str]
    exactness_review_flags: list[str]
    support_shape_flags: list[str]


def _load_scaffold(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {path} is missing 'records'")
    return [cast("JsonDict", item) for item in records_obj if isinstance(item, dict)]


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            out.append(text)
    return out


def _is_deterministic(record: JsonDict) -> bool:
    answer_type = str(record.get("answer_type") or "").strip().lower()
    return answer_type in {"boolean", "number", "date", "name", "names"}


def _risk_score(record: JsonDict) -> int:
    score = 0
    route_family = str(record.get("route_family") or "").strip().lower()
    answer_type = str(record.get("answer_type") or "").strip().lower()
    question = str(record.get("question") or "").strip().lower()
    failure_class = str(record.get("failure_class") or "").strip().lower()
    manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
    labels = {item.lower() for item in _coerce_str_list(record.get("manual_exactness_labels"))}
    exactness_flags = _coerce_str_list(record.get("exactness_review_flags"))
    required_page_anchor = record.get("required_page_anchor")

    if route_family == "model":
        score += 100
    if failure_class == "support_undercoverage":
        score += 80
    if "platform_exact_risk" in labels:
        score += 80
    if "page_specific_exact_risk" in labels:
        score += 70
    if "suffix_risk" in labels:
        score += 60
    if isinstance(required_page_anchor, dict) and required_page_anchor:
        score += 60
    if any(term in question for term in ("page 2", "second page", "title page", "cover page", "first page")):
        score += 60
    if answer_type in {"name", "names"}:
        score += 45
    if "claim number" in question:
        score += 45
    if exactness_flags:
        score += 30
    if not manual_verdict:
        score += 20
    if manual_verdict == "correct":
        score -= 10
    unresolved_risk_labels = {"platform_exact_risk", "page_specific_exact_risk", "suffix_risk"}
    if "semantic_correct" in labels and not labels.intersection(unresolved_risk_labels):
        # Reviewed correct cases should not crowd out unresolved answer-risk candidates.
        score -= 120
    return score


def _risk_reasons(record: JsonDict) -> list[str]:
    reasons: list[str] = []
    route_family = str(record.get("route_family") or "").strip().lower()
    answer_type = str(record.get("answer_type") or "").strip().lower()
    question = str(record.get("question") or "").strip().lower()
    failure_class = str(record.get("failure_class") or "").strip().lower()
    manual_verdict = str(record.get("manual_verdict") or "").strip().lower()
    labels = {item.lower() for item in _coerce_str_list(record.get("manual_exactness_labels"))}
    exactness_flags = _coerce_str_list(record.get("exactness_review_flags"))
    required_page_anchor = record.get("required_page_anchor")

    if route_family == "model":
        reasons.append("model_route")
    if failure_class == "support_undercoverage":
        reasons.append("support_undercoverage")
    if "platform_exact_risk" in labels:
        reasons.append("platform_exact_risk")
    if "page_specific_exact_risk" in labels:
        reasons.append("page_specific_exact_risk")
    if "suffix_risk" in labels:
        reasons.append("suffix_risk")
    if isinstance(required_page_anchor, dict) and required_page_anchor:
        reasons.append("required_page_anchor")
    if any(term in question for term in ("page 2", "second page", "title page", "cover page", "first page")):
        reasons.append("explicit_page_reference")
    if answer_type in {"name", "names"}:
        reasons.append("name_exactness")
    if "claim number" in question:
        reasons.append("claim_number_reference")
    if exactness_flags:
        reasons.extend(f"flag:{flag}" for flag in exactness_flags)
    if not manual_verdict:
        reasons.append("manual_verdict_missing")
    return reasons


def build_candidates(records: list[JsonDict]) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for record in records:
        if not _is_deterministic(record):
            continue
        score = _risk_score(record)
        if score <= 0:
            continue
        candidates.append(
            ReviewCandidate(
                question_id=str(record.get("question_id") or "").strip(),
                score=score,
                reasons=_risk_reasons(record),
                answer_type=str(record.get("answer_type") or "").strip(),
                route_family=str(record.get("route_family") or "").strip(),
                question=str(record.get("question") or "").strip(),
                current_answer_text=str(record.get("current_answer_text") or record.get("current_answer") or "").strip(),
                expected_answer=str(record.get("expected_answer") or "").strip(),
                manual_verdict=str(record.get("manual_verdict") or "").strip(),
                failure_class=str(record.get("failure_class") or "").strip(),
                manual_exactness_labels=_coerce_str_list(record.get("manual_exactness_labels")),
                exactness_review_flags=_coerce_str_list(record.get("exactness_review_flags")),
                support_shape_flags=_coerce_str_list(record.get("support_shape_flags")),
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.question_id))
    return candidates


def render_report(candidates: list[ReviewCandidate], *, limit: int) -> str:
    lines = [
        "# Exactness Review Queue",
        "",
        f"- Candidates: `{min(len(candidates), limit)}` shown / `{len(candidates)}` total",
        "",
    ]
    for candidate in candidates[:limit]:
        lines.extend(
            [
                f"## {candidate.question_id}",
                f"- risk_score: `{candidate.score}`",
                f"- answer_type: `{candidate.answer_type}`",
                f"- route_family: `{candidate.route_family}`",
                f"- question: {candidate.question}",
                f"- current_answer: `{candidate.current_answer_text}`",
                f"- expected_answer: `{candidate.expected_answer or 'None'}`",
                f"- manual_verdict: `{candidate.manual_verdict or '(blank)'}`",
                f"- failure_class: `{candidate.failure_class or '(blank)'}`",
                f"- manual_exactness_labels: `{', '.join(candidate.manual_exactness_labels) if candidate.manual_exactness_labels else '(none)'}`",
                f"- exactness_review_flags: `{', '.join(candidate.exactness_review_flags) if candidate.exactness_review_flags else '(none)'}`",
                f"- support_shape_flags: `{', '.join(candidate.support_shape_flags) if candidate.support_shape_flags else '(none)'}`",
                f"- reasons: `{', '.join(candidate.reasons) if candidate.reasons else '(none)'}`",
                "",
            ]
        )
    return "\n".join(lines)


def _summary_counts(candidates: list[ReviewCandidate], *, attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = str(getattr(candidate, attr) or "").strip() or "(blank)"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _json_payload(*, scaffold: Path, candidates: list[ReviewCandidate], limit: int) -> JsonDict:
    shown = candidates[:limit]
    return {
        "scaffold": str(scaffold),
        "candidate_count": len(candidates),
        "shown_count": len(shown),
        "route_family_counts": _summary_counts(candidates, attr="route_family"),
        "answer_type_counts": _summary_counts(candidates, attr="answer_type"),
        "manual_verdict_counts": _summary_counts(candidates, attr="manual_verdict"),
        "candidates": [asdict(candidate) for candidate in shown],
        "all_candidate_qids": [candidate.question_id for candidate in candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic exactness review queue from a truth-audit scaffold.")
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-qids-file", type=Path, default=None)
    args = parser.parse_args()

    candidates = build_candidates(_load_scaffold(args.scaffold))
    report = render_report(candidates, limit=args.limit)
    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    if args.out_json is not None:
        payload = _json_payload(scaffold=args.scaffold, candidates=candidates, limit=args.limit)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_qids_file is not None:
        args.out_qids_file.write_text("".join(f"{candidate.question_id}\n" for candidate in candidates), encoding="utf-8")


if __name__ == "__main__":
    main()
