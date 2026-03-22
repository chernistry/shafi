# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            items.append(text)
    return items


def _normalize_answer(text: str) -> str:
    return " ".join((text or "").strip().split())


def _load_truth_routes(path: Path | None) -> dict[str, JsonDict]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    by_qid: dict[str, JsonDict] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("question_id") or "").strip()
        if qid:
            by_qid[qid] = cast("JsonDict", raw)
    return by_qid


def _load_artifact(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    if isinstance(payload, list):
        by_qid: dict[str, JsonDict] = {}
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            telemetry = cast("JsonDict", raw.get("telemetry") or {})
            case = cast("JsonDict", raw.get("case") or {})
            qid = str(telemetry.get("question_id") or case.get("case_id") or case.get("question_id") or "").strip()
            if not qid:
                continue
            by_qid[qid] = {
                "question": str(case.get("question") or "").strip(),
                "answer_type": str(case.get("answer_type") or telemetry.get("answer_type") or "").strip(),
                "answer_text": str(raw.get("answer_text") or "").strip(),
                "used_page_ids": _coerce_str_list(telemetry.get("used_page_ids")),
                "context_page_ids": _coerce_str_list(telemetry.get("context_page_ids")),
                "cited_page_ids": _coerce_str_list(telemetry.get("cited_page_ids")),
                "retrieved_page_ids": _coerce_str_list(telemetry.get("retrieved_page_ids")),
                "doc_refs": _coerce_str_list(telemetry.get("doc_refs")),
                "model_llm": str(telemetry.get("model_llm") or "").strip(),
                "generation_mode": str(telemetry.get("generation_mode") or "").strip(),
            }
        return by_qid

    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        by_qid = {}
        for raw in cast("list[object]", payload["cases"]):
            if not isinstance(raw, dict):
                continue
            telemetry = cast("JsonDict", raw.get("telemetry") or {})
            qid = str(raw.get("question_id") or raw.get("case_id") or telemetry.get("question_id") or "").strip()
            if not qid:
                continue
            by_qid[qid] = {
                "question": str(raw.get("question") or "").strip(),
                "answer_type": str(raw.get("answer_type") or telemetry.get("answer_type") or "").strip(),
                "answer_text": str(raw.get("answer") or raw.get("answer_text") or "").strip(),
                "used_page_ids": _coerce_str_list(raw.get("used_pages") or telemetry.get("used_page_ids")),
                "context_page_ids": _coerce_str_list(telemetry.get("context_page_ids")),
                "cited_page_ids": _coerce_str_list(telemetry.get("cited_page_ids")),
                "retrieved_page_ids": _coerce_str_list(telemetry.get("retrieved_page_ids")),
                "doc_refs": _coerce_str_list(telemetry.get("doc_refs")),
                "model_llm": str(telemetry.get("model_llm") or "").strip(),
                "generation_mode": str(telemetry.get("generation_mode") or "").strip(),
            }
        return by_qid

    raise ValueError(f"Unsupported artifact shape in {path}")


def build_diff(
    *,
    artifact_a: dict[str, JsonDict],
    artifact_b: dict[str, JsonDict],
    truth_routes: dict[str, JsonDict],
    label_a: str,
    label_b: str,
) -> JsonDict:
    all_qids = sorted(set(artifact_a) | set(artifact_b))
    cases: list[JsonDict] = []
    answer_changed = 0
    used_pages_changed = 0
    null_behavior_changed = 0
    route_family_counts: dict[str, int] = {}

    for qid in all_qids:
        left = artifact_a.get(qid, {})
        right = artifact_b.get(qid, {})
        truth = truth_routes.get(qid, {})
        route_family = str(truth.get("route_family") or "").strip() or "unknown"
        route_family_counts[route_family] = route_family_counts.get(route_family, 0) + 1

        left_answer = _normalize_answer(str(left.get("answer_text") or ""))
        right_answer = _normalize_answer(str(right.get("answer_text") or ""))
        left_used = _coerce_str_list(left.get("used_page_ids"))
        right_used = _coerce_str_list(right.get("used_page_ids"))
        left_null = left_answer.lower() in {"", "null"}
        right_null = right_answer.lower() in {"", "null"}
        if left_answer != right_answer:
            answer_changed += 1
        if left_used != right_used:
            used_pages_changed += 1
        if left_null != right_null:
            null_behavior_changed += 1

        cases.append(
            {
                "question_id": qid,
                "route_family": route_family,
                "model_route": str(truth.get("model_route") or "").strip(),
                "question": str(left.get("question") or right.get("question") or truth.get("question") or "").strip(),
                "answer_type": str(left.get("answer_type") or right.get("answer_type") or truth.get("answer_type") or "").strip(),
                f"{label_a}_answer": left_answer,
                f"{label_b}_answer": right_answer,
                f"{label_a}_used_page_ids": left_used,
                f"{label_b}_used_page_ids": right_used,
                f"{label_a}_doc_refs": _coerce_str_list(left.get("doc_refs")),
                f"{label_b}_doc_refs": _coerce_str_list(right.get("doc_refs")),
                "answer_changed": left_answer != right_answer,
                "used_pages_changed": left_used != right_used,
                "null_behavior_changed": left_null != right_null,
            }
        )

    summary: JsonDict = {
        "label_a": label_a,
        "label_b": label_b,
        "case_count": len(all_qids),
        "answer_changed_count": answer_changed,
        "used_pages_changed_count": used_pages_changed,
        "null_behavior_changed_count": null_behavior_changed,
        "route_family_counts": route_family_counts,
    }
    return {"summary": summary, "cases": cases}


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload["summary"])
    cases = cast("list[JsonDict]", payload["cases"])
    label_a = str(summary["label_a"])
    label_b = str(summary["label_b"])
    lines = [
        "# Artifact Diff",
        "",
        f"- label_a: `{label_a}`",
        f"- label_b: `{label_b}`",
        f"- case_count: `{summary['case_count']}`",
        f"- answer_changed_count: `{summary['answer_changed_count']}`",
        f"- used_pages_changed_count: `{summary['used_pages_changed_count']}`",
        f"- null_behavior_changed_count: `{summary['null_behavior_changed_count']}`",
        "",
        "## Route Families",
        "",
    ]
    for key, value in sorted(cast("dict[str, int]", summary["route_family_counts"]).items()):
        lines.append(f"- `{key}`: `{value}`")

    changed = [case for case in cases if bool(case["answer_changed"]) or bool(case["used_pages_changed"]) or bool(case["null_behavior_changed"])]
    lines.extend(["", "## Changed Cases", ""])
    if not changed:
        lines.append("- none")
        return "\n".join(lines) + "\n"

    for case in changed:
        lines.extend(
            [
                f"### {case['question_id']}",
                f"- route_family: `{case['route_family']}`",
                f"- answer_type: `{case['answer_type']}`",
                f"- question: {case['question']}",
                f"- {label_a}_used_pages: `{case[f'{label_a}_used_page_ids']}`",
                f"- {label_b}_used_pages: `{case[f'{label_b}_used_page_ids']}`",
                f"- {label_a}_answer: `{case[f'{label_a}_answer']}`",
                f"- {label_b}_answer: `{case[f'{label_b}_answer']}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two artifact outputs by qid.")
    parser.add_argument("--artifact-a", type=Path, required=True)
    parser.add_argument("--artifact-b", type=Path, required=True)
    parser.add_argument("--label-a", type=str, required=True)
    parser.add_argument("--label-b", type=str, required=True)
    parser.add_argument("--truth-scaffold-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_diff(
        artifact_a=_load_artifact(args.artifact_a),
        artifact_b=_load_artifact(args.artifact_b),
        truth_routes=_load_truth_routes(args.truth_scaffold_json),
        label_a=args.label_a,
        label_b=args.label_b,
    )
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
