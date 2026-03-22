# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]

_COMPARE_RE = re.compile(
    r"\b(?:earlier|higher|lower|same|both|compare|which case has|which of the two|appeared in both)\b",
    re.IGNORECASE,
)
_CASE_REF_RE = re.compile(r"\b(?:CFI|CA|ARB|ENF|SCT)\s+\d+/\d+\b", re.IGNORECASE)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_dict_list(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in value if isinstance(item, dict)]


def _coerce_preview_texts(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    previews: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        snippet = str(item.get("snippet") or "").strip()
        if snippet:
            previews.append(snippet)
    return previews


def _load_truth_records(path: Path | None) -> dict[str, JsonDict]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    return {
        str(record.get("question_id") or "").strip(): record
        for record in _coerce_dict_list(payload.get("records"))
        if str(record.get("question_id") or "").strip()
    }


def _extractive_family(question: str) -> str:
    q = " ".join((question or "").split()).casefold()
    if "who made this law" in q or "made this law" in q:
        return "who_made"
    if "official law number" in q:
        return "official_law_number"
    if "citation title" in q or q.startswith("what is the title of ") or q.startswith("what is the official title of "):
        return "citation_title"
    if "enacted" in q or "date of enactment" in q:
        return "enactment_date"
    if "commencement" in q or "come into force" in q or "effective date" in q:
        return "commencement"
    if "appointing and dismissing the registrar" in q or ("appoint" in q and "dismiss" in q and "registrar" in q):
        return "registrar_authority"
    if "administered by" in q or "who administers" in q or "responsible for administering" in q:
        return "administration"
    if "liability" in q and "partner" in q:
        return "liability"
    if "maximum fine" in q or "what is the maximum fine" in q:
        return "maximum_fine"
    if "what was the outcome" in q or "what was the result of the application" in q:
        return "case_outcome"
    if "what must it provide" in q and "english translation" in q:
        return "translation_requirement"
    return ""


def _classify_case(case: JsonDict, truth: JsonDict | None) -> tuple[str, str]:
    question = str(case.get("question") or "").strip()
    answer_type = str(case.get("answer_type") or "").strip().lower()
    answer_text = str(case.get("answer_text") or "").strip().lower()
    used_page_ids = case.get("used_page_ids")
    support_previews = _coerce_preview_texts((truth or {}).get("support_page_previews"))

    if not support_previews and (
        answer_text in {"", "null"}
        or not isinstance(used_page_ids, list)
        or not cast("list[object]", used_page_ids)
    ):
        return "upstream_retrieval_support_miss", ""

    if answer_type in {"boolean", "name", "names"} and (
        _COMPARE_RE.search(question) is not None or len(_CASE_REF_RE.findall(question)) >= 2
    ):
        return "compare", ""

    family = _extractive_family(question)
    if family:
        return "deterministic_extractive", family

    return "genuine_synthesis", ""


def build_report(*, model_route_pack: Path, truth_scaffold: Path | None) -> JsonDict:
    pack_payload = _load_json(model_route_pack)
    if not isinstance(pack_payload, dict):
        raise ValueError(f"Expected JSON object in {model_route_pack}")
    cases = _coerce_dict_list(pack_payload.get("cases"))
    truth_by_qid = _load_truth_records(truth_scaffold)

    labeled_cases: list[JsonDict] = []
    class_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}

    for case in cases:
        qid = str(case.get("question_id") or "").strip()
        truth = truth_by_qid.get(qid)
        bucket, family = _classify_case(case, truth)
        class_counts[bucket] = class_counts.get(bucket, 0) + 1
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
        labeled_cases.append(
            {
                "question_id": qid,
                "question": str(case.get("question") or "").strip(),
                "answer_type": str(case.get("answer_type") or "").strip(),
                "model_name": str(case.get("model_name") or "").strip(),
                "classification": bucket,
                "extractive_family": family,
                "support_preview_count": len(_coerce_preview_texts((truth or {}).get("support_page_previews"))),
                "route_family_truth": str((truth or {}).get("route_family") or "").strip(),
            }
        )

    return {
        "summary": {
            "case_count": len(labeled_cases),
            "classification_counts": class_counts,
            "extractive_family_counts": family_counts,
            "deterministic_extractive_case_count": class_counts.get("deterministic_extractive", 0),
        },
        "cases": labeled_cases,
    }


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload["summary"])
    cases = cast("list[JsonDict]", payload["cases"])
    lines = [
        "# Model-Route Pack Decomposition",
        "",
        f"- case_count: `{summary['case_count']}`",
        f"- deterministic_extractive_case_count: `{summary['deterministic_extractive_case_count']}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key, value in sorted(cast("dict[str, int]", summary["classification_counts"]).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Extractive Families", ""])
    for key, value in sorted(cast("dict[str, int]", summary["extractive_family_counts"]).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Cases", ""])
    for case in cases:
        family = str(case["extractive_family"] or "-")
        lines.extend(
            [
                f"### {case['question_id']}",
                f"- classification: `{case['classification']}`",
                f"- extractive_family: `{family}`",
                f"- answer_type: `{case['answer_type']}`",
                f"- model_name: `{case['model_name']}`",
                f"- support_preview_count: `{case['support_preview_count']}`",
                f"- question: {case['question']}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partition the model-route pack into bounded slices.")
    parser.add_argument("--model-route-pack", type=Path, required=True)
    parser.add_argument("--truth-scaffold-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_report(model_route_pack=args.model_route_pack, truth_scaffold=args.truth_scaffold_json)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
