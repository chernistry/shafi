# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class BaselineComparison:
    baseline_label: str
    baseline_path: str
    baseline_sha256: str
    answer_changed_qids: list[str]
    page_changed_qids: list[str]
    equivalent_to_champion: bool
    verdict: str


def _load_json_dict(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _answers_by_id(payload: JsonDict) -> dict[str, JsonDict]:
    answers_obj = payload.get("answers")
    answers = cast("list[object]", answers_obj) if isinstance(answers_obj, list) else []
    out: dict[str, JsonDict] = {}
    for row in answers:
        if not isinstance(row, dict):
            continue
        record = cast("JsonDict", row)
        qid = str(record.get("question_id") or "").strip()
        if qid:
            out[qid] = record
    return out


def _answer_value(record: JsonDict) -> str:
    return json.dumps(record.get("answer"), ensure_ascii=False, sort_keys=True)


def _page_projection(record: JsonDict) -> str:
    telemetry = cast("JsonDict", record.get("telemetry")) if isinstance(record.get("telemetry"), dict) else {}
    retrieval = cast("JsonDict", telemetry.get("retrieval")) if isinstance(telemetry.get("retrieval"), dict) else {}
    return json.dumps(
        {
            "retrieved_chunk_pages": retrieval.get("retrieved_chunk_pages", []),
            "used_page_ids": telemetry.get("used_page_ids", []),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_against_champion(
    *,
    champion_submission: dict[str, JsonDict],
    baseline_path: Path,
) -> BaselineComparison:
    baseline_submission = _answers_by_id(_load_json_dict(baseline_path))
    if set(champion_submission) != set(baseline_submission):
        raise ValueError(f"Champion and baseline question_id sets do not match: {baseline_path}")

    answer_changed_qids: list[str] = []
    page_changed_qids: list[str] = []
    for qid in sorted(champion_submission):
        if _answer_value(champion_submission[qid]) != _answer_value(baseline_submission[qid]):
            answer_changed_qids.append(qid)
        if _page_projection(champion_submission[qid]) != _page_projection(baseline_submission[qid]):
            page_changed_qids.append(qid)

    equivalent_to_champion = not answer_changed_qids and not page_changed_qids
    verdict = "equivalent_to_champion" if equivalent_to_champion else "not_equivalent_to_champion"
    return BaselineComparison(
        baseline_label=baseline_path.stem,
        baseline_path=str(baseline_path.resolve()),
        baseline_sha256=_sha256(baseline_path),
        answer_changed_qids=answer_changed_qids,
        page_changed_qids=page_changed_qids,
        equivalent_to_champion=equivalent_to_champion,
        verdict=verdict,
    )


def _render_markdown(
    *,
    champion_label: str,
    champion_path: Path,
    champion_sha256: str,
    comparisons: list[BaselineComparison],
) -> str:
    safe_paths = [comparison.baseline_path for comparison in comparisons if comparison.equivalent_to_champion]
    lines = [
        "# Champion Equivalence Report",
        "",
        f"- practical_champion_label: `{champion_label}`",
        f"- practical_champion_submission: `{champion_path.resolve()}`",
        f"- practical_champion_sha256: `{champion_sha256}`",
        f"- safe_baselines: `{', '.join(safe_paths) if safe_paths else '(none)'}`",
        "- submission_policy: `NO_SUBMIT_WITHOUT_USER_APPROVAL`",
        "",
    ]
    for comparison in comparisons:
        lines.extend(
            [
                f"## {comparison.baseline_label}",
                f"- baseline_path: `{comparison.baseline_path}`",
                f"- baseline_sha256: `{comparison.baseline_sha256}`",
                f"- equivalent_to_champion: `{comparison.equivalent_to_champion}`",
                f"- verdict: `{comparison.verdict}`",
                f"- answer_changed_count: `{len(comparison.answer_changed_qids)}`",
                f"- page_changed_count: `{len(comparison.page_changed_qids)}`",
                f"- answer_changed_qids: `{', '.join(comparison.answer_changed_qids) if comparison.answer_changed_qids else '(none)'}`",
                f"- page_changed_qids: `{', '.join(comparison.page_changed_qids) if comparison.page_changed_qids else '(none)'}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify which baselines are exactly equivalent to the practical champion artifact.")
    parser.add_argument("--champion-label", required=True)
    parser.add_argument("--champion-submission", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    champion_path = args.champion_submission.resolve()
    champion_submission = _answers_by_id(_load_json_dict(champion_path))
    champion_sha256 = _sha256(champion_path)
    comparisons = [
        _compare_against_champion(
            champion_submission=champion_submission,
            baseline_path=baseline.resolve(),
        )
        for baseline in args.baseline
    ]

    safe_baselines = [comparison.baseline_path for comparison in comparisons if comparison.equivalent_to_champion]
    safe_sha256 = [comparison.baseline_sha256 for comparison in comparisons if comparison.equivalent_to_champion]
    payload: JsonDict = {
        "practical_champion_label": str(args.champion_label).strip(),
        "practical_champion_submission": str(champion_path),
        "practical_champion_sha256": champion_sha256,
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "safe_baselines": safe_baselines,
        "safe_baseline_sha256": safe_sha256,
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(
        _render_markdown(
            champion_label=str(args.champion_label).strip(),
            champion_path=champion_path,
            champion_sha256=champion_sha256,
            comparisons=comparisons,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
