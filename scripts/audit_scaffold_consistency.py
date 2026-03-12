from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]

DEFAULT_FIELDS = (
    "manual_verdict",
    "expected_answer",
    "failure_class",
    "manual_exactness_labels",
)


@dataclass(frozen=True)
class FieldObservation:
    path: str
    value: object


@dataclass(frozen=True)
class FieldConsistency:
    field: str
    consistent: bool
    variants: dict[str, list[str]]


@dataclass(frozen=True)
class QuestionConsistency:
    question_id: str
    found_in: int
    missing_paths: list[str]
    fields: list[FieldConsistency]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _records_by_id(path: Path) -> dict[str, JsonDict]:
    payload = _load_json(path)
    records_obj = payload.get("records")
    if not isinstance(records_obj, list):
        raise ValueError(f"Scaffold at {path} is missing 'records'")
    out: dict[str, JsonDict] = {}
    for raw_obj in cast("list[object]", records_obj):
        if not isinstance(raw_obj, dict):
            continue
        raw = cast("JsonDict", raw_obj)
        question_id = str(raw.get("question_id") or "").strip()
        if question_id and question_id not in out:
            out[question_id] = raw
    return out


def _stable_variant_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _collect_field_consistency(
    *,
    question_id: str,
    field: str,
    records_by_path: dict[Path, dict[str, JsonDict]],
) -> FieldConsistency:
    variants: dict[str, list[str]] = {}
    for path, records in records_by_path.items():
        record = records.get(question_id)
        if record is None:
            continue
        value = record.get(field)
        key = _stable_variant_key(value)
        variants.setdefault(key, []).append(str(path))
    return FieldConsistency(
        field=field,
        consistent=len(variants) <= 1,
        variants=variants,
    )


def build_consistency_report(
    *,
    scaffold_paths: list[Path],
    question_ids: list[str],
    fields: list[str],
) -> list[QuestionConsistency]:
    records_by_path = {path: _records_by_id(path) for path in scaffold_paths}
    reports: list[QuestionConsistency] = []
    for question_id in question_ids:
        missing_paths = [
            str(path)
            for path, records in records_by_path.items()
            if question_id not in records
        ]
        field_reports = [
            _collect_field_consistency(
                question_id=question_id,
                field=field,
                records_by_path=records_by_path,
            )
            for field in fields
        ]
        found_in = len(scaffold_paths) - len(missing_paths)
        reports.append(
            QuestionConsistency(
                question_id=question_id,
                found_in=found_in,
                missing_paths=missing_paths,
                fields=field_reports,
            )
        )
    return reports


def render_markdown(
    *,
    scaffold_paths: list[Path],
    reports: list[QuestionConsistency],
) -> str:
    lines = [
        "# Truth-Audit Scaffold Consistency Report",
        "",
        f"- Scaffolds scanned: `{len(scaffold_paths)}`",
        "",
    ]
    for report in reports:
        lines.extend(
            [
                f"## {report.question_id}",
                f"- Found in: `{report.found_in}`",
                f"- Missing in: `{len(report.missing_paths)}`",
            ]
        )
        if report.missing_paths:
            lines.append(f"- Missing paths: `{', '.join(report.missing_paths)}`")
        lines.append("")
        for field in report.fields:
            lines.append(f"### {field.field}")
            lines.append(f"- consistent: `{field.consistent}`")
            for variant_key, paths in sorted(field.variants.items(), key=lambda item: item[0]):
                lines.append(f"- variant: `{variant_key}`")
                lines.append(f"  - paths: `{', '.join(sorted(paths))}`")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit consistency of manual truth-audit fields across scaffold artifacts.")
    parser.add_argument("--scaffold", type=Path, action="append", default=[])
    parser.add_argument("--scaffold-glob", default="platform_runs/warmup/truth_audit_scaffold*.json")
    parser.add_argument("--qid", action="append", default=[])
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    scaffold_paths = sorted({path.resolve() for path in args.scaffold} | {path.resolve() for path in Path(".").glob(str(args.scaffold_glob))})
    if not scaffold_paths:
        raise ValueError("No scaffold paths found")

    question_ids = sorted({str(question_id).strip() for question_id in args.qid if str(question_id).strip()})
    if not question_ids:
        raise ValueError("At least one --qid is required")

    fields = [str(field).strip() for field in args.field if str(field).strip()] or list(DEFAULT_FIELDS)
    reports = build_consistency_report(
        scaffold_paths=scaffold_paths,
        question_ids=question_ids,
        fields=fields,
    )
    markdown = render_markdown(scaffold_paths=scaffold_paths, reports=reports)
    payload = {
        "scaffolds": [str(path) for path in scaffold_paths],
        "question_ids": question_ids,
        "fields": fields,
        "reports": [asdict(report) for report in reports],
    }

    if args.out is not None:
        args.out.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
