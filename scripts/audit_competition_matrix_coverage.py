# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _fmt_float(value: object) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.6f}"


def build_audit(payload: JsonDict) -> JsonDict:
    rows_obj = payload.get("rows")
    if not isinstance(rows_obj, list):
        raise ValueError("Expected 'rows' list in matrix payload")
    rows = [cast("JsonDict", row) for row in rows_obj if isinstance(row, dict)]

    candidate_statuses = {"candidate", "ceiling", "rejected"}
    missing_supported: list[str] = []
    missing_unsupported: list[str] = []
    duplicate_groups: dict[tuple[str, str, str], list[str]] = {}

    for row in rows:
        status = str(row.get("status") or "")
        if status not in candidate_statuses:
            continue
        label = str(row.get("label") or "")
        notes = str(row.get("notes") or "")
        estimates = (
            row.get("platform_like_total_estimate"),
            row.get("strict_total_estimate"),
            row.get("paranoid_total_estimate"),
        )
        if all(value in (None, "") for value in estimates):
            if "[estimates=unsupported_local_envelope]" in notes or "[estimates=not_applicable_invalid]" in notes:
                missing_unsupported.append(label)
            else:
                missing_supported.append(label)
            continue
        triple = tuple(_fmt_float(value) for value in estimates)
        duplicate_groups.setdefault(triple, []).append(label)

    duplicate_rows = [
        {
            "platform_like_total_estimate": triple[0],
            "strict_total_estimate": triple[1],
            "paranoid_total_estimate": triple[2],
            "labels": labels,
        }
        for triple, labels in sorted(duplicate_groups.items())
        if len(labels) > 1
    ]

    return {
        "rows_checked": len(rows),
        "candidate_like_rows": sum(1 for row in rows if str(row.get("status") or "") in candidate_statuses),
        "missing_supported_estimates": missing_supported,
        "missing_unsupported_estimates": missing_unsupported,
        "duplicate_estimate_groups": duplicate_rows,
    }


def render_markdown(audit: JsonDict) -> str:
    lines = [
        "# Competition Matrix Coverage Audit",
        "",
        f"- rows_checked: `{audit['rows_checked']}`",
        f"- candidate_like_rows: `{audit['candidate_like_rows']}`",
        f"- missing_supported_estimates: `{len(cast('list[object]', audit['missing_supported_estimates']))}`",
        f"- missing_unsupported_estimates: `{len(cast('list[object]', audit['missing_unsupported_estimates']))}`",
        "",
        "## Missing Supported Estimates",
        "",
    ]
    missing_supported = cast("list[object]", audit["missing_supported_estimates"])
    if missing_supported:
        for label in missing_supported:
            lines.append(f"- `{label}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Missing Unsupported Estimates", ""])
    missing_unsupported = cast("list[object]", audit["missing_unsupported_estimates"])
    if missing_unsupported:
        for label in missing_unsupported:
            lines.append(f"- `{label}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Duplicate Estimate Groups", ""])
    duplicate_groups = cast("list[object]", audit["duplicate_estimate_groups"])
    if duplicate_groups:
        for raw_group in duplicate_groups:
            group = cast("JsonDict", raw_group)
            labels = cast("list[object]", group["labels"])
            lines.append(
                "- "
                f"platform_like=`{group['platform_like_total_estimate']}` "
                f"strict=`{group['strict_total_estimate']}` "
                f"paranoid=`{group['paranoid_total_estimate']}` "
                f"labels={','.join(str(label) for label in labels)}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical competition matrix estimate coverage.")
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_json(args.matrix_json)
    audit = build_audit(payload)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
