from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

try:
    from scripts.scan_private_doc_anomalies import build_cluster_collapsed_review
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.scan_private_doc_anomalies import build_cluster_collapsed_review

JsonDict = dict[str, Any]


def _load_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", payload)


def _load_json_list(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [cast("JsonDict", item) for item in cast("list[object]", payload) if isinstance(item, dict)]


def _load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(cast("JsonDict", payload))
    return rows


def build_underqueried_family_doctrine_pack(
    *,
    gap_report: JsonDict,
    scan_records: list[JsonDict],
    variants: list[JsonDict],
) -> JsonDict:
    family_rows = cast("list[JsonDict]", gap_report.get("family_coverage_report") or [])
    underqueried_rows = [
        row
        for row in family_rows
        if str(row.get("coverage_bucket") or "") in {"zero-hit", "one-hit"}
    ]
    underqueried_rows.sort(
        key=lambda row: (
            0 if str(row.get("coverage_bucket") or "") == "zero-hit" else 1,
            -int(row.get("doc_count") or 0),
            str(row.get("family_tag") or ""),
        )
    )

    doc_by_id = {
        str(record.get("doc_id") or ""): record
        for record in scan_records
        if str(record.get("doc_id") or "")
    }
    variants_by_family: dict[str, list[JsonDict]] = defaultdict(list)
    for variant in variants:
        gold_doc_ids = cast("list[str]", variant.get("expected_gold_doc_ids") or [])
        mapped_families = {
            family_tag
            for doc_id in gold_doc_ids
            for family_tag in cast("list[str]", doc_by_id.get(doc_id, {}).get("doc_family_tags") or [])
        }
        for family_tag in mapped_families:
            variants_by_family[family_tag].append(variant)

    entries: list[JsonDict] = []
    for row in underqueried_rows:
        family_tag = str(row.get("family_tag") or "")
        family_records = [
            record for record in scan_records if family_tag in cast("list[str]", record.get("doc_family_tags") or [])
        ]
        representative_docs = build_cluster_collapsed_review(family_records, limit=3)
        family_variants = variants_by_family.get(family_tag, [])
        entries.append(
            {
                "family_tag": family_tag,
                "coverage_bucket": str(row.get("coverage_bucket") or "unscored"),
                "doc_count": int(row.get("doc_count") or 0),
                "question_count": int(row.get("question_count") or 0),
                "coverage_ratio": float(row.get("coverage_ratio") or 0.0),
                "representative_docs": representative_docs,
                "anti_overfit_variants": [
                    {
                        "id": str(variant.get("id") or ""),
                        "question": str(variant.get("question") or ""),
                        "variant_type": str(variant.get("variant_type") or ""),
                        "expected_gold_doc_ids": cast("list[str]", variant.get("expected_gold_doc_ids") or []),
                    }
                    for variant in family_variants[:5]
                ],
                "measurement_only": True,
                "guidance": (
                    "Use this family as a measurement-only doctrine review lane. "
                    "Do not patch runtime behavior from this pack alone."
                ),
            }
        )

    return {
        "measurement_only": True,
        "underqueried_family_count": len(entries),
        "zero_hit_families": [entry["family_tag"] for entry in entries if entry["coverage_bucket"] == "zero-hit"],
        "one_hit_families": [entry["family_tag"] for entry in entries if entry["coverage_bucket"] == "one-hit"],
        "entries": entries,
    }


def render_underqueried_family_doctrine_pack_markdown(pack: JsonDict) -> str:
    lines = [
        "# Underqueried Family Doctrine Pack",
        "",
        "- Measurement only: `True`",
        f"- Underqueried families: `{pack['underqueried_family_count']}`",
        "",
    ]
    entries = cast("list[JsonDict]", pack.get("entries") or [])
    if not entries:
        lines.append("No zero-hit or one-hit families were found.")
        return "\n".join(lines) + "\n"

    for entry in entries:
        lines.append(f"## {entry['family_tag']}")
        lines.append("")
        lines.append(f"- Coverage bucket: `{entry['coverage_bucket']}`")
        lines.append(f"- Doc count: `{entry['doc_count']}`")
        lines.append(f"- Question count: `{entry['question_count']}`")
        lines.append(f"- Coverage ratio: `{entry['coverage_ratio']}`")
        lines.append(f"- Guidance: {entry['guidance']}")
        lines.append("")

        representative_docs = cast("list[JsonDict]", entry.get("representative_docs") or [])
        lines.append("### Representative Docs")
        lines.append("")
        if representative_docs:
            lines.extend(
                [
                    "| rank | cluster_kind | representative_doc_id | filename | score | member_count | top_reasons |",
                    "| ---: | --- | --- | --- | ---: | ---: | --- |",
                ]
            )
            for rank, doc_entry in enumerate(representative_docs, start=1):
                top_reasons = ", ".join(cast("list[str]", doc_entry.get("reason_tags") or [])[:3]) or "-"
                lines.append(
                    f"| {rank} | {doc_entry['cluster_kind']} | {doc_entry['representative_doc_id']} | "
                    f"{doc_entry['representative_filename']} | {doc_entry['suspicion_score']} | "
                    f"{doc_entry['member_count']} | {top_reasons} |"
                )
        else:
            lines.append("- No representative docs available.")
        lines.append("")

        variants = cast("list[JsonDict]", entry.get("anti_overfit_variants") or [])
        lines.append("### Anti-Overfit Variants")
        lines.append("")
        if variants:
            for variant in variants:
                lines.append(
                    f"- `{variant['id']}` ({variant['variant_type']}): {variant['question']}"
                )
        else:
            lines.append("- No public-derived variants mapped to this family yet.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a measurement-only doctrine pack for underqueried families.")
    parser.add_argument("--gap-report-json", type=Path, required=True)
    parser.add_argument("--scan-results-jsonl", type=Path, required=True)
    parser.add_argument("--variants-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    pack = build_underqueried_family_doctrine_pack(
        gap_report=_load_json_object(args.gap_report_json),
        scan_records=_load_jsonl(args.scan_results_jsonl),
        variants=_load_json_list(args.variants_json),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_underqueried_family_doctrine_pack_markdown(pack), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
