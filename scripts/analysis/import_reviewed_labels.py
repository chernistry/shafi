"""Import reviewed golden labels into a canonical repo-local bundle.

The big-model relabeling pass produces three files outside the repo. This script
copies them into the repo-local reviewed-gold folder, validates alignment, and
derives confidence-aware golden and page-benchmark slices for downstream gates.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def _label_weight_from_confidence(confidence: str) -> float:
    """Map reviewed confidence into the training weight policy.

    Args:
        confidence: Raw confidence string from the reviewed labels.

    Returns:
        Numeric training weight for the reviewed row.
    """
    normalized = confidence.strip().lower()
    if normalized == "high":
        return 1.0
    if normalized == "medium":
        return 0.5
    return 0.0


def _coerce_str(value: object) -> str:
    """Return a trimmed string representation.

    Args:
        value: Raw object.

    Returns:
        Trimmed string value.
    """
    return str(value or "").strip()


def _coerce_str_list(value: object) -> list[str]:
    """Return a trimmed string list.

    Args:
        value: Raw list-like object.

    Returns:
        List of non-empty strings.
    """
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _coerce_str(item))]


def _load_json(path: Path) -> Any:
    """Load a JSON file.

    Args:
        path: Source JSON path.

    Returns:
        Decoded JSON value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_reviewed_inputs(
    *,
    reviewed_labels: Sequence[dict[str, object]],
    reviewed_benchmark_cases: Sequence[dict[str, object]],
    expected_count: int,
) -> tuple[list[str], int]:
    """Validate reviewed labels and benchmark alignment.

    Args:
        reviewed_labels: Reviewed label rows.
        reviewed_benchmark_cases: Reviewed page benchmark rows.
        expected_count: Expected row count.

    Returns:
        Tuple of reviewed label order and benchmark-page mismatch count.

    Raises:
        ValueError: If the reviewed inputs are malformed.
    """
    if len(reviewed_labels) != expected_count:
        raise ValueError(f"Expected {expected_count} reviewed labels, got {len(reviewed_labels)}")
    if len(reviewed_benchmark_cases) != expected_count:
        raise ValueError(f"Expected {expected_count} reviewed benchmark rows, got {len(reviewed_benchmark_cases)}")

    ordered_ids: list[str] = []
    seen_label_ids: set[str] = set()
    for item in reviewed_labels:
        question_id = _coerce_str(item.get("question_id"))
        if not question_id:
            raise ValueError("Reviewed labels contain an empty question_id")
        if question_id in seen_label_ids:
            raise ValueError(f"Duplicate reviewed question_id: {question_id}")
        seen_label_ids.add(question_id)
        ordered_ids.append(question_id)

        for page_id in _coerce_str_list(item.get("golden_page_ids")):
            if not isinstance(page_id, str):
                raise ValueError(f"Non-string page id for question_id={question_id}")

    benchmark_by_id: dict[str, list[str]] = {}
    seen_benchmark_ids: set[str] = set()
    for item in reviewed_benchmark_cases:
        question_id = _coerce_str(item.get("question_id"))
        if not question_id:
            raise ValueError("Reviewed page benchmark contains an empty question_id")
        if question_id in seen_benchmark_ids:
            raise ValueError(f"Duplicate reviewed benchmark question_id: {question_id}")
        seen_benchmark_ids.add(question_id)
        benchmark_by_id[question_id] = _coerce_str_list(item.get("gold_page_ids"))

    if seen_label_ids != seen_benchmark_ids:
        missing_from_benchmark = sorted(seen_label_ids - seen_benchmark_ids)
        missing_from_labels = sorted(seen_benchmark_ids - seen_label_ids)
        raise ValueError(
            "Reviewed labels and page benchmark do not share the same ID set: "
            f"missing_from_benchmark={missing_from_benchmark[:5]} "
            f"missing_from_labels={missing_from_labels[:5]}"
        )

    mismatch_count = 0
    for item in reviewed_labels:
        question_id = _coerce_str(item.get("question_id"))
        if benchmark_by_id.get(question_id, []) != _coerce_str_list(item.get("golden_page_ids")):
            mismatch_count += 1
    return ordered_ids, mismatch_count


def _normalize_reviewed_labels(
    reviewed_labels: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Normalize reviewed labels into one canonical schema.

    Args:
        reviewed_labels: Raw reviewed label rows.

    Returns:
        Normalized reviewed rows.
    """
    normalized: list[dict[str, object]] = []
    for item in reviewed_labels:
        confidence = _coerce_str(item.get("confidence")).lower() or "low"
        normalized.append(
            {
                "question_id": _coerce_str(item.get("question_id")),
                "question": _coerce_str(item.get("question")),
                "answer_type": _coerce_str(item.get("answer_type")) or "free_text",
                "golden_answer": item.get("golden_answer"),
                "golden_page_ids": _coerce_str_list(item.get("golden_page_ids")),
                "confidence": confidence,
                "label_status": _coerce_str(item.get("label_status")).lower(),
                "audit_note": _coerce_str(item.get("audit_note")),
                "current_label_problem": _coerce_str(item.get("current_label_problem")),
                "trust_tier": confidence,
                "label_weight": _label_weight_from_confidence(confidence),
            }
        )
    return normalized


def _filter_reviewed_labels(
    rows: Sequence[dict[str, object]],
    *,
    allowed_confidences: set[str],
) -> list[dict[str, object]]:
    """Filter reviewed rows by confidence.

    Args:
        rows: Canonical reviewed rows.
        allowed_confidences: Allowed confidence tier set.

    Returns:
        Filtered reviewed rows preserving order.
    """
    return [row for row in rows if _coerce_str(row.get("confidence")).lower() in allowed_confidences]


def _build_page_benchmark(
    rows: Sequence[dict[str, object]],
    *,
    trust_tier: str,
) -> dict[str, object]:
    """Build page-benchmark payload from reviewed rows.

    Args:
        rows: Canonical reviewed rows.
        trust_tier: Trust tier to apply to every exported case.

    Returns:
        Page benchmark payload.
    """
    return {
        "cases": [
            {
                "question_id": _coerce_str(row.get("question_id")),
                "gold_page_ids": _coerce_str_list(row.get("golden_page_ids")),
                "trust_tier": trust_tier,
            }
            for row in rows
        ]
    }


def import_reviewed_labels(
    *,
    reviewed_golden_path: Path,
    reviewed_page_benchmark_path: Path,
    audit_report_path: Path,
    output_dir: Path,
    expected_count: int = 100,
) -> dict[str, object]:
    """Import reviewed labels and derive confidence-aware slices.

    Args:
        reviewed_golden_path: Reviewed golden-label source path.
        reviewed_page_benchmark_path: Reviewed page-benchmark source path.
        audit_report_path: Markdown audit-report path.
        output_dir: Canonical reviewed-gold output directory.
        expected_count: Expected reviewed row count.

    Returns:
        JSON-serializable import manifest.

    Raises:
        FileNotFoundError: If a required source file is missing.
        ValueError: If validation fails.
    """
    for source_path in (reviewed_golden_path, reviewed_page_benchmark_path, audit_report_path):
        if not source_path.exists():
            raise FileNotFoundError(source_path)

    reviewed_labels_obj = _load_json(reviewed_golden_path)
    reviewed_benchmark_obj = _load_json(reviewed_page_benchmark_path)
    if not isinstance(reviewed_labels_obj, list):
        raise ValueError("Reviewed golden labels must be a JSON list")
    if not isinstance(reviewed_benchmark_obj, dict):
        raise ValueError("Reviewed page benchmark must be a JSON object")

    reviewed_labels = [item for item in reviewed_labels_obj if isinstance(item, dict)]
    reviewed_benchmark_cases_obj = reviewed_benchmark_obj.get("cases")
    if not isinstance(reviewed_benchmark_cases_obj, list):
        raise ValueError("Reviewed page benchmark is missing cases[]")
    reviewed_benchmark_cases = [item for item in reviewed_benchmark_cases_obj if isinstance(item, dict)]

    ordered_ids, benchmark_page_mismatch_count = _validate_reviewed_inputs(
        reviewed_labels=reviewed_labels,
        reviewed_benchmark_cases=reviewed_benchmark_cases,
        expected_count=expected_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = {
        "corrected_golden_labels_v3.json": reviewed_golden_path,
        "corrected_page_benchmark_v3.json": reviewed_page_benchmark_path,
        "label_audit_report.md": audit_report_path,
    }
    for target_name, source_path in copied_paths.items():
        shutil.copy2(source_path, output_dir / target_name)

    reviewed_all = _normalize_reviewed_labels(reviewed_labels)
    reviewed_high_confidence = _filter_reviewed_labels(reviewed_all, allowed_confidences={"high"})
    reviewed_medium_plus_high = _filter_reviewed_labels(reviewed_all, allowed_confidences={"high", "medium"})

    derived_files: dict[str, object] = {
        "reviewed_all_100.json": reviewed_all,
        "reviewed_high_confidence_81.json": reviewed_high_confidence,
        "reviewed_medium_plus_high_95.json": reviewed_medium_plus_high,
        "reviewed_page_benchmark_all_100.json": _build_page_benchmark(reviewed_all, trust_tier="reviewed"),
        "reviewed_page_benchmark_high_confidence_81.json": _build_page_benchmark(
            reviewed_high_confidence,
            trust_tier="trusted",
        ),
        "reviewed_page_benchmark_medium_plus_high_95.json": _build_page_benchmark(
            reviewed_medium_plus_high,
            trust_tier="reviewed",
        ),
    }
    for filename, payload in derived_files.items():
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "reviewed_golden_source": str(reviewed_golden_path),
        "reviewed_page_benchmark_source": str(reviewed_page_benchmark_path),
        "audit_report_source": str(audit_report_path),
        "output_dir": str(output_dir),
        "expected_count": expected_count,
        "row_count": len(reviewed_all),
        "ordered_question_ids": ordered_ids,
        "slice_counts": {
            "reviewed_all_100": len(reviewed_all),
            "reviewed_high_confidence_81": len(reviewed_high_confidence),
            "reviewed_medium_plus_high_95": len(reviewed_medium_plus_high),
        },
        "confidence_counts": {
            "high": sum(1 for row in reviewed_all if row["confidence"] == "high"),
            "medium": sum(1 for row in reviewed_all if row["confidence"] == "medium"),
            "low": sum(1 for row in reviewed_all if row["confidence"] == "low"),
        },
        "benchmark_page_mismatch_count": benchmark_page_mismatch_count,
    }
    (output_dir / "import_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured argument parser.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-golden", type=Path, required=True)
    parser.add_argument("--reviewed-page-benchmark", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / ".sdd" / "golden" / "reviewed",
    )
    parser.add_argument("--expected-count", type=int, default=100)
    return parser


def main() -> int:
    """Run the reviewed-label import CLI.

    Returns:
        Process exit code.
    """
    args = build_arg_parser().parse_args()
    manifest = import_reviewed_labels(
        reviewed_golden_path=args.reviewed_golden,
        reviewed_page_benchmark_path=args.reviewed_page_benchmark,
        audit_report_path=args.audit_report,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
