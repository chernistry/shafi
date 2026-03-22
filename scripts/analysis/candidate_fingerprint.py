from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

JsonDict = dict[str, Any]


def _load_json(path: Path) -> JsonDict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def _load_raw_results(path: Path) -> list[JsonDict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected raw-results list in {path}")
    return [cast("JsonDict", item) for item in cast("list[object]", obj) if isinstance(item, dict)]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sorted_unique_strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(item).strip() for item in cast("list[object]", values) if str(item).strip()})


def _build_answers_rows(submission_payload: JsonDict) -> list[JsonDict]:
    answers_obj = submission_payload.get("answers")
    answers = cast("list[object]", answers_obj) if isinstance(answers_obj, list) else []
    rows: list[JsonDict] = []
    for item in answers:
        if not isinstance(item, dict):
            continue
        row = cast("JsonDict", item)
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            continue
        rows.append(
            {
                "question_id": question_id,
                "answer": row.get("answer"),
            }
        )
    rows.sort(key=lambda item: str(item["question_id"]))
    return rows


def _build_page_rows(raw_results: list[JsonDict], *, key: str, fallback_key: str | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in raw_results:
        telemetry_obj = item.get("telemetry")
        telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        question_id = str(telemetry.get("question_id") or "").strip()
        if not question_id:
            continue
        page_ids = _sorted_unique_strings(telemetry.get(key))
        if not page_ids and fallback_key is not None:
            page_ids = _sorted_unique_strings(telemetry.get(fallback_key))
        rows.append({"question_id": question_id, "page_ids": page_ids})
    rows.sort(key=lambda item: str(item["question_id"]))
    return rows


def _build_route_rows(raw_results: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for item in raw_results:
        telemetry_obj = item.get("telemetry")
        telemetry = cast("JsonDict", telemetry_obj) if isinstance(telemetry_obj, dict) else {}
        question_id = str(telemetry.get("question_id") or "").strip()
        if not question_id:
            continue
        rows.append(
            {
                "question_id": question_id,
                "answer_type": str(telemetry.get("answer_type") or "").strip(),
                "model_embed": str(telemetry.get("model_embed") or "").strip(),
                "model_rerank": str(telemetry.get("model_rerank") or "").strip(),
                "model_llm": str(telemetry.get("model_llm") or telemetry.get("model_name") or "").strip(),
                "generation_mode": str(telemetry.get("generation_mode") or "").strip(),
                "llm_provider": str(telemetry.get("llm_provider") or "").strip(),
            }
        )
    rows.sort(key=lambda item: str(item["question_id"]))
    return rows


def _load_known_fingerprints(paths: list[Path]) -> list[JsonDict]:
    known: list[JsonDict] = []
    for path in paths:
        payload = _load_json(path)
        nested = payload.get("candidate_fingerprint")
        fingerprint_payload = cast("JsonDict", nested) if isinstance(nested, dict) else payload
        known.append(fingerprint_payload)
    return known


def _find_duplicate(*, current: JsonDict, known_payloads: list[JsonDict]) -> tuple[str | None, str | None]:
    current_fingerprint = str(current.get("fingerprint") or "").strip()
    if not current_fingerprint:
        return None, None
    for payload in known_payloads:
        fingerprint = str(payload.get("fingerprint") or "").strip()
        label = str(payload.get("label") or "").strip()
        if fingerprint and fingerprint == current_fingerprint and label and label != str(current.get("label") or "").strip():
            return label, fingerprint
    return None, None


def _render_markdown(payload: JsonDict) -> str:
    duplicate_label = str(payload.get("duplicate_of_label") or "").strip()
    lines = [
        "# Candidate Fingerprint",
        "",
        f"- label: `{payload.get('label')}`",
        f"- fingerprint: `{payload.get('fingerprint')}`",
        f"- answers_hash: `{payload.get('answers_hash')}`",
        f"- used_pages_hash: `{payload.get('used_pages_hash')}`",
        f"- context_pages_hash: `{payload.get('context_pages_hash')}`",
        f"- route_map_hash: `{payload.get('route_map_hash')}`",
        f"- should_skip: `{payload.get('should_skip')}`",
        f"- duplicate_of_label: `{duplicate_label or 'none'}`",
    ]
    return "\n".join(lines) + "\n"


def build_candidate_fingerprint(
    *,
    label: str,
    submission_json: Path,
    raw_results_json: Path,
    known_fingerprint_jsons: list[Path],
) -> JsonDict:
    submission_payload = _load_json(submission_json)
    raw_results = _load_raw_results(raw_results_json)
    answers_rows = _build_answers_rows(submission_payload)
    used_page_rows = _build_page_rows(raw_results, key="used_page_ids")
    context_page_rows = _build_page_rows(raw_results, key="context_page_ids", fallback_key="retrieved_page_ids")
    route_rows = _build_route_rows(raw_results)

    payload: JsonDict = {
        "schema_version": 1,
        "label": label,
        "submission_path": str(submission_json),
        "raw_results_path": str(raw_results_json),
        "question_count": len(answers_rows),
        "answers_hash": _sha256_text(_canonical_json(answers_rows)),
        "used_pages_hash": _sha256_text(_canonical_json(used_page_rows)),
        "context_pages_hash": _sha256_text(_canonical_json(context_page_rows)),
        "route_map_hash": _sha256_text(_canonical_json(route_rows)),
    }
    payload["fingerprint"] = _sha256_text(
        _canonical_json(
            {
                "answers_hash": payload["answers_hash"],
                "used_pages_hash": payload["used_pages_hash"],
                "context_pages_hash": payload["context_pages_hash"],
                "route_map_hash": payload["route_map_hash"],
            }
        )
    )
    duplicate_of_label, matched_fingerprint = _find_duplicate(current=payload, known_payloads=_load_known_fingerprints(known_fingerprint_jsons))
    payload["duplicate_of_label"] = duplicate_of_label
    payload["matched_fingerprint"] = matched_fingerprint
    payload["should_skip"] = duplicate_of_label is not None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a candidate fingerprint from submission answers and raw-results telemetry.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--submission-json", type=Path, required=True)
    parser.add_argument("--raw-results-json", type=Path, required=True)
    parser.add_argument("--known-candidate-fingerprint-json", action="append", default=[])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    payload = {
        "candidate_fingerprint": build_candidate_fingerprint(
            label=str(args.label),
            submission_json=args.submission_json.resolve(),
            raw_results_json=args.raw_results_json.resolve(),
            known_fingerprint_jsons=[Path(str(item)).resolve() for item in cast("list[object]", args.known_candidate_fingerprint_json)],
        )
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(payload["candidate_fingerprint"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
