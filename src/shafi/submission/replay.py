"""Helpers for answer-stable submission replay and drift accounting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from shafi.submission.common import count_submission_sentences

JsonDict = dict[str, Any]

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ReplayDriftSummary:
    """Summarize answer/page drift between two submission payloads.

    Args:
        answer_changed_qids: Question IDs whose answer payload changed.
        page_changed_qids: Question IDs whose retrieved page projection changed.
        used_page_count_deltas: Per-question used page-count deltas.

    Returns:
        ReplayDriftSummary: Immutable drift summary.
    """

    answer_changed_qids: list[str]
    page_changed_qids: list[str]
    used_page_count_deltas: dict[str, int]

    @property
    def answer_changed_count(self) -> int:
        """Return the number of answer-changed questions.

        Returns:
            int: Count of question IDs with answer drift.
        """
        return len(self.answer_changed_qids)

    @property
    def page_changed_count(self) -> int:
        """Return the number of page-changed questions.

        Returns:
            int: Count of question IDs with page-projection drift.
        """
        return len(self.page_changed_qids)


def as_dict(value: object) -> JsonDict:
    """Coerce an object into a JSON dict.

    Args:
        value: Arbitrary JSON-like value.

    Returns:
        JsonDict: The original mapping or an empty dict.
    """

    return cast("JsonDict", value) if isinstance(value, dict) else {}


def as_dict_list(value: object) -> list[JsonDict]:
    """Coerce an object into a list of JSON dicts.

    Args:
        value: Arbitrary JSON-like value.

    Returns:
        list[JsonDict]: Dict items only.
    """

    if not isinstance(value, list):
        return []
    return [cast("JsonDict", item) for item in cast("list[object]", value) if isinstance(item, dict)]


def coerce_str_list(value: object) -> list[str]:
    """Convert a JSON array into a cleaned list of strings.

    Args:
        value: Arbitrary JSON-like value.

    Returns:
        list[str]: Non-empty stripped strings.
    """

    if not isinstance(value, list):
        return []
    return [text for item in cast("list[object]", value) if (text := str(item).strip())]


def load_json_dict(path: Path) -> JsonDict:
    """Load a JSON object from disk.

    Args:
        path: JSON file path.

    Returns:
        JsonDict: Parsed object.

    Raises:
        ValueError: If the JSON payload is not an object.
    """

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast("JsonDict", obj)


def load_json_list(path: Path) -> list[JsonDict]:
    """Load a JSON array of objects from disk.

    Args:
        path: JSON file path.

    Returns:
        list[JsonDict]: Parsed rows.

    Raises:
        ValueError: If the JSON payload is not an array.
    """

    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array in {path}")
    return as_dict_list(cast("object", obj))


def sha256_json_file(path: Path) -> str:
    """Return the SHA256 hash for a JSON artifact on disk.

    Args:
        path: Artifact path.

    Returns:
        str: Lowercase hex digest, or an empty string when the file is absent.
    """

    if not path.exists():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def answers_by_id(payload: JsonDict) -> dict[str, JsonDict]:
    """Index submission answers by question ID.

    Args:
        payload: Submission payload with an ``answers`` list.

    Returns:
        dict[str, JsonDict]: Question-ID keyed answer records.
    """

    out: dict[str, JsonDict] = {}
    for raw in as_dict_list(payload.get("answers")):
        qid = str(raw.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def raw_results_by_id(records: list[JsonDict]) -> dict[str, JsonDict]:
    """Index raw-result records by question ID.

    Args:
        records: Raw results rows.

    Returns:
        dict[str, JsonDict]: Question-ID keyed raw-result rows.
    """

    out: dict[str, JsonDict] = {}
    for raw in records:
        case = as_dict(raw.get("case"))
        qid = str(case.get("case_id") or case.get("question_id") or "").strip()
        if qid:
            out[qid] = raw
    return out


def deepcopy_json(value: object) -> object:
    """Deep-copy a JSON-compatible structure.

    Args:
        value: JSON-compatible value.

    Returns:
        object: Deep-copied JSON structure.
    """

    return json.loads(json.dumps(value, ensure_ascii=False))


def page_count(answer_record: JsonDict) -> int:
    """Count retrieved pages from a submission answer record.

    Args:
        answer_record: Submission answer row.

    Returns:
        int: Total retrieved page count.
    """

    retrieval = as_dict(as_dict(as_dict(answer_record.get("telemetry")).get("retrieval")))
    pages = as_dict_list(retrieval.get("retrieved_chunk_pages"))
    count = 0
    for raw in pages:
        page_numbers_obj = raw.get("page_numbers")
        if isinstance(page_numbers_obj, list):
            count += len([item for item in cast("list[object]", page_numbers_obj) if isinstance(item, int | float)])
    return count


def qid_allowlist(*, values: list[str], file_path: Path | None) -> set[str]:
    """Build a QID allowlist from flags and an optional file.

    Args:
        values: Explicit QID values.
        file_path: Optional file with one QID per line.

    Returns:
        set[str]: Cleaned allowlist.
    """

    out = {text for raw in values if (text := str(raw).strip())}
    if file_path is not None:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                out.add(text)
    return out


def validate_replay_contract(
    *,
    answer_source_submission: JsonDict,
    answer_source_raw_results: list[JsonDict],
    page_source_submission: JsonDict,
    page_source_raw_results: list[JsonDict],
) -> None:
    """Validate that answer/page artifacts can participate in a clean replay.

    Args:
        answer_source_submission: Baseline submission payload.
        answer_source_raw_results: Baseline raw results.
        page_source_submission: Candidate submission payload.
        page_source_raw_results: Candidate raw results.

    Raises:
        ValueError: If question sets or required raw-results fields do not align.
    """

    answer_submission_by_id = answers_by_id(answer_source_submission)
    page_submission_by_id = answers_by_id(page_source_submission)
    answer_raw_by_id = raw_results_by_id(answer_source_raw_results)
    page_raw_by_id = raw_results_by_id(page_source_raw_results)

    submission_qids = set(answer_submission_by_id)
    if submission_qids != set(page_submission_by_id):
        raise ValueError("Answer-source and page-source submission question_id sets do not match")
    if submission_qids != set(answer_raw_by_id):
        raise ValueError("Answer-source raw_results question_id set does not match its submission")
    if submission_qids != set(page_raw_by_id):
        raise ValueError("Page-source raw_results question_id set does not match its submission")

    for label, raw_by_id in (("answer_source", answer_raw_by_id), ("page_source", page_raw_by_id)):
        for qid, raw in raw_by_id.items():
            if "answer_text" not in raw:
                raise ValueError(f"{label} raw_results missing answer_text for question_id={qid}")
            telemetry = as_dict(raw.get("telemetry"))
            if not telemetry:
                raise ValueError(f"{label} raw_results missing telemetry for question_id={qid}")
            if "used_page_ids" not in telemetry or "context_page_ids" not in telemetry:
                raise ValueError(f"{label} raw_results missing used/context page IDs for question_id={qid}")


def build_counterfactual_preflight(
    *,
    merged_payload: JsonDict,
    answer_source_preflight: JsonDict,
    page_source_preflight: JsonDict,
    answer_source_submission: Path,
    page_source_submission: Path,
    allowlisted_qids: set[str],
    page_allowlisted_qids: set[str],
) -> JsonDict:
    """Build a preflight summary for a counterfactual replay artifact.

    Args:
        merged_payload: Replay submission payload.
        answer_source_preflight: Baseline preflight payload.
        page_source_preflight: Candidate preflight payload.
        answer_source_submission: Baseline submission path.
        page_source_submission: Candidate submission path.
        allowlisted_qids: QIDs allowed to take candidate answers.
        page_allowlisted_qids: QIDs allowed to take candidate page projections.

    Returns:
        JsonDict: Counterfactual preflight payload.
    """

    answers = answers_by_id(merged_payload)
    answer_type_counts: Counter[str] = Counter()
    null_answer_counts: Counter[str] = Counter()
    empty_pages_counts: Counter[str] = Counter()
    page_counts: list[int] = []
    free_text_char_counts: list[int] = []
    free_text_sentence_counts: list[int] = []
    model_name_empty_count = 0

    answer_type_by_qid: dict[str, str] = {}
    for qid, raw in answers.items():
        telemetry = as_dict(raw.get("telemetry"))
        answer_type = str(telemetry.get("answer_type") or "").strip().lower() or "free_text"
        answer_type_by_qid[qid] = answer_type

    for qid, raw in answers.items():
        answer_type = answer_type_by_qid[qid]
        answer_type_counts[answer_type] += 1
        answer_value = raw.get("answer")
        if answer_value is None:
            null_answer_counts[answer_type] += 1
        pages = page_count(raw)
        page_counts.append(pages)
        if pages == 0:
            empty_pages_counts[answer_type] += 1
        telemetry = as_dict(raw.get("telemetry"))
        if not str(telemetry.get("model_name") or "").strip():
            model_name_empty_count += 1
        if answer_type == "free_text" and isinstance(answer_value, str):
            free_text_char_counts.append(len(answer_value))
            free_text_sentence_counts.append(count_submission_sentences(answer_value))

    def _percentile_int(values: list[int], q: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
        return int(ordered[idx])

    return {
        "phase": page_source_preflight.get("phase") or answer_source_preflight.get("phase"),
        "questions_count": len(answers),
        "answer_type_counts": dict(answer_type_counts),
        "null_answer_counts_by_type": dict(null_answer_counts),
        "empty_retrieved_chunk_pages_counts_by_type": dict(empty_pages_counts),
        "page_count_distribution": {
            "min": min(page_counts, default=0),
            "p50": _percentile_int(page_counts, 0.50),
            "p95": _percentile_int(page_counts, 0.95),
            "max": max(page_counts, default=0),
        },
        "free_text_char_distribution": {
            "min": min(free_text_char_counts, default=0),
            "p50": _percentile_int(free_text_char_counts, 0.50),
            "p95": _percentile_int(free_text_char_counts, 0.95),
            "max": max(free_text_char_counts, default=0),
        },
        "free_text_sentence_distribution": {
            "min": min(free_text_sentence_counts, default=0),
            "p50": _percentile_int(free_text_sentence_counts, 0.50),
            "p95": _percentile_int(free_text_sentence_counts, 0.95),
            "max": max(free_text_sentence_counts, default=0),
        },
        "model_name_empty_count": model_name_empty_count,
        "submission_sha256": "",
        "code_archive_sha256": page_source_preflight.get("code_archive_sha256")
        or answer_source_preflight.get("code_archive_sha256")
        or "",
        "score_settings_sha256": page_source_preflight.get("score_settings_sha256")
        or answer_source_preflight.get("score_settings_sha256")
        or "",
        "score_settings_fingerprint": deepcopy_json(
            page_source_preflight.get("score_settings_fingerprint")
            or answer_source_preflight.get("score_settings_fingerprint")
            or {}
        ),
        "questions_sha256": page_source_preflight.get("questions_sha256")
        or answer_source_preflight.get("questions_sha256")
        or "",
        "documents_zip_sha256": page_source_preflight.get("documents_zip_sha256")
        or answer_source_preflight.get("documents_zip_sha256")
        or "",
        "pdf_count": page_source_preflight.get("pdf_count") or answer_source_preflight.get("pdf_count") or 0,
        "phase_collection_name": page_source_preflight.get("phase_collection_name")
        or answer_source_preflight.get("phase_collection_name"),
        "qdrant_point_count": page_source_preflight.get("qdrant_point_count")
        or answer_source_preflight.get("qdrant_point_count"),
        "truth_audit_workbook_path": page_source_preflight.get("truth_audit_workbook_path")
        or answer_source_preflight.get("truth_audit_workbook_path"),
        "raw_results_path": "",
        "counterfactual_projection": {
            "answer_source_submission": str(answer_source_submission),
            "page_source_submission": str(page_source_submission),
            "answer_source_code_archive_sha256": answer_source_preflight.get("code_archive_sha256") or "",
            "page_source_code_archive_sha256": page_source_preflight.get("code_archive_sha256") or "",
            "answer_source_score_settings_sha256": answer_source_preflight.get("score_settings_sha256") or "",
            "page_source_score_settings_sha256": page_source_preflight.get("score_settings_sha256") or "",
            "page_source_answer_qids": sorted(allowlisted_qids),
            "page_source_page_qids": sorted(page_allowlisted_qids),
            "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        },
    }


def merge_answer_stable_records(
    *,
    answer_source_submission: JsonDict,
    answer_source_raw_results: list[JsonDict],
    page_source_submission: JsonDict,
    page_source_raw_results: list[JsonDict],
    allowlisted_qids: set[str],
    page_allowlisted_qids: set[str],
    page_source_pages_default: str,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:
    """Build a merged replay artifact with frozen answers and swapped page projections.

    Args:
        answer_source_submission: Baseline submission payload.
        answer_source_raw_results: Baseline raw results.
        page_source_submission: Candidate submission payload.
        page_source_raw_results: Candidate raw results.
        allowlisted_qids: QIDs that may take candidate answers.
        page_allowlisted_qids: QIDs that may take candidate page projections.
        page_source_pages_default: ``all`` or ``none`` page-source default.

    Returns:
        tuple[JsonDict, list[JsonDict], JsonDict]: Submission payload, raw results, and merge report.
    """

    validate_replay_contract(
        answer_source_submission=answer_source_submission,
        answer_source_raw_results=answer_source_raw_results,
        page_source_submission=page_source_submission,
        page_source_raw_results=page_source_raw_results,
    )

    answer_submission_by_id = answers_by_id(answer_source_submission)
    page_submission_by_id = answers_by_id(page_source_submission)
    answer_raw_by_id = raw_results_by_id(answer_source_raw_results)
    page_raw_by_id = raw_results_by_id(page_source_raw_results)

    merged_answers: list[JsonDict] = []
    merged_raw_results: list[JsonDict] = []
    changed_answer_qids: list[str] = []
    changed_page_qids: list[str] = []

    for qid, answer_record in answer_submission_by_id.items():
        page_record = page_submission_by_id[qid]
        answer_raw = answer_raw_by_id[qid]
        page_raw = page_raw_by_id[qid]

        use_page_source_answer = qid in allowlisted_qids
        use_page_source_pages = page_source_pages_default == "all" and not page_allowlisted_qids
        if page_allowlisted_qids:
            use_page_source_pages = qid in page_allowlisted_qids

        chosen_answer_record = page_record if use_page_source_answer else answer_record
        chosen_answer_raw = page_raw if use_page_source_answer else answer_raw

        merged_answer_record = cast("JsonDict", deepcopy_json(chosen_answer_record))
        merged_answer_telemetry = as_dict(merged_answer_record.get("telemetry"))
        page_telemetry = as_dict(page_record.get("telemetry"))
        answer_telemetry = as_dict(answer_record.get("telemetry"))
        chosen_answer_case = as_dict(chosen_answer_raw.get("case"))
        chosen_answer_raw_telemetry = as_dict(chosen_answer_raw.get("telemetry"))
        answer_type = (
            str(chosen_answer_case.get("answer_type") or chosen_answer_raw_telemetry.get("answer_type") or "")
            .strip()
            .lower()
        )
        if answer_type:
            merged_answer_telemetry["answer_type"] = answer_type
        answer_retrieval = as_dict(answer_telemetry.get("retrieval"))
        page_retrieval = as_dict(page_telemetry.get("retrieval"))
        retrieval_source = page_retrieval if use_page_source_pages else answer_retrieval
        merged_answer_telemetry["retrieval"] = deepcopy_json(retrieval_source)
        merged_answer_record["telemetry"] = merged_answer_telemetry
        merged_answers.append(merged_answer_record)

        merged_raw_record = cast("JsonDict", deepcopy_json(chosen_answer_raw))
        merged_raw_telemetry = as_dict(merged_raw_record.get("telemetry"))
        page_raw_telemetry = as_dict(page_raw.get("telemetry"))
        answer_raw_telemetry = as_dict(answer_raw.get("telemetry"))
        for field in (
            "retrieved_chunk_ids",
            "retrieved_page_ids",
            "context_chunk_ids",
            "context_page_ids",
            "used_chunk_ids",
            "used_page_ids",
            "must_include_chunk_ids",
            "doc_shortlist",
            "support_shape_flags",
            "support_shape_class",
            "localized_support_chunk_ids",
            "localized_support_page_ids",
        ):
            source_telemetry = page_raw_telemetry if use_page_source_pages else answer_raw_telemetry
            if field in source_telemetry:
                merged_raw_telemetry[field] = deepcopy_json(source_telemetry[field])
        merged_raw_record["telemetry"] = merged_raw_telemetry
        merged_raw_results.append(merged_raw_record)

        if json.dumps(answer_record.get("answer"), ensure_ascii=False, sort_keys=True) != json.dumps(
            merged_answer_record.get("answer"), ensure_ascii=False, sort_keys=True
        ):
            changed_answer_qids.append(qid)
        baseline_pages = as_dict(as_dict(answer_record.get("telemetry")).get("retrieval")).get(
            "retrieved_chunk_pages", []
        )
        merged_pages = as_dict(as_dict(merged_answer_record.get("telemetry")).get("retrieval")).get(
            "retrieved_chunk_pages",
            [],
        )
        if json.dumps(baseline_pages, ensure_ascii=False, sort_keys=True) != json.dumps(
            merged_pages,
            ensure_ascii=False,
            sort_keys=True,
        ):
            changed_page_qids.append(qid)

    report: JsonDict = {
        "answer_source_count": len(answer_submission_by_id),
        "page_source_count": len(page_submission_by_id),
        "merged_count": len(merged_answers),
        "answer_changed_count_vs_answer_source": len(changed_answer_qids),
        "page_projection_changed_count_vs_answer_source": len(changed_page_qids),
        "answer_changed_qids": changed_answer_qids,
        "page_projection_changed_qids": changed_page_qids,
        "page_source_answer_qids": sorted(allowlisted_qids),
        "page_source_page_qids": sorted(page_allowlisted_qids),
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
    }
    merged_submission = {
        "architecture_summary": deepcopy_json(
            answer_source_submission.get("architecture_summary")
            or page_source_submission.get("architecture_summary")
            or {}
        ),
        "answers": merged_answers,
    }
    return merged_submission, merged_raw_results, report


def submission_answer_value(record: JsonDict) -> str:
    """Render a submission answer into a stable comparison string.

    Args:
        record: Submission answer row.

    Returns:
        str: Stable JSON string for answer comparison.
    """

    return json.dumps(record.get("answer"), ensure_ascii=False, sort_keys=True)


def submission_page_projection(record: JsonDict) -> str:
    """Render the retrieved page projection into a stable comparison string.

    Args:
        record: Submission answer row.

    Returns:
        str: Stable JSON string for retrieved page projection comparison.
    """

    retrieval = as_dict(as_dict(record.get("telemetry")).get("retrieval"))
    return json.dumps(retrieval.get("retrieved_chunk_pages", []), ensure_ascii=False, sort_keys=True)


def used_page_ids_from_raw_result(raw_record: JsonDict) -> list[str]:
    """Extract used page IDs from a raw-result row.

    Args:
        raw_record: Raw results row.

    Returns:
        list[str]: Used page IDs.
    """

    return coerce_str_list(as_dict(raw_record.get("telemetry")).get("used_page_ids"))


def compare_submission_drift(
    *,
    baseline_submission: JsonDict,
    candidate_submission: JsonDict,
    baseline_raw_results: list[JsonDict],
    candidate_raw_results: list[JsonDict],
) -> ReplayDriftSummary:
    """Compare answer/page drift between baseline and replay candidate artifacts.

    Args:
        baseline_submission: Baseline submission payload.
        candidate_submission: Replay candidate submission payload.
        baseline_raw_results: Baseline raw results.
        candidate_raw_results: Replay candidate raw results.

    Returns:
        ReplayDriftSummary: Drift summary for answers and page projections.

    Raises:
        ValueError: If the question sets do not match.
    """

    baseline_by_id = answers_by_id(baseline_submission)
    candidate_by_id = answers_by_id(candidate_submission)
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("Baseline and candidate submission question_id sets do not match")

    baseline_raw_by_id = raw_results_by_id(baseline_raw_results)
    candidate_raw_by_id = raw_results_by_id(candidate_raw_results)
    if set(baseline_by_id) != set(baseline_raw_by_id) or set(candidate_by_id) != set(candidate_raw_by_id):
        raise ValueError("Submission and raw_results question_id sets do not match during drift comparison")

    answer_changed_qids: list[str] = []
    page_changed_qids: list[str] = []
    used_page_count_deltas: dict[str, int] = {}
    for qid in sorted(baseline_by_id):
        if submission_answer_value(baseline_by_id[qid]) != submission_answer_value(candidate_by_id[qid]):
            answer_changed_qids.append(qid)
        if submission_page_projection(baseline_by_id[qid]) != submission_page_projection(candidate_by_id[qid]):
            page_changed_qids.append(qid)
        used_page_count_deltas[qid] = len(used_page_ids_from_raw_result(candidate_raw_by_id[qid])) - len(
            used_page_ids_from_raw_result(baseline_raw_by_id[qid])
        )
    return ReplayDriftSummary(
        answer_changed_qids=answer_changed_qids,
        page_changed_qids=page_changed_qids,
        used_page_count_deltas=used_page_count_deltas,
    )
