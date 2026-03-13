from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from rag_challenge.submission.common import SubmissionCase
from rag_challenge.submission.generate import _project_submission_result
from rag_challenge.submission.platform import PlatformCaseResult, _project_platform_answer, _result_anomaly_flags


def test_unsupported_synthetic_pack_projection_and_anomaly_contract() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "unsupported_synthetic_pack.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = cast("list[dict[str, Any]]", payload["cases"])

    for case_payload in cases:
        case = SubmissionCase(
            case_id=str(case_payload["case_id"]),
            question=str(case_payload["question"]),
            answer_type=str(case_payload["answer_type"]),
        )
        answer_text = str(case_payload["answer_text"])
        telemetry = cast("dict[str, object]", case_payload.get("telemetry") or {})

        submission_result = _project_submission_result(
            case_id=case.case_id,
            answer_type=case.answer_type,
            answer_text=answer_text,
            telemetry=telemetry,
        )
        platform_result = PlatformCaseResult(
            case=case,
            answer_text=answer_text,
            telemetry=telemetry,
            total_ms=0,
        )
        projected = _project_platform_answer(platform_result)

        assert submission_result["answer"] == case_payload["expected_submission_answer"]
        assert submission_result["retrieved_chunk_ids"] == case_payload["expected_submission_pages"]
        assert projected["answer"] == case_payload["expected_platform_answer"]
        projected_telemetry = cast("dict[str, object]", projected["telemetry"])
        projected_retrieval = cast("dict[str, object]", projected_telemetry["retrieval"])
        assert projected_retrieval["retrieved_chunk_pages"] == case_payload["expected_platform_pages"]
        assert _result_anomaly_flags(platform_result) == case_payload["expected_anomaly_flags"]
