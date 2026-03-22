from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast

import httpx

from rag_challenge.submission.common import (
    SubmissionAnswer,
    SubmissionCase,
    as_int,
    chunk_id_to_page_id,
    chunk_ids_to_page_ids,
    classify_unanswerable_answer,
    coerce_answer_type,
    coerce_str_list,
    load_cases,
    normalize_free_text_answer,
    select_submission_used_pages,
)

_DEFAULT_UNANSWERABLE_FREE_TEXT = "There is no information on this question in the provided documents."


def _parse_sse_body(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped.split("data:", maxsplit=1)[1].strip()
        if not payload:
            continue
        try:
            obj: object = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(cast("dict[str, object]", obj))
    return events


_as_int = as_int
_chunk_id_to_page_id = chunk_id_to_page_id
_chunk_ids_to_page_ids = chunk_ids_to_page_ids
_coerce_answer_type = coerce_answer_type
_coerce_str_list = coerce_str_list
_select_submission_used_pages = select_submission_used_pages
_classify_unanswerable_answer = classify_unanswerable_answer
_load_cases = load_cases
_normalize_free_text_answer = normalize_free_text_answer


def _project_submission_result(
    *,
    case_id: str,
    answer_type: str,
    answer_text: str,
    telemetry: dict[str, object],
    total_ms_fallback: int = 0,
) -> dict[str, Any]:
    used_pages = _select_submission_used_pages(telemetry)
    is_unanswerable_strict, is_unanswerable_free_text = _classify_unanswerable_answer(answer_text, answer_type)

    answer_out: SubmissionAnswer = None if is_unanswerable_strict else _coerce_answer_type(answer_text, answer_type)
    if is_unanswerable_free_text:
        answer_out = _DEFAULT_UNANSWERABLE_FREE_TEXT
    if is_unanswerable_strict or is_unanswerable_free_text:
        used_pages = []

    return {
        "id": case_id,
        "answer_type": answer_type,
        "answer": answer_out,
        "ttft_ms": _as_int(telemetry.get("ttft_ms"), 0),
        "time_per_output_token_ms": _as_int(telemetry.get("time_per_output_token_ms"), 0),
        "total_ms": _as_int(telemetry.get("total_ms"), total_ms_fallback),
        "retrieved_chunk_ids": used_pages,
        "prompt_tokens": _as_int(telemetry.get("prompt_tokens"), 0),
        "completion_tokens": _as_int(telemetry.get("completion_tokens"), 0),
        "model_name": str(telemetry.get("model_llm") or ""),
    }


async def _run_case(case: SubmissionCase, client: httpx.AsyncClient, endpoint: str) -> dict[str, Any]:
    payload = {
        "question": case.question,
        "request_id": case.case_id,
        "question_id": case.case_id,
        "answer_type": case.answer_type,
    }
    t0 = time.perf_counter()
    resp = await client.post(endpoint, json=payload, timeout=60.0)
    total_ms = int((time.perf_counter() - t0) * 1000.0)
    resp.raise_for_status()

    events = _parse_sse_body(resp.text)
    answer_final_evt = next((evt for evt in events if evt.get("type") == "answer_final"), None)
    if answer_final_evt is not None:
        answer_text = str(answer_final_evt.get("text", "")).strip()
    else:
        answer_text = "".join(str(evt.get("text", "")) for evt in events if evt.get("type") == "token").strip()
    telemetry_evt = next((evt for evt in events if evt.get("type") == "telemetry"), None)
    telemetry: dict[str, object] = {}
    if isinstance(telemetry_evt, dict):
        payload_obj = telemetry_evt.get("payload")
        if isinstance(payload_obj, dict):
            telemetry = cast("dict[str, object]", payload_obj)

    return _project_submission_result(
        case_id=case.case_id,
        answer_type=case.answer_type,
        answer_text=answer_text,
        telemetry=telemetry,
        total_ms_fallback=total_ms,
    )


async def generate_submission_json(
    *,
    questions_path: Path,
    endpoint: str,
    out_path: Path,
    concurrency: int,
) -> None:
    cases = _load_cases(questions_path)
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    results: list[dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]

    async with httpx.AsyncClient() as client:
        async def _worker(idx: int, case: SubmissionCase) -> None:
            async with sem:
                results[idx] = await _run_case(case, client, endpoint)

        await asyncio.gather(*[_worker(i, case) for i, case in enumerate(cases)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate competition submission JSON by querying the live API.")
    parser.add_argument("--questions", required=True, help="Path to questions JSON (e.g. dataset/public_dataset.json)")
    parser.add_argument("--endpoint", default="http://localhost:8000/query", help="POST /query endpoint URL")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(argv)

    asyncio.run(
        generate_submission_json(
            questions_path=Path(args.questions),
            endpoint=str(args.endpoint),
            out_path=Path(args.out),
            concurrency=int(args.concurrency),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
