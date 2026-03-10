from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from rag_challenge.api.app import app_state
from rag_challenge.models import QueryRequest, SSEEventType
from rag_challenge.telemetry import TelemetryCollector

logger = logging.getLogger(__name__)

router = APIRouter()

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _sse(event_type: SSEEventType, data: dict[str, Any]) -> str:
    payload = json.dumps({"type": event_type.value, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _stream_pipeline(request: QueryRequest) -> AsyncIterator[str]:
    question_id = request.question_id.strip() or request.request_id
    collector = TelemetryCollector(
        request_id=request.request_id,
        question_id=question_id,
        answer_type=request.answer_type,
    )

    try:
        pipeline = app_state.pipeline
        async for event in pipeline.astream(  # pyright: ignore[reportUnknownMemberType]
            {
                "query": request.question,
                "request_id": request.request_id,
                "question_id": question_id,
                "answer_type": request.answer_type,
                "collector": collector,
            },
            stream_mode="custom",
        ):
            event_type = event.get("type")
            if event_type == "token":
                yield _sse(SSEEventType.TOKEN, {"text": str(event.get("text", ""))})
            elif event_type == "answer_final":
                yield _sse(SSEEventType.ANSWER_FINAL, {"text": str(event.get("text", ""))})
            elif event_type == "telemetry":
                payload = event.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {}
                yield _sse(SSEEventType.TELEMETRY, {"payload": payload})

        yield _sse(SSEEventType.DONE, {})
    except Exception as exc:  # pragma: no cover - traceback formatting branch
        logger.error("Pipeline error: %s\n%s", exc, traceback.format_exc())
        telemetry = collector.finalize()
        # Telemetry is mandatory for scoring; emit it even on failures.
        yield _sse(SSEEventType.TELEMETRY, {"payload": telemetry.model_dump()})
        yield _sse(
            SSEEventType.ERROR,
            {"message": str(exc), "payload": telemetry.model_dump()},
        )
        yield _sse(SSEEventType.DONE, {})


@router.post("/query")
async def query_endpoint(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_pipeline(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request.request_id,
        },
    )


@router.get("/health")
async def health_check() -> JSONResponse:
    checks: dict[str, dict[str, object]] = {}
    overall = True

    try:
        qdrant_ok = await app_state.store.health_check()
        checks["qdrant"] = {"status": "ok" if qdrant_ok else "error"}
        if not qdrant_ok:
            overall = False
    except Exception as exc:
        checks["qdrant"] = {"status": "error", "detail": str(exc)}
        overall = False

    try:
        count = await app_state.store.count_points()
        checks["collection"] = {"status": "ok", "points": int(count)}
    except Exception as exc:
        checks["collection"] = {"status": "error", "detail": str(exc)}

    return JSONResponse(
        status_code=200 if overall else 503,
        content={
            "status": "healthy" if overall else "unhealthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": checks,
        },
    )
