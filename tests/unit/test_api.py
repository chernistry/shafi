from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from rag_challenge.api.app import create_app
from rag_challenge.core.request_limiter import AsyncRequestLimiter
from rag_challenge.models import TelemetryPayload

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _FakePipeline:
    def __init__(self, events: list[dict[str, object]] | None = None, error: Exception | None = None) -> None:
        self._events = events or []
        self._error = error

    async def astream(self, *_args: object, **_kwargs: object) -> AsyncIterator[dict[str, object]]:
        if self._error is not None:
            raise self._error
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_query_endpoint_streams_token_telemetry_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_challenge.api import routes

    telemetry = TelemetryPayload(
        request_id="req-1",
        question_id="q-1",
        answer_type="boolean",
        ttft_ms=10,
        total_ms=20,
        embed_ms=1,
        qdrant_ms=2,
        rerank_ms=3,
        llm_ms=4,
    ).model_dump()
    fake_pipeline = _FakePipeline(
        events=[
            {"type": "token", "text": "Hello"},
            {"type": "token", "text": " world"},
            {"type": "telemetry", "payload": telemetry},
        ]
    )
    monkeypatch.setattr(routes.app_state, "pipeline", fake_pipeline, raising=False)
    monkeypatch.setattr(routes.app_state, "query_limiter", AsyncRequestLimiter(concurrency_limit=100, min_interval_s=0.0), raising=False)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={
                "question": "What is the rule?",
                "request_id": "req-1",
                "question_id": "q-1",
                "answer_type": "boolean",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"] == "req-1"

    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line.removeprefix("data: ")) for line in lines]
    assert [p["type"] for p in payloads] == ["token", "token", "telemetry", "done"]
    assert payloads[0]["text"] == "Hello"
    assert payloads[1]["text"] == " world"
    assert payloads[2]["payload"]["request_id"] == "req-1"
    assert payloads[2]["payload"]["question_id"] == "q-1"
    assert payloads[2]["payload"]["answer_type"] == "boolean"


@pytest.mark.asyncio
async def test_query_endpoint_streams_error_and_done_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_challenge.api import routes

    monkeypatch.setattr(routes.app_state, "pipeline", _FakePipeline(error=RuntimeError("boom")), raising=False)
    monkeypatch.setattr(routes.app_state, "query_limiter", AsyncRequestLimiter(concurrency_limit=100, min_interval_s=0.0), raising=False)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"question": "Q", "request_id": "req-err", "question_id": "q-err", "answer_type": "number"},
        )

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line.removeprefix("data: ")) for line in lines]
    assert [p["type"] for p in payloads] == ["telemetry", "error", "done"]
    assert payloads[0]["payload"]["request_id"] == "req-err"
    assert payloads[0]["payload"]["question_id"] == "q-err"
    assert payloads[0]["payload"]["answer_type"] == "number"
    assert payloads[1]["message"] == "boom"


@pytest.mark.asyncio
async def test_health_endpoint_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_challenge.api import routes

    fake_store = AsyncMock()
    fake_store.health_check = AsyncMock(return_value=True)
    fake_store.count_points = AsyncMock(return_value=500)
    monkeypatch.setattr(routes.app_state, "store", fake_store, raising=False)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["checks"]["qdrant"]["status"] == "ok"
    assert data["checks"]["collection"]["points"] == 500


@pytest.mark.asyncio
async def test_health_endpoint_unhealthy_when_qdrant_down(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_challenge.api import routes

    fake_store = AsyncMock()
    fake_store.health_check = AsyncMock(return_value=False)
    fake_store.count_points = AsyncMock(side_effect=RuntimeError("collection unavailable"))
    monkeypatch.setattr(routes.app_state, "store", fake_store, raising=False)

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["checks"]["qdrant"]["status"] == "error"
    assert data["checks"]["collection"]["status"] == "error"
