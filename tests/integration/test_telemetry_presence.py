from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from rag_challenge.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _FailingPipeline:
    async def astream(self, *_args: object, **_kwargs: object) -> AsyncIterator[dict[str, object]]:
        raise RuntimeError("forced failure")
        yield {}


@pytest.mark.asyncio
async def test_query_error_path_still_emits_telemetry_and_done(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_challenge.api import routes

    monkeypatch.setattr(routes.app_state, "pipeline", _FailingPipeline(), raising=False)
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={
                "question": "Q?",
                "request_id": "req-telemetry",
                "question_id": "q-telemetry",
                "answer_type": "free_text",
            },
        )

    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [item["type"] for item in payloads] == ["telemetry", "error", "done"]
    assert payloads[0]["payload"]["request_id"] == "req-telemetry"
