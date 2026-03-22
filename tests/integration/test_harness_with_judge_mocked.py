from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rag_challenge.eval import harness as harness_mod
from rag_challenge.eval.judge import JudgeOutcome, JudgeResult, JudgeScores


def _sse(*events: dict[str, object]) -> str:
    return "".join(f"data: {json.dumps(evt)}\n" for evt in events)


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHTTPClient:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del args, kwargs

    async def __aenter__(self) -> _FakeHTTPClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        del exc_type, exc, tb
        return False

    async def post(self, endpoint_url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        del endpoint_url, timeout
        answer_type = str(json.get("answer_type") or "free_text").strip().lower()
        answer = "Yes" if answer_type == "boolean" else "Hello"
        telemetry = {
            "ttft_ms": 123,
            "classify_ms": 0.1,
            "embed_ms": 0.0,
            "qdrant_ms": 0.0,
            "rerank_ms": 0.0,
            "llm_ms": 10.0,
            "verify_ms": 0.0,
            "retrieved_chunk_ids": ["doc:0:0:x"],
            "context_chunk_ids": ["doc:0:0:x"],
            "cited_chunk_ids": ["doc:0:0:x"],
            "cited_page_ids": ["doc_1"],
            "context_page_ids": ["doc_1"],
        }
        body = _sse(
            {"type": "token", "text": answer},
            {"type": "telemetry", "payload": telemetry},
            {"type": "done"},
        )
        return _FakeResponse(text=body)


class _FakeNoSupportHTTPClient:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del args, kwargs

    async def __aenter__(self) -> _FakeNoSupportHTTPClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        del exc_type, exc, tb
        return False

    async def post(self, endpoint_url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        del endpoint_url, timeout
        answer = str(json.get("question") or "").strip()
        telemetry = {
            "ttft_ms": 123,
            "classify_ms": 0.1,
            "embed_ms": 0.0,
            "qdrant_ms": 0.0,
            "rerank_ms": 0.0,
            "llm_ms": 10.0,
            "verify_ms": 0.0,
            "retrieved_chunk_ids": [],
            "context_chunk_ids": [],
            "cited_chunk_ids": [],
            "cited_page_ids": [],
            "context_page_ids": [],
            "used_page_ids": [],
        }
        body = _sse(
            {"type": "token", "text": answer},
            {"type": "telemetry", "payload": telemetry},
            {"type": "done"},
        )
        return _FakeResponse(text=body)


class _FakeQdrantClient:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del args, kwargs

    async def get_collections(self) -> object:
        raise RuntimeError("no qdrant in unit test")


class _FakeJudgeClient:
    def __init__(self) -> None:
        pass

    async def close(self) -> None:
        return None

    async def evaluate(self, *args, **kwargs) -> JudgeOutcome:  # type: ignore[no-untyped-def]
        del args, kwargs
        result = JudgeResult(
            verdict="PASS",
            scores=JudgeScores(accuracy=5, grounding=5, clarity=4, uncertainty_handling=5),
            format_issues=[],
            unsupported_claims=[],
            grounding_evidence=[],
            recommended_fix="",
        )
        return JudgeOutcome(result=result, model="fake-judge", failure="")


def _fake_settings(tmp_path) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        qdrant=SimpleNamespace(
            url="http://unused-qdrant",
            api_key="",
            timeout_s=1,
            check_compatibility=False,
            collection="unused-collection",
        ),
        judge=SimpleNamespace(
            enabled=False,
            concurrency=1,
            docs_dir=str(tmp_path),
            sources_max_chars_per_page=20000,
            sources_max_pages=6,
            sources_max_chars_total=60000,
        ),
    )


@pytest.mark.asyncio
async def test_harness_with_judge_mocked(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Q1?",
                    "answer_type": "free_text",
                    "gold_chunk_ids": ["doc:0:0:x"],
                },
                {
                    "id": "q2",
                    "question": "Q2?",
                    "answer_type": "boolean",
                    "gold_chunk_ids": ["doc:0:0:x"],
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(harness_mod, "AsyncQdrantClient", _FakeQdrantClient)
    monkeypatch.setattr(harness_mod.httpx, "AsyncClient", _FakeHTTPClient)
    monkeypatch.setattr(harness_mod, "JudgeClient", _FakeJudgeClient)
    monkeypatch.setattr(harness_mod, "get_settings", lambda: _fake_settings(tmp_path))

    async def _fake_sources(*args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        del args, kwargs
        return "SOURCE TEXT"

    monkeypatch.setattr(harness_mod, "build_sources_text", _fake_sources)
    monkeypatch.setattr(harness_mod, "select_used_pages", lambda payload, max_pages: ["doc_1"])

    result = await harness_mod.run_evaluation(
        golden_path=golden,
        endpoint_url="http://unused/query",
        concurrency=2,
        emit_cases=True,
        judge_enabled=True,
        judge_scope="free_text",
        judge_docs_dir=tmp_path,
    )
    summary = result.summary()

    assert "judge" in summary
    assert summary["judge"]["cases"] == 1
    assert summary["judge"]["pass_rate"] == 1.0
    assert summary["support_page_guard"]["zero_support_cases"] == 0
    assert summary["support_page_guard"]["unjustified_zero_support_cases"] == 0
    assert summary["grounding_g_score_beta_2_5"] == 1.0
    assert summary["grounding_g_score_beta_2_5_by_answer_type"]["free_text"] == 1.0
    assert summary["grounding_g_score_beta_2_5_by_answer_type"]["boolean"] == 1.0

    judged = [row for row in result.cases if isinstance(row.get("judge"), dict)]
    assert len(judged) == 1
    assert judged[0]["case_id"] == "q1"
    assert judged[0]["used_pages"] == ["doc_1"]
    assert judged[0]["zero_support_allowed"] is True
    assert judged[0]["zero_support_reason"] == "support_present"


@pytest.mark.asyncio
async def test_harness_flags_unjustified_and_justified_zero_support(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "There is no information on this question.",
                    "answer_type": "free_text",
                    "gold_chunk_ids": ["doc:0:0:x"],
                },
                {
                    "id": "q2",
                    "question": "Yes",
                    "answer_type": "boolean",
                    "gold_chunk_ids": ["doc:0:0:x"],
                },
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(harness_mod, "AsyncQdrantClient", _FakeQdrantClient)
    monkeypatch.setattr(harness_mod.httpx, "AsyncClient", _FakeNoSupportHTTPClient)
    monkeypatch.setattr(harness_mod, "get_settings", lambda: _fake_settings(tmp_path))

    result = await harness_mod.run_evaluation(
        golden_path=golden,
        endpoint_url="http://unused/query",
        concurrency=1,
        emit_cases=True,
        judge_enabled=False,
    )
    summary = result.summary()

    assert summary["support_page_guard"] == {
        "checked_cases": 2,
        "zero_support_cases": 2,
        "justified_zero_support_cases": 1,
        "unjustified_zero_support_cases": 1,
        "zero_support_reason_counts": {
            "missing_support_pages": 1,
            "no_information": 1,
        },
        "top_unjustified_zero_support_cases": [
            {
                "case_id": "q2",
                "answer_type": "boolean",
                "reason": "missing_support_pages",
                "question": "Yes",
                "answer": "Yes",
            }
        ],
    }
    by_id = {str(row["case_id"]): row for row in result.cases}
    assert by_id["q1"]["zero_support_allowed"] is True
    assert by_id["q1"]["zero_support_reason"] == "no_information"
    assert by_id["q2"]["zero_support_allowed"] is False
    assert by_id["q2"]["zero_support_reason"] == "missing_support_pages"
