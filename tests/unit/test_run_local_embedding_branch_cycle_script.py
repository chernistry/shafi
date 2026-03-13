from __future__ import annotations

from scripts.run_local_embedding_branch_cycle import _phase_collection_name, _probe_ollama_dimension


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, path: str, json: dict[str, object]) -> _FakeResponse:
        del json
        self.calls.append(path)
        return self._responses.pop(0)


def test_probe_ollama_dimension_uses_embed_endpoint(monkeypatch) -> None:
    fake_client = _FakeClient([_FakeResponse(200, {"embeddings": [[0.1, 0.2, 0.3]]})])

    class _Factory:
        def __call__(self, *, base_url: str, timeout: float) -> _FakeClient:
            del base_url, timeout
            return fake_client

    monkeypatch.setattr("scripts.run_local_embedding_branch_cycle.httpx.Client", _Factory())
    dim = _probe_ollama_dimension(base_url="http://localhost:11434", model="embeddinggemma:latest", timeout_s=5.0)
    assert dim == 3
    assert fake_client.calls == ["/api/embed"]


def test_probe_ollama_dimension_falls_back_to_legacy(monkeypatch) -> None:
    fake_client = _FakeClient(
        [
            _FakeResponse(404, {}),
            _FakeResponse(200, {"embedding": [0.1, 0.2]}),
        ]
    )

    class _Factory:
        def __call__(self, *, base_url: str, timeout: float) -> _FakeClient:
            del base_url, timeout
            return fake_client

    monkeypatch.setattr("scripts.run_local_embedding_branch_cycle.httpx.Client", _Factory())
    dim = _probe_ollama_dimension(base_url="http://localhost:11434", model="embeddinggemma:latest", timeout_s=5.0)
    assert dim == 2
    assert fake_client.calls == ["/api/embed", "/api/embeddings"]


def test_phase_collection_name_uses_prefix_and_phase() -> None:
    assert _phase_collection_name(collection_prefix="legal_chunks_embeddinggemma_iter14", phase="warmup") == (
        "legal_chunks_embeddinggemma_iter14_warmup"
    )
