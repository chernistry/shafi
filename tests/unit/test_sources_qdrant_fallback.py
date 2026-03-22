from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_challenge.eval.sources import QdrantPageTextFallback


class _FakeQdrantClient:
    def __init__(self) -> None:
        self._call = 0

    async def scroll(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args
        self._call += 1
        if self._call == 1:
            # Return out-of-order chunk_idx to verify ordering.
            records = [
                SimpleNamespace(payload={"chunk_id": "doc:0:2:x", "chunk_text": "C"}),
                SimpleNamespace(payload={"chunk_id": "doc:0:0:x", "chunk_text": "A"}),
            ]
            return records, 1
        if self._call == 2:
            records = [
                SimpleNamespace(payload={"chunk_id": "doc:0:1:x", "chunk_text": "B"}),
            ]
            return records, None
        return [], None


@pytest.mark.asyncio
async def test_qdrant_fallback_sorts_by_chunk_idx_and_paginates() -> None:
    client = _FakeQdrantClient()
    fallback = QdrantPageTextFallback(
        qdrant_client=client,  # type: ignore[arg-type]
        collection="col",
        max_chars_per_page=10_000,
    )
    text = await fallback.get_page_text(doc_id="doc", page=1)
    assert text == "A\n\nB\n\nC"

