from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from rag_challenge.core.sparse_bm25 import BM25SparseEncoder


def test_bm25_sparse_encoder_defaults_cache_dir_to_workspace_cache(tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    class _FakeSparseTextEmbedding:
        def __init__(self, model_name: str, *, cache_dir: str | None, threads: int | None, lazy_load: bool) -> None:
            recorded["model_name"] = model_name
            recorded["cache_dir"] = cache_dir
            recorded["threads"] = threads
            recorded["lazy_load"] = lazy_load

    with (
        patch("rag_challenge.core.sparse_bm25.Path.cwd", return_value=tmp_path),
        patch("fastembed.sparse.SparseTextEmbedding", _FakeSparseTextEmbedding),
    ):
        encoder = BM25SparseEncoder(model_name="Qdrant/bm25")

    expected = (tmp_path / ".cache" / "fastembed").resolve()
    assert encoder._cache_dir == str(expected)
    assert expected.is_dir()
    assert recorded == {
        "model_name": "Qdrant/bm25",
        "cache_dir": str(expected),
        "threads": None,
        "lazy_load": True,
    }


def test_bm25_sparse_encoder_expands_custom_cache_dir(tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    class _FakeSparseTextEmbedding:
        def __init__(self, model_name: str, *, cache_dir: str | None, threads: int | None, lazy_load: bool) -> None:
            recorded["cache_dir"] = cache_dir

    custom_dir = tmp_path / "nested" / "bm25-cache"
    with patch("fastembed.sparse.SparseTextEmbedding", _FakeSparseTextEmbedding):
        encoder = BM25SparseEncoder(model_name="Qdrant/bm25", cache_dir=str(custom_dir))

    expected = custom_dir.resolve()
    assert encoder._cache_dir == str(expected)
    assert expected.is_dir()
    assert recorded["cache_dir"] == str(expected)


def test_bm25_sparse_encoder_falls_back_when_custom_cache_dir_is_unavailable(tmp_path: Path) -> None:
    recorded: dict[str, object] = {}

    class _FakeSparseTextEmbedding:
        def __init__(self, model_name: str, *, cache_dir: str | None, threads: int | None, lazy_load: bool) -> None:
            recorded["cache_dir"] = cache_dir

    bad_cache = Path("/home/appuser/.cache/fastembed")
    real_prepare = BM25SparseEncoder._prepare_cache_dir

    def _fake_prepare(path: Path) -> str:
        if path == bad_cache:
            raise OSError(45, "Operation not supported")
        return real_prepare(path)

    with (
        patch("rag_challenge.core.sparse_bm25.Path.cwd", return_value=tmp_path),
        patch("rag_challenge.core.sparse_bm25.BM25SparseEncoder._prepare_cache_dir", side_effect=_fake_prepare),
        patch("fastembed.sparse.SparseTextEmbedding", _FakeSparseTextEmbedding),
    ):
        encoder = BM25SparseEncoder(model_name="Qdrant/bm25", cache_dir=str(bad_cache))

    expected = (tmp_path / ".cache" / "fastembed").resolve()
    assert encoder._cache_dir == str(expected)
    assert expected.is_dir()
    assert recorded["cache_dir"] == str(expected)
