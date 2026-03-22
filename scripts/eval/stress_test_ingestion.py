#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

from rag_challenge.ingestion.chunker import LegalChunker
from rag_challenge.ingestion.parser import DocumentParser
from rag_challenge.ingestion.pipeline import IngestionPipeline, IngestionStats

if TYPE_CHECKING:
    from rag_challenge.models import Chunk, ParsedDocument

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "docs"
EMBED_DIM = 8

JsonDict = dict[str, Any]


@dataclass
class StageTimings:
    parse_s: float = 0.0
    chunk_s: float = 0.0
    sac_s: float = 0.0
    embed_s: float = 0.0
    upsert_s: float = 0.0
    delete_s: float = 0.0

    def to_dict(self) -> JsonDict:
        return {
            "parse": round(self.parse_s, 4),
            "chunk": round(self.chunk_s, 4),
            "sac": round(self.sac_s, 4),
            "embed": round(self.embed_s, 4),
            "upsert": round(self.upsert_s, 4),
            "delete_stale": round(self.delete_s, 4),
        }


def _chunker_settings_stub() -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            chunk_size_tokens=450,
            chunk_overlap_tokens=45,
        )
    )


def _pipeline_settings_stub(manifest_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            ingest_version="stress-proxy-v1",
            manifest_filename=".rag_challenge_ingestion_manifest.json",
            manifest_dir=str(manifest_dir),
            manifest_hash_chunk_size_bytes=1024 * 1024,
            manifest_schema_version=1,
            sac_concurrency=1,
        ),
        qdrant=SimpleNamespace(
            enable_sparse_bm25=False,
        ),
    )


class TimingParser:
    def __init__(self, parser: DocumentParser, timings: StageTimings) -> None:
        self._parser = parser
        self._timings = timings

    def list_supported_files(self, doc_dir: Path) -> list[Path]:
        return self._parser.list_supported_files(doc_dir)

    def parse_file(self, file_path: Path) -> ParsedDocument:
        started = time.perf_counter()
        try:
            return self._parser.parse_file(file_path)
        finally:
            self._timings.parse_s += max(0.0, time.perf_counter() - started)

    def parse_directory(self, doc_dir: Path) -> list[ParsedDocument]:
        started = time.perf_counter()
        try:
            return self._parser.parse_directory(doc_dir)
        finally:
            self._timings.parse_s += max(0.0, time.perf_counter() - started)


class TimingChunker:
    def __init__(self, chunker: LegalChunker, timings: StageTimings) -> None:
        self._chunker = chunker
        self._timings = timings

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        started = time.perf_counter()
        try:
            return self._chunker.chunk_document(doc)
        finally:
            self._timings.chunk_s += max(0.0, time.perf_counter() - started)


class FakeSAC:
    def __init__(self, timings: StageTimings) -> None:
        self._timings = timings

    async def generate_doc_summary(self, doc: ParsedDocument) -> str:
        started = time.perf_counter()
        try:
            head = " ".join(line.strip() for line in doc.full_text.splitlines()[:2] if line.strip())
            return f"{doc.title}: {head[:160]}".strip()
        finally:
            self._timings.sac_s += max(0.0, time.perf_counter() - started)

    def augment_chunks(self, chunks: list[Chunk], summary: str) -> list[Chunk]:
        return [
            chunk.model_copy(
                update={
                    "chunk_text_for_embedding": f"[DOC_SUMMARY]\n{summary}\n\n[CHUNK]\n{chunk.chunk_text}",
                    "doc_summary": summary,
                }
            )
            for chunk in chunks
        ]


class FakeEmbedder:
    def __init__(self, timings: StageTimings) -> None:
        self._timings = timings

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            vectors: list[list[float]] = []
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vector = [round(digest[idx] / 255.0, 6) for idx in range(EMBED_DIM)]
                vectors.append(vector)
            return vectors
        finally:
            self._timings.embed_s += max(0.0, time.perf_counter() - started)

    async def close(self) -> None:
        return None


class FakeStore:
    def __init__(self, timings: StageTimings) -> None:
        self._timings = timings

    async def ensure_collection(self) -> None:
        return None

    async def ensure_payload_indexes(self) -> None:
        return None

    async def ensure_shadow_collection(self) -> None:
        return None

    async def ensure_shadow_payload_indexes(self) -> None:
        return None

    async def ensure_page_collection(self) -> None:
        return None

    async def ensure_page_payload_indexes(self) -> None:
        return None

    async def upsert_pages(self, pages: list[object], dense_vectors: list[list[float]], *, sparse_vectors: list[object] | None = None) -> int:
        del dense_vectors, sparse_vectors
        return len(pages)

    async def delete_pages_by_doc_id(self, doc_id: str) -> None:
        del doc_id

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        sparse_vectors: list[object] | None = None,
    ) -> int:
        del vectors, sparse_vectors
        started = time.perf_counter()
        try:
            return len(chunks)
        finally:
            self._timings.upsert_s += max(0.0, time.perf_counter() - started)

    async def upsert_shadow_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        sparse_vectors: list[object] | None = None,
    ) -> int:
        del vectors, sparse_vectors
        started = time.perf_counter()
        try:
            return len(chunks)
        finally:
            self._timings.upsert_s += max(0.0, time.perf_counter() - started)

    async def delete_stale_doc_versions(self, doc_id: str, *, keep_ingest_version: str) -> None:
        del doc_id, keep_ingest_version
        started = time.perf_counter()
        try:
            return None
        finally:
            self._timings.delete_s += max(0.0, time.perf_counter() - started)

    async def close(self) -> None:
        return None


def _copy_fixture_docs(*, fixture_dir: Path, corpus_dir: Path, target_docs: int) -> list[Path]:
    fixture_files = sorted(path for path in fixture_dir.iterdir() if path.is_file())
    if not fixture_files:
        raise ValueError(f"No fixture files found in {fixture_dir}")
    copied: list[Path] = []
    for idx in range(target_docs):
        template = fixture_files[idx % len(fixture_files)]
        target = corpus_dir / f"{idx:04d}_{template.name}"
        shutil.copyfile(template, target)
        copied.append(target)
    return copied


def _rss_mb_from_tracemalloc_bytes(value: int) -> float:
    return round(value / (1024 * 1024), 3)


def _largest_stage_name(stage_timings: StageTimings) -> str:
    stage_map = stage_timings.to_dict()
    if not stage_map:
        return "none"
    return max(stage_map.items(), key=lambda item: item[1])[0]


def _stage_share_map(stage_timings: StageTimings) -> JsonDict:
    stage_map = stage_timings.to_dict()
    total = sum(float(value) for value in stage_map.values())
    if total <= 0.0:
        return {name: 0.0 for name in stage_map}
    return {
        name: round((float(value) / total), 4)
        for name, value in stage_map.items()
    }


def _blocking_bottleneck(stats: IngestionStats) -> str:
    if stats.docs_failed > 0:
        return "parse_or_chunk_failures_detected"
    if stats.docs_parsed == 0:
        return "proxy_input_invalid"
    return "none_local_blocker_detected"


def _readiness_verdict(stats: IngestionStats) -> str:
    if stats.docs_failed > 0:
        return "not_ready_local_failures_detected"
    if stats.docs_parsed == 0 or stats.chunks_upserted == 0:
        return "not_ready_proxy_incomplete"
    return "proxy_ready_with_external_uncertainty"


def _build_report(
    *,
    target_docs: int,
    fixture_templates: int,
    stats: IngestionStats,
    stage_timings: StageTimings,
    python_peak_alloc_mb: float,
) -> JsonDict:
    docs_per_s = stats.docs_parsed / stats.elapsed_s if stats.elapsed_s > 0 else 0.0
    chunks_per_s = stats.chunks_created / stats.elapsed_s if stats.elapsed_s > 0 else 0.0
    largest_stage = _largest_stage_name(stage_timings)
    return {
        "proxy_scope": "real_parser_chunker_mocked_external_services",
        "target_docs": target_docs,
        "fixture_templates": fixture_templates,
        "docs_parsed": stats.docs_parsed,
        "docs_failed": stats.docs_failed,
        "docs_skipped_unchanged": stats.docs_skipped_unchanged,
        "docs_deleted": stats.docs_deleted,
        "chunks_created": stats.chunks_created,
        "chunks_embedded": stats.chunks_embedded,
        "chunks_upserted": stats.chunks_upserted,
        "sac_summaries_generated": stats.sac_summaries_generated,
        "elapsed_s": round(stats.elapsed_s, 4),
        "docs_per_s": round(docs_per_s, 4),
        "chunks_per_s": round(chunks_per_s, 4),
        "python_peak_alloc_mb": python_peak_alloc_mb,
        "stage_timings_s": stage_timings.to_dict(),
        "stage_share": _stage_share_map(stage_timings),
        "largest_measured_stage": largest_stage,
        "blocking_bottleneck": _blocking_bottleneck(stats),
        "largest_uncertainty": "external_services_mocked",
        "readiness_verdict": _readiness_verdict(stats),
        "errors": list(stats.errors),
        "notes": [
            "Parser and chunker are real; SAC, embeddings, and Qdrant writes are mocked for a bounded local readiness proxy.",
            "This proxy is valid for local CPU/memory and orchestration stability only.",
            "External embedding, reranker, and Qdrant network throughput remain unmeasured.",
        ],
    }


def _render_markdown(report: JsonDict) -> str:
    lines = [
        "# Ingestion Stress Proxy",
        "",
        f"- `proxy_scope`: `{report['proxy_scope']}`",
        f"- `target_docs`: `{report['target_docs']}`",
        f"- `fixture_templates`: `{report['fixture_templates']}`",
        f"- `docs_parsed`: `{report['docs_parsed']}`",
        f"- `docs_failed`: `{report['docs_failed']}`",
        f"- `chunks_created`: `{report['chunks_created']}`",
        f"- `elapsed_s`: `{report['elapsed_s']}`",
        f"- `docs_per_s`: `{report['docs_per_s']}`",
        f"- `chunks_per_s`: `{report['chunks_per_s']}`",
        f"- `python_peak_alloc_mb`: `{report['python_peak_alloc_mb']}`",
        f"- `largest_measured_stage`: `{report['largest_measured_stage']}`",
        f"- `blocking_bottleneck`: `{report['blocking_bottleneck']}`",
        f"- `largest_uncertainty`: `{report['largest_uncertainty']}`",
        f"- `readiness_verdict`: `{report['readiness_verdict']}`",
        "",
        "## Stage Timings",
        "",
    ]
    stage_timings = cast("JsonDict", report["stage_timings_s"])
    stage_share = cast("JsonDict", report["stage_share"])
    for stage_name, stage_s in stage_timings.items():
        lines.append(f"- `{stage_name}`: `{stage_s}` s (`share={stage_share.get(stage_name, 0.0)}`)")

    lines.extend(["", "## Notes", ""])
    for note in cast("list[str]", report["notes"]):
        lines.append(f"- {note}")

    errors = cast("list[str]", report.get("errors") or [])
    if errors:
        lines.extend(["", "## Errors", ""])
        for error in errors:
            lines.append(f"- {error}")

    return "\n".join(lines).rstrip() + "\n"


async def _run_proxy(*, fixture_dir: Path, target_docs: int) -> JsonDict:
    timings = StageTimings()
    with tempfile.TemporaryDirectory(prefix="ingestion-stress-proxy-") as tmp_dir:
        temp_root = Path(tmp_dir)
        corpus_dir = temp_root / "docs"
        manifest_dir = temp_root / "manifests"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        copied = _copy_fixture_docs(fixture_dir=fixture_dir, corpus_dir=corpus_dir, target_docs=target_docs)

        parser = TimingParser(DocumentParser(), timings)
        with patch("rag_challenge.ingestion.chunker.get_settings", return_value=_chunker_settings_stub()):
            chunker = TimingChunker(LegalChunker(), timings)

        sac = FakeSAC(timings)
        embedder = FakeEmbedder(timings)
        store = FakeStore(timings)

        with patch("rag_challenge.ingestion.pipeline.get_settings", return_value=_pipeline_settings_stub(manifest_dir)):
            pipeline = IngestionPipeline(
                parser=cast("Any", parser),
                chunker=cast("Any", chunker),
                sac=cast("Any", sac),
                embedder=cast("Any", embedder),
                store=cast("Any", store),
            )
            tracemalloc.start()
            try:
                stats = await pipeline.run(corpus_dir)
                _current_alloc, peak_alloc = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
                await pipeline.close()

    return _build_report(
        target_docs=target_docs,
        fixture_templates=len({path.name.split("_", 1)[1] for path in copied}),
        stats=stats,
        stage_timings=timings,
        python_peak_alloc_mb=_rss_mb_from_tracemalloc_bytes(peak_alloc),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--target-docs", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.target_docs <= 0:
        raise ValueError("--target-docs must be > 0")

    report = asyncio.run(_run_proxy(fixture_dir=args.fixture_dir.resolve(), target_docs=int(args.target_docs)))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ingestion_stress_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "ingestion_stress_report.md").write_text(_render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
