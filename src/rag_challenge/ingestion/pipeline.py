from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rag_challenge.config import get_settings
from rag_challenge.config.logging import setup_logging
from rag_challenge.core.embedding import EmbeddingClient
from rag_challenge.core.qdrant import QdrantStore
from rag_challenge.core.sparse_bm25 import BM25SparseEncoder
from rag_challenge.ingestion.chunker import LegalChunker
from rag_challenge.ingestion.parser import DocumentParser
from rag_challenge.ingestion.sac import SACGenerator
from rag_challenge.llm import LLMProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rag_challenge.models import Chunk, ParsedDocument

logger = logging.getLogger(__name__)


def _error_list_factory() -> list[str]:
    return []


def _manifest_docs_factory() -> dict[str, _ManifestEntry]:
    return {}


def _path_list_factory() -> list[Path]:
    return []


def _deleted_entries_factory() -> dict[str, _ManifestEntry]:
    return {}


def _fingerprint_map_factory() -> dict[str, _FileFingerprint]:
    return {}


@dataclass(frozen=True)
class _FileFingerprint:
    sha256: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class _ManifestEntry:
    sha256: str
    size_bytes: int
    mtime_ns: int
    doc_id: str
    status: str


@dataclass
class _IngestionManifest:
    schema_version: int = 1
    ingest_version: str = ""
    documents: dict[str, _ManifestEntry] = field(default_factory=_manifest_docs_factory)


@dataclass
class _IngestionPlan:
    changed_files: list[Path] = field(default_factory=_path_list_factory)
    unchanged_count: int = 0
    deleted_entries: dict[str, _ManifestEntry] = field(default_factory=_deleted_entries_factory)
    fingerprints: dict[str, _FileFingerprint] = field(default_factory=_fingerprint_map_factory)


@dataclass
class IngestionStats:
    docs_parsed: int = 0
    docs_failed: int = 0
    docs_skipped_unchanged: int = 0
    docs_deleted: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_upserted: int = 0
    sac_summaries_generated: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=_error_list_factory)


class IngestionPipeline:
    """Orchestrates parse -> chunk -> SAC -> embed -> Qdrant upsert."""

    def __init__(
        self,
        *,
        parser: DocumentParser | None = None,
        chunker: LegalChunker | None = None,
        sac: SACGenerator | None = None,
        embedder: EmbeddingClient | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self._settings = get_settings()
        self._parser = parser or DocumentParser()
        self._chunker = chunker or LegalChunker()
        self._embedder = embedder or EmbeddingClient()
        self._store = store or QdrantStore()
        self._sparse_encoder: BM25SparseEncoder | None = None
        qdrant_settings = getattr(self._settings, "qdrant", None)
        if qdrant_settings is not None and bool(getattr(qdrant_settings, "enable_sparse_bm25", True)):
            cache_dir = str(getattr(qdrant_settings, "fastembed_cache_dir", "")).strip() or None
            threads = self._coerce_int(getattr(qdrant_settings, "sparse_threads", None))
            try:
                self._sparse_encoder = BM25SparseEncoder(
                    model_name=str(getattr(qdrant_settings, "sparse_model", "Qdrant/bm25")),
                    cache_dir=cache_dir,
                    threads=threads,
                )
            except Exception:
                logger.warning("Failed initializing BM25 sparse encoder; continuing dense-only ingestion", exc_info=True)
                self._sparse_encoder = None

        self._owned_llm: LLMProvider | None = None
        if sac is None:
            self._owned_llm = LLMProvider()
            self._sac = SACGenerator(self._owned_llm)
        else:
            self._sac = sac

    async def run(self, doc_dir: Path) -> IngestionStats:
        started = time.perf_counter()
        stats = IngestionStats()
        doc_dir = Path(doc_dir)
        manifest_path = self._manifest_path_for(doc_dir)

        logger.info(
            "ingestion_start",
            extra={
                "doc_dir": str(doc_dir),
                "manifest_path": str(manifest_path),
                "ingest_version": self._settings.ingestion.ingest_version,
            },
        )
        await self._store.ensure_collection()
        await self._store.ensure_payload_indexes()

        manifest = self._load_manifest(manifest_path)
        ingest_version = self._settings.ingestion.ingest_version
        if manifest.ingest_version and manifest.ingest_version != ingest_version:
            logger.info(
                "Ingest version changed %r -> %r; invalidating incremental manifest",
                manifest.ingest_version,
                ingest_version,
            )
            manifest = _IngestionManifest(ingest_version=ingest_version)
        elif not manifest.ingest_version:
            manifest.ingest_version = ingest_version

        source_files = self._list_source_files(doc_dir)
        plan = self._build_ingestion_plan(doc_dir, source_files, manifest)
        stats.docs_skipped_unchanged = plan.unchanged_count

        docs = self._parse_files(plan.changed_files)
        stats.docs_parsed = len(docs)
        logger.info(
            "ingestion_plan",
            extra={
                "files_total": len(source_files),
                "files_changed": len(plan.changed_files),
                "files_unchanged": plan.unchanged_count,
                "files_deleted": len(plan.deleted_entries),
                "docs_parsed": stats.docs_parsed,
            },
        )

        if docs:
            total_docs = len(docs)
            progress_started = time.perf_counter()
            for processed_docs, doc in enumerate(docs, start=1):
                doc_started = time.perf_counter()
                rel_path = self._relative_manifest_key(doc_dir, Path(doc.source_path))
                fingerprint = plan.fingerprints.get(rel_path)
                if fingerprint is None:
                    logger.warning(
                        "ingestion_doc_missing_fingerprint",
                        extra={"rel_path": rel_path, "doc_id": doc.doc_id},
                    )
                else:
                    try:
                        chunks = self._chunker.chunk_document(doc)
                        if not chunks:
                            logger.warning(
                                "ingestion_doc_empty_chunks",
                                extra={"doc_id": doc.doc_id, "doc_title": doc.title},
                            )
                            self._set_manifest_entry(
                                manifest=manifest,
                                rel_path=rel_path,
                                doc_id=doc.doc_id,
                                fingerprint=fingerprint,
                                status="empty",
                            )
                            self._save_manifest(manifest_path, manifest)
                        else:
                            summary = await self._sac.generate_doc_summary(doc)
                            stats.sac_summaries_generated += 1
                            augmented = self._sac.augment_chunks(chunks, summary)

                            stats.chunks_created += len(augmented)
                            logger.info(
                                "ingestion_doc_processed",
                                extra={
                                    "doc_id": doc.doc_id,
                                    "doc_title": doc.title,
                                    "chunks": len(augmented),
                                    "summary_chars": len(summary),
                                },
                            )

                            embedding_texts = [chunk.chunk_text_for_embedding for chunk in augmented]
                            vectors = await self._embedder.embed_documents(embedding_texts)
                            stats.chunks_embedded += len(vectors)

                            sparse_vectors = None
                            if self._sparse_encoder is not None:
                                try:
                                    sparse_vectors = self._sparse_encoder.encode_documents(
                                        [chunk.chunk_text_for_embedding for chunk in augmented]
                                    )
                                except Exception:
                                    logger.warning(
                                        "BM25 sparse encoding failed; falling back to dense-only for doc_id=%s",
                                        doc.doc_id,
                                        exc_info=True,
                                    )
                                    sparse_vectors = None

                            upserted = await self._store.upsert_chunks(
                                augmented,
                                vectors,
                                sparse_vectors=sparse_vectors,
                            ) if sparse_vectors is not None else await self._store.upsert_chunks(augmented, vectors)
                            stats.chunks_upserted += upserted
                            await self._cleanup_stale_versions([doc.doc_id])

                            self._set_manifest_entry(
                                manifest=manifest,
                                rel_path=rel_path,
                                doc_id=doc.doc_id,
                                fingerprint=fingerprint,
                                status="indexed",
                            )
                            self._save_manifest(manifest_path, manifest)
                    except Exception as exc:
                        stats.docs_failed += 1
                        error_text = f"{doc.doc_id} ({doc.title}): {exc}"
                        stats.errors.append(error_text)
                        logger.error(
                            "ingestion_doc_failed",
                            extra={"doc_id": doc.doc_id, "doc_title": doc.title},
                            exc_info=True,
                        )
                elapsed_progress_s = max(0.0, time.perf_counter() - progress_started)
                avg_doc_s = elapsed_progress_s / max(1, processed_docs)
                remaining_docs = max(0, total_docs - processed_docs)
                logger.info(
                    "ingestion_progress",
                    extra={
                        "done_docs": processed_docs,
                        "total_docs": total_docs,
                        "remaining_docs": remaining_docs,
                        "progress_pct": round((processed_docs / max(1, total_docs)) * 100.0, 2),
                        "doc_elapsed_s": round(max(0.0, time.perf_counter() - doc_started), 3),
                        "avg_doc_s": round(avg_doc_s, 3),
                        "eta_s": int(avg_doc_s * remaining_docs),
                        "docs_failed": stats.docs_failed,
                        "chunks_upserted_total": stats.chunks_upserted,
                    },
                )

        deleted_rel_paths = await self._delete_removed_documents(plan.deleted_entries, stats)
        for rel_path in deleted_rel_paths:
            manifest.documents.pop(rel_path, None)
        manifest.ingest_version = self._settings.ingestion.ingest_version
        self._save_manifest(manifest_path, manifest)

        stats.elapsed_s = time.perf_counter() - started
        logger.info(
            "ingestion_complete",
            extra={
                "docs_parsed": stats.docs_parsed,
                "docs_skipped_unchanged": stats.docs_skipped_unchanged,
                "docs_deleted": stats.docs_deleted,
                "docs_failed": stats.docs_failed,
                "chunks_created": stats.chunks_created,
                "chunks_upserted": stats.chunks_upserted,
                "elapsed_s": round(stats.elapsed_s, 3),
            },
        )
        return stats

    def _manifest_path_for(self, doc_dir: Path) -> Path:
        filename = self._settings.ingestion.manifest_filename
        configured_dir = str(self._settings.ingestion.manifest_dir).strip()
        if configured_dir:
            base_dir = Path(configured_dir).expanduser()
            if not base_dir.is_absolute():
                base_dir = Path.cwd() / base_dir
            base_dir.mkdir(parents=True, exist_ok=True)
            return base_dir / filename

        candidate_dir = doc_dir.parent
        if self._is_dir_writable(candidate_dir):
            return candidate_dir / filename

        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        fallback_dir = cache_root / "rag_challenge"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / filename

    @staticmethod
    def _is_dir_writable(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("1", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def _list_source_files(self, doc_dir: Path) -> list[Path]:
        list_supported_files_obj: object = getattr(self._parser, "list_supported_files", None)
        if callable(list_supported_files_obj):
            list_supported_files = cast("Callable[[Path], object]", list_supported_files_obj)
            listed_obj: object = list_supported_files(doc_dir)
            listed_type_name = type(listed_obj).__name__
            if isinstance(listed_obj, list):
                listed_type_name = "list"
                listed_paths: list[Path] = []
                for item in cast("list[object]", listed_obj):
                    if not isinstance(item, Path):
                        break
                    listed_paths.append(item)
                else:
                    return listed_paths
            logger.warning(
                "Parser.list_supported_files returned unexpected value (%s); falling back to parse_directory path",
                listed_type_name,
            )

        docs = self._parser.parse_directory(doc_dir)
        return [Path(doc.source_path) for doc in docs if doc.source_path]

    def _parse_files(self, files: list[Path]) -> list[ParsedDocument]:
        if not files:
            return []

        parsed: list[ParsedDocument] = []
        for file_path in files:
            try:
                parsed.append(self._parser.parse_file(file_path))
            except Exception:
                logger.warning("Failed to parse %s; skipping", file_path, exc_info=True)
        return parsed

    def _build_ingestion_plan(
        self,
        doc_dir: Path,
        source_files: list[Path],
        manifest: _IngestionManifest,
    ) -> _IngestionPlan:
        plan = _IngestionPlan()

        for file_path in source_files:
            rel_key = self._relative_manifest_key(doc_dir, file_path)
            stat_result = file_path.stat()
            fingerprint = _FileFingerprint(
                sha256=self._hash_file(file_path),
                size_bytes=int(stat_result.st_size),
                mtime_ns=int(stat_result.st_mtime_ns),
            )
            plan.fingerprints[rel_key] = fingerprint

            previous = manifest.documents.get(rel_key)
            if previous is not None and previous.sha256 == fingerprint.sha256:
                plan.unchanged_count += 1
                continue
            plan.changed_files.append(file_path)

        for rel_key, entry in manifest.documents.items():
            if rel_key not in plan.fingerprints:
                plan.deleted_entries[rel_key] = entry

        return plan

    async def _delete_removed_documents(
        self,
        deleted_entries: dict[str, _ManifestEntry],
        stats: IngestionStats,
    ) -> list[str]:
        deleted_rel_paths: list[str] = []
        for rel_path, entry in deleted_entries.items():
            try:
                await self._store.delete_by_doc_id(entry.doc_id)
            except Exception as exc:
                stats.errors.append(f"{rel_path} ({entry.doc_id}): delete failed: {exc}")
                logger.warning(
                    "Failed deleting removed document rel_path=%s doc_id=%s",
                    rel_path,
                    entry.doc_id,
                    exc_info=True,
                )
                continue

            stats.docs_deleted += 1
            deleted_rel_paths.append(rel_path)

        return deleted_rel_paths

    def _update_manifest_entries(
        self,
        *,
        manifest: _IngestionManifest,
        doc_dir: Path,
        docs_by_rel_path: dict[str, ParsedDocument],
        successful_doc_ids: list[str],
        chunked_doc_ids: set[str],
        fingerprints: dict[str, _FileFingerprint],
        deleted_rel_paths: list[str],
    ) -> None:
        successful_doc_id_set = set(successful_doc_ids)

        for rel_path in deleted_rel_paths:
            manifest.documents.pop(rel_path, None)

        for rel_path, doc in docs_by_rel_path.items():
            if doc.doc_id not in successful_doc_id_set:
                continue

            fingerprint = fingerprints.get(rel_path)
            if fingerprint is None:
                logger.warning(
                    "No fingerprint found for parsed doc rel_path=%s doc_id=%s; skipping manifest update",
                    rel_path,
                    doc.doc_id,
                )
                continue

            manifest.documents[rel_path] = _ManifestEntry(
                sha256=fingerprint.sha256,
                size_bytes=fingerprint.size_bytes,
                mtime_ns=fingerprint.mtime_ns,
                doc_id=doc.doc_id,
                status="indexed" if doc.doc_id in chunked_doc_ids else "empty",
            )

        manifest.ingest_version = self._settings.ingestion.ingest_version

    def _set_manifest_entry(
        self,
        *,
        manifest: _IngestionManifest,
        rel_path: str,
        doc_id: str,
        fingerprint: _FileFingerprint,
        status: str,
    ) -> None:
        manifest.documents[rel_path] = _ManifestEntry(
            sha256=fingerprint.sha256,
            size_bytes=fingerprint.size_bytes,
            mtime_ns=fingerprint.mtime_ns,
            doc_id=doc_id,
            status=status,
        )
        manifest.ingest_version = self._settings.ingestion.ingest_version

    @staticmethod
    def _relative_manifest_key(doc_dir: Path, file_path: Path) -> str:
        return file_path.relative_to(doc_dir).as_posix()

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        chunk_size = max(1, int(self._settings.ingestion.manifest_hash_chunk_size_bytes))
        with path.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _load_manifest(self, manifest_path: Path) -> _IngestionManifest:
        expected_schema_version = int(self._settings.ingestion.manifest_schema_version)
        if not manifest_path.exists():
            return _IngestionManifest(schema_version=expected_schema_version)

        try:
            raw_obj: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load ingestion manifest %s; starting fresh", manifest_path, exc_info=True)
            return _IngestionManifest(schema_version=expected_schema_version)

        if not isinstance(raw_obj, dict):
            logger.warning("Ingestion manifest is not an object: %s", manifest_path)
            return _IngestionManifest(schema_version=expected_schema_version)

        data = cast("dict[str, object]", raw_obj)
        schema_version = data.get("schema_version")
        if schema_version != expected_schema_version:
            logger.info(
                "Ignoring ingestion manifest %s with schema_version=%r (expected=%d)",
                manifest_path,
                schema_version,
                expected_schema_version,
            )
            return _IngestionManifest(schema_version=expected_schema_version)

        ingest_version_obj = data.get("ingest_version")
        documents_obj = data.get("documents")
        manifest = _IngestionManifest(
            schema_version=expected_schema_version,
            ingest_version=str(ingest_version_obj or ""),
        )
        if not isinstance(documents_obj, dict):
            return manifest

        documents_map = cast("dict[object, object]", documents_obj)
        for key_obj, entry_obj in documents_map.items():
            if not isinstance(key_obj, str) or not isinstance(entry_obj, dict):
                continue
            entry_dict = cast("dict[str, object]", entry_obj)
            sha256_obj = entry_dict.get("sha256")
            doc_id_obj = entry_dict.get("doc_id")
            if not isinstance(sha256_obj, str) or not isinstance(doc_id_obj, str):
                continue

            size_bytes = self._coerce_int(entry_dict.get("size_bytes", 0))
            mtime_ns = self._coerce_int(entry_dict.get("mtime_ns", 0))
            if size_bytes is None or mtime_ns is None:
                continue

            status_obj = entry_dict.get("status", "indexed")
            status = status_obj if isinstance(status_obj, str) else "indexed"
            manifest.documents[key_obj] = _ManifestEntry(
                sha256=sha256_obj,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                doc_id=doc_id_obj,
                status=status,
            )

        return manifest

    def _save_manifest(self, manifest_path: Path, manifest: _IngestionManifest) -> None:
        payload = {
            "schema_version": manifest.schema_version,
            "ingest_version": manifest.ingest_version,
            "documents": {
                rel_path: {
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                    "mtime_ns": entry.mtime_ns,
                    "doc_id": entry.doc_id,
                    "status": entry.status,
                }
                for rel_path, entry in sorted(manifest.documents.items())
            },
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(manifest_path)

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    async def close(self) -> None:
        close_errors: list[Exception] = []

        for closer in (self._embedder.close, self._store.close):
            try:
                await closer()
            except Exception as exc:
                close_errors.append(exc)
                logger.warning("Failed to close resource: %s", exc, exc_info=True)

        if self._owned_llm is not None:
            try:
                await self._owned_llm.close()
            except Exception as exc:
                close_errors.append(exc)
                logger.warning("Failed to close LLM provider: %s", exc, exc_info=True)

        if close_errors:
            raise RuntimeError(f"Failed closing {len(close_errors)} ingestion resources")

    async def _chunk_and_augment_docs(
        self,
        docs: list[ParsedDocument],
        stats: IngestionStats,
    ) -> tuple[dict[str, list[Chunk]], list[str]]:
        semaphore = asyncio.Semaphore(max(1, int(self._settings.ingestion.sac_concurrency)))
        successful_doc_ids: list[str] = []

        async def process_doc(doc: ParsedDocument) -> tuple[str, list[Chunk]]:
            try:
                chunks = self._chunker.chunk_document(doc)
                if not chunks:
                    logger.warning("Doc %s produced 0 chunks", doc.doc_id)
                    successful_doc_ids.append(doc.doc_id)
                    return doc.doc_id, []

                async with semaphore:
                    summary = await self._sac.generate_doc_summary(doc)
                stats.sac_summaries_generated += 1

                augmented = self._sac.augment_chunks(chunks, summary)
                logger.info(
                    "Processed doc %s title=%r chunks=%d summary_chars=%d",
                    doc.doc_id,
                    doc.title,
                    len(augmented),
                    len(summary),
                )
                successful_doc_ids.append(doc.doc_id)
                return doc.doc_id, augmented
            except Exception as exc:
                stats.docs_failed += 1
                error_text = f"{doc.doc_id} ({doc.title}): {exc}"
                stats.errors.append(error_text)
                logger.error("Failed processing doc %s: %s", doc.doc_id, exc, exc_info=True)
                return doc.doc_id, []

        results = await asyncio.gather(*[process_doc(doc) for doc in docs])
        return {doc_id: chunks for doc_id, chunks in results if chunks}, successful_doc_ids

    async def _cleanup_stale_versions(self, doc_ids: Iterable[str]) -> None:
        delete_stale = getattr(self._store, "delete_stale_doc_versions", None)
        if delete_stale is None:
            logger.info("QdrantStore has no stale-version cleanup method; skipping")
            return

        ingest_version = self._settings.ingestion.ingest_version
        for doc_id in doc_ids:
            try:
                await delete_stale(doc_id, keep_ingest_version=ingest_version)
            except Exception:
                logger.warning(
                    "Failed stale-version cleanup for doc_id=%s ingest_version=%s",
                    doc_id,
                    ingest_version,
                    exc_info=True,
                )


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="rag_challenge ingestion pipeline")
    parser.add_argument("--doc-dir", type=Path, required=True, help="Directory containing source documents")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.app.log_level, settings.app.log_format)

    if not args.doc_dir.is_dir():
        logger.error("Not a directory: %s", args.doc_dir)
        return 1

    pipeline = IngestionPipeline()
    try:
        stats = await pipeline.run(args.doc_dir)
        logger.info("Stats: %s", stats)
    finally:
        await pipeline.close()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
