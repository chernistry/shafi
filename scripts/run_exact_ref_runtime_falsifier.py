# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, cast

from rag_challenge.config import get_settings
from rag_challenge.core.qdrant import QdrantStore
from rag_challenge.core.retriever import HybridRetriever
from rag_challenge.core.sparse_bm25 import BM25SparseEncoder
from rag_challenge.models import Chunk, DocType, RetrievedChunk

JsonDict = dict[str, Any]


class _ZeroEmbedder:
    def __init__(self) -> None:
        self._dimensions = int(get_settings().embedding.dimensions)

    async def embed_query(self, text: str) -> list[float]:
        del text
        return [0.0] * self._dimensions


def _load_pack(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict at {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Expected 'cases' list at {path}")
    return [cast("JsonDict", case) for case in cases if isinstance(case, dict)]


def _load_corpus(corpus_dir: Path, cases: list[JsonDict]) -> list[Chunk]:
    dimensions = int(get_settings().embedding.dimensions)
    del dimensions
    chunks: list[Chunk] = []
    seen_doc_ids: set[str] = set()
    for case in cases:
        source_fixtures = [str(item).strip() for item in cast("list[object]", case.get("source_fixtures") or []) if str(item).strip()]
        doc_refs = [str(item).strip() for item in cast("list[object]", case.get("doc_refs") or []) if str(item).strip()]
        for source_fixture in source_fixtures:
            filename = Path(source_fixture).name
            source_path = corpus_dir / filename
            if not source_path.exists():
                continue
            doc_id = source_path.stem
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            text = source_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            title = _canonical_title(filename=filename, text=text, doc_refs=doc_refs)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:page1",
                    doc_id=doc_id,
                    doc_title=title,
                    doc_type=DocType.STATUTE,
                    jurisdiction="DIFC",
                    section_path="page:1",
                    chunk_text=text,
                    chunk_text_for_embedding=text,
                    doc_summary=title,
                    citations=[title, *doc_refs],
                    anchors=[title, *doc_refs],
                    token_count=len(text.split()),
                )
            )
    return chunks


def _canonical_title(*, filename: str, text: str, doc_refs: list[str]) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line and len(first_line) <= 96:
        return first_line
    if doc_refs:
        return doc_refs[0]
    return filename.replace("_", " ").replace(".txt", "").title()


def _dense_zeros(count: int) -> list[list[float]]:
    dims = int(get_settings().embedding.dimensions)
    return [[0.0] * dims for _ in range(count)]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def _gold_hit(chunks: list[RetrievedChunk], gold_texts: list[str]) -> bool:
    normalized_gold = [_normalize(text) for text in gold_texts if _normalize(text)]
    if not normalized_gold:
        return False
    for chunk in chunks:
        hay = _normalize(str(chunk.text or ""))
        if any(gold in hay for gold in normalized_gold):
            return True
    return False


async def run_runtime_falsifier(*, pack_path: Path, corpus_dir: Path) -> JsonDict:
    settings = get_settings()
    cases = _load_pack(pack_path)
    chunks = _load_corpus(corpus_dir, cases)
    encoder = BM25SparseEncoder(model_name=str(getattr(settings.qdrant, "sparse_model", "Qdrant/bm25")))
    sparse_vectors = encoder.encode_documents([chunk.chunk_text_for_embedding for chunk in chunks])

    store = QdrantStore()
    temp_collection = f"{store.collection_name}_ticket147_{uuid.uuid4().hex[:8]}"
    store._settings.collection = temp_collection  # type: ignore[attr-defined]
    await store.ensure_collection()
    await store.ensure_payload_indexes()
    await store.upsert_chunks(chunks, _dense_zeros(len(chunks)), sparse_vectors=sparse_vectors)

    retriever = HybridRetriever(store=store, embedder=cast("Any", _ZeroEmbedder()))
    results: list[JsonDict] = []
    harmful = 0
    fail_open = 0

    try:
        for case in cases:
            question = str(case.get("question") or "").strip()
            doc_refs = [str(item).strip() for item in cast("list[object]", case.get("doc_refs") or []) if str(item).strip()]
            gold_texts = [str(item).strip() for item in cast("list[object]", case.get("gold_texts") or []) if str(item).strip()]
            retrieved = await retriever.retrieve(
                question,
                doc_refs=doc_refs,
                sparse_only=True,
                top_k=3,
                doc_type_filter=DocType.STATUTE,
            )
            debug = retriever.get_last_retrieval_debug()
            fail_open_triggered = bool(debug.get("fail_open_triggered"))
            gold_hit = _gold_hit(retrieved, gold_texts)
            harmful_case = fail_open_triggered and not gold_hit
            if fail_open_triggered:
                fail_open += 1
            if harmful_case:
                harmful += 1
            results.append(
                {
                    "case_id": str(case.get("case_id") or "").strip(),
                    "question": question,
                    "doc_refs": doc_refs,
                    "gold_texts": gold_texts,
                    "retrieved_doc_titles": [chunk.doc_title for chunk in retrieved],
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved],
                    "fail_open_triggered": fail_open_triggered,
                    "fail_open_stages": list(cast("list[object]", debug.get("fail_open_stages") or [])),
                    "final_doc_ref_filter_applied": debug.get("final_doc_ref_filter_applied"),
                    "initial_chunk_count": debug.get("initial_chunk_count"),
                    "final_chunk_count": debug.get("final_chunk_count"),
                    "gold_hit": gold_hit,
                    "harmful_fail_open": harmful_case,
                }
            )
    finally:
        await store.client.delete_collection(collection_name=temp_collection)
        await store.close()

    return {
        "summary": {
            "case_count": len(results),
            "doc_count": len(chunks),
            "fail_open_case_count": fail_open,
            "harmful_fail_open_case_count": harmful,
            "verdict": "no_harmful_runtime_fail_open_pattern" if harmful == 0 else "harmful_runtime_fail_open_pattern",
        },
        "cases": results,
    }


def _render_markdown(payload: JsonDict) -> str:
    summary = cast("JsonDict", payload["summary"])
    lines = [
        "# Exact-Ref Runtime Falsifier",
        "",
        f"- case_count: `{summary['case_count']}`",
        f"- doc_count: `{summary['doc_count']}`",
        f"- fail_open_case_count: `{summary['fail_open_case_count']}`",
        f"- harmful_fail_open_case_count: `{summary['harmful_fail_open_case_count']}`",
        f"- verdict: `{summary['verdict']}`",
        "",
        "## Cases",
        "",
    ]
    for case in cast("list[JsonDict]", payload["cases"]):
        lines.extend(
            [
                f"### {case['case_id']}",
                f"- fail_open_triggered: `{case['fail_open_triggered']}`",
                f"- fail_open_stages: `{case['fail_open_stages']}`",
                f"- gold_hit: `{case['gold_hit']}`",
                f"- harmful_fail_open: `{case['harmful_fail_open']}`",
                f"- retrieved_doc_titles: `{case['retrieved_doc_titles']}`",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Qdrant-backed exact-ref runtime falsifier.")
    parser.add_argument("--pack-json", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = asyncio.run(run_runtime_falsifier(pack_path=args.pack_json, corpus_dir=args.corpus_dir))
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
