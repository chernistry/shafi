# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportPrivateUsage=false
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag_challenge.models import RetrievedChunk


CASE_REF_PREFIXES = ("CFI", "CA", "SCT", "ENF", "DEC", "TCD", "ARB")


@dataclass(frozen=True)
class ProbeCase:
    qid: str
    question: str
    target_doc_ids: tuple[str, ...]
    miss_family: str
    route: str


def _set_probe_env_defaults(*, collection: str) -> None:
    os.environ.setdefault("ISAACUS_API_KEY", "probe-only-dummy")
    os.environ.setdefault("ZEROENTROPY_API_KEY", "probe-only-dummy")
    os.environ.setdefault("COHERE_API_KEY", "probe-only-dummy")
    os.environ["QDRANT_COLLECTION"] = collection


def _load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected question list in {path}")
    questions: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        if qid and question:
            questions[qid] = question
    return questions


def _is_case_ref(ref: str) -> bool:
    prefix = ref.split(" ", maxsplit=1)[0].upper()
    return prefix in CASE_REF_PREFIXES


def _select_target_cases(
    *,
    miss_pack_path: Path,
    questions_by_id: dict[str, str],
    exact_refs_fn: Callable[[str], list[str]],
    doc_refs_fn: Callable[[str], list[str]],
) -> list[ProbeCase]:
    payload = json.loads(miss_pack_path.read_text(encoding="utf-8"))
    cases_obj = payload.get("cases")
    if not isinstance(cases_obj, list):
        raise ValueError(f"Expected 'cases' list in {miss_pack_path}")

    selected: list[ProbeCase] = []
    seen: set[str] = set()
    for raw in cases_obj:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("qid") or "").strip()
        if not qid or qid in seen:
            continue
        question = questions_by_id.get(qid, "").strip()
        if not question:
            continue
        exact_refs = exact_refs_fn(question)
        if not exact_refs:
            continue
        doc_refs = doc_refs_fn(question)
        if any(_is_case_ref(ref) for ref in doc_refs):
            continue
        target_doc_ids = tuple(str(item).strip() for item in raw.get("target_doc_ids") or [] if str(item).strip())
        if not target_doc_ids:
            continue
        selected.append(
            ProbeCase(
                qid=qid,
                question=question,
                target_doc_ids=target_doc_ids,
                miss_family=str(raw.get("miss_family") or "").strip() or "unknown",
                route=str(raw.get("route") or "").strip() or "unknown",
            )
        )
        seen.add(qid)
    return selected


def _top_unique_doc_ids(chunks: list[RetrievedChunk], *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        ordered.append(chunk.doc_id)
        if len(ordered) >= limit:
            break
    return ordered


def _best_target_doc_rank(chunks: list[RetrievedChunk], target_doc_ids: tuple[str, ...]) -> int | None:
    target_set = set(target_doc_ids)
    seen: set[str] = set()
    rank = 0
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        rank += 1
        if chunk.doc_id in target_set:
            return rank
    return None


def _classify_outcome(*, baseline_rank: int | None, boosted_rank: int | None) -> str:
    if baseline_rank is None and boosted_rank is None:
        return "no_hit"
    if baseline_rank is None and boosted_rank is not None:
        return "hit_gained"
    if baseline_rank is not None and boosted_rank is None:
        return "hit_lost"
    assert baseline_rank is not None and boosted_rank is not None
    if boosted_rank < baseline_rank:
        return "rank_improved"
    if boosted_rank > baseline_rank:
        return "rank_worsened"
    return "rank_unchanged"


def _select_control_cases(
    *,
    questions_by_id: dict[str, str],
    target_qids: set[str],
    doc_refs_fn: Callable[[str], list[str]],
    limit: int,
) -> list[dict[str, str]]:
    controls: list[dict[str, str]] = []
    for qid, question in questions_by_id.items():
        if qid in target_qids:
            continue
        refs = doc_refs_fn(question)
        if not any(_is_case_ref(ref) for ref in refs):
            continue
        controls.append({"qid": qid, "question": question})
        if len(controls) >= limit:
            break
    return controls


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sparse-only live probe for the exact-legal-reference BM25 boost on the warm-up collection."
    )
    parser.add_argument("--questions-json", type=Path, required=True)
    parser.add_argument("--miss-pack-json", type=Path, required=True)
    parser.add_argument("--collection", default="legal_chunks_platform_warmup")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--control-limit", type=int, default=5)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args(argv)


class _DummyEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        raise RuntimeError("dense path should not be used in the sparse-only exact-legal-reference probe")


async def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    _set_probe_env_defaults(collection=str(args.collection))

    from rag_challenge.core.classifier import QueryClassifier
    from rag_challenge.core.qdrant import QdrantStore
    from rag_challenge.core.retriever import HybridRetriever

    questions_by_id = _load_questions(args.questions_json)
    target_cases = _select_target_cases(
        miss_pack_path=args.miss_pack_json,
        questions_by_id=questions_by_id,
        exact_refs_fn=QueryClassifier.extract_exact_legal_refs,
        doc_refs_fn=QueryClassifier.extract_doc_refs,
    )
    control_cases = _select_control_cases(
        questions_by_id=questions_by_id,
        target_qids={case.qid for case in target_cases},
        doc_refs_fn=QueryClassifier.extract_doc_refs,
        limit=int(args.control_limit),
    )

    result: dict[str, Any] = {
        "label": "ticket50_exact_legal_reference_live_probe",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "collection": str(args.collection),
        "limit": int(args.limit),
        "dense_api_key_present": bool(os.getenv("ISAACUS_API_KEY", "").strip() and os.getenv("ISAACUS_API_KEY") != "probe-only-dummy"),
        "target_case_count": len(target_cases),
        "control_case_count": len(control_cases),
        "target_cases": [],
        "control_cases": [],
        "summary": {},
    }

    store = QdrantStore()
    retriever = HybridRetriever(store=store, embedder=cast("Any", _DummyEmbedder()))
    try:
        result["qdrant_collection_exists"] = await store.client.collection_exists(store.collection_name)
        result["bm25_enabled"] = bool(getattr(retriever, "_bm25_enabled", False))
        result["sparse_encoder_available"] = getattr(retriever, "_sparse_encoder", None) is not None
        if not result["qdrant_collection_exists"]:
            raise RuntimeError(f"Qdrant collection {store.collection_name!r} does not exist")
        if not result["sparse_encoder_available"]:
            raise RuntimeError("BM25 sparse encoder unavailable")

        outcome_counts = {
            "hit_gained": 0,
            "hit_lost": 0,
            "rank_improved": 0,
            "rank_worsened": 0,
            "rank_unchanged": 0,
            "no_hit": 0,
            "query_changed": 0,
            "query_unchanged": 0,
        }

        for case in target_cases:
            doc_refs = QueryClassifier.extract_doc_refs(case.question)
            exact_refs = QueryClassifier.extract_exact_legal_refs(case.question)
            expanded_refs = retriever._expand_doc_ref_variants(doc_refs)
            where = retriever._build_filter(
                doc_type_filter=None,
                jurisdiction_filter=None,
                doc_refs=expanded_refs,
            )
            baseline_query = case.question.strip()
            boosted_query = HybridRetriever._build_sparse_query(query=case.question, extracted_refs=doc_refs)
            query_changed = boosted_query != baseline_query
            outcome_counts["query_changed" if query_changed else "query_unchanged"] += 1

            baseline_chunks = retriever._map_results(
                await retriever._query_sparse_only(query=baseline_query, limit=int(args.limit), where=where)
            )
            boosted_chunks = retriever._map_results(
                await retriever._query_sparse_only(query=boosted_query, limit=int(args.limit), where=where)
            )
            baseline_rank = _best_target_doc_rank(baseline_chunks, case.target_doc_ids)
            boosted_rank = _best_target_doc_rank(boosted_chunks, case.target_doc_ids)
            outcome = _classify_outcome(baseline_rank=baseline_rank, boosted_rank=boosted_rank)
            outcome_counts[outcome] += 1

            result["target_cases"].append(
                {
                    **asdict(case),
                    "doc_refs": doc_refs,
                    "exact_refs": exact_refs,
                    "baseline_query": baseline_query,
                    "boosted_query": boosted_query,
                    "query_changed": query_changed,
                    "baseline": {
                        "best_target_doc_rank": baseline_rank,
                        "top_unique_doc_ids": _top_unique_doc_ids(baseline_chunks),
                    },
                    "boosted": {
                        "best_target_doc_rank": boosted_rank,
                        "top_unique_doc_ids": _top_unique_doc_ids(boosted_chunks),
                    },
                    "outcome": outcome,
                }
            )

        for control in control_cases:
            question = control["question"]
            doc_refs = QueryClassifier.extract_doc_refs(question)
            boosted_query = HybridRetriever._build_sparse_query(query=question, extracted_refs=doc_refs)
            result["control_cases"].append(
                {
                    **control,
                    "doc_refs": doc_refs,
                    "query_changed": boosted_query != question.strip(),
                }
            )

        result["summary"] = {
            **outcome_counts,
            "controls_changed_count": sum(1 for case in result["control_cases"] if case["query_changed"]),
        }
        return result
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = asyncio.run(_run_probe(args))
    except Exception as exc:
        payload = {
            "label": "ticket50_exact_legal_reference_live_probe",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "collection": str(args.collection),
            "limit": int(args.limit),
            "runtime_blocker": str(exc),
            "python": sys.version,
        }
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1

    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
