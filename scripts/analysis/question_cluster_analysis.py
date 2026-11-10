"""
Question Cluster Analysis — EYAL ticket 3004

Embeds 900 private questions with Kanon-2, clusters via K-means,
joins with V15_HYBRID performance metrics, identifies weakest clusters.

Usage:
    uv run python scripts/question_cluster_analysis.py
    uv run python scripts/question_cluster_analysis.py --load-embeddings embeddings_cache.npy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("cluster_analysis")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_questions() -> list[dict[str, Any]]:
    path = ROOT / "dataset/private/questions.json"
    with open(path) as f:
        return json.load(f)


def load_v15_hybrid_perf() -> dict[str, dict[str, Any]]:
    """Returns qid -> {ttft_ms, is_null, page_count, answer_type, answer}."""
    path = ROOT / "data/private_submission_V15_HYBRID.json"
    with open(path) as f:
        data = json.load(f)
    perf: dict[str, dict[str, Any]] = {}
    for a in data["answers"]:
        qid = a["question_id"]
        timing = a.get("telemetry", {}).get("timing", {})
        retrieval = a.get("telemetry", {}).get("retrieval", {})
        pages = retrieval.get("retrieved_chunk_pages", [])
        page_count = sum(len(p.get("page_numbers", [])) for p in pages) if pages else 0
        perf[qid] = {
            "ttft_ms": timing.get("ttft_ms", 0),
            "is_null": a.get("answer") is None,
            "no_pages": page_count == 0,
            "page_count": page_count,
            "answer_type": a.get("answer_type", ""),
            "total_time_ms": timing.get("total_time_ms", 0),
            "over_5s": timing.get("ttft_ms", 0) > 5000,
        }
    return perf


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_questions(questions: list[dict[str, Any]]) -> np.ndarray:
    """Embed all questions using Kanon-2 API (batched)."""
    from shafi.core.embedding import EmbeddingClient
    texts = [q["question"] for q in questions]
    log.info("Embedding %d questions via Kanon-2...", len(texts))
    async with EmbeddingClient() as client:  # type: ignore[attr-defined]
        vecs = await client.embed_documents(texts)
    log.info("Embedding complete. Shape: (%d, %d)", len(vecs), len(vecs[0]))
    return np.array(vecs, dtype=np.float32)


def embed_with_context_manager(questions: list[dict[str, Any]]) -> np.ndarray:
    """Wrapper that handles client lifecycle correctly."""
    from shafi.core.embedding import EmbeddingClient
    import httpx
    texts = [q["question"] for q in questions]
    log.info("Embedding %d questions via Kanon-2...", len(texts))

    async def _run() -> list[list[float]]:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {_get_api_key()}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as http_client:
            client = EmbeddingClient(client=http_client)
            return await client.embed_documents(texts)

    vecs = asyncio.run(_run())
    log.info("Embedding complete. Shape: (%d, %d)", len(vecs), len(vecs[0]))
    return np.array(vecs, dtype=np.float32)


def _get_api_key() -> str:
    from shafi.config import get_settings
    return get_settings().embedding.api_key.get_secret_value()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_questions(
    embeddings: np.ndarray,
    n_clusters: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """K-means clustering. Returns label array of length N."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    # L2-normalize before clustering (cosine similarity → Euclidean K-means)
    normed = normalize(embeddings, norm="l2")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(normed)
    log.info("Clustering complete: %d clusters", n_clusters)
    return labels


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def describe_cluster(qs: list[dict[str, Any]], perf: dict[str, dict[str, Any]]) -> str:
    """Quick text heuristic: find the most common pattern in a cluster."""
    keywords: dict[str, int] = {}
    for q in qs:
        text = q["question"].lower()
        for kw in [
            "how many", "what is", "which", "who", "when", "where", "whether",
            "did", "does", "was", "were", "has", "have", "can", "could",
            "penalty", "fine", "date", "case", "article", "section", "claimant",
            "defendant", "judge", "court", "appeal", "award", "costs", "claim",
        ]:
            if kw in text:
                keywords[kw] = keywords.get(kw, 0) + 1
    if not keywords:
        return "misc"
    top = sorted(keywords.items(), key=lambda x: -x[1])[:3]
    return " | ".join(f"{k}({v})" for k, v in top)


def analyze_clusters(
    questions: list[dict[str, Any]],
    labels: np.ndarray,
    perf: dict[str, dict[str, Any]],
    n_clusters: int,
) -> list[dict[str, Any]]:
    """Aggregate metrics per cluster."""
    # Build qid → question lookup
    qid_to_q: dict[str, dict[str, Any]] = {q["id"]: q for q in questions}

    # Group questions by cluster
    clusters: dict[int, list[dict[str, Any]]] = {i: [] for i in range(n_clusters)}
    for q, label in zip(questions, labels):
        clusters[int(label)].append(q)

    results = []
    for cid, qs in clusters.items():
        qids = [q["id"] for q in qs]
        perfs = [perf[qid] for qid in qids if qid in perf]
        if not perfs:
            continue

        ttfts = [p["ttft_ms"] for p in perfs if p["ttft_ms"] > 0]
        null_count = sum(1 for p in perfs if p["is_null"])
        nopg_count = sum(1 for p in perfs if p["no_pages"])
        over5s_count = sum(1 for p in perfs if p["over_5s"])
        type_dist: dict[str, int] = {}
        for p in perfs:
            t = p["answer_type"]
            type_dist[t] = type_dist.get(t, 0) + 1
        dominant_type = max(type_dist.items(), key=lambda x: x[1])[0] if type_dist else "?"

        results.append({
            "cluster_id": cid,
            "size": len(qs),
            "null_count": null_count,
            "null_rate": null_count / len(qs),
            "nopg_count": nopg_count,
            "nopg_rate": nopg_count / len(qs),
            "over5s_count": over5s_count,
            "over5s_rate": over5s_count / len(qs),
            "ttft_p50_ms": float(np.median(ttfts)) if ttfts else 0,
            "ttft_p95_ms": float(np.percentile(ttfts, 95)) if ttfts else 0,
            "ttft_mean_ms": float(np.mean(ttfts)) if ttfts else 0,
            "dominant_type": dominant_type,
            "type_dist": type_dist,
            "description": describe_cluster(qs, perf),
            "sample_questions": [q["question"] for q in qs[:3]],
        })

    return results


def score_cluster_weakness(c: dict[str, Any]) -> float:
    """Composite weakness score: high null/nopg/ttft = worse."""
    return (
        c["null_rate"] * 10.0          # nulls are worst
        + c["nopg_rate"] * 5.0         # no-pages bad for grounding
        + c["over5s_rate"] * 3.0       # >5s kills F coefficient
        + c["ttft_p95_ms"] / 10000.0   # normalize ttft
    )


def format_cluster_table(clusters: list[dict[str, Any]]) -> str:
    """Format clusters as markdown table sorted by weakness."""
    ranked = sorted(clusters, key=score_cluster_weakness, reverse=True)
    lines = [
        "## Question Cluster Analysis — V15_HYBRID Performance\n",
        f"| Rank | C# | Size | Null% | NoPg% | >5s% | TTFT p50 | TTFT p95 | DomType | Pattern |",
        "|------|-----|------|-------|-------|------|----------|----------|---------|---------|",
    ]
    for rank, c in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {c['cluster_id']:2d} | {c['size']:4d} | "
            f"{c['null_rate']*100:5.1f}% | {c['nopg_rate']*100:5.1f}% | "
            f"{c['over5s_rate']*100:4.1f}% | "
            f"{c['ttft_p50_ms']:7.0f}ms | {c['ttft_p95_ms']:7.0f}ms | "
            f"{c['dominant_type']:9s} | {c['description'][:50]} |"
        )
    lines.append("")

    # Top-3 weakest clusters: sample questions
    lines.append("## Weakest Clusters — Sample Questions\n")
    for rank, c in enumerate(ranked[:3], 1):
        lines.append(f"### Rank {rank}: Cluster {c['cluster_id']} (size={c['size']}, "
                     f"null={c['null_count']}, nopg={c['nopg_count']})\n")
        lines.append(f"Pattern: `{c['description']}`  |  Types: {c['type_dist']}\n")
        for i, q in enumerate(c["sample_questions"], 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-embeddings", type=Path, help="Load pre-computed embeddings (.npy)")
    parser.add_argument("--save-embeddings", type=Path, default=Path("data/question_embeddings.npy"),
                        help="Save embeddings to file")
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("data/cluster_analysis.md"))
    args = parser.parse_args()

    questions = load_questions()
    log.info("Loaded %d questions", len(questions))

    perf = load_v15_hybrid_perf()
    log.info("Loaded V15_HYBRID perf for %d questions", len(perf))

    # Embeddings
    if args.load_embeddings and args.load_embeddings.exists():
        log.info("Loading embeddings from %s", args.load_embeddings)
        embeddings = np.load(str(args.load_embeddings))
    else:
        embeddings = embed_with_context_manager(questions)
        log.info("Saving embeddings to %s", args.save_embeddings)
        np.save(str(args.save_embeddings), embeddings)

    log.info("Embeddings shape: %s", embeddings.shape)

    # Clustering
    labels = cluster_questions(embeddings, n_clusters=args.n_clusters)

    # Analysis
    clusters = analyze_clusters(questions, labels, perf, n_clusters=args.n_clusters)
    report = format_cluster_table(clusters)

    # Print to stdout
    print(report)

    # Save report
    args.output.write_text(report)
    log.info("Report saved to %s", args.output)

    # Summary for BULLETIN
    ranked = sorted(clusters, key=score_cluster_weakness, reverse=True)
    top3 = ranked[:3]
    print("\n## BULLETIN SUMMARY (copy to BULLETIN)\n")
    summary_parts = []
    for rank, c in enumerate(top3, 1):
        summary_parts.append(
            f"Cluster-{c['cluster_id']}(n={c['size']},null={c['null_count']},nopg={c['nopg_count']},"
            f"p95={c['ttft_p95_ms']:.0f}ms,pattern='{c['description'][:40]}')"
        )
    total_null = sum(c["null_count"] for c in clusters)
    total_nopg = sum(c["nopg_count"] for c in clusters)
    print(
        f"[EYAL] Cluster analysis done ({args.n_clusters} clusters, {len(questions)} questions). "
        f"Total null={total_null}, nopg={total_nopg}. "
        f"Top-3 weakest: {'; '.join(summary_parts)}"
    )


if __name__ == "__main__":
    main()
