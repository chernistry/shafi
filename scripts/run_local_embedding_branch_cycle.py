from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import httpx


def _probe_ollama_dimension(*, base_url: str, model: str, timeout_s: float) -> int:
    base = base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=timeout_s) as client:
        resp = client.post("/api/embed", json={"model": model, "input": "dimension probe"})
        if resp.status_code == 404:
            resp = client.post("/api/embeddings", json={"model": model, "prompt": "dimension probe"})
        resp.raise_for_status()
        data = resp.json()
    if isinstance(data, dict):
        data_dict = cast("dict[str, object]", data)
        embeddings = data_dict.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return len(cast("list[object]", embeddings[0]))
        embedding = data_dict.get("embedding")
        if isinstance(embedding, list):
            return len(cast("list[object]", embedding))
        data_rows = data_dict.get("data")
        if isinstance(data_rows, list) and data_rows and isinstance(data_rows[0], dict):
            row = cast("dict[str, object]", data_rows[0])
            embedding_row = row.get("embedding")
            if isinstance(embedding_row, list):
                return len(cast("list[object]", embedding_row))
    raise ValueError(f"Unable to determine embedding dimension for model={model!r}")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _phase_collection_name(*, collection_prefix: str, phase: str) -> str:
    normalized_prefix = collection_prefix.strip()
    normalized_phase = phase.strip().lower()
    if not normalized_prefix:
        raise ValueError("collection_prefix must be non-empty")
    if not normalized_phase:
        raise ValueError("phase must be non-empty")
    return f"{normalized_prefix}_{normalized_phase}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline local-embedding branch cycle with Ollama.")
    parser.add_argument("--model", default="embeddinggemma:latest")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--artifact-suffix", required=True)
    parser.add_argument("--doc-dir", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--baseline-raw-results", required=True)
    parser.add_argument("--baseline-scaffold", required=True)
    parser.add_argument("--baseline-preflight", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--exactness-report", required=True)
    parser.add_argument("--backlog-dir", required=True)
    parser.add_argument("--ledger-json", required=True)
    parser.add_argument("--seed-qids-file", required=True)
    parser.add_argument("--research-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--query-concurrency", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    dim = _probe_ollama_dimension(
        base_url=args.ollama_base_url,
        model=args.model,
        timeout_s=float(args.timeout_s),
    )

    env = dict(os.environ)
    env["QDRANT_URL"] = env.get("QDRANT_URL", "http://localhost:6333")
    phase = str(env.get("EVAL_PHASE", "warmup")).strip().lower() or "warmup"
    phase_collection = _phase_collection_name(collection_prefix=args.collection, phase=phase)
    env["EMBED_PROVIDER"] = "ollama"
    env["EMBED_MODEL"] = args.model
    env["EMBED_OLLAMA_BASE_URL"] = args.ollama_base_url
    env["EMBED_DIMENSIONS"] = str(dim)
    env["QDRANT_COLLECTION"] = phase_collection
    env["EVAL_COLLECTION_PREFIX"] = args.collection
    env["INGEST_INGEST_VERSION"] = args.artifact_suffix

    ingest_cmd = [
        sys.executable,
        "-m",
        "rag_challenge.ingestion.pipeline",
        "--doc-dir",
        str(Path(args.doc_dir).resolve()),
    ]
    _run(ingest_cmd, cwd=root, env=env)

    cycle_cmd = [
        sys.executable,
        "scripts/run_offline_hypothesis_cycle.py",
        "--artifact-suffix",
        args.artifact_suffix,
        "--baseline-label",
        args.baseline_label,
        "--baseline-submission",
        args.baseline_submission,
        "--baseline-raw-results",
        args.baseline_raw_results,
        "--baseline-scaffold",
        args.baseline_scaffold,
        "--baseline-preflight",
        args.baseline_preflight,
        "--benchmark",
        args.benchmark,
        "--seed-qids-file",
        args.seed_qids_file,
        "--leaderboard",
        args.leaderboard,
        "--team",
        args.team,
        "--exactness-report",
        args.exactness_report,
        "--backlog-dir",
        args.backlog_dir,
        "--ledger-json",
        args.ledger_json,
        "--query-concurrency",
        str(args.query_concurrency),
        "--research-dir",
        args.research_dir,
    ]
    _run(cycle_cmd, cwd=root, env=env)

    summary = {
        "submission_policy": "NO_SUBMIT_WITHOUT_USER_APPROVAL",
        "model": args.model,
        "collection_prefix": args.collection,
        "phase_collection": phase_collection,
        "phase": phase,
        "artifact_suffix": args.artifact_suffix,
        "embedding_dimensions": dim,
        "research_dir": str(Path(args.research_dir).resolve()),
    }
    summary_path = Path(args.research_dir).resolve() / "local_embedding_cycle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
