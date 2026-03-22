#!/bin/bash
set -e
export $(grep -v '^#' profiles/private_v9_rerank12.env | xargs)
find . -name "*manifest.json" -delete
exec uv run python -m rag_challenge.ingestion.pipeline --doc-dir data/private
