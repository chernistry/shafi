#!/bin/bash
set -e
export $(grep -v '^#' profiles/private_v9_rerank12.env | xargs)

pdf_files=(data/private/*.pdf)
total=${#pdf_files[@]}
batch_size=10

echo "Starting batched ingest of $total files..."

for ((i=0; i<total; i+=batch_size)); do
    batch=("${pdf_files[@]:i:batch_size}")
    batch_dir="data/private_batch_$i"
    mkdir -p "$batch_dir"
    cp "${batch[@]}" "$batch_dir/"
    
    echo "Processing batch $i to $((i+batch_size))..."
    # manifest skips processed files. --skip-deletion keeps existing points.
    uv run python -m rag_challenge.ingestion.pipeline --doc-dir "$batch_dir" --skip-deletion
    
    rm -rf "$batch_dir"
done

echo "BATCHED INGEST COMPLETE"
