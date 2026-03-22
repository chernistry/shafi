#!/bin/bash
export $(grep -v '^#' profiles/private_v9_rerank12.env | xargs)
export PYTHONPATH=$(pwd)/src

pdf_dir=$(pwd)/data/private
find -L "$pdf_dir" -maxdepth 1 -name "*.pdf" | sort > data/private_file_list.txt
total=$(wc -l < data/private_file_list.txt)

MANIFEST_DIR="$(pwd)/platform_runs/private/manifests/reliable_final"
mkdir -p "$MANIFEST_DIR"

echo "Starting RELIABLE V6 ingest..."

# Read file list into an array for direct access
IFS=$'\n' read -d '' -r -a files < data/private_file_list.txt

for ((i=0; i<total; i++)); do
    file="${files[$i]}"
    echo "[index $i] --- File: $file ---"
    
    # Check if already in manifest (skip without starting pipeline if possible)
    # Actually just let the pipeline handle it, it's fast.
    
    batch_dir="$(pwd)/data/private_file_$i"
    mkdir -p "$batch_dir"
    cp "$file" "$batch_dir/"
    
    # Run pipeline for THIS SINGLE FILE
    echo "Running pipeline for $file..."
    INGEST_MANIFEST_DIR="$MANIFEST_DIR" timeout 300 uv run python -m rag_challenge.ingestion.pipeline --doc-dir "$batch_dir"
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "Pipeline failed with exit code $exit_code for file $i. Continuing anyway."
    fi
    
    rm -rf "$batch_dir"
    
    # Points check every 10
    if [ $((i % 10)) -eq 0 ]; then
        POINTS=$(curl -s http://localhost:6333/collections/legal_chunks_private_1792 | grep -oE '"points_count"\s*:\s*[0-9]+' | grep -oE '[0-9]+' | head -n 1)
        echo "Progress: $i/$total files. Qdrant points: $POINTS"
    fi
done

echo "ALL FILES FINISHED"
