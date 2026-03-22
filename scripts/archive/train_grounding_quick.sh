#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT_ROOT="${GROUNDING_MODEL_ROOT:-$ROOT_DIR}"
DATA_DIR="${GROUNDING_DATA_DIR:-$ROOT_DIR/data/derived/grounding_ml/v1}"
ROUTER_DIR="$ARTIFACT_ROOT/models/grounding_router/v1_quick"
SCORER_DIR="$ARTIFACT_ROOT/models/page_scorer/v1_quick"
EVAL_PATH="$ARTIFACT_ROOT/models/grounding_eval/v1_quick_metrics.json"

if [[ ! -f "$DATA_DIR/train.jsonl" || ! -f "$DATA_DIR/dev.jsonl" ]]; then
  uv run python "$ROOT_DIR/scripts/export_grounding_ml_dataset.py" --output-dir "$DATA_DIR"
fi

uv run --extra dev python "$ROOT_DIR/scripts/train_grounding_router.py" \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --dev-jsonl "$DATA_DIR/dev.jsonl" \
  --output-dir "$ROUTER_DIR" \
  --max-train-rows 40 \
  --max-dev-rows 10

uv run --extra dev python "$ROOT_DIR/scripts/train_page_scorer.py" \
  --train-jsonl "$DATA_DIR/train.jsonl" \
  --dev-jsonl "$DATA_DIR/dev.jsonl" \
  --output-dir "$SCORER_DIR" \
  --max-train-rows 40 \
  --max-dev-rows 10

uv run --extra dev python "$ROOT_DIR/scripts/eval_grounding_models.py" \
  --dev-jsonl "$DATA_DIR/dev.jsonl" \
  --router-model "$ROUTER_DIR/router.joblib" \
  --page-scorer-model "$SCORER_DIR/page_scorer.joblib" \
  --output-path "$EVAL_PATH" \
  --max-dev-rows 10
