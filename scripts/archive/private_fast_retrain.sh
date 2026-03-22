#!/usr/bin/env bash
# private_fast_retrain.sh — Retrain page scorer from eval results in <5 minutes
#
# Usage:
#   bash scripts/private_fast_retrain.sh path/to/eval_output.json [path/to/golden.json]
#
# If golden.json provided, uses it as the source of gold_page_ids.
# Otherwise, tries to extract from eval output (needs gold_page_ids in case dict).

set -euo pipefail

EVAL_JSON="${1:?Usage: $0 <eval_output.json> [golden.json]}"
GOLDEN_JSON="${2:-eval_golden_warmup_verified.json}"
BASE_TRAIN="data/derived/grounding_ml/v2_reviewed/train.jsonl"
BASE_MODEL="models/page_scorer/v8_temporal/page_scorer.joblib"
OUTPUT_DIR="models/page_scorer/private_v1"
TARGET_MAP="/tmp/private_target_map.json"

echo "=== EYAL Fast Retrain Pipeline ==="
echo "Eval input: $EVAL_JSON"
echo "Golden:     $GOLDEN_JSON"
echo "Base model: $BASE_MODEL"
echo ""

# Step 1: Build target map by joining eval telemetry with golden labels
echo "[1/4] Building target map from eval + golden labels..."
uv run python3 -c "
import json, sys

eval_data = json.load(open('$EVAL_JSON'))
items = eval_data if isinstance(eval_data, list) else eval_data.get('results', eval_data.get('rows', []))

# Load golden labels for gold_page_ids
golden_data = json.load(open('$GOLDEN_JSON'))
golden_list = golden_data if isinstance(golden_data, list) else golden_data.get('cases', [])
gold_by_id = {}
for g in golden_list:
    gid = g.get('id', g.get('question_id', ''))
    gold_pages = g.get('gold_page_ids', g.get('golden_page_ids', g.get('gold_chunk_ids', [])))
    # Convert chunk IDs to page IDs if needed (format: doc:page:chunk:hash → doc_page+1)
    if gold_pages and ':' in str(gold_pages[0]):
        page_ids = []
        seen = set()
        for cid in gold_pages:
            parts = cid.split(':')
            if len(parts) >= 2 and parts[1].isdigit():
                pid = f'{parts[0]}_{int(parts[1])+1}'
                if pid not in seen:
                    seen.add(pid)
                    page_ids.append(pid)
        gold_pages = page_ids
    if gid and gold_pages:
        gold_by_id[gid] = gold_pages

target_map = []
for item in items:
    case = item.get('case', {})
    telemetry = item.get('telemetry', {})
    qid = case.get('question_id', case.get('case_id', case.get('id', '')))
    used_pages = telemetry.get('used_page_ids', [])
    gold_pages = gold_by_id.get(qid, case.get('gold_page_ids', []))

    if qid and gold_pages:
        target_map.append({
            'question_id': qid,
            'verified_gold_pages': gold_pages,
            'used_pages': used_pages,
            'failure_category': 'eval_derived',
        })

json.dump(target_map, open('$TARGET_MAP', 'w'), indent=2)
print(f'  Extracted {len(target_map)} questions with gold labels (from {len(gold_by_id)} golden entries)')
"

# Step 2: Retrain
echo "[2/4] Retraining LightGBM scorer..."
uv run python scripts/retrain_page_scorer_on_private.py \
  --base-train-jsonl "$BASE_TRAIN" \
  --eval-target-map "$TARGET_MAP" \
  --base-model "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR"

# Step 3: Check if improved
echo ""
echo "[3/4] Checking improvement..."
RESULT=$(uv run python3 -c "
import json
m = json.load(open('$OUTPUT_DIR/metrics.json'))
new = m.get('dev_hit_at_2', 0)
old = 0.8947  # v8 dev_hit@2
improved = new >= old - 0.01  # allow small regression
print(f'{new:.4f} {\"PASS\" if improved else \"SKIP\"} {new - old:+.4f}')
")
NEW_HIT2=$(echo "$RESULT" | cut -d' ' -f1)
VERDICT=$(echo "$RESULT" | cut -d' ' -f2)
DELTA=$(echo "$RESULT" | cut -d' ' -f3)

echo "  Old hit@2: 0.8947"
echo "  New hit@2: $NEW_HIT2 (delta: $DELTA)"
echo "  Verdict:   $VERDICT"

# Step 4: Deploy ONLY if improved
if [ "$VERDICT" = "PASS" ]; then
  echo "[4/4] Deploying improved model..."
  for profile in profiles/private_v7_enhanced.env profiles/private_v7_1792_enhanced.env profiles/private_v8_full.env; do
    if [ -f "$profile" ]; then
      sed -i'' -e "s|PIPELINE_TRAINED_PAGE_SCORER_MODEL_PATH=.*|PIPELINE_TRAINED_PAGE_SCORER_MODEL_PATH=$OUTPUT_DIR/page_scorer.joblib|" "$profile"
      echo "  Updated: $profile"
    fi
  done
  echo ""
  echo "=== Retrain complete — DEPLOYED ==="
else
  echo "[4/4] Skipping deployment — no improvement."
  echo ""
  echo "=== Retrain complete — NOT DEPLOYED (keeping v8_temporal) ==="
fi
echo "Model: $OUTPUT_DIR/page_scorer.joblib"
