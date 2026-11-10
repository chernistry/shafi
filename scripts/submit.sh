#!/bin/bash
#
# Tzur Labs — Competition Submission Script
# Usage: bash scripts/submit.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Files ──────────────────────────────────────────────────────────
PRIMARY="data/submit/tzur_labs_primary.json"
SECONDARY="data/submit/tzur_labs_secondary.json"
CODE="data/submit/tzur_labs_code.zip"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║      TZUR LABS — COMPETITION SUBMISSION      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Preflight ──────────────────────────────────────────────────────
echo -e "${CYAN}[1/4] Preflight checks...${NC}"

FAIL=0
for f in "$PRIMARY" "$SECONDARY" "$CODE"; do
  if [ ! -f "$f" ]; then
    echo -e "  ${RED}✗ MISSING: $f${NC}"
    FAIL=1
  else
    SIZE=$(du -h "$f" | cut -f1)
    echo -e "  ${GREEN}✓${NC} $f ($SIZE)"
  fi
done

if [ "$FAIL" -eq 1 ]; then
  echo -e "\n${RED}ABORT: Missing files.${NC}"
  exit 1
fi

# Check .env has API key
if ! grep -q "EVAL_API_KEY" .env 2>/dev/null; then
  echo -e "  ${RED}✗ No EVAL_API_KEY in .env${NC}"
  exit 1
fi
echo -e "  ${GREEN}✓${NC} EVAL_API_KEY found in .env"

# Quick validate primary
COUNT=$(python3 -c "import json; print(len(json.load(open('$PRIMARY'))['answers']))")
if [ "$COUNT" -ne 900 ]; then
  echo -e "  ${RED}✗ PRIMARY has $COUNT answers (need 900)${NC}"
  exit 1
fi
echo -e "  ${GREEN}✓${NC} PRIMARY: $COUNT answers"

# Quick validate secondary
COUNT2=$(python3 -c "import json; print(len(json.load(open('$SECONDARY'))['answers']))")
if [ "$COUNT2" -ne 900 ]; then
  echo -e "  ${RED}✗ SECONDARY has $COUNT2 answers (need 900)${NC}"
  exit 1
fi
echo -e "  ${GREEN}✓${NC} SECONDARY: $COUNT2 answers"

# Check code archive for secrets
SECRETS=$(unzip -l "$CODE" 2>/dev/null | grep -iE '\.env$|\.env\.local$' | grep -v example || true)
if [ -n "$SECRETS" ]; then
  echo -e "  ${RED}✗ SECRETS IN CODE ARCHIVE:${NC}"
  echo "  $SECRETS"
  exit 1
fi
echo -e "  ${GREEN}✓${NC} Code archive: clean (no secrets)"

echo -e "\n${GREEN}All preflight checks passed.${NC}"

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[2/4] Submission plan:${NC}"
echo ""
echo -e "  ${BOLD}Submit #1 (primary):${NC}"
echo -e "    Answers: $PRIMARY"
echo -e "    Code:    $CODE"
echo ""
echo -e "  ${BOLD}Submit #2 (secondary):${NC}"
echo -e "    Answers: $SECONDARY"
echo -e "    Code:    $CODE"
echo ""
echo -e "  Platform: https://platform.agentic-challenge.ai/api/v1"
echo ""

# ── Submit #1 ──────────────────────────────────────────────────────
echo -e "${CYAN}[3/4] Submit #1 — PRIMARY${NC}"
echo -e "${YELLOW}This will upload tzur_labs_primary.json to the competition platform.${NC}"
echo ""
read -p "$(echo -e ${BOLD}Press ENTER to submit primary, or Ctrl+C to abort: ${NC})"

echo -e "\n${CYAN}Uploading primary submission...${NC}"
EVAL_PHASE=final uv run python -m rag_challenge.submission.platform \
  --submit-existing \
  --submission-path "$PRIMARY" \
  --code-archive-path "$CODE" \
  --force-submit-existing \
  --poll

echo -e "\n${GREEN}✓ Primary submission uploaded.${NC}"

# ── Submit #2 ──────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4/4] Submit #2 — SECONDARY${NC}"
echo -e "${YELLOW}This will upload tzur_labs_secondary.json to the competition platform.${NC}"
echo ""
read -p "$(echo -e ${BOLD}Press ENTER to submit secondary, or Ctrl+C to skip: ${NC})"

echo -e "\n${CYAN}Uploading secondary submission...${NC}"
EVAL_PHASE=final uv run python -m rag_challenge.submission.platform \
  --submit-existing \
  --submission-path "$SECONDARY" \
  --code-archive-path "$CODE" \
  --force-submit-existing \
  --poll

echo -e "\n${GREEN}✓ Secondary submission uploaded.${NC}"

# ── Done ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║          BOTH SUBMISSIONS UPLOADED           ║${NC}"
echo -e "${BOLD}║   Check platform for evaluation status.      ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""
