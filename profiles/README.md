# Score Profiles

These files contain only non-secret, score-affecting overrides that must be
loaded explicitly when reproducing a measured runtime profile. Keep
`.env.local` machine-only and never hide benchmark behavior there.

Usage:

```bash
set -a
source ./.env
source ./profiles/main_baseline_20260319.env
export QDRANT_URL=http://localhost:6333
set +a
uv run python -m rag_challenge.submission.platform --artifact-suffix explicit_profile
```

For the current calibrated replay path, build the answer source under
`main_baseline_20260319.env`, build the page-source challenger under its own
profile, then merge them with:

```bash
uv run python scripts/build_calibrated_replay_candidate.py \
  --phase warmup \
  --answer-source-suffix explicit_profile_main_20260319 \
  --page-source-suffix kanon2_1792_ablation_20260319
```
