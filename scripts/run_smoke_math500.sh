#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen2.5-7B}"
MODEL_TYPE="${MODEL_TYPE:-smoke}"
RUN_NAME="${RUN_NAME:-math500_smoke}"
BACKEND="${BACKEND:-vllm}"
SAMPLES_PER_PROBLEM="${SAMPLES_PER_PROBLEM:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python scripts/experiment.py \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --model-type "$MODEL_TYPE" \
  --run-name "$RUN_NAME" \
  --limit 20 \
  --samples-per-problem "$SAMPLES_PER_PROBLEM" \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --ks "1,2,4,8" \
  "$@"
