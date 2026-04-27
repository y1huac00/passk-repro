#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:?set BASE_MODEL, e.g. BASE_MODEL=Qwen/Qwen2.5-7B}"
RLVR_MODEL="${RLVR_MODEL:?set RLVR_MODEL to the same-lineage RLVR checkpoint}"
BASE_RUN_NAME="${BASE_RUN_NAME:-math500_base_full}"
RLVR_RUN_NAME="${RLVR_RUN_NAME:-math500_rlvr_full}"
BACKEND="${BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python scripts/experiment.py \
  --backend "$BACKEND" \
  --model "$BASE_MODEL" \
  --model-type base \
  --run-name "$BASE_RUN_NAME" \
  --samples-per-problem 128 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 16384 \
  --batch-size "$BATCH_SIZE" \
  --ks "1,2,4,8,16,32,64,128" \
  "$@"

python scripts/experiment.py \
  --backend "$BACKEND" \
  --model "$RLVR_MODEL" \
  --model-type rlvr \
  --run-name "$RLVR_RUN_NAME" \
  --samples-per-problem 128 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-new-tokens 16384 \
  --batch-size "$BATCH_SIZE" \
  --ks "1,2,4,8,16,32,64,128" \
  "$@"
