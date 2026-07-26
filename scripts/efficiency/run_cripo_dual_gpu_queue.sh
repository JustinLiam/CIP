#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/liam/pythonProject/VCIP-ICML-main/results/efficiency_kdd26/20260726}"
QUEUE_LOG="$RUN_ROOT/supervisor"
mkdir -p "$QUEUE_LOG"

run_queue() {
  local gpu="$1"
  shift
  local spec dataset seed
  for spec in "$@"; do
    dataset="${spec%%:*}"
    seed="${spec##*:}"
    printf '[%s] start gpu=%s dataset=%s seed=%s\n' \
      "$(date -Iseconds)" "$gpu" "$dataset" "$seed" \
      | tee -a "$QUEUE_LOG/gpu${gpu}.log"
    RUN_ROOT="$RUN_ROOT" "$SCRIPT_DIR/run_cripo_efficiency_train.sh" \
      "$dataset" "$seed" "$gpu" \
      >> "$QUEUE_LOG/gpu${gpu}.log" 2>&1
    printf '[%s] complete gpu=%s dataset=%s seed=%s\n' \
      "$(date -Iseconds)" "$gpu" "$dataset" "$seed" \
      | tee -a "$QUEUE_LOG/gpu${gpu}.log"
  done
}

# MIMIC dominates runtime, so both GPUs start with MIMIC. Tumor fills the
# remaining queue after each GPU finishes its assigned MIMIC seeds.
run_queue 0 \
  mimic:10 mimic:1010 mimic:101010 \
  tumor:10 tumor:1010 tumor:101010 &
pid0=$!

run_queue 1 \
  mimic:101 mimic:10101 \
  tumor:101 tumor:10101 &
pid1=$!

printf '%s\n' "$pid0" > "$QUEUE_LOG/gpu0.pid"
printf '%s\n' "$pid1" > "$QUEUE_LOG/gpu1.pid"

wait "$pid0"
wait "$pid1"
date -Iseconds > "$QUEUE_LOG/cripo_training_complete.txt"
