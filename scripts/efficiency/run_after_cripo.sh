#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
GPU="${1:?gpu index}"
CRIPO_PID_FILE="$RUN_ROOT/supervisor/gpu${GPU}.pid"

while [[ ! -s "$CRIPO_PID_FILE" ]]; do
  sleep 2
done
cripo_pid="$(cat "$CRIPO_PID_FILE")"
while kill -0 "$cripo_pid" 2>/dev/null; do
  sleep 10
done

if [[ "$GPU" == "0" ]]; then
  seeds=(10 1010 101010)
else
  seeds=(101 10101)
fi
for dataset in mimic tumor; do
  for seed in "${seeds[@]}"; do
    "$PROJECT_ROOT/scripts/efficiency/run_cripo_efficiency_profile.sh" \
      "$dataset" "$seed" "$GPU"
  done
done

exec "$PROJECT_ROOT/scripts/efficiency/run_baseline_dual_gpu_queue.sh" "$GPU"
