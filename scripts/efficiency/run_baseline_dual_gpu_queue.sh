#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
RUNNER="$PROJECT_ROOT/scripts/efficiency/run_baseline_efficiency_train.sh"
SEEDS=(10 101 1010 10101 101010)
QUEUE_ID="${1:?gpu index or rebalanced queue id}"
GPU="$QUEUE_ID"
SUPERVISOR_LOG="$RUN_ROOT/supervisor/baseline_gpu${QUEUE_ID}.log"

mkdir -p "$(dirname "$SUPERVISOR_LOG")"

if [[ -n "${WAIT_PID:-}" ]]; then
  printf '[%s] waiting for retained task pid=%s queue=%s\n' \
    "$(date -Is)" "$WAIT_PID" "$QUEUE_ID" >> "$SUPERVISOR_LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

EXPLICIT_TASKS=()
if [[ "$QUEUE_ID" == "0" ]]; then
  TASK_GROUPS=(
    "tumor gift"
    "mimic scrl"
    "tumor ct"
    "mimic crn"
    "mimic gift"
  )
elif [[ "$QUEUE_ID" == "1" ]]; then
  TASK_GROUPS=(
    "tumor scrl"
    "tumor vcip"
    "tumor crn"
    "tumor rmsn"
    "tumor actin"
    "mimic vcip"
    "mimic rmsn"
    "mimic actin"
    "mimic ct"
  )
elif [[ "$QUEUE_ID" == "r0" ]]; then
  GPU=0
  EXPLICIT_TASKS=(
    "mimic actin 10"
    "mimic actin 101"
    "mimic actin 1010"
    "mimic actin 10101"
    "mimic actin 101010"
    "tumor ct 10"
    "tumor ct 101"
    "tumor ct 1010"
    "tumor ct 10101"
    "tumor ct 101010"
  )
elif [[ "$QUEUE_ID" == "r1" ]]; then
  GPU=1
  EXPLICIT_TASKS=(
    "mimic rmsn 10101"
    "mimic rmsn 101010"
    "mimic ct 10"
    "mimic ct 101"
    "mimic ct 1010"
    "mimic ct 10101"
    "mimic ct 101010"
  )
else
  printf 'unsupported queue=%s\n' "$QUEUE_ID" >&2
  exit 2
fi

run_task() {
  local dataset="$1"
  local model="$2"
  local seed="$3"
  local status_file="$RUN_ROOT/$dataset/$model/seed_$seed/status.txt"
  while [[ -s "$status_file" ]] \
    && [[ "$(cat "$status_file")" == "RUNNING" ]]; do
    printf '[%s] wait running gpu=%s dataset=%s model=%s seed=%s\n' \
      "$(date -Is)" "$GPU" "$dataset" "$model" "$seed" >> "$SUPERVISOR_LOG"
    sleep 30
  done
  if [[ -s "$status_file" ]] \
    && [[ "$(cat "$status_file")" == "COMPLETED" ]]; then
    printf '[%s] skip completed gpu=%s dataset=%s model=%s seed=%s\n' \
      "$(date -Is)" "$GPU" "$dataset" "$model" "$seed" >> "$SUPERVISOR_LOG"
    return
  fi
    printf '[%s] start gpu=%s dataset=%s model=%s seed=%s\n' \
      "$(date -Is)" "$GPU" "$dataset" "$model" "$seed" >> "$SUPERVISOR_LOG"
    if "$RUNNER" "$dataset" "$model" "$seed" "$GPU"; then
      printf '[%s] complete gpu=%s dataset=%s model=%s seed=%s\n' \
        "$(date -Is)" "$GPU" "$dataset" "$model" "$seed" >> "$SUPERVISOR_LOG"
    else
      status=$?
      printf '[%s] failed=%s gpu=%s dataset=%s model=%s seed=%s\n' \
        "$(date -Is)" "$status" "$GPU" "$dataset" "$model" "$seed" >> "$SUPERVISOR_LOG"
    fi
}

if (( ${#EXPLICIT_TASKS[@]} )); then
  for task in "${EXPLICIT_TASKS[@]}"; do
    read -r dataset model seed <<<"$task"
    run_task "$dataset" "$model" "$seed"
  done
else
  for group in "${TASK_GROUPS[@]}"; do
    read -r dataset model <<<"$group"
    for seed in "${SEEDS[@]}"; do
      run_task "$dataset" "$model" "$seed"
    done
  done
fi

printf '[%s] baseline queue complete gpu=%s queue=%s\n' \
  "$(date -Is)" "$GPU" "$QUEUE_ID" \
  >> "$SUPERVISOR_LOG"
