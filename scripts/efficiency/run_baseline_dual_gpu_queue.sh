#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
RUNNER="$PROJECT_ROOT/scripts/efficiency/run_baseline_efficiency_train.sh"
SEEDS=(10 101 1010 10101 101010)
GPU="${1:?gpu index}"
SUPERVISOR_LOG="$RUN_ROOT/supervisor/baseline_gpu${GPU}.log"

mkdir -p "$(dirname "$SUPERVISOR_LOG")"

if [[ "$GPU" == "0" ]]; then
  TASK_GROUPS=(
    "tumor gift"
    "mimic scrl"
    "tumor ct"
    "mimic crn"
    "mimic gift"
  )
elif [[ "$GPU" == "1" ]]; then
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
else
  printf 'unsupported gpu=%s\n' "$GPU" >&2
  exit 2
fi

for group in "${TASK_GROUPS[@]}"; do
  read -r dataset model <<<"$group"
  for seed in "${SEEDS[@]}"; do
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
  done
done

printf '[%s] baseline queue complete gpu=%s\n' "$(date -Is)" "$GPU" \
  >> "$SUPERVISOR_LOG"
