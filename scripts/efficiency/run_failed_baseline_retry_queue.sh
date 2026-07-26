#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
RUNNER="$PROJECT_ROOT/scripts/efficiency/run_baseline_efficiency_train.sh"
GPU="${1:?gpu index}"
SEEDS=(10 101 1010 10101 101010)
SUPERVISOR_DIR="$RUN_ROOT/supervisor"
LOG="$SUPERVISOR_DIR/baseline_retry_gpu${GPU}.log"
PID_FILE="$SUPERVISOR_DIR/baseline_retry_gpu${GPU}.pid"
ORIGINAL_PID_FILES=(
  "$SUPERVISOR_DIR/baseline_relaunch_gpu0.pid"
  "$SUPERVISOR_DIR/baseline_relaunch_gpu1.pid"
)

mkdir -p "$SUPERVISOR_DIR"
printf '%s\n' "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

original_queue_is_live() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" \
    | grep -Fq 'run_baseline_dual_gpu_queue.sh'
}

printf '[%s] retry worker waiting gpu=%s\n' "$(date -Is)" "$GPU" >> "$LOG"
while true; do
  live=0
  for pid_file in "${ORIGINAL_PID_FILES[@]}"; do
    if [[ -s "$pid_file" ]] && original_queue_is_live "$(cat "$pid_file")"; then
      live=$((live + 1))
    fi
  done
  (( live == 0 )) && break
  sleep 60
done

if [[ "$GPU" == "0" ]]; then
  MODEL="crn"
elif [[ "$GPU" == "1" ]]; then
  MODEL="ct"
else
  printf 'unsupported gpu=%s\n' "$GPU" >&2
  exit 2
fi

for seed in "${SEEDS[@]}"; do
  printf '[%s] retry start gpu=%s dataset=tumor model=%s seed=%s env=pytorch-lightning\n' \
    "$(date -Is)" "$GPU" "$MODEL" "$seed" >> "$LOG"
  if CONDA_ENV=pytorch-lightning "$RUNNER" tumor "$MODEL" "$seed" "$GPU"; then
    printf '[%s] retry complete gpu=%s dataset=tumor model=%s seed=%s\n' \
      "$(date -Is)" "$GPU" "$MODEL" "$seed" >> "$LOG"
  else
    status=$?
    printf '[%s] retry failed=%s gpu=%s dataset=tumor model=%s seed=%s\n' \
      "$(date -Is)" "$status" "$GPU" "$MODEL" "$seed" >> "$LOG"
  fi
done

printf '[%s] retry queue complete gpu=%s\n' "$(date -Is)" "$GPU" >> "$LOG"
