#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SUPERVISOR_DIR="$RUN_ROOT/supervisor"
LOG="$SUPERVISOR_DIR/finalizer.log"
STATUS="$SUPERVISOR_DIR/finalizer.status"
PID_FILE="$SUPERVISOR_DIR/finalizer.pid"
QUEUE_PID_FILES=(
  "$SUPERVISOR_DIR/baseline_relaunch_gpu0.pid"
  "$SUPERVISOR_DIR/baseline_relaunch_gpu1.pid"
  "$SUPERVISOR_DIR/baseline_retry_gpu0.pid"
  "$SUPERVISOR_DIR/baseline_retry_gpu1.pid"
)

mkdir -p "$SUPERVISOR_DIR"
printf '%s\n' "$$" > "$PID_FILE"
printf 'WAITING\n' > "$STATUS"

cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

queue_is_live() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" \
    | grep -Eq 'run_baseline_dual_gpu_queue\.sh|run_failed_baseline_retry_queue\.sh'
}

printf '[%s] finalizer started\n' "$(date -Is)" >> "$LOG"
while true; do
  live=0
  for pid_file in "${QUEUE_PID_FILES[@]}"; do
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      if queue_is_live "$pid"; then
        live=$((live + 1))
      fi
    else
      live=$((live + 1))
    fi
  done
  (( live == 0 )) && break
  sleep "$POLL_SECONDS"
done

printf 'FINALIZING\n' > "$STATUS"
printf '[%s] queues stopped; re-profiling CRIPO at fixed batch sizes\n' \
  "$(date -Is)" >> "$LOG"
cd "$PROJECT_ROOT"
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

profile_dataset() {
  local dataset="$1"
  local gpu="$2"
  local seed
  for seed in 10 101 1010 10101 101010; do
    "$PROJECT_ROOT/scripts/efficiency/run_cripo_efficiency_profile.sh" \
      "$dataset" "$seed" "$gpu"
  done
}

profile_dataset tumor 0 >> "$LOG" 2>&1 &
tumor_profile_pid=$!
profile_dataset mimic 1 >> "$LOG" 2>&1 &
mimic_profile_pid=$!
profile_failed=0
wait "$tumor_profile_pid" || profile_failed=1
wait "$mimic_profile_pid" || profile_failed=1
if (( profile_failed != 0 )); then
  printf 'NEEDS_ATTENTION\n' > "$STATUS"
  printf '[%s] fixed-batch CRIPO profiling failed\n' "$(date -Is)" >> "$LOG"
  exit 1
fi

printf '[%s] fixed-batch profiles complete; generating artifacts\n' \
  "$(date -Is)" >> "$LOG"
python scripts/efficiency/normalize_efficiency_metadata.py >> "$LOG" 2>&1
python scripts/efficiency/summarize_efficiency.py "$RUN_ROOT" >> "$LOG" 2>&1
python scripts/efficiency/render_efficiency_table.py \
  "$RUN_ROOT/efficiency_summary.csv" \
  "$RUN_ROOT/efficiency_table.tex" >> "$LOG" 2>&1

if python scripts/efficiency/audit_efficiency.py "$RUN_ROOT" >> "$LOG" 2>&1; then
  printf 'COMPLETED\n' > "$STATUS"
  printf '[%s] strict audit passed\n' "$(date -Is)" >> "$LOG"
else
  printf 'NEEDS_ATTENTION\n' > "$STATUS"
  printf '[%s] strict audit failed; inspection required\n' "$(date -Is)" >> "$LOG"
fi
