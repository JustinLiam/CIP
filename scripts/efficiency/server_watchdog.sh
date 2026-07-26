#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
MAX_CYCLES="${MAX_CYCLES:-180}"
LOG="$RUN_ROOT/supervisor/server_watchdog.log"
PID_FILE="$RUN_ROOT/supervisor/server_watchdog.pid"

mkdir -p "$RUN_ROOT/supervisor"
printf '%s\n' "$$" > "$PID_FILE"

cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

for ((cycle=1; cycle<=MAX_CYCLES; cycle++)); do
  {
    printf '=== %s cycle=%d ===\n' "$(date -Is)" "$cycle"
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader
    printf 'workers='
    pgrep -fc 'train_ct_iql_em.py|experiments/train.py|profile_cripo_efficiency.py' || true
    printf 'queues='
    pgrep -fc 'run_cripo_dual_gpu_queue.sh|run_baseline_dual_gpu_queue.sh|run_after_cripo.sh' || true
    find "$RUN_ROOT" -name status.txt -type f -print0 \
      | xargs -0 -r cat \
      | sort \
      | uniq -c \
      | sed 's/^/status /'
    error_count="$(
      {
        find "$RUN_ROOT" -path '*/logs/train.log' -type f -print0 \
          | xargs -0 -r rg -l \
            'Traceback|RuntimeError|CUDA out of memory|Error executing job|Killed' \
          || true
      } | wc -l
    )"
    printf 'logs_with_error_signatures=%s\n' "$error_count"
  } >> "$LOG" 2>&1

  if [[ -f "$RUN_ROOT/audit_report.json" ]] \
    && python - "$RUN_ROOT/audit_report.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") else 1)
PY
  then
    printf '=== %s all tasks passed audit; watchdog stopping ===\n' \
      "$(date -Is)" >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done

printf '=== %s max cycles reached; watchdog stopping ===\n' "$(date -Is)" >> "$LOG"
