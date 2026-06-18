#!/usr/bin/env bash
# Launch em_iql_lr_ablation in tmux, watch for completion, auto-analyze summary.csv.
#
# Usage (from repo root):
#   bash scripts/cancer/train/run_em_iql_lr_ablation_autotmux.sh [GAMMA] [NUM_GPUS]
#
# Optional env (forwarded to worker jobs):
#   GRID_MLFLOW_URI=http://host:5000
#   GRID_MLFLOW_EXPERIMENT=em_iql_lr_ablation
#   GRID_FORCE=1
#   GRID_SEEDS="2 20 202 2020 20202 202020"
#
# Creates tmux session "lr_ablation" with windows:
#   gpu0, gpu1, ...  - one worker per GPU
#   watch            - polls .done count and runs analysis when all 12 jobs finish
#
# Detach: Ctrl-b d   Reattach: tmux attach -t lr_ablation

set -euo pipefail

GAMMA="${1:-4}"
NUM_GPUS="${2:-2}"
SESSION="${LR_ABLATION_TMUX_SESSION:-lr_ablation}"
TOTAL_JOBS=12

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${ROOT}/grid_results/em_iql_lr_ablation/gamma_${GAMMA}"
WATCH_LOG="${GRID_ROOT}/watch.log"
ANALYSIS_LOG="${GRID_ROOT}/analysis_report.txt"
RUN_SH="${ROOT}/scripts/cancer/train/run_em_iql_lr_ablation.sh"
COLLECT_PY="${ROOT}/scripts/cancer/train/collect_lr_ablation_summary.py"
ANALYZE_PY="${ROOT}/scripts/cancer/train/analyze_lr_ablation_results.py"

mkdir -p "${GRID_ROOT}/done" "${GRID_ROOT}/logs"

MLFLOW_URI_EXPORT=""
if [[ -n "${GRID_MLFLOW_URI:-}" ]]; then
  MLFLOW_URI_EXPORT="export GRID_MLFLOW_URI='${GRID_MLFLOW_URI}';"
fi
MLFLOW_EXP_EXPORT="export GRID_MLFLOW_EXPERIMENT='${GRID_MLFLOW_EXPERIMENT:-em_iql_lr_ablation}';"
FORCE_EXPORT=""
if [[ "${GRID_FORCE:-0}" == "1" ]]; then
  FORCE_EXPORT="export GRID_FORCE=1;"
fi
SEEDS_EXPORT=""
if [[ -n "${GRID_SEEDS:-}" ]]; then
  SEEDS_EXPORT="export GRID_SEEDS='${GRID_SEEDS}';"
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found. Install tmux or run run_em_iql_lr_ablation.sh directly." >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session '${SESSION}' already exists." >&2
  echo "  tmux attach -t ${SESSION}" >&2
  echo "  tmux kill-session -t ${SESSION}   # to restart" >&2
  exit 1
fi

echo "Starting tmux session '${SESSION}' with ${NUM_GPUS} GPU worker(s), ${TOTAL_JOBS} total jobs."
echo "  grid_root=${GRID_ROOT}"
echo "  watch_log=${WATCH_LOG}"
echo "  analysis_report=${ANALYSIS_LOG}"

tmux new-session -d -s "${SESSION}" -n "gpu0" "bash -lc '
  eval \"\$(conda shell.bash hook)\"
  conda activate vcip
  cd \"${ROOT}\"
  ${MLFLOW_URI_EXPORT}
  ${MLFLOW_EXP_EXPORT}
  ${FORCE_EXPORT}
  ${SEEDS_EXPORT}
  export GRID_WORKER_ID=0 GRID_NUM_WORKERS=${NUM_GPUS}
  echo \"[gpu0] worker 0/${NUM_GPUS} starting...\"
  bash \"${RUN_SH}\" 0 ${GAMMA}
  echo \"[gpu0] worker finished. Press Enter to close.\"
  read
'"

for (( gpu=1; gpu<NUM_GPUS; gpu++ )); do
  tmux new-window -t "${SESSION}" -n "gpu${gpu}" "bash -lc '
    eval \"\$(conda shell.bash hook)\"
    conda activate vcip
    cd \"${ROOT}\"
    ${MLFLOW_URI_EXPORT}
    ${MLFLOW_EXP_EXPORT}
    ${FORCE_EXPORT}
    ${SEEDS_EXPORT}
    export GRID_WORKER_ID=${gpu} GRID_NUM_WORKERS=${NUM_GPUS}
    echo \"[gpu${gpu}] worker ${gpu}/${NUM_GPUS} starting...\"
    bash \"${RUN_SH}\" ${gpu} ${GAMMA}
    echo \"[gpu${gpu}] worker finished. Press Enter to close.\"
    read
  '"
done

tmux new-window -t "${SESSION}" -n "watch" "bash -lc '
  eval \"\$(conda shell.bash hook)\"
  conda activate vcip
  cd \"${ROOT}\"
  GRID_ROOT=\"${GRID_ROOT}\"
  TOTAL=${TOTAL_JOBS}
  WATCH_LOG=\"${WATCH_LOG}\"
  ANALYSIS_LOG=\"${ANALYSIS_LOG}\"
  COLLECT_PY=\"${COLLECT_PY}\"
  ANALYZE_PY=\"${ANALYZE_PY}\"
  GAMMA=${GAMMA}

  exec > >(tee -a \"\${WATCH_LOG}\") 2>&1
  echo \"=== lr ablation watch started \$(date -Iseconds) ===\"
  echo \"waiting for \${TOTAL} done markers in \${GRID_ROOT}/done/\"

  while true; do
    done_n=\$(find \"\${GRID_ROOT}/done\" -name \"*.done\" 2>/dev/null | wc -l | tr -d \" \")
    running=\$(pgrep -fc \"run_em_iql_lr_ablation.sh|train_ct_iql_em.py.*em_iql_lr_ablation\" 2>/dev/null || true)
    echo \"[\$(date +%H:%M:%S)] done=\${done_n}/\${TOTAL} running_procs=\${running}\"

    python \"\${COLLECT_PY}\" --grid-root \"\${GRID_ROOT}\" --gamma \"\${GAMMA}\" 2>/dev/null || true

    if (( done_n >= TOTAL )); then
      echo \"=== all jobs done at \$(date -Iseconds) ===\"
      python \"\${COLLECT_PY}\" --grid-root \"\${GRID_ROOT}\" --gamma \"\${GAMMA}\"
      python \"\${ANALYZE_PY}\" --grid-root \"\${GRID_ROOT}\" --output \"\${ANALYSIS_LOG}\"
      echo \"=== analysis written to \${ANALYSIS_LOG} ===\"
      cat \"\${ANALYSIS_LOG}\"
      echo \"=== watch complete. Press Enter to close. ===\"
      read
      break
    fi

    if (( running == 0 )) && (( done_n < TOTAL )); then
      echo \"WARN: no training processes but only \${done_n}/\${TOTAL} done.\"
      echo \"      Check gpu* windows for errors. Retrying watch in 60s...\"
    fi
    sleep 60
  done
'"

tmux select-window -t "${SESSION}:watch"
echo ""
echo "tmux session '${SESSION}' started."
echo "  attach:  tmux attach -t ${SESSION}"
echo "  windows: gpu0..gpu$((NUM_GPUS-1)) (workers), watch (auto-analyze)"
echo "  tail:    tail -f ${WATCH_LOG}"
