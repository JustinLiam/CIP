#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/liam/pythonProject/VCIP-ICML-main

TRAIN_RUN_ROOT="${1:?Usage: $0 TRAIN_RUN_ROOT EVAL_ROOT TARGET_FILE}"
EVAL_ROOT="${2:?missing EVAL_ROOT}"
TARGET_FILE="${3:?missing TARGET_FILE}"
mkdir -p "$EVAL_ROOT/logs"

log(){ printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$EVAL_ROOT/logs/dual_target_supervisor.log"; }

log "dual target evaluation start"
bash scripts/epi_abm/run_cripo_external_target_variant.sh \
  "$TRAIN_RUN_ROOT" "$EVAL_ROOT" factual_replay_final \
  factual_final 1.0 0 "$TARGET_FILE" &
pid_factual=$!
bash scripts/epi_abm/run_cripo_external_target_variant.sh \
  "$TRAIN_RUN_ROOT" "$EVAL_ROOT" half_factual_replay_final \
  half_factual_final 0.5 1 "$TARGET_FILE" &
pid_half=$!
printf '%s\n' "$pid_factual" > "$EVAL_ROOT/logs/factual_replay_final.pid"
printf '%s\n' "$pid_half" > "$EVAL_ROOT/logs/half_factual_replay_final.pid"

failed=0
if ! wait "$pid_factual"; then
  log "factual_replay_final failed"
  failed=1
fi
if ! wait "$pid_half"; then
  log "half_factual_replay_final failed"
  failed=1
fi
(( failed == 0 )) || { log "dual target evaluation failed; logs preserved"; exit 1; }
log "dual target evaluation complete"
