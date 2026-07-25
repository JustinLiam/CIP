#!/usr/bin/env bash
# Adaptive two-GPU Tumor WeightNet ablation with dataset.seed=exp.seed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:-weightnet_expseed_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/ablation_studies/tumor/weightnet_alignment/noise0_seq60_expseed/${RUN_ID}}"
PROTOCOL="scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh"
SEEDS_RAW="${GRID_SEEDS:-10 101 1010 10101 101010}"
read -r -a SEEDS <<<"${SEEDS_RAW}"
DEFAULT_CONDITIONS="sinkhorn:1 sinkhorn:2 sinkhorn:3 sinkhorn:4 mmd:1 mmd:2 mmd:3 mmd:4 uniform:1 uniform:2 uniform:3 uniform:4"
read -r -a CONDITIONS <<<"${CONDITION_LIST:-${DEFAULT_CONDITIONS}}"
GPU_FREE_THRESHOLD_MB="${GPU_FREE_THRESHOLD_MB:-1024}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
SEED_PARALLELISM="${SEED_PARALLELISM:-2}"
EM_E_EPOCHS="${EM_E_EPOCHS:-10}"
EM_E_W_LR="${EM_E_W_LR:-0.003}"
EVAL_TAU_LIST="${EVAL_TAU_LIST:-1 2 3 4 5 6 7 8 9 10 11 12}"
PYTHON_BIN="${PYTHON_BIN:-/home/liam/anaconda3/envs/vcip/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${RESULT_ROOT}"
SCHEDULER_LOG="${RESULT_ROOT}/scheduler.log"
exec >>"${SCHEDULER_LOG}" 2>&1

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

gpu_used_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

variant_settings() {
  case "$1" in
    sinkhorn) printf 'true sinkhorn\n' ;;
    mmd) printf 'true mmd\n' ;;
    uniform) printf 'false sinkhorn\n' ;;
    *) return 2 ;;
  esac
}

merge_condition() {
  local condition_root=$1 variant=$2 gamma=$3
  "${PYTHON_BIN}" - "${condition_root}" "${variant}" "${gamma}" "${SEEDS[*]}" "${EVAL_TAU_LIST}" <<'PY'
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
variant = sys.argv[2]
gamma = int(sys.argv[3])
seeds = [int(value) for value in sys.argv[4].split()]
taus = [int(value) for value in sys.argv[5].split()]
rows = []
for seed in seeds:
    path = root / "seeds" / f"seed_{seed}" / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))

latest = {}
for row in rows:
    if row.get("rmse_uns") in {None, "", "NA"}:
        continue
    key = (row["split"], int(row["seed"]), int(row["eval_tau"]))
    latest[key] = row

expected = {(split, seed, tau) for split in ("val", "test") for seed in seeds for tau in taus}
if set(latest) != expected:
    raise RuntimeError(f"Incomplete {variant=} {gamma=}: missing={sorted(expected - set(latest))}")

ordered = [latest[key] for key in sorted(latest, key=lambda key: (key[0], key[1], key[2]))]
summary_path = root / "summary.csv"
tmp_path = root / ".summary.csv.tmp"
with tmp_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
    writer.writeheader()
    writer.writerows(ordered)
os.replace(tmp_path, summary_path)

raw_path = root / "val_test_rmse_by_seed.csv"
with raw_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["variant", "gamma", "split", "seed", "dataset_seed", "tau", "rmse_uns"])
    for row in ordered:
        writer.writerow([variant, gamma, row["split"], row["seed"], row["dataset_seed"], row["eval_tau"], row["rmse_uns"]])

groups = defaultdict(list)
for row in ordered:
    groups[(row["split"], int(row["eval_tau"]))].append(float(row["rmse_uns"]))
aggregate_path = root / "val_test_rmse_summary.csv"
with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["variant", "gamma", "split", "tau", "n", "mean_rmse", "std_rmse"])
    for (split, tau), values in sorted(groups.items()):
        writer.writerow([variant, gamma, split, tau, len(values), statistics.mean(values), statistics.stdev(values)])
print(f"merged {len(ordered)} rows into {summary_path}")
PY
}

run_condition() {
  local variant=$1 gamma=$2 gpu=$3 use_weight_net align_loss seed seed_root pid failed=0
  local condition_root="${RESULT_ROOT}/${variant}/kappa_${gamma}"
  local -a pids=()
  read -r use_weight_net align_loss <<<"$(variant_settings "${variant}")"
  mkdir -p "${condition_root}/seeds"
  cat >"${condition_root}/experiment_meta.txt" <<META
method=CRIPO
ablation=weightnet_alignment
variant=${variant}
ct_use_weight_net=${use_weight_net}
ct_align_loss=${align_loss}
kappa=${gamma}
noise=0.0
max_seq_length=60
dataset_seed_mode=exp_seed
train_val_test=1000/200/200
exp_seeds=${SEEDS[*]}
seed_parallelism=${SEED_PARALLELISM}
eval_splits=val test
eval_tau=${EVAL_TAU_LIST}
checkpoint_selection=max_rmse_uns_tau1_2_3_4_5_6
em_e_epochs=${EM_E_EPOCHS}
em_e_w_lr=${EM_E_W_LR}
code_commit=$(git rev-parse HEAD)
started_at=$(date -Iseconds)
META

  log "Launching variant=${variant}, kappa=${gamma}, seeds=${SEEDS[*]} on GPU${gpu}."
  for seed in "${SEEDS[@]}"; do
    seed_root="${condition_root}/seeds/seed_${seed}"
    mkdir -p "${seed_root}"
    env \
      GRID_ROOT="${seed_root}" GRID_SEEDS="${seed}" \
      DATASET_SEED_MODE=exp_seed DATASET_SEED="${seed}" \
      DATASET_TRAIN=1000 DATASET_VAL=200 DATASET_TEST=200 \
      MAX_SEQ_LENGTH=60 TEST_SPLIT=both TUMOR_NOISE_SCALE=0.0 \
      CT_USE_WEIGHT_NET="${use_weight_net}" CT_ALIGN_LOSS="${align_loss}" \
      EM_E_EPOCHS="${EM_E_EPOCHS}" EM_E_W_LR="${EM_E_W_LR}" \
      EVAL_TAU_LIST="${EVAL_TAU_LIST}" USE_MLFLOW=false \
      GPU_WAIT_MEMORY_MB=76000 GPU_WAIT_SECONDS=30 FORCE=0 \
      MLFLOW_EXPERIMENT="cripo_weightnet_${variant}_kappa${gamma}_expseed" \
      bash "${PROTOCOL}" "${gpu}" "${gamma}" \
      >"${seed_root}/launcher.log" 2>&1 &
    pids+=("$!")
    if ((${#pids[@]} >= SEED_PARALLELISM)); then
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then failed=1; fi
      done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if ((failed)); then
    log "ERROR condition failed: variant=${variant}, kappa=${gamma}, GPU${gpu}."
    return 1
  fi
  merge_condition "${condition_root}" "${variant}" "${gamma}"
  date -Iseconds >"${condition_root}/DONE.txt"
  log "Completed variant=${variant}, kappa=${gamma} on GPU${gpu}."
}

log "Ablation scheduler started: RUN_ID=${RUN_ID}, root=${RESULT_ROOT}"
next=0
declare -A gpu_pid=()
declare -A gpu_label=()

while ((next < ${#CONDITIONS[@]} || ${#gpu_pid[@]} > 0)); do
  for gpu in 0 1; do
    if [[ -n "${gpu_pid[$gpu]:-}" ]] && ! kill -0 "${gpu_pid[$gpu]}" 2>/dev/null; then
      if wait "${gpu_pid[$gpu]}"; then
        log "Worker finished: ${gpu_label[$gpu]} on GPU${gpu}."
      else
        log "ERROR worker failed: ${gpu_label[$gpu]} on GPU${gpu}."
        exit 1
      fi
      unset 'gpu_pid[$gpu]' 'gpu_label[$gpu]'
    fi

    if [[ -z "${gpu_pid[$gpu]:-}" ]] && ((next < ${#CONDITIONS[@]})); then
      used="$(gpu_used_mb "${gpu}")"
      if ((used < GPU_FREE_THRESHOLD_MB)); then
        condition="${CONDITIONS[$next]}"
        variant="${condition%%:*}"
        gamma="${condition##*:}"
        run_condition "${variant}" "${gamma}" "${gpu}" &
        gpu_pid[$gpu]=$!
        gpu_label[$gpu]="${variant}:kappa${gamma}"
        next=$((next + 1))
      fi
    fi
  done
  sleep "${GPU_POLL_SECONDS}"
done

log "All WeightNet ablation conditions completed."
