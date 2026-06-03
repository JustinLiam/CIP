#!/usr/bin/env bash
# IQL hyperparameter grid search with per-seed CT retraining.
#
# Pipeline:
#   1) train_ct.py (once per seed) -> grid_results/.../ct_checkpoints/seed_${seed}/coeff_${gamma}/
#   2) train_iql_planner.py (per combo) -> grid_results/.../iql_ckpts_work/<tag>/
#   3) eval_iql_planner.py (optional downstream MAE on val/test)
#
# Cross-world metrics (sim vs predictor) are read from iql_val_trace.json after IQL training:
#   Spearman rho(sim, predictor), Top-1/3/5 overlap, picked step (selection_world), best step (sim).
#
# Usage (from repo root):
#   bash scripts/cancer/train/gridsearch_iql.sh [gamma] [gpu] [test_split]
#
# Optional env:
#   GRID_SKIP_CT=1           - skip CT if ct_best_encoder.pt already exists for the seed
#   GRID_SKIP_IQL=1          - only train CT (debug)
#   GRID_SEEDS="10 101"      - override SEEDS
#   GRID_WORKER_ID=0         - worker shard [0, GRID_NUM_WORKERS-1]
#   GRID_NUM_WORKERS=1
#
# Multi-GPU example:
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_iql.sh 4 0
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_iql.sh 4 1

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate vcip

gamma=${1:-4}
gpu=${2:-0}
TEST_SPLIT=${3:-false}

GRID_WORKER_ID=${GRID_WORKER_ID:-0}
GRID_NUM_WORKERS=${GRID_NUM_WORKERS:-1}
if (( GRID_NUM_WORKERS < 1 )) || (( GRID_WORKER_ID < 0 )) || (( GRID_WORKER_ID >= GRID_NUM_WORKERS )); then
  echo "ERROR: invalid sharding: GRID_WORKER_ID=${GRID_WORKER_ID} GRID_NUM_WORKERS=${GRID_NUM_WORKERS}" >&2
  exit 1
fi
echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${gamma} gpu=${gpu}"

# ---------- IQL search space ----------
IQL_TAU_LIST=("0.5")
IQL_BETA_LIST=("5.0")
IQL_TARGET_TAU_LIST=("0.001")
IQL_LR_LIST=("3e-4")
IQL_DISCOUNT_LIST=("0.95")

if [[ -n "${GRID_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${GRID_SEEDS}"
else
  SEEDS=(10 101 1010 10101 101010 20 202 2020 20202 202020)
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

grid_root="${ROOT}/grid_results/iql_search/gamma_${gamma}"
mkdir -p "${grid_root}/logs" "${grid_root}/iql_ckpts" "${grid_root}/iql_ckpts_work" \
         "${grid_root}/ct_checkpoints"

summary_csv="${grid_root}/summary.csv"
summary_lock="${grid_root}/.summary.lock"

EXPECTED_HEADER="combo_id,seed,iql_tau,iql_beta,iql_target_tau,iql_lr,iql_discount,ct_ckpt_path,iql_picked_step,iql_best_step_sim,rho_sim_predictor,top1_overlap_sim_predictor,top3_overlap_sim_predictor,top5_overlap_sim_predictor,iql_best_val_mae_uns,iql_best_val_mae_predictor,mae_uns_eval_split,eval_split"

with_lock() {
  ( flock -x 9; "$@" ) 9> "${summary_lock}"
}

if [[ -f "${summary_csv}" ]]; then
  existing_header=$(head -n1 "${summary_csv}" || true)
  if [[ "${existing_header}" != "${EXPECTED_HEADER}" ]]; then
    bak="${summary_csv}.$(date +%s).bak"
    mv "${summary_csv}" "${bak}"
    echo "[schema] archived old summary to ${bak}"
  fi
fi
if [[ ! -f "${summary_csv}" ]]; then
  echo "${EXPECTED_HEADER}" > "${summary_csv}"
fi

parse_last() {
  local file="$1" pat="$2"
  grep -oP "${pat}" "${file}" 2>/dev/null | tail -n1 || true
}

shard_owner() {
  local tag="$1"
  local h
  h=$(printf '%s' "${tag}" | cksum | awk '{print $1}')
  echo $(( h % GRID_NUM_WORKERS ))
}

# Canonical CT path for this grid run (train + IQL load must match).
ct_ckpt_for_seed() {
  local seed="$1"
  echo "${grid_root}/ct_checkpoints/seed_${seed}/coeff_${gamma}/ct_best_encoder.pt"
}

ct_work_dir_for_seed() {
  local seed="$1"
  echo "${grid_root}/ct_checkpoints/seed_${seed}/coeff_${gamma}"
}

read_iql_crossworld_metrics() {
  local trace_json="$1"
  python - "${trace_json}" <<'PY'
import json
import sys
from typing import List, Optional

import numpy as np

path = sys.argv[1]


def spearman_rho(a: List[float], b: List[float]) -> Optional[float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.size < 2 or bb.size < 2 or aa.size != bb.size:
        return None
    ra = aa.argsort().argsort().astype(np.float64)
    rb = bb.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return None
    return float((ra * rb).sum() / denom)


def top_k_overlap(metric_a: List[float], metric_b: List[float], k: int) -> Optional[float]:
    if k <= 0 or len(metric_a) == 0 or len(metric_a) != len(metric_b):
        return None
    k = min(k, len(metric_a))
    idx_a = set(np.argsort(np.asarray(metric_a))[:k].tolist())
    idx_b = set(np.argsort(np.asarray(metric_b))[:k].tolist())
    return float(len(idx_a & idx_b) / k)


def na():
    print("NA NA NA NA NA NA NA NA")
    return


try:
    with open(path, "r") as f:
        trace = json.load(f)
except FileNotFoundError:
    na()
    raise SystemExit(0)

metric_key = str(trace.get("val_metric", "mae_uns"))
worlds = list(trace.get("worlds", ["sim", "predictor"]))
history = trace.get("history", [])
best_pw = trace.get("best_per_world", {})
selection_world = str(trace.get("selection_world", worlds[0] if worlds else "sim"))

if "sim" not in worlds or "predictor" not in worlds or len(history) < 2:
    na()
    raise SystemExit(0)

series_sim = [float(entry["sim"][metric_key]) for entry in history]
series_pred = [float(entry["predictor"][metric_key]) for entry in history]

rho = spearman_rho(series_sim, series_pred)
t1 = top_k_overlap(series_sim, series_pred, 1)
t3 = top_k_overlap(series_sim, series_pred, 3)
t5 = top_k_overlap(series_sim, series_pred, 5)

picked_step = best_pw.get(selection_world, {}).get("step")
best_step_sim = best_pw.get("sim", {}).get("step")
picked_mae = best_pw.get(selection_world, {}).get("metric")
pred_mae = best_pw.get("predictor", {}).get("metric")

def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


print(
    " ".join(
        [
            fmt(picked_step),
            fmt(best_step_sim),
            fmt(rho),
            fmt(t1),
            fmt(t3),
            fmt(t5),
            fmt(picked_mae),
            fmt(pred_mae),
        ]
    )
)
PY
}

train_ct_for_seed() {
  local seed="$1"
  local tag="ct__seed_${seed}"
  local owner
  owner=$(shard_owner "${tag}")
  if (( owner != GRID_WORKER_ID )); then
    return 0
  fi

  local ct_work_dir
  ct_work_dir=$(ct_work_dir_for_seed "${seed}")
  local ct_ckpt
  ct_ckpt=$(ct_ckpt_for_seed "${seed}")
  local log_file="${grid_root}/logs/${tag}.log"

  mkdir -p "${ct_work_dir}"

  if [[ "${GRID_SKIP_CT:-0}" == "1" && -f "${ct_ckpt}" ]]; then
    echo "[w${GRID_WORKER_ID}][skip CT] ${ct_ckpt} exists"
    return 0
  fi

  echo "================================================================"
  echo "=== [w${GRID_WORKER_ID}] train_ct seed=${seed} -> ${ct_ckpt} ==="
  echo "================================================================"

  CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/train_ct.py \
    +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${gamma}*" \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    "+exp.ct_ckpt_dir=${ct_work_dir}" \
    2>&1 | tee "${log_file}"

  if [[ ! -f "${ct_ckpt}" ]]; then
    echo "ERROR: CT training did not produce ${ct_ckpt}" >&2
    return 1
  fi
  echo "[w${GRID_WORKER_ID}] CT saved: ${ct_ckpt}"
}

run_iql_combo() {
  local tau="$1" beta="$2" t_tau="$3" lr="$4" disc="$5" seed="$6"
  local combo_id="tau-${tau}_beta-${beta}_ttau-${t_tau}_lr-${lr}_disc-${disc}"
  local tag="${combo_id}__seed_${seed}"
  local log_file="${grid_root}/logs/${tag}.log"
  local iql_work_dir="${grid_root}/iql_ckpts_work/${tag}"
  local iql_ckpt_src="${iql_work_dir}/iql_planner.pt"
  local iql_ckpt_grid="${grid_root}/iql_ckpts/${tag}.pt"
  local trace_json="${iql_work_dir}/iql_val_trace.json"
  local ct_ckpt
  ct_ckpt=$(ct_ckpt_for_seed "${seed}")

  local owner
  owner=$(shard_owner "${tag}")
  if (( owner != GRID_WORKER_ID )); then
    return 0
  fi

  if with_lock grep -q "^${combo_id},${seed}," "${summary_csv}"; then
    echo "[w${GRID_WORKER_ID}][skip] ${tag} already in summary"
    return 0
  fi

  if [[ ! -f "${ct_ckpt}" ]]; then
    echo "ERROR: missing CT ckpt for seed=${seed}: ${ct_ckpt} (run CT first or unset GRID_SKIP_CT)" >&2
    return 0
  fi

  mkdir -p "${iql_work_dir}"

  echo "================================================================"
  echo "=== [w${GRID_WORKER_ID}] train_iql ${tag} | CT=${ct_ckpt} ==="
  echo "================================================================"

  CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/train_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${gamma}*" \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    exp.iql_tau="${tau}" \
    exp.iql_beta="${beta}" \
    exp.iql_target_tau="${t_tau}" \
    exp.iql_actor_lr="${lr}" exp.iql_qf_lr="${lr}" exp.iql_vf_lr="${lr}" \
    exp.iql_discount="${disc}" \
    exp.iql_inference_ckpt="${ct_ckpt}" \
    "+exp.iql_save_dir=${iql_work_dir}" \
    2>&1 | tee "${log_file}"

  local picked_step="NA" best_step_sim="NA" rho="NA" t1="NA" t3="NA" t5="NA"
  local picked_mae="NA" pred_mae="NA"
  local mae_eval="NA"

  if [[ -f "${trace_json}" ]]; then
    read -r picked_step best_step_sim rho t1 t3 t5 picked_mae pred_mae <<< "$(read_iql_crossworld_metrics "${trace_json}")" || true
  else
    echo "[w${GRID_WORKER_ID}] warning: ${trace_json} missing; cross-world metrics = NA"
    picked_step=$(parse_last "${log_file}" 'Saved BEST IQL planner.* at step \K[0-9]+')
    [[ -z "${picked_step}" ]] && picked_step="NA"
    picked_mae=$(parse_last "${log_file}" 'Saved BEST IQL planner.*\(mae_uns=\K[0-9.eE+-]+')
    [[ -z "${picked_mae}" ]] && picked_mae="NA"
  fi

  if [[ ! -f "${iql_ckpt_src}" ]]; then
    echo "ERROR: IQL ckpt missing for ${tag}" >&2
    with_lock bash -c "echo '${combo_id},${seed},${tau},${beta},${t_tau},${lr},${disc},${ct_ckpt},NA,NA,NA,NA,NA,NA,NA,NA,NA,${TEST_SPLIT}' >> '${summary_csv}'"
    return 0
  fi
  cp "${iql_ckpt_src}" "${iql_ckpt_grid}"

  echo "================================================================"
  echo "=== [w${GRID_WORKER_ID}] eval_iql ${tag} (exp.test=${TEST_SPLIT}) ==="
  echo "================================================================"
  CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/eval_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${gamma}*" \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    exp.test="${TEST_SPLIT}" \
    exp.iql_inference_ckpt="${ct_ckpt}" \
    exp.iql_eval_ckpt="${iql_ckpt_grid}" \
    2>&1 | tee -a "${log_file}"

  mae_eval=$(parse_last "${log_file}" 'MAE unscaled: \K[0-9.eE+-]+')
  [[ -z "${mae_eval}" ]] && mae_eval="NA"

  with_lock bash -c "echo '${combo_id},${seed},${tau},${beta},${t_tau},${lr},${disc},${ct_ckpt},${picked_step},${best_step_sim},${rho},${t1},${t3},${t5},${picked_mae},${pred_mae},${mae_eval},${TEST_SPLIT}' >> '${summary_csv}'"
}

# Phase 1: CT per seed (shared across all IQL combos for that seed)
for seed in "${SEEDS[@]}"; do
  train_ct_for_seed "${seed}"
done

if [[ "${GRID_SKIP_IQL:-0}" == "1" ]]; then
  echo "GRID_SKIP_IQL=1: CT phase done; skipping IQL grid."
  exit 0
fi

# Phase 2: IQL grid per combo x seed
for tau in "${IQL_TAU_LIST[@]}"; do
  for t_tau in "${IQL_TARGET_TAU_LIST[@]}"; do
    for beta in "${IQL_BETA_LIST[@]}"; do
      for disc in "${IQL_DISCOUNT_LIST[@]}"; do
        for lr in "${IQL_LR_LIST[@]}"; do
          for seed in "${SEEDS[@]}"; do
            run_iql_combo "${tau}" "${beta}" "${t_tau}" "${lr}" "${disc}" "${seed}"
          done
        done
      done
    done
  done
done

echo "Grid search done. Summary: ${summary_csv}"
python - <<PY
import csv
from collections import defaultdict

import numpy as np

path = "${summary_csv}"


def to_f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


groups = defaultdict(list)
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        groups[r["combo_id"]].append(to_f(r.get("mae_uns_eval_split")))

rows = []
for cid, maes in groups.items():
    maes = [m for m in maes if m is not None]
    if maes:
        rows.append((np.mean(maes), np.median(maes), len(maes), cid))
rows.sort(key=lambda x: x[0])

print("\n--- Top combos by mean mae_uns_eval_split ---")
print(f"{'Rank':<4} {'Mean':<10} {'Median':<10} {'N':<4} combo_id")
for i, (mean, med, n, cid) in enumerate(rows[:15], 1):
    print(f"{i:<4} {mean:<10.6f} {med:<10.6f} {n:<4} {cid}")
PY
