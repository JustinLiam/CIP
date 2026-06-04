#!/usr/bin/env bash
# CT (train_ct) -> IQL (train_iql_planner) -> eval (eval_iql_planner) for cancer_sim_cont.
#
# Usage (from repo root):
#   bash scripts/cancer/train/train_ct_iql.sh [test] [gamma] [gpu]
#     test  - exp.test for eval only (false=val, true=test), default false
#     gamma - dataset.coeff, default 4
#     gpu   - CUDA_VISIBLE_DEVICES, default 0
#
# Eval runs at exp.tau = 1,2,3,4,5,6,8,10,12 (one row per seed x tau in ct_iql_eval_by_tau.csv).
# IQL training metrics (independent of eval tau) go to ct_iql_training_summary.csv once per seed.
# Override taus: CT_IQL_EVAL_TAUS="4 6 12"
#
# Outputs:
#   results/ct_iql_eval_by_tau.csv      - eval metrics that MUST vary with exp.tau
#   results/ct_iql_training_summary.csv - IQL val / cross-world metrics (one row per seed)
# Eval-only (checkpoints must exist):
#   CT_IQL_SKIP_TRAIN=1 bash scripts/cancer/train/train_ct_iql.sh false 4 0
#
# Optional env:
#   CT_IQL_SEEDS="10 101"       - override default seed list
#   CT_IQL_EVAL_TAUS="4 6 12"   - override default eval horizon list
#   CT_IQL_SKIP_TRAIN=1         - skip train_ct + train_iql, eval only
#   CT_IQL_GRID_CT=1            - Plan B: grid-best CT hparams (gamma_4 summary.csv top-1)
#   CT_IQL_CT_EXTRA="exp.ct_lr=1e-4 ..."  - extra Hydra overrides for train_ct only
#
# Checkpoints (canonical paths, match gridsearch layout under tumor_generator):
#   ct_checkpoints/tumor_generator/seed_${seed}/coeff_${gamma}/ct_best_encoder.pt
#   iql_models/tumor_generator/seed_${seed}/coeff_${gamma}/iql_planner.pt

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

test=${1:-false}
gamma=${2:-4}
gpu=${3:-0}

# Default IQL hyperparams (vcip_cancer.yaml); recorded in summary CSV combo_id row fields.
COMBO_ID="vcip_cancer_default"
IQL_TAU="0.5"
IQL_BETA="5.0"
IQL_TARGET_TAU="0.005"
IQL_LR="3e-4"
IQL_DISCOUNT="0.95"

if [[ -n "${CT_IQL_EVAL_TAUS:-}" ]]; then
  read -r -a EVAL_TAUS <<< "${CT_IQL_EVAL_TAUS}"
else
  EVAL_TAUS=(1 2 3 4 5 6 8 10 12)
fi

if [[ -n "${CT_IQL_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${CT_IQL_SEEDS}"
else
  SEEDS=(10 101 1010 10101 101010 20 202 2020 20202 202020)
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

# Extra Hydra args forwarded to train_ct.py only.
CT_TRAIN_EXTRA=()
if [[ "${CT_IQL_GRID_CT:-0}" == "1" ]]; then
  COMBO_ID="grid_ct_best_k1"
  CT_TRAIN_EXTRA=(
    exp.ct_lr=1e-4
    exp.ct_w_lr=5e-3
    exp.ct_multi_k_max=1
    exp.ct_multi_eta=0.3
    exp.ct_es_metric=mae_uw
    exp.ct_anchor_weight=0.5
    exp.ct_dyn_hidden=128
    exp.ct_dyn_consistency_weight=0.1
  )
elif [[ -n "${CT_IQL_CT_EXTRA:-}" ]]; then
  read -r -a CT_TRAIN_EXTRA <<< "${CT_IQL_CT_EXTRA}"
fi

DATASET_NAME="tumor_generator"

CSV_DIR="${ROOT}/results"
EVAL_CSV="${CSV_DIR}/ct_iql_eval_by_tau.csv"
TRAIN_CSV="${CSV_DIR}/ct_iql_training_summary.csv"
CSV_LOCK="${CSV_DIR}/.ct_iql_csv.lock"
mkdir -p "${CSV_DIR}"
touch "${CSV_LOCK}"

# eval_tau = rollout horizon (exp.tau). iql_expectile is NOT eval horizon.
EVAL_HEADER="seed,gamma,eval_tau,tau_verified_from_log,mae_norm_eval,mae_uns_eval,rmse_uns_eval,eval_split,ct_ckpt_path,iql_ckpt_path,combo_id"
TRAIN_HEADER="combo_id,seed,gamma,iql_expectile,iql_beta,iql_target_tau,iql_lr,iql_discount,ct_ckpt_path,iql_ckpt_path,iql_picked_step,iql_best_step_sim,rho_sim_predictor,top1_overlap_sim_predictor,top3_overlap_sim_predictor,top5_overlap_sim_predictor,iql_best_val_mae_uns,iql_best_val_mae_predictor"

ensure_csv_header() {
  local path="$1" expected="$2"
  if [[ -f "${path}" ]]; then
    local existing_header
    existing_header=$(head -n1 "${path}" || true)
    if [[ "${existing_header}" != "${expected}" ]]; then
      local bak="${path}.$(date +%s).bak"
      mv "${path}" "${bak}"
      echo "[schema] archived old CSV to ${bak}"
    fi
  fi
  if [[ ! -f "${path}" ]]; then
    echo "${expected}" > "${path}"
  fi
}

append_csv_row() {
  local path="$1" expected_header="$2" row="$3"
  (
    flock -x 200
    ensure_csv_header "${path}" "${expected_header}"
    printf "%s\n" "${row}" >> "${path}"
  ) 200>"${CSV_LOCK}"
}

parse_eval_log() {
  local log_file="$1" expected_tau="$2"
  python - "${log_file}" "${expected_tau}" <<'PY'
import re
import sys

path, expected = sys.argv[1], str(sys.argv[2])
text = open(path, encoding="utf-8", errors="replace").read()

tau_m = re.search(r"\(tau=(\d+),\s*max_tau=", text)
tau_verified = tau_m.group(1) if tau_m else "NA"

agg = re.search(
    r"MAE normalized:\s*([0-9.eE+-]+)\s*\|\s*MAE unscaled:\s*([0-9.eE+-]+)\s*\|\s*RMSE unscaled:\s*([0-9.eE+-]+)",
    text,
)
if agg:
    mae_norm, mae_uns, rmse_uns = agg.groups()
else:
    mae_norm = mae_uns = rmse_uns = "NA"

tau_ok = "1" if tau_verified == expected else "0"
print(tau_verified, mae_norm, mae_uns, rmse_uns, tau_ok)
PY
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

ct_dir_for_seed() {
  local seed="$1"
  echo "${ROOT}/ct_checkpoints/${DATASET_NAME}/seed_${seed}/coeff_${gamma}"
}

ct_ckpt_for_seed() {
  echo "$(ct_dir_for_seed "$1")/ct_best_encoder.pt"
}

iql_dir_for_seed() {
  local seed="$1"
  echo "${ROOT}/iql_models/${DATASET_NAME}/seed_${seed}/coeff_${gamma}"
}

iql_ckpt_for_seed() {
  echo "$(iql_dir_for_seed "$1")/iql_planner.pt"
}

for seed in "${SEEDS[@]}"; do
  CT_DIR="$(ct_dir_for_seed "${seed}")"
  CT_CKPT="$(ct_ckpt_for_seed "${seed}")"
  IQL_DIR="$(iql_dir_for_seed "${seed}")"
  IQL_CKPT="$(iql_ckpt_for_seed "${seed}")"
  TRACE_JSON="${IQL_DIR}/iql_val_trace.json"

  echo "================================================================"
  echo "=== seed=${seed} coeff=${gamma} | CT -> IQL -> eval taus=(${EVAL_TAUS[*]}) exp.test=${test} ==="
  echo "================================================================"

  if [[ "${CT_IQL_SKIP_TRAIN:-0}" != "1" ]]; then
    mkdir -p "${CT_DIR}"

    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_ct.py \
      +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      "+exp.ct_ckpt_dir=${CT_DIR}" \
      "${CT_TRAIN_EXTRA[@]}"

    if [[ ! -f "${CT_CKPT}" ]]; then
      echo "ERROR: CT checkpoint missing after train_ct: ${CT_CKPT}" >&2
      exit 1
    fi

    mkdir -p "${IQL_DIR}"

    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.iql_inference_ckpt="${CT_CKPT}" \
      "+exp.iql_save_dir=${IQL_DIR}"

    if [[ ! -f "${IQL_CKPT}" ]]; then
      echo "ERROR: IQL checkpoint missing after train_iql_planner: ${IQL_CKPT}" >&2
      exit 1
    fi
  else
    if [[ ! -f "${CT_CKPT}" ]]; then
      echo "ERROR: CT checkpoint missing (CT_IQL_SKIP_TRAIN=1): ${CT_CKPT}" >&2
      exit 1
    fi
    if [[ ! -f "${IQL_CKPT}" ]]; then
      echo "ERROR: IQL checkpoint missing (CT_IQL_SKIP_TRAIN=1): ${IQL_CKPT}" >&2
      exit 1
    fi
  fi

  picked_step="NA" best_step_sim="NA" rho="NA" t1="NA" t3="NA" t5="NA"
  picked_mae="NA" pred_mae="NA"

  if [[ -f "${TRACE_JSON}" ]]; then
    read -r picked_step best_step_sim rho t1 t3 t5 picked_mae pred_mae \
      <<< "$(read_iql_crossworld_metrics "${TRACE_JSON}")" || true
  else
    echo "warning: ${TRACE_JSON} missing; cross-world metrics = NA"
  fi

  append_csv_row "${TRAIN_CSV}" "${TRAIN_HEADER}" \
    "${COMBO_ID},${seed},${gamma},${IQL_TAU},${IQL_BETA},${IQL_TARGET_TAU},${IQL_LR},${IQL_DISCOUNT},${CT_CKPT},${IQL_CKPT},${picked_step},${best_step_sim},${rho},${t1},${t3},${t5},${picked_mae},${pred_mae}"

  for eval_tau in "${EVAL_TAUS[@]}"; do
    echo "--- seed=${seed} eval ++exp.tau=${eval_tau} exp.test=${test} ---"

    eval_log="$(mktemp)"
    # ++exp.tau forces override over vcip_cancer / hparams defaults (rollout horizon).
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/eval_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.test="${test}" \
      "++exp.tau=${eval_tau}" \
      exp.iql_inference_ckpt="${CT_CKPT}" \
      exp.iql_eval_ckpt="${IQL_CKPT}" \
      2>&1 | tee "${eval_log}"

    read -r tau_verified mae_norm mae_uns rmse_uns tau_ok \
      <<< "$(parse_eval_log "${eval_log}" "${eval_tau}")" || true
    [[ -z "${tau_verified}" ]] && tau_verified="NA"
    [[ -z "${mae_norm}" ]] && mae_norm="NA"
    [[ -z "${mae_uns}" ]] && mae_uns="NA"
    [[ -z "${rmse_uns}" ]] && rmse_uns="NA"

    if [[ "${tau_ok}" != "1" ]]; then
      echo "WARNING: seed=${seed} requested eval_tau=${eval_tau} but log reports tau=${tau_verified}" >&2
    fi

    append_csv_row "${EVAL_CSV}" "${EVAL_HEADER}" \
      "${seed},${gamma},${eval_tau},${tau_verified},${mae_norm},${mae_uns},${rmse_uns},${test},${CT_CKPT},${IQL_CKPT},${COMBO_ID}"

    rm -f "${eval_log}"
  done
done

echo "Done."
echo "  Eval by tau:  ${EVAL_CSV}"
echo "  IQL training: ${TRAIN_CSV}"
