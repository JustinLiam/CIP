#!/usr/bin/env bash
# Focused one-stage EM+IQL sweep for the local-global CTHistoryEncoder.
#
# Defaults match the current focused run request:
#   local-global only: model.inference.local_conv_layers=1
#   exp.iql_tau: 0.7, 0.9
#   exp.iql_actor_lr: 3e-4, 1e-3
#   exp.iql_qf_lr: 3e-4, 1e-3
#   exp.iql_vf_lr: tied to actor_lr unless IQL_VF_LR is set
#   exp.iql_max_grad_norm: 5.0
#   exp.em_m_steps_per_outer: 1000
#   seeds: 20, 2020, 202020
#
# Evaluation logs the existing MAE/RMSE metrics and GIFT-aligned tumor RMSE:
#   gift_rmse         = unscaled cancer-volume RMSE
#   gift_rmse_percent = gift_rmse / 1150 * 100
#
# Usage from repo root:
#   bash scripts/cancer/train/run_em_iql_local_global_focused_sweep.sh [GPU] [GAMMA]

set -euo pipefail

if [[ -f /home/liam/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/liam/anaconda3/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)"
fi
conda activate vcip

GPU="${1:-1}"
GAMMA="${2:-4}"

TEST_SPLIT="${TEST_SPLIT:-false}"
SEEDS_RAW="${GRID_SEEDS:-20 2020 202020}"
IQL_TAU_RAW="${FOCUS_IQL_TAU_LIST:-0.7 0.9}"
ACTOR_LR_RAW="${FOCUS_ACTOR_LR_LIST:-3e-4 1e-3}"
QF_LR_RAW="${FOCUS_QF_LR_LIST:-3e-4 1e-3}"
EVAL_TAU_LIST="${EVAL_TAU_LIST:-[1,2,3,4,5,6]}"

read -r -a SEEDS <<< "${SEEDS_RAW}"
read -r -a IQL_TAU_LIST <<< "${IQL_TAU_RAW}"
read -r -a ACTOR_LR_LIST <<< "${ACTOR_LR_RAW}"
read -r -a QF_LR_LIST <<< "${QF_LR_RAW}"

IQL_BETA="${IQL_BETA:-2.0}"
IQL_MAX_GRAD="${IQL_MAX_GRAD:-5.0}"
EM_M_STEPS="${EM_M_STEPS:-1000}"
EM_HER_SAMPLES="${EM_HER_SAMPLES:-1}"
EM_HER_REFRESH="${EM_HER_REFRESH:-0}"
GPU_WAIT_MEMORY_MB="${GPU_WAIT_MEMORY_MB:-1000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-em_iql_local_global_focused}"
MLFLOW_URI="${MLFLOW_URI:-}"
FORCE="${FORCE:-0}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${GRID_ROOT:-${ROOT}/grid_results/em_iql_local_global_focused/gamma_${GAMMA}}"
mkdir -p "${GRID_ROOT}/logs" "${GRID_ROOT}/ckpts" "${GRID_ROOT}/done"

SUMMARY="${GRID_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "combo_id,seed,local_conv_layers,iql_tau,iql_actor_lr,iql_qf_lr,iql_vf_lr,iql_beta,iql_max_grad_norm,em_m_steps_per_outer,em_her_samples_per_transition,em_her_refresh_every,best_outer,best_val_mae_uns,eval_tau,eval_mae_norm,eval_mae_uns,eval_rmse_norm,eval_rmse_norm_x_std,eval_rmse_uns,gift_rmse,gift_rmse_percent,gift_mae_percent,em_ckpt,train_log,eval_log,finished_at" > "${SUMMARY}"
fi

MLFLOW_URI_ARGS=()
if [[ -n "${MLFLOW_URI}" ]]; then
  MLFLOW_URI_ARGS=("exp.mlflow_uri=${MLFLOW_URI}")
fi

encode_float() {
  case "$1" in
    0.7) echo "07" ;;
    0.9) echo "09" ;;
    2.0|2) echo "20" ;;
    5.0|5) echo "50" ;;
    1000) echo "1k" ;;
    *) echo "${1//./}" | tr '-' '_' ;;
  esac
}

encode_lr() {
  case "$1" in
    3e-4) echo "3e4" ;;
    1e-3) echo "1e3" ;;
    1e-4) echo "1e4" ;;
    *) echo "${1//-/_}" ;;
  esac
}

read_em_metrics() {
  local ckpt_path="$1"
  python - "${ckpt_path}" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    print("NA NA")
    raise SystemExit(0)
extra = c.get("extra") or {}
outer = c.get("outer_iter", extra.get("outer_iter", "NA"))
val = extra.get("best_val_mae_uns", extra.get("val_mae_uns", "NA"))
def f(v):
    try:
        return f"{float(v):.8g}"
    except Exception:
        return str(v)
print(f(outer), f(val))
PY
}

wait_for_gpu() {
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | head -n1 | tr -d ' ')"
    if [[ -z "${used}" || "${used}" -lt "${GPU_WAIT_MEMORY_MB}" ]]; then
      return 0
    fi
    echo "[wait] GPU ${GPU} memory.used=${used}MB >= ${GPU_WAIT_MEMORY_MB}MB; sleeping ${GPU_WAIT_SECONDS}s"
    sleep "${GPU_WAIT_SECONDS}"
  done
}

append_eval_rows() {
  local combo_id="$1"
  local seed="$2"
  local local_layers="$3"
  local iql_tau="$4"
  local actor_lr="$5"
  local qf_lr="$6"
  local vf_lr="$7"
  local best_outer="$8"
  local best_val="$9"
  local em_ckpt="${10}"
  local train_log="${11}"
  local eval_log="${12}"
  local finished_at="${13}"

  python - "${SUMMARY}" "${combo_id}" "${seed}" "${local_layers}" "${iql_tau}" "${actor_lr}" "${qf_lr}" "${vf_lr}" \
    "${IQL_BETA}" "${IQL_MAX_GRAD}" "${EM_M_STEPS}" "${EM_HER_SAMPLES}" "${EM_HER_REFRESH}" \
    "${best_outer}" "${best_val}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}" <<'PY'
import csv
import os
import re
import sys

(
    summary_path, combo_id, seed, local_layers, iql_tau, actor_lr, qf_lr, vf_lr,
    iql_beta, iql_max_grad, em_m_steps, em_her_samples, em_her_refresh,
    best_outer, best_val, em_ckpt, train_log, eval_log, finished_at
) = sys.argv[1:]

def empty_metrics():
    return {
        "eval_tau": "NA",
        "eval_mae_norm": "NA",
        "eval_mae_uns": "NA",
        "eval_rmse_norm": "NA",
        "eval_rmse_norm_x_std": "NA",
        "eval_rmse_uns": "NA",
        "gift_rmse": "NA",
        "gift_rmse_percent": "NA",
        "gift_mae_percent": "NA",
    }

rows = []
current = None
with open(eval_log, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = re.search(r"IQL eval autoregressive action rollout: .*?\(tau=(\d+),", line)
        if m:
            if current is not None:
                rows.append(current)
            current = empty_metrics()
            current["eval_tau"] = m.group(1)
            continue
        if current is None:
            continue
        m = re.search(r"Global RMSE on stacked batches \(normalized space\): ([0-9.eE+-]+)", line)
        if m:
            current["eval_rmse_norm"] = m.group(1)
            continue
        m = re.search(r"Global RMSE .*VCIP-style\): ([0-9.eE+-]+)", line)
        if m:
            current["eval_rmse_norm_x_std"] = m.group(1)
            continue
        m = re.search(
            r"MAE normalized: ([0-9.eE+-]+) \| MAE unscaled: ([0-9.eE+-]+) \| RMSE unscaled: ([0-9.eE+-]+)",
            line,
        )
        if m:
            current["eval_mae_norm"] = m.group(1)
            current["eval_mae_uns"] = m.group(2)
            current["eval_rmse_uns"] = m.group(3)
            continue
        m = re.search(
            r"GIFT-style tumor RMSE unscaled: ([0-9.eE+-]+) \| GIFT-style tumor RMSE \(% of [0-9.eE+-]+\): ([0-9.eE+-]+) \| GIFT-style tumor MAE .*: ([0-9.eE+-]+)",
            line,
        )
        if m:
            current["gift_rmse"] = m.group(1)
            current["gift_rmse_percent"] = m.group(2)
            current["gift_mae_percent"] = m.group(3)
            continue
if current is not None:
    rows.append(current)
if not rows:
    rows = [empty_metrics()]

fieldnames = [
    "combo_id", "seed", "local_conv_layers", "iql_tau", "iql_actor_lr", "iql_qf_lr", "iql_vf_lr",
    "iql_beta", "iql_max_grad_norm", "em_m_steps_per_outer", "em_her_samples_per_transition",
    "em_her_refresh_every", "best_outer", "best_val_mae_uns", "eval_tau", "eval_mae_norm",
    "eval_mae_uns", "eval_rmse_norm", "eval_rmse_norm_x_std", "eval_rmse_uns", "gift_rmse",
    "gift_rmse_percent", "gift_mae_percent", "em_ckpt", "train_log", "eval_log", "finished_at",
]
with open(summary_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for metrics in rows:
        row = {
            "combo_id": combo_id,
            "seed": seed,
            "local_conv_layers": local_layers,
            "iql_tau": iql_tau,
            "iql_actor_lr": actor_lr,
            "iql_qf_lr": qf_lr,
            "iql_vf_lr": vf_lr,
            "iql_beta": iql_beta,
            "iql_max_grad_norm": iql_max_grad,
            "em_m_steps_per_outer": em_m_steps,
            "em_her_samples_per_transition": em_her_samples,
            "em_her_refresh_every": em_her_refresh,
            "best_outer": best_outer,
            "best_val_mae_uns": best_val,
            "em_ckpt": em_ckpt,
            "train_log": train_log,
            "eval_log": eval_log,
            "finished_at": finished_at,
        }
        row.update(metrics)
        writer.writerow(row)
print(f"appended {len(rows)} eval row(s) from {os.path.basename(eval_log)}")
PY
}

run_one() {
  local iql_tau="$1"
  local actor_lr="$2"
  local qf_lr="$3"
  local seed="$4"
  local vf_lr="${IQL_VF_LR:-${actor_lr}}"
  local local_layers=1

  local tau_id actor_id qf_id vf_id beta_id grad_id
  tau_id="$(encode_float "${iql_tau}")"
  actor_id="$(encode_lr "${actor_lr}")"
  qf_id="$(encode_lr "${qf_lr}")"
  vf_id="$(encode_lr "${vf_lr}")"
  beta_id="$(encode_float "${IQL_BETA}")"
  grad_id="$(encode_float "${IQL_MAX_GRAD}")"
  local combo_id="lg_b${beta_id}_t${tau_id}_alr${actor_id}_qlr${qf_id}_vlr${vf_id}_g${grad_id}_m$(encode_float "${EM_M_STEPS}")"
  local tag="${combo_id}_seed${seed}"
  local em_dir="${GRID_ROOT}/ckpts/${tag}"
  local em_ckpt="${em_dir}/ct_iql_em_best.pt"
  local log_dir="${GRID_ROOT}/logs/${tag}"
  local train_log="${log_dir}/train.log"
  local eval_log="${log_dir}/eval.log"
  local done_flag="${GRID_ROOT}/done/${tag}.done"

  if [[ "${FORCE}" != "1" ]]; then
    if [[ -f "${done_flag}" ]]; then
      echo "[skip] ${tag} (done marker exists)"
      return 0
    fi
    if grep -q "^${combo_id},${seed}," "${SUMMARY}" 2>/dev/null; then
      echo "[skip] ${tag} already recorded in ${SUMMARY}"
      return 0
    fi
  fi

  mkdir -p "${em_dir}" "${log_dir}"

  echo "========== ${tag} =========="
  echo "  seed=${seed} gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
  echo "  local_conv_layers=${local_layers}"
  echo "  iql_tau=${iql_tau} actor_lr=${actor_lr} qf_lr=${qf_lr} vf_lr=${vf_lr}"
  echo "  iql_beta=${IQL_BETA} max_grad=${IQL_MAX_GRAD} m_steps=${EM_M_STEPS}"

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/train_ct_iql_em.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_her_refresh_every=${EM_HER_REFRESH}" \
    "exp.em_her_samples_per_transition=${EM_HER_SAMPLES}" \
    "exp.ct_num_workers=0" \
    "exp.iql_beta=${IQL_BETA}" \
    "exp.iql_tau=${iql_tau}" \
    "exp.iql_actor_lr=${actor_lr}" \
    "exp.iql_qf_lr=${qf_lr}" \
    "exp.iql_vf_lr=${vf_lr}" \
    "exp.iql_max_grad_norm=${IQL_MAX_GRAD}" \
    "exp.em_m_steps_per_outer=${EM_M_STEPS}" \
    "+exp.em_ckpt_dir=${em_dir}" \
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.mlflow_combo_id=${combo_id}" \
    "${MLFLOW_URI_ARGS[@]}" \
    2>&1 | tee "${train_log}"

  if [[ ! -f "${em_ckpt}" ]]; then
    echo "ERROR: EM checkpoint missing after train: ${em_ckpt}" >&2
    return 1
  fi

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/eval_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" \
    exp.test="${TEST_SPLIT}" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_eval_ckpt=${em_ckpt}" \
    "exp.iql_eval_tau_list=${EVAL_TAU_LIST}" \
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.mlflow_combo_id=${combo_id}" \
    "${MLFLOW_URI_ARGS[@]}" \
    2>&1 | tee "${eval_log}"

  local best_outer best_val finished_at
  read -r best_outer best_val <<< "$(read_em_metrics "${em_ckpt}")"
  finished_at="$(date -Iseconds)"
  append_eval_rows \
    "${combo_id}" "${seed}" "${local_layers}" "${iql_tau}" "${actor_lr}" "${qf_lr}" "${vf_lr}" \
    "${best_outer}" "${best_val}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}"

  {
    echo "finished_at=${finished_at}"
    echo "combo_id=${combo_id}"
    echo "seed=${seed}"
    echo "local_conv_layers=${local_layers}"
    echo "iql_tau=${iql_tau}"
    echo "iql_actor_lr=${actor_lr}"
    echo "iql_qf_lr=${qf_lr}"
    echo "iql_vf_lr=${vf_lr}"
    echo "em_ckpt=${em_ckpt}"
    echo "train_log=${train_log}"
    echo "eval_log=${eval_log}"
  } > "${done_flag}"
  echo "[done] ${tag}"
}

echo "[focused] one-stage EM+IQL local-global sweep"
echo "[focused] gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
echo "[focused] seeds=(${SEEDS[*]})"
echo "[focused] iql_tau=(${IQL_TAU_LIST[*]}) actor_lr=(${ACTOR_LR_LIST[*]}) qf_lr=(${QF_LR_LIST[*]})"
echo "[focused] beta=${IQL_BETA} grad=${IQL_MAX_GRAD} m_steps=${EM_M_STEPS} eval_tau_list=${EVAL_TAU_LIST}"
echo "[focused] grid_root=${GRID_ROOT}"

for iql_tau in "${IQL_TAU_LIST[@]}"; do
  for actor_lr in "${ACTOR_LR_LIST[@]}"; do
    for qf_lr in "${QF_LR_LIST[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "${iql_tau}" "${actor_lr}" "${qf_lr}" "${seed}"
      done
    done
  done
done

python - "${SUMMARY}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    if row.get("gift_rmse_percent") in ("", "NA", None):
        continue
    groups[(row["combo_id"], row["eval_tau"])].append(float(row["gift_rmse_percent"]))

print(f"\nSummary: {path}")
for (combo_id, eval_tau), vals in sorted(groups.items()):
    if vals:
        print(
            f"{combo_id} tau={eval_tau}: n={len(vals)} "
            f"gift_rmse_percent_mean={statistics.mean(vals):.6f} "
            f"vals={[round(v, 6) for v in vals]}"
        )
PY
