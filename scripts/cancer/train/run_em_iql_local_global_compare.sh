#!/usr/bin/env bash
# One-stage EM+IQL comparison for current CT encoder vs local-global encoder.
#
# Protocol follows scripts/cancer/train/run_em_iql_lr_ablation.sh runA_sym3e4:
#   actor/q/v lr = 3e-4, beta = 2.0, tau = 0.7, max_grad = 5.0,
#   em_m_steps_per_outer = 1000, HER refresh = 0, HER samples = 1.
# Only the CTHistoryEncoder layout changes:
#   baseline     -> model.inference.local_conv_layers=0
#   local_global -> model.inference.local_conv_layers=1

set -euo pipefail

if [[ -f /home/liam/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/liam/anaconda3/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)"
fi
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"
TEST_SPLIT="${TEST_SPLIT:-false}"
SEEDS_RAW="${GRID_SEEDS:-20 2020 202020}"
METHODS_RAW="${METHODS:-baseline local_global}"
read -r -a SEEDS <<< "${SEEDS_RAW}"
read -r -a METHODS <<< "${METHODS_RAW}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${GRID_ROOT:-${ROOT}/results/tumor/em_iql_local_global_compare/gamma_${GAMMA}}"
mkdir -p "${GRID_ROOT}/logs" "${GRID_ROOT}/ckpts" "${GRID_ROOT}/done"

SUMMARY="${GRID_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "method,seed,local_conv_layers,iql_actor_lr,iql_qf_lr,iql_vf_lr,iql_beta,iql_tau,iql_max_grad_norm,em_m_steps_per_outer,em_her_samples_per_transition,em_her_refresh_every,best_outer,best_val_mae_uns,eval_mae_norm,eval_mae_uns,eval_rmse_uns,em_ckpt,train_log,eval_log,finished_at" > "${SUMMARY}"
fi

IQL_ACTOR_LR="${IQL_ACTOR_LR:-3e-4}"
IQL_QF_LR="${IQL_QF_LR:-3e-4}"
IQL_VF_LR="${IQL_VF_LR:-3e-4}"
IQL_BETA="${IQL_BETA:-2.0}"
IQL_TAU="${IQL_TAU:-0.7}"
IQL_MAX_GRAD="${IQL_MAX_GRAD:-5.0}"
EM_M_STEPS="${EM_M_STEPS:-1000}"
EM_HER_SAMPLES="${EM_HER_SAMPLES:-1}"
EM_HER_REFRESH="${EM_HER_REFRESH:-0}"
GPU_WAIT_MEMORY_MB="${GPU_WAIT_MEMORY_MB:-1000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-em_iql_local_global_compare}"

parse_last() {
  local file="$1" pat="$2"
  grep -oP "${pat}" "${file}" 2>/dev/null | tail -n1 || true
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

local_layers_for_method() {
  case "$1" in
    baseline) echo 0 ;;
    local_global) echo 1 ;;
    *) echo "ERROR: unknown method '$1' (expected baseline|local_global)" >&2; return 1 ;;
  esac
}

run_one() {
  local method="$1" seed="$2"
  local local_layers
  local_layers="$(local_layers_for_method "${method}")"

  local tag="${method}_runA_sym3e4_seed${seed}"
  local em_dir="${GRID_ROOT}/ckpts/${tag}"
  local em_ckpt="${em_dir}/ct_iql_em_best.pt"
  local log_dir="${GRID_ROOT}/logs/${tag}"
  local done_flag="${GRID_ROOT}/done/${tag}.done"
  local train_log="${log_dir}/train.log"
  local eval_log="${log_dir}/eval.log"

  if [[ -f "${done_flag}" ]]; then
    echo "[skip] ${tag} (done marker exists)"
    return 0
  fi
  if grep -q "^${method},${seed}," "${SUMMARY}" 2>/dev/null; then
    echo "[skip] ${tag} already recorded in ${SUMMARY}"
    return 0
  fi

  mkdir -p "${em_dir}" "${log_dir}"

  echo "========== ${tag} =========="
  echo "  local_conv_layers=${local_layers}"
  echo "  seed=${seed} gamma=${GAMMA}"
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/train_ct_iql_em.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_her_refresh_every=${EM_HER_REFRESH}" \
    "exp.em_her_samples_per_transition=${EM_HER_SAMPLES}" \
    "exp.ct_num_workers=0" \
    "exp.iql_beta=${IQL_BETA}" \
    "exp.iql_tau=${IQL_TAU}" \
    "exp.iql_actor_lr=${IQL_ACTOR_LR}" \
    "exp.iql_qf_lr=${IQL_QF_LR}" \
    "exp.iql_vf_lr=${IQL_VF_LR}" \
    "exp.iql_max_grad_norm=${IQL_MAX_GRAD}" \
    "exp.em_m_steps_per_outer=${EM_M_STEPS}" \
    "+exp.em_ckpt_dir=${em_dir}" \
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.mlflow_combo_id=${method}_runA_sym3e4" \
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
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.mlflow_combo_id=${method}_runA_sym3e4" \
    2>&1 | tee "${eval_log}"

  local best_outer best_val eval_mae_norm eval_mae_uns eval_rmse_uns finished_at
  read -r best_outer best_val <<< "$(read_em_metrics "${em_ckpt}")"
  eval_mae_norm="$(parse_last "${eval_log}" 'MAE normalized: \K[0-9.eE+-]+')"
  eval_mae_uns="$(parse_last "${eval_log}" 'MAE unscaled: \K[0-9.eE+-]+')"
  eval_rmse_uns="$(parse_last "${eval_log}" 'RMSE unscaled: \K[0-9.eE+-]+')"
  [[ -z "${eval_mae_norm}" ]] && eval_mae_norm="NA"
  [[ -z "${eval_mae_uns}" ]] && eval_mae_uns="NA"
  [[ -z "${eval_rmse_uns}" ]] && eval_rmse_uns="NA"
  finished_at="$(date -Iseconds)"

  echo "${method},${seed},${local_layers},${IQL_ACTOR_LR},${IQL_QF_LR},${IQL_VF_LR},${IQL_BETA},${IQL_TAU},${IQL_MAX_GRAD},${EM_M_STEPS},${EM_HER_SAMPLES},${EM_HER_REFRESH},${best_outer},${best_val},${eval_mae_norm},${eval_mae_uns},${eval_rmse_uns},${em_ckpt},${train_log},${eval_log},${finished_at}" >> "${SUMMARY}"
  {
    echo "finished_at=${finished_at}"
    echo "method=${method}"
    echo "seed=${seed}"
    echo "local_conv_layers=${local_layers}"
    echo "em_ckpt=${em_ckpt}"
    echo "train_log=${train_log}"
    echo "eval_log=${eval_log}"
  } > "${done_flag}"
  echo "[done] ${tag}: eval_mae_uns=${eval_mae_uns}"
}

echo "[compare] one-stage EM+IQL"
echo "[compare] gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
echo "[compare] methods=(${METHODS[*]}) seeds=(${SEEDS[*]})"
echo "[compare] grid_root=${GRID_ROOT}"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${method}" "${seed}"
  done
done

python - "${SUMMARY}" <<'PY'
import csv, statistics, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
print("\nSummary:", p)
for m in sorted({r["method"] for r in rows}):
    vals = [float(r["eval_mae_uns"]) for r in rows if r["method"] == m and r["eval_mae_uns"] not in ("", "NA")]
    if vals:
        print(f"{m}: n={len(vals)} mean={statistics.mean(vals):.6f} median={statistics.median(vals):.6f} vals={[round(v, 6) for v in vals]}")
PY
