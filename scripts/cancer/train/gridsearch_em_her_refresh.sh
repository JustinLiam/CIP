#!/usr/bin/env bash
# Grid search over exp.em_her_refresh_every for CT+IQL EM training.
#
# Sweeps:
#   exp.em_her_refresh_every  - 0, 1, 5, 10, 20 (override via GRID_HER_REFRESH_LIST)
#   exp.seed                  - 3, 30, 303, 3030, 30303 (override via GRID_SEEDS)
#
# Results are logged to MLflow only (no summary.csv):
#   experiment: exp.mlflow_experiment=her_refresh
#   tag combo_id: her{N}  (e.g. her0, her10)
#   train run metrics: best/val_mae_uns, val/sim/mae_uns per outer
#   eval run metrics:  eval/tau{tau}/mae_uns
#
# Checkpoints (isolated per combo):
#   grid_results/em_her_refresh/gamma_{GAMMA}/ckpts/her{N}_seed{SEED}/ct_iql_em_best.pt
#
# Usage (from repo root):
#   bash scripts/cancer/train/gridsearch_em_her_refresh.sh [GPU] [GAMMA]
#
# Optional env:
#   GRID_SEEDS="3 30"                    - override seed list
#   GRID_HER_REFRESH_LIST="0 10"         - override her_refresh list
#   GRID_MLFLOW_EXPERIMENT=her_refresh   - MLflow experiment name
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2   - shard jobs across workers/GPUs
#   GRID_SKIP_EVAL=1                     - train only (no eval_iql_planner)
#   GRID_FORCE=1                         - re-run even if .done marker exists
#
# Multi-GPU example:
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_em_her_refresh.sh 0 4
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_em_her_refresh.sh 1 4

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"

GRID_WORKER_ID="${GRID_WORKER_ID:-0}"
GRID_NUM_WORKERS="${GRID_NUM_WORKERS:-1}"
GRID_MLFLOW_EXPERIMENT="${GRID_MLFLOW_EXPERIMENT:-her_refresh}"
GRID_SKIP_EVAL="${GRID_SKIP_EVAL:-0}"
GRID_FORCE="${GRID_FORCE:-0}"

if (( GRID_NUM_WORKERS < 1 )) || (( GRID_WORKER_ID < 0 )) || (( GRID_WORKER_ID >= GRID_NUM_WORKERS )); then
  echo "ERROR: invalid sharding: GRID_WORKER_ID=${GRID_WORKER_ID} GRID_NUM_WORKERS=${GRID_NUM_WORKERS}" >&2
  exit 1
fi

if [[ -n "${GRID_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${GRID_SEEDS}"
else
  SEEDS=(3 30 303 3030 30303)
fi

if [[ -n "${GRID_HER_REFRESH_LIST:-}" ]]; then
  read -r -a HER_REFRESH_LIST <<< "${GRID_HER_REFRESH_LIST}"
else
  HER_REFRESH_LIST=(0 1 5 10 20)
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${ROOT}/grid_results/em_her_refresh/gamma_${GAMMA}"
mkdir -p "${GRID_ROOT}/ckpts" "${GRID_ROOT}/logs" "${GRID_ROOT}/done"

echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${GAMMA} gpu=${GPU}"
echo "  seeds=(${SEEDS[*]})"
echo "  em_her_refresh_every=(${HER_REFRESH_LIST[*]})"
echo "  mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
echo "  grid_root=${GRID_ROOT}"

job_idx=0
for her in "${HER_REFRESH_LIST[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if (( job_idx % GRID_NUM_WORKERS != GRID_WORKER_ID )); then
      ((job_idx++)) || true
      continue
    fi
    ((job_idx++)) || true

    combo_id="her${her}"
    tag="her${her}_seed${seed}"
    em_dir="${GRID_ROOT}/ckpts/${tag}"
    em_ckpt="${em_dir}/ct_iql_em_best.pt"
    log_dir="${GRID_ROOT}/logs/${tag}"
    done_flag="${GRID_ROOT}/done/${tag}.done"
    mkdir -p "${em_dir}" "${log_dir}"

    if [[ "${GRID_FORCE}" != "1" && -f "${done_flag}" ]]; then
      echo "[SKIP] ${tag} (done marker exists; set GRID_FORCE=1 to re-run)"
      continue
    fi

    echo "========== ${tag} | em_her_refresh_every=${her} seed=${seed} =========="

    train_log="${log_dir}/train.log"
    CUDA_VISIBLE_DEVICES="${GPU}" python runnables/train_ct_iql_em.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
      exp.seed="${seed}" dataset.coeff="${GAMMA}" \
      "exp.em_her_refresh_every=${her}" \
      "+exp.em_ckpt_dir=${em_dir}" \
      "exp.mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}" \
      "exp.mlflow_combo_id=${combo_id}" \
      2>&1 | tee "${train_log}"

    if [[ ! -f "${em_ckpt}" ]]; then
      echo "ERROR: EM checkpoint missing after train: ${em_ckpt}" >&2
      exit 1
    fi

    if [[ "${GRID_SKIP_EVAL}" == "1" ]]; then
      echo "[train-only] ${tag} -> ${em_ckpt}"
      date -Iseconds > "${done_flag}"
      continue
    fi

    eval_log="${log_dir}/eval.log"
    CUDA_VISIBLE_DEVICES="${GPU}" python runnables/eval_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
      exp.seed="${seed}" dataset.coeff="${GAMMA}" \
      exp.test=false \
      "exp.em_eval_ckpt=${em_ckpt}" \
      "exp.mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}" \
      "exp.mlflow_combo_id=${combo_id}" \
      2>&1 | tee "${eval_log}"

    {
      echo "finished_at=$(date -Iseconds)"
      echo "em_her_refresh_every=${her}"
      echo "seed=${seed}"
      echo "em_ckpt=${em_ckpt}"
      echo "train_log=${train_log}"
      echo "eval_log=${eval_log}"
      echo "mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
      echo "mlflow_combo_id=${combo_id}"
    } > "${done_flag}"

    echo "[DONE] ${tag}"
  done
done

echo "Grid search worker ${GRID_WORKER_ID} finished."
echo "View results in MLflow experiment '${GRID_MLFLOW_EXPERIMENT}' (filter tags: combo_id, seed)."
