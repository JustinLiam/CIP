#!/usr/bin/env bash
# Grid search over exp.em_her_samples_per_transition for static HER replay.
#
# Sweeps:
#   exp.em_her_samples_per_transition - 1, 2, 4, 8 (override via GRID_HER_SAMPLES_LIST)
#   exp.seed                          - 3, 30, 303, 3030, 30303 (override via GRID_SEEDS)
#
# Fixed:
#   exp.em_her_refresh_every=0, so the enlarged HER replay is built once and
#   kept unchanged throughout EM training.
#
# Results are logged to MLflow only (no summary.csv):
#   experiment: exp.mlflow_experiment=her_samples
#   tag combo_id: herk{K}  (e.g. herk4)
#
# Checkpoints (isolated per combo):
#   grid_results/em_her_samples/gamma_{GAMMA}/ckpts/herk{K}_seed{SEED}/ct_iql_em_best.pt
#
# Usage (from repo root):
#   bash scripts/cancer/train/gridsearch_em_her_samples.sh [GPU] [GAMMA]
#
# Optional env:
#   GRID_SEEDS="3 30"                    - override seed list
#   GRID_HER_SAMPLES_LIST="1 4"          - override samples-per-transition list
#   GRID_MLFLOW_EXPERIMENT=her_samples   - MLflow experiment name
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2  - shard jobs across workers/GPUs
#   GRID_SKIP_EVAL=1                     - train only (no eval_iql_planner)
#   GRID_FORCE=1                         - re-run even if .done marker exists
#
# Multi-GPU example:
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_em_her_samples.sh 0 4
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_em_her_samples.sh 1 4

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"

GRID_WORKER_ID="${GRID_WORKER_ID:-0}"
GRID_NUM_WORKERS="${GRID_NUM_WORKERS:-1}"
GRID_MLFLOW_EXPERIMENT="${GRID_MLFLOW_EXPERIMENT:-her_samples}"
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

if [[ -n "${GRID_HER_SAMPLES_LIST:-}" ]]; then
  read -r -a HER_SAMPLES_LIST <<< "${GRID_HER_SAMPLES_LIST}"
else
  HER_SAMPLES_LIST=(1 2 4 8)
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${ROOT}/grid_results/em_her_samples/gamma_${GAMMA}"
mkdir -p "${GRID_ROOT}/ckpts" "${GRID_ROOT}/logs" "${GRID_ROOT}/done"

echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${GAMMA} gpu=${GPU}"
echo "  seeds=(${SEEDS[*]})"
echo "  em_her_refresh_every=0"
echo "  em_her_samples_per_transition=(${HER_SAMPLES_LIST[*]})"
echo "  mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
echo "  grid_root=${GRID_ROOT}"

job_idx=0
for k in "${HER_SAMPLES_LIST[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if (( job_idx % GRID_NUM_WORKERS != GRID_WORKER_ID )); then
      ((job_idx++)) || true
      continue
    fi
    ((job_idx++)) || true

    combo_id="herk${k}"
    tag="herk${k}_seed${seed}"
    em_dir="${GRID_ROOT}/ckpts/${tag}"
    em_ckpt="${em_dir}/ct_iql_em_best.pt"
    log_dir="${GRID_ROOT}/logs/${tag}"
    done_flag="${GRID_ROOT}/done/${tag}.done"
    mkdir -p "${em_dir}" "${log_dir}"

    if [[ "${GRID_FORCE}" != "1" && -f "${done_flag}" ]]; then
      echo "[SKIP] ${tag} (done marker exists; set GRID_FORCE=1 to re-run)"
      continue
    fi

    echo "========== ${tag} | em_her_refresh_every=0 em_her_samples_per_transition=${k} seed=${seed} =========="

    train_log="${log_dir}/train.log"
    CUDA_VISIBLE_DEVICES="${GPU}" python runnables/train_ct_iql_em.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
      exp.seed="${seed}" dataset.coeff="${GAMMA}" \
      "exp.em_her_refresh_every=0" \
      "exp.em_her_samples_per_transition=${k}" \
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
      echo "em_her_refresh_every=0"
      echo "em_her_samples_per_transition=${k}"
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
