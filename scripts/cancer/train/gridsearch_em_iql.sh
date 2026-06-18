#!/usr/bin/env bash
# Grid search over IQL + EM M-step hyperparameters (train_ct_iql_em.py).
#
# Sweeps (defaults):
#   exp.iql_beta              - 1.0, 2.0, 3.0
#   exp.iql_tau               - 0.5, 0.7, 0.9
#   exp.iql_actor_lr          - 1e-4, 3e-4, 1e-3
#   exp.iql_qf_lr             - 1e-4, 3e-4, 1e-3
#   exp.iql_max_grad_norm     - 2.0, 5.0
#   exp.em_m_steps_per_outer  - 500, 1000, 2000
#   exp.seed                  - 2020, 20202, 202020 (override via GRID_SEEDS)
#
# Fixed (not swept):
#   exp.em_her_refresh_every=0
#   exp.em_her_samples_per_transition=1  (vcip.yaml default)
#   exp.iql_vf_lr = exp.iql_actor_lr  (tied each combo)
#
# Combos per seed: 3*3*3*3*2*3 = 486  |  Total jobs (3 seeds): 1458
#
# Results: MLflow experiment (default: em_iql_grid_her0)
#   train stage tag: ct_iql_em | eval stage tag: eval
#   tags: combo_id, seed, gamma
#
# Checkpoints (per machine unless grid_results is on shared NFS):
#   grid_results/em_iql_grid_her0/gamma_{GAMMA}/ckpts/{tag}/ct_iql_em_best.pt
#
# Usage (from repo root):
#   bash scripts/cancer/train/gridsearch_em_iql.sh [GPU] [GAMMA]
#
# Optional env:
#   GRID_SEEDS="3 30"                         - override seed list
#   GRID_BETA_LIST="2.0 3.0"                  - override beta list
#   GRID_TAU_LIST="0.5 0.7"                   - override tau list
#   GRID_ACTOR_LR_LIST="3e-4 1e-3"            - override actor_lr list
#   GRID_QF_LR_LIST="1e-4 3e-4"               - override qf_lr list
#   GRID_MAX_GRAD_LIST="2.0 5.0"              - override max_grad list
#   GRID_M_STEPS_LIST="500 1000"              - override m_steps list
#   GRID_MLFLOW_EXPERIMENT=em_iql_grid_her0   - MLflow experiment name
#   GRID_MLFLOW_URI=http://host:5000          - central tracking server (see MLflow note below)
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=6       - shard jobs across cluster workers/GPUs
#   GRID_SKIP_EVAL=1                          - train only
#   GRID_FORCE=1                              - re-run even if .done marker exists
#
# --- 3-machine / 6-GPU example (GRID_NUM_WORKERS must match total parallel workers) ---
#   Machine A (2x4090): GRID_WORKER_ID=0 GPU=0  |  GRID_WORKER_ID=1 GPU=1
#   Machine B (2x4090): GRID_WORKER_ID=2 GPU=0  |  GRID_WORKER_ID=3 GPU=1
#   Machine C (2xH800):  GRID_WORKER_ID=4 GPU=0  |  GRID_WORKER_ID=5 GPU=1
#
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=6 bash scripts/cancer/train/gridsearch_em_iql.sh 0 4
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=6 bash scripts/cancer/train/gridsearch_em_iql.sh 1 4
#   ... (unique WORKER_ID 0..5 on each machine, one process per GPU)
#
# --- MLflow across machines (frp / no shared LAN) ---
#   Keeping exp.mlflow_uri=http://127.0.0.1:5000 on every machine writes to LOCAL mlruns only;
#   a MacBook cannot see all runs in one UI that way.
#   Use ONE central server and point all workers at it via GRID_MLFLOW_URI, e.g.:
#     - Run ``mlflow server --host 0.0.0.0 --port 5000`` on MacBook (or a always-on host).
#     - Expose port 5000 through frp so training machines reach the same URI.
#     - Launch grid with: GRID_MLFLOW_URI=http://<frp-host>:<frp-port> bash ...gridsearch_em_iql.sh ...
#   If the tracking server is unreachable, training continues (metrics may be dropped).

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"

GRID_WORKER_ID="${GRID_WORKER_ID:-0}"
GRID_NUM_WORKERS="${GRID_NUM_WORKERS:-1}"
GRID_MLFLOW_EXPERIMENT="${GRID_MLFLOW_EXPERIMENT:-em_iql_grid_her0}"
GRID_MLFLOW_URI="${GRID_MLFLOW_URI:-}"
GRID_SKIP_EVAL="${GRID_SKIP_EVAL:-0}"
GRID_FORCE="${GRID_FORCE:-0}"

if (( GRID_NUM_WORKERS < 1 )) || (( GRID_WORKER_ID < 0 )) || (( GRID_WORKER_ID >= GRID_NUM_WORKERS )); then
  echo "ERROR: invalid sharding: GRID_WORKER_ID=${GRID_WORKER_ID} GRID_NUM_WORKERS=${GRID_NUM_WORKERS}" >&2
  exit 1
fi

if [[ -n "${GRID_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${GRID_SEEDS}"
else
  SEEDS=(2020 20202 202020)
fi

if [[ -n "${GRID_BETA_LIST:-}" ]]; then
  read -r -a BETA_LIST <<< "${GRID_BETA_LIST}"
else
  BETA_LIST=(1.0 2.0 3.0)
fi

if [[ -n "${GRID_TAU_LIST:-}" ]]; then
  read -r -a TAU_LIST <<< "${GRID_TAU_LIST}"
else
  TAU_LIST=(0.5 0.7 0.9)
fi

if [[ -n "${GRID_ACTOR_LR_LIST:-}" ]]; then
  read -r -a ACTOR_LR_LIST <<< "${GRID_ACTOR_LR_LIST}"
else
  ACTOR_LR_LIST=(1e-4 3e-4 1e-3)
fi

if [[ -n "${GRID_QF_LR_LIST:-}" ]]; then
  read -r -a QF_LR_LIST <<< "${GRID_QF_LR_LIST}"
else
  QF_LR_LIST=(1e-4 3e-4 1e-3)
fi

if [[ -n "${GRID_MAX_GRAD_LIST:-}" ]]; then
  read -r -a MAX_GRAD_LIST <<< "${GRID_MAX_GRAD_LIST}"
else
  MAX_GRAD_LIST=(2.0 5.0)
fi

if [[ -n "${GRID_M_STEPS_LIST:-}" ]]; then
  read -r -a M_STEPS_LIST <<< "${GRID_M_STEPS_LIST}"
else
  M_STEPS_LIST=(500 1000 2000)
fi

# Encode float/string params for combo_id / filesystem tag (no dots).
encode_beta() {
  case "$1" in
    1.0|1) echo "b10" ;;
    2.0|2) echo "b20" ;;
    3.0|3) echo "b30" ;;
    *) echo "b${1//./}" ;;
  esac
}

encode_tau() {
  case "$1" in
    0.5) echo "t05" ;;
    0.7) echo "t07" ;;
    0.9) echo "t09" ;;
    *) echo "t${1//./}" ;;
  esac
}

encode_lr() {
  case "$1" in
    1e-4) echo "1e4" ;;
    3e-4) echo "3e4" ;;
    1e-3) echo "1e3" ;;
    *) echo "${1//-/_}" ;;
  esac
}

encode_grad() {
  case "$1" in
    2.0|2) echo "g2" ;;
    5.0|5) echo "g5" ;;
    *) echo "g${1//./}" ;;
  esac
}

encode_m_steps() {
  case "$1" in
    500) echo "m500" ;;
    1000) echo "m1k" ;;
    2000) echo "m2k" ;;
    *) echo "m${1}" ;;
  esac
}

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${ROOT}/grid_results/em_iql_grid_her0/gamma_${GAMMA}"
mkdir -p "${GRID_ROOT}/ckpts" "${GRID_ROOT}/logs" "${GRID_ROOT}/done"

MLFLOW_URI_ARGS=()
if [[ -n "${GRID_MLFLOW_URI}" ]]; then
  MLFLOW_URI_ARGS=("exp.mlflow_uri=${GRID_MLFLOW_URI}")
fi

echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${GAMMA} gpu=${GPU}"
echo "  seeds=(${SEEDS[*]})"
echo "  iql_beta=(${BETA_LIST[*]})"
echo "  iql_tau=(${TAU_LIST[*]})"
echo "  iql_actor_lr=(${ACTOR_LR_LIST[*]})"
echo "  iql_qf_lr=(${QF_LR_LIST[*]})"
echo "  iql_max_grad_norm=(${MAX_GRAD_LIST[*]})"
echo "  em_m_steps_per_outer=(${M_STEPS_LIST[*]})"
echo "  em_her_refresh_every=1 (fixed)"
echo "  mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
if [[ -n "${GRID_MLFLOW_URI}" ]]; then
  echo "  mlflow_uri=${GRID_MLFLOW_URI}"
else
  echo "  mlflow_uri=(vcip.yaml default; typically http://127.0.0.1:5000 on this host)"
fi
echo "  grid_root=${GRID_ROOT}"

job_idx=0
for beta in "${BETA_LIST[@]}"; do
  for tau in "${TAU_LIST[@]}"; do
    for actor_lr in "${ACTOR_LR_LIST[@]}"; do
      for qf_lr in "${QF_LR_LIST[@]}"; do
        for max_grad in "${MAX_GRAD_LIST[@]}"; do
          for m_steps in "${M_STEPS_LIST[@]}"; do
            beta_id="$(encode_beta "${beta}")"
            tau_id="$(encode_tau "${tau}")"
            alr_id="$(encode_lr "${actor_lr}")"
            qlr_id="$(encode_lr "${qf_lr}")"
            grad_id="$(encode_grad "${max_grad}")"
            m_id="$(encode_m_steps "${m_steps}")"
            combo_id="${beta_id}_${tau_id}_alr${alr_id}_qlr${qlr_id}_${grad_id}_${m_id}"

            for seed in "${SEEDS[@]}"; do
              if (( job_idx % GRID_NUM_WORKERS != GRID_WORKER_ID )); then
                ((job_idx++)) || true
                continue
              fi
              ((job_idx++)) || true

              tag="${combo_id}_seed${seed}"
              em_dir="${GRID_ROOT}/ckpts/${tag}"
              em_ckpt="${em_dir}/ct_iql_em_best.pt"
              log_dir="${GRID_ROOT}/logs/${tag}"
              done_flag="${GRID_ROOT}/done/${tag}.done"
              mkdir -p "${em_dir}" "${log_dir}"

              if [[ "${GRID_FORCE}" != "1" && -f "${done_flag}" ]]; then
                echo "[SKIP] ${tag} (done marker exists; set GRID_FORCE=1 to re-run)"
                continue
              fi

              echo "========== ${tag} =========="
              echo "  iql_beta=${beta} iql_tau=${tau}"
              echo "  iql_actor_lr=${actor_lr} iql_qf_lr=${qf_lr} iql_vf_lr=${actor_lr}"
              echo "  iql_max_grad_norm=${max_grad} em_m_steps_per_outer=${m_steps}"
              echo "  em_her_refresh_every=0 em_her_samples_per_transition=1 seed=${seed}"

              train_log="${log_dir}/train.log"
              CUDA_VISIBLE_DEVICES="${GPU}" python runnables/train_ct_iql_em.py \
                +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
                exp.seed="${seed}" dataset.coeff="${GAMMA}" \
                "exp.em_her_refresh_every=0" \
                "exp.em_her_samples_per_transition=1" \
                "exp.iql_beta=${beta}" \
                "exp.iql_tau=${tau}" \
                "exp.iql_actor_lr=${actor_lr}" \
                "exp.iql_qf_lr=${qf_lr}" \
                "exp.iql_vf_lr=${actor_lr}" \
                "exp.iql_max_grad_norm=${max_grad}" \
                "exp.em_m_steps_per_outer=${m_steps}" \
                "+exp.em_ckpt_dir=${em_dir}" \
                "exp.mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}" \
                "exp.mlflow_combo_id=${combo_id}" \
                "${MLFLOW_URI_ARGS[@]}" \
                2>&1 | tee "${train_log}"

              if [[ ! -f "${em_ckpt}" ]]; then
                echo "ERROR: EM checkpoint missing after train: ${em_ckpt}" >&2
                exit 1
              fi

              if [[ "${GRID_SKIP_EVAL}" == "1" ]]; then
                echo "[train-only] ${tag} -> ${em_ckpt}"
                {
                  echo "finished_at=$(date -Iseconds)"
                  echo "combo_id=${combo_id}"
                  echo "seed=${seed}"
                  echo "iql_beta=${beta}"
                  echo "iql_tau=${tau}"
                  echo "iql_actor_lr=${actor_lr}"
                  echo "iql_qf_lr=${qf_lr}"
                  echo "iql_max_grad_norm=${max_grad}"
                  echo "em_m_steps_per_outer=${m_steps}"
                  echo "em_her_refresh_every=0"
                  echo "em_her_samples_per_transition=1"
                  echo "em_ckpt=${em_ckpt}"
                  echo "train_log=${train_log}"
                } > "${done_flag}"
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
                "${MLFLOW_URI_ARGS[@]}" \
                2>&1 | tee "${eval_log}"

              {
                echo "finished_at=$(date -Iseconds)"
                echo "combo_id=${combo_id}"
                echo "seed=${seed}"
                echo "iql_beta=${beta}"
                echo "iql_tau=${tau}"
                echo "iql_actor_lr=${actor_lr}"
                echo "iql_qf_lr=${qf_lr}"
                echo "iql_max_grad_norm=${max_grad}"
                echo "em_m_steps_per_outer=${m_steps}"
                echo "em_her_refresh_every=0"
                echo "em_her_samples_per_transition=1"
                echo "em_ckpt=${em_ckpt}"
                echo "train_log=${train_log}"
                echo "eval_log=${eval_log}"
                echo "mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
                echo "mlflow_combo_id=${combo_id}"
              } > "${done_flag}"

              echo "[DONE] ${tag}"
            done
          done
        done
      done
    done
  done
done

echo "Grid search worker ${GRID_WORKER_ID} finished."
echo "View results in MLflow experiment '${GRID_MLFLOW_EXPERIMENT}' (filter tags: combo_id, seed)."
