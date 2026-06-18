#!/usr/bin/env bash
# IQL learning-rate ablation under EM (her=0): symmetric 3e-4 vs asymmetric 1e-3/3e-4/1e-3.
#
# Runs (default):
#   Run A (baseline):  iql_actor_lr=iql_qf_lr=iql_vf_lr=3e-4
#   Run B (proposed):  iql_actor_lr=1e-3, iql_qf_lr=3e-4, iql_vf_lr=1e-3
#   Seeds:            2, 20, 202, 2020, 20202, 202020
#
# Total jobs: 2 configs x 6 seeds = 12
#
# Other IQL/EM settings follow configs/model/vcip.yaml defaults unless overridden below.
# Fixed for this study: em_her_refresh_every=0, em_her_samples_per_transition=1, ct_num_workers=0
#
# Outputs:
#   grid_results/em_iql_lr_ablation/gamma_{GAMMA}/
#     ckpts/{tag}/ct_iql_em_best.pt
#     logs/{tag}/train.log, eval.log
#     done/{tag}.done
#     summary.csv          (refreshed after each job + at end)
#
# MLflow experiment (default): em_iql_lr_ablation
#   tags: combo_id (runA_sym3e4 | runB_alr1e3_qlr3e4_vf1e3), seed, gamma, ablation=lr
#
# Usage (from repo root):
#   bash scripts/cancer/train/run_em_iql_lr_ablation.sh [GPU] [GAMMA]
#
# Optional env:
#   GRID_SEEDS="2 20 202"              - override seed list
#   GRID_MLFLOW_EXPERIMENT=em_iql_lr_ablation
#   GRID_MLFLOW_URI=http://host:5000   - central MLflow (recommended)
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2
#   GRID_SKIP_EVAL=1                   - train only
#   GRID_FORCE=1                       - re-run even if .done exists
#   GRID_RUNS="A"                      - A | B | AB (default AB)
#
# Multi-GPU example (2 workers):
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2 bash scripts/cancer/train/run_em_iql_lr_ablation.sh 0 4
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=2 bash scripts/cancer/train/run_em_iql_lr_ablation.sh 1 4

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"

GRID_WORKER_ID="${GRID_WORKER_ID:-0}"
GRID_NUM_WORKERS="${GRID_NUM_WORKERS:-1}"
GRID_MLFLOW_EXPERIMENT="${GRID_MLFLOW_EXPERIMENT:-em_iql_lr_ablation}"
GRID_MLFLOW_URI="${GRID_MLFLOW_URI:-}"
GRID_SKIP_EVAL="${GRID_SKIP_EVAL:-0}"
GRID_FORCE="${GRID_FORCE:-0}"
GRID_RUNS="${GRID_RUNS:-AB}"

if (( GRID_NUM_WORKERS < 1 )) || (( GRID_WORKER_ID < 0 )) || (( GRID_WORKER_ID >= GRID_NUM_WORKERS )); then
  echo "ERROR: invalid sharding: GRID_WORKER_ID=${GRID_WORKER_ID} GRID_NUM_WORKERS=${GRID_NUM_WORKERS}" >&2
  exit 1
fi

if [[ -n "${GRID_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${GRID_SEEDS}"
else
  SEEDS=(2 20 202 2020 20202 202020)
fi

# vcip.yaml-aligned defaults for this ablation (override via Hydra CLI if needed).
IQL_BETA="${IQL_BETA:-2.0}"
IQL_TAU="${IQL_TAU:-0.7}"
IQL_MAX_GRAD="${IQL_MAX_GRAD:-5.0}"
EM_M_STEPS="${EM_M_STEPS:-1000}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${ROOT}/grid_results/em_iql_lr_ablation/gamma_${GAMMA}"
mkdir -p "${GRID_ROOT}/ckpts" "${GRID_ROOT}/logs" "${GRID_ROOT}/done"

COLLECT_PY="${ROOT}/scripts/cancer/train/collect_lr_ablation_summary.py"

MLFLOW_URI_ARGS=()
if [[ -n "${GRID_MLFLOW_URI}" ]]; then
  MLFLOW_URI_ARGS=("exp.mlflow_uri=${GRID_MLFLOW_URI}")
fi

declare -a RUN_KEYS=()
if [[ "${GRID_RUNS}" == *A* ]]; then
  RUN_KEYS+=("A")
fi
if [[ "${GRID_RUNS}" == *B* ]]; then
  RUN_KEYS+=("B")
fi
if [[ ${#RUN_KEYS[@]} -eq 0 ]]; then
  echo "ERROR: GRID_RUNS must include A and/or B, got: ${GRID_RUNS}" >&2
  exit 1
fi

run_params() {
  local key="$1"
  case "${key}" in
    A)
      RUN_LABEL="runA_sym3e4"
      ACTOR_LR="3e-4"
      QF_LR="3e-4"
      VF_LR="3e-4"
      ;;
    B)
      RUN_LABEL="runB_alr1e3_qlr3e4_vf1e3"
      ACTOR_LR="1e-3"
      QF_LR="3e-4"
      VF_LR="1e-3"
      ;;
    *)
      echo "ERROR: unknown run key ${key}" >&2
      exit 1
      ;;
  esac
}

echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${GAMMA} gpu=${GPU}"
echo "  runs=(${RUN_KEYS[*]})"
echo "  seeds=(${SEEDS[*]})"
echo "  iql_beta=${IQL_BETA} iql_tau=${IQL_TAU} iql_max_grad_norm=${IQL_MAX_GRAD}"
echo "  em_m_steps_per_outer=${EM_M_STEPS}"
echo "  em_her_refresh_every=0 em_her_samples_per_transition=1 ct_num_workers=0"
echo "  mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
if [[ -n "${GRID_MLFLOW_URI}" ]]; then
  echo "  mlflow_uri=${GRID_MLFLOW_URI}"
fi
echo "  grid_root=${GRID_ROOT}"

job_idx=0
for run_key in "${RUN_KEYS[@]}"; do
  run_params "${run_key}"
  for seed in "${SEEDS[@]}"; do
    if (( job_idx % GRID_NUM_WORKERS != GRID_WORKER_ID )); then
      ((job_idx++)) || true
      continue
    fi
    ((job_idx++)) || true

    tag="${RUN_LABEL}_seed${seed}"
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
    echo "  run=${RUN_LABEL} seed=${seed}"
    echo "  iql_actor_lr=${ACTOR_LR} iql_qf_lr=${QF_LR} iql_vf_lr=${VF_LR}"

    train_log="${log_dir}/train.log"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" python runnables/train_ct_iql_em.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
      exp.seed="${seed}" dataset.coeff="${GAMMA}" \
      "exp.em_her_refresh_every=0" \
      "exp.em_her_samples_per_transition=1" \
      "exp.ct_num_workers=0" \
      "exp.iql_beta=${IQL_BETA}" \
      "exp.iql_tau=${IQL_TAU}" \
      "exp.iql_actor_lr=${ACTOR_LR}" \
      "exp.iql_qf_lr=${QF_LR}" \
      "exp.iql_vf_lr=${VF_LR}" \
      "exp.iql_max_grad_norm=${IQL_MAX_GRAD}" \
      "exp.em_m_steps_per_outer=${EM_M_STEPS}" \
      "+exp.em_ckpt_dir=${em_dir}" \
      "exp.mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}" \
      "exp.mlflow_combo_id=${RUN_LABEL}" \
      "${MLFLOW_URI_ARGS[@]}" \
      2>&1 | tee "${train_log}"
    train_exit=${PIPESTATUS[0]}
    set -e

    if [[ ! -f "${em_ckpt}" ]]; then
      echo "ERROR: EM checkpoint missing after train (exit=${train_exit}): ${em_ckpt}" >&2
      exit 1
    fi
    if [[ "${train_exit}" -ne 0 ]]; then
      echo "[WARN] Train exited with code ${train_exit}; ckpt exists, continuing."
    fi

    eval_log="${log_dir}/eval.log"
    if [[ "${GRID_SKIP_EVAL}" == "1" ]]; then
      echo "[train-only] ${tag} -> ${em_ckpt}"
    else
      set +e
      CUDA_VISIBLE_DEVICES="${GPU}" python runnables/eval_iql_planner.py \
        +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
        exp.seed="${seed}" dataset.coeff="${GAMMA}" \
        exp.test=false \
        "exp.em_eval_ckpt=${em_ckpt}" \
        "exp.mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}" \
        "exp.mlflow_combo_id=${RUN_LABEL}" \
        "${MLFLOW_URI_ARGS[@]}" \
        2>&1 | tee "${eval_log}"
      eval_exit=${PIPESTATUS[0]}
      set -e
      if [[ "${eval_exit}" -ne 0 ]]; then
        echo "ERROR: eval failed (exit=${eval_exit}) for ${tag}" >&2
        exit 1
      fi
    fi

    {
      echo "finished_at=$(date -Iseconds)"
      echo "ablation=lr"
      echo "run_label=${RUN_LABEL}"
      echo "combo_id=${RUN_LABEL}"
      echo "seed=${seed}"
      echo "gamma=${GAMMA}"
      echo "iql_actor_lr=${ACTOR_LR}"
      echo "iql_qf_lr=${QF_LR}"
      echo "iql_vf_lr=${VF_LR}"
      echo "iql_beta=${IQL_BETA}"
      echo "iql_tau=${IQL_TAU}"
      echo "iql_max_grad_norm=${IQL_MAX_GRAD}"
      echo "em_m_steps_per_outer=${EM_M_STEPS}"
      echo "em_her_refresh_every=0"
      echo "em_her_samples_per_transition=1"
      echo "train_exit=${train_exit}"
      echo "em_ckpt=${em_ckpt}"
      echo "train_log=${train_log}"
      echo "eval_log=${eval_log}"
      echo "mlflow_experiment=${GRID_MLFLOW_EXPERIMENT}"
    } > "${done_flag}"

    python "${COLLECT_PY}" --grid-root "${GRID_ROOT}" --gamma "${GAMMA}" || true

    echo "[DONE] ${tag}"
  done
done

python "${COLLECT_PY}" --grid-root "${GRID_ROOT}" --gamma "${GAMMA}"
echo "Worker ${GRID_WORKER_ID} finished. Summary: ${GRID_ROOT}/summary.csv"
