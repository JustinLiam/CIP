#!/usr/bin/env bash
# CT (train_ct) -> IQL (train_iql_planner) -> eval (eval_iql_planner) for cancer_sim_cont.
#
# Edit the constants below, then from repo root:
#   bash scripts/cancer/train/train_ct_iql.sh
#
# Metrics: IQL training + eval (scheme 3: eval/tau{k}/mae_uns) log to MLflow when exp.use_mlflow=true.
# Checkpoints (per seed):
#   ct_checkpoints/tumor_generator/seed_${SEED}/coeff_${GAMMA}/ct_best_encoder.pt
#   iql_models/tumor_generator/seed_${SEED}/coeff_${GAMMA}/iql_planner.pt

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

# --- configuration (edit in place) ---
GPU=0
GAMMA=4
SEEDS=(10 10101 101010)
EVAL_TAUS=(1 2 3 4 5 6 8 10 12 15)
# false = val split, true = test split; eval runs once per entry (separate MLflow run each)
EVAL_TESTS=(false true)

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

DATASET_NAME="tumor_generator"
IQL_EVAL_TAU_LIST="[$(IFS=,; echo "${EVAL_TAUS[*]}")]"

for SEED in "${SEEDS[@]}"; do
  CT_DIR="${ROOT}/ct_checkpoints/${DATASET_NAME}/seed_${SEED}/coeff_${GAMMA}/one_step"
  CT_CKPT="${CT_DIR}/ct_best_encoder.pt"
  IQL_DIR="${ROOT}/iql_models/${DATASET_NAME}/seed_${SEED}/coeff_${GAMMA}/one_step"
  IQL_CKPT="${IQL_DIR}/iql_planner.pt"

  echo "================================================================"
  echo "=== seed=${SEED} coeff=${GAMMA} | CT -> IQL -> eval taus=(${EVAL_TAUS[*]}) splits=(${EVAL_TESTS[*]}) ==="
  echo "================================================================"

  mkdir -p "${CT_DIR}"
  CUDA_VISIBLE_DEVICES=${GPU} python runnables/train_ct.py \
    +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${SEED}" dataset.coeff="${GAMMA}" \
    exp.ct_rollout_mode=none \
    exp.ct_weight_mode=offline_periodic \
    exp.ct_es_metric=mae_uw \
    "+exp.ct_ckpt_dir=${CT_DIR}"

  if [[ ! -f "${CT_CKPT}" ]]; then
    echo "ERROR: CT checkpoint missing: ${CT_CKPT}" >&2
    exit 1
  fi

  mkdir -p "${IQL_DIR}"
  CUDA_VISIBLE_DEVICES=${GPU} python runnables/train_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${SEED}" dataset.coeff="${GAMMA}" \
    exp.iql_inference_ckpt="${CT_CKPT}" \
    "+exp.iql_save_dir=${IQL_DIR}"

  if [[ ! -f "${IQL_CKPT}" ]]; then
    echo "ERROR: IQL checkpoint missing: ${IQL_CKPT}" >&2
    exit 1
  fi

  for EVAL_TEST in "${EVAL_TESTS[@]}"; do
    echo "--- seed=${SEED} eval taus=${IQL_EVAL_TAU_LIST} exp.test=${EVAL_TEST} (one MLflow run, scheme 3) ---"
    CUDA_VISIBLE_DEVICES=${GPU} python runnables/eval_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip_cancer "+model/hparams/cancer=${GAMMA}*" \
      exp.seed="${SEED}" dataset.coeff="${GAMMA}" \
      exp.test="${EVAL_TEST}" \
      "exp.iql_eval_tau_list=${IQL_EVAL_TAU_LIST}" \
      exp.iql_inference_ckpt="${CT_CKPT}" \
      exp.iql_eval_ckpt="${IQL_CKPT}"
  done
done

echo "Done. seeds=(${SEEDS[*]}) gamma=${GAMMA} eval_splits=(${EVAL_TESTS[*]})"
