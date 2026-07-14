#!/usr/bin/env bash
# Canonical Tumor main/ablation experiment for WeightNet alignment.
# Full CRIPO (Sinkhorn) is stored with main experiments; MMD and uniform-weight
# controls are stored with ablation studies. No output is written to grid_results.
#
# Usage:
#   bash scripts/cancer/train/run_cripo_weightnet_ablation.sh 0 1

set -euo pipefail

GPU_A="${1:-0}"
GPU_B="${2:-1}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

SEEDS_A="10 101 1010"
SEEDS_B="10101 101010"
DATASET_SEED=10
GAMMAS="1 2 3 4"
VARIANTS="sinkhorn mmd uniform"
PROTOCOL="scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh"
EXPERIMENT_TAG="noise0_seq60_fixed_dseed10"
MAIN_ROOT="${MAIN_RESULTS_ROOT:-${ROOT}/results/main_experiments/tumor/cripo/${EXPERIMENT_TAG}}"
ABLATION_ROOT="${ABLATION_RESULTS_ROOT:-${ROOT}/results/ablation_studies/tumor/weightnet_alignment/${EXPERIMENT_TAG}}"
FORCE="${FORCE:-0}"

export TUMOR_NOISE_SCALE=0.0

variant_settings() {
  case "$1" in
    sinkhorn) printf 'true sinkhorn\n' ;;
    mmd) printf 'true mmd\n' ;;
    uniform) printf 'false sinkhorn\n' ;;
    *) echo "ERROR: unknown variant $1" >&2; return 2 ;;
  esac
}

variant_root() {
  local variant="$1" gamma="$2"
  if [[ "${variant}" == "sinkhorn" ]]; then
    printf '%s/gamma_%s/sinkhorn\n' "${MAIN_ROOT}" "${gamma}"
  else
    printf '%s/%s/gamma_%s\n' "${ABLATION_ROOT}" "${variant}" "${gamma}"
  fi
}

for variant in ${VARIANTS}; do
  read -r use_weight_net align_loss <<< "$(variant_settings "${variant}")"
  for gamma in ${GAMMAS}; do
    run_root="$(variant_root "${variant}" "${gamma}")"
    mkdir -p "${run_root}"
    cat > "${run_root}/experiment_meta.txt" <<META
method=CRIPO
ablation=weightnet_alignment
variant=${variant}
ct_use_weight_net=${use_weight_net}
ct_align_loss=${align_loss}
gamma=${gamma}
noise=0.0
max_seq_length=60
dataset_seed=10
dataset_seed_mode=fixed
train_val_test=1000/200/200
exp_seeds=10 101 1010 10101 101010
test_split=true
eval_tau=1 2 3 4 5 6
checkpoint_selection=max_rmse_uns_tau1_2_3_4_5_6
protocol=${PROTOCOL}
started_at=$(date -Iseconds)
META

    echo "[ablation] variant=${variant} gamma=${gamma} root=${run_root}"
    env \
      GRID_ROOT="${run_root}/gpu${GPU_A}" \
      GRID_SEEDS="${SEEDS_A}" \
      DATASET_SEED="${DATASET_SEED}" \
      DATASET_SEED_MODE=fixed \
      MAX_SEQ_LENGTH=60 \
      TEST_SPLIT=true \
      CT_USE_WEIGHT_NET="${use_weight_net}" \
      CT_ALIGN_LOSS="${align_loss}" \
      FORCE="${FORCE}" \
      MLFLOW_EXPERIMENT="cripo_weightnet_ablation_${variant}_gamma${gamma}" \
      bash "${PROTOCOL}" "${GPU_A}" "${gamma}" \
      > "${run_root}/gpu${GPU_A}_launcher.log" 2>&1 &
    pid_a=$!

    env \
      GRID_ROOT="${run_root}/gpu${GPU_B}" \
      GRID_SEEDS="${SEEDS_B}" \
      DATASET_SEED="${DATASET_SEED}" \
      DATASET_SEED_MODE=fixed \
      MAX_SEQ_LENGTH=60 \
      TEST_SPLIT=true \
      CT_USE_WEIGHT_NET="${use_weight_net}" \
      CT_ALIGN_LOSS="${align_loss}" \
      FORCE="${FORCE}" \
      MLFLOW_EXPERIMENT="cripo_weightnet_ablation_${variant}_gamma${gamma}" \
      bash "${PROTOCOL}" "${GPU_B}" "${gamma}" \
      > "${run_root}/gpu${GPU_B}_launcher.log" 2>&1 &
    pid_b=$!

    wait "${pid_a}"
    wait "${pid_b}"
    date -Iseconds > "${run_root}/DONE.txt"
  done
done

python runnables/summarize_cripo_weightnet_ablation.py \
  --main-root "${MAIN_ROOT}" \
  --ablation-root "${ABLATION_ROOT}"

echo "[ablation] all conditions completed"
