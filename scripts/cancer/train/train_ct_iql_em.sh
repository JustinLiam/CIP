#!/usr/bin/env bash
# CT+IQL unified EM training -> eval (cancer_sim_cont).
# Usage from repo root:
#   bash scripts/cancer/train/train_ct_iql_em.sh [GPU] [GAMMA] [SEED]
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"
SEED="${3:-10}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

EM_DIR="${ROOT}/em_checkpoints/seed_${SEED}_gamma_${GAMMA}"
EM_CKPT="${EM_DIR}/ct_iql_em_best.pt"

mkdir -p "${EM_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python runnables/train_ct_iql_em.py \
  +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
  exp.seed="${SEED}" dataset.coeff="${GAMMA}" \
  "+exp.em_ckpt_dir=${EM_DIR}"

if [[ ! -f "${EM_CKPT}" ]]; then
  echo "ERROR: EM checkpoint missing: ${EM_CKPT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" python runnables/eval_iql_planner.py \
  +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
  exp.seed="${SEED}" dataset.coeff="${GAMMA}" exp.test=false \
  "exp.em_eval_ckpt=${EM_CKPT}"

echo "Done. EM ckpt: ${EM_CKPT}"
