#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
DATASET="${1:?dataset: tumor or mimic}"
SEED="${2:?seed}"
GPU="${3:?gpu index}"
TASK_ROOT="$RUN_ROOT/$DATASET/cripo/seed_$SEED"
CKPT="$TASK_ROOT/checkpoints/ct_iql_em_best.pt"
OUTPUT="$TASK_ROOT/inference_profile.json"
LOG="$TASK_ROOT/logs/inference_profile.log"

if [[ ! -s "$CKPT" ]]; then
  printf 'missing checkpoint: %s\n' "$CKPT" >&2
  exit 2
fi

source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip
cd "$PROJECT_ROOT"

common=(
  "+model=vcip"
  "exp.seed=$SEED"
  "exp.test=true"
  "exp.use_mlflow=false"
  "+exp.em_eval_ckpt=$CKPT"
  "+exp.efficiency_output=$OUTPUT"
)
if [[ "$DATASET" == "tumor" ]]; then
  dataset_args=(
    "+dataset=cancer_sim_cont"
    "dataset.seed=$SEED"
    "dataset.coeff=4"
    "exp.load_data=false"
    "exp.batch_size_val=200"
  )
elif [[ "$DATASET" == "mimic" ]]; then
  dataset_args=(
    "+dataset=mimic3_synthetic_gift"
    "dataset.max_number=500"
    "dataset.data_seed=10"
    "exp.batch_size_val=100"
    "+model.inference.local_conv_layers=1"
  )
else
  printf 'unsupported dataset=%s\n' "$DATASET" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
HYDRA_FULL_ERROR=1 \
python -u scripts/efficiency/profile_cripo_efficiency.py \
  "${dataset_args[@]}" "${common[@]}" > "$LOG" 2>&1

[[ -s "$OUTPUT" ]]
