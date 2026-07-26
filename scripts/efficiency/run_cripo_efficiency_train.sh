#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/20260726}"
DATASET="${1:?dataset: tumor or mimic}"
SEED="${2:?seed}"
GPU="${3:?gpu index}"
OUTER_ITERS="${OUTER_ITERS:-18}"
M_STEPS="${M_STEPS:-1000}"

TASK_ROOT="$RUN_ROOT/$DATASET/cripo/seed_$SEED"
CKPT_DIR="$TASK_ROOT/checkpoints"
LOG_DIR="$TASK_ROOT/logs"
TRAIN_LOG="$LOG_DIR/train.log"
RESOURCE_LOG="$LOG_DIR/resources.tsv"
STATUS_FILE="$TASK_ROOT/status.txt"
METADATA_FILE="$TASK_ROOT/metadata.tsv"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip
cd "$PROJECT_ROOT"

printf 'dataset\t%s\nseed\t%s\ngpu\t%s\nouter_iters\t%s\nm_steps\t%s\ngit_commit\t%s\n' \
  "$DATASET" "$SEED" "$GPU" "$OUTER_ITERS" "$M_STEPS" "$(git rev-parse HEAD)" \
  > "$METADATA_FILE"

monitor_gpu() {
  local child_pid="$1"
  local max_used=0
  local sample_count=0
  printf 'unix_time\tgpu_used_mib\tgpu_util_percent\n' > "$RESOURCE_LOG"
  while kill -0 "$child_pid" 2>/dev/null; do
    local row used util
    row="$(nvidia-smi --id="$GPU" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | head -n 1)"
    used="${row%%,*}"
    util="${row##*,}"
    used="${used// /}"
    util="${util// /}"
    [[ "$used" =~ ^[0-9]+$ ]] || used=0
    (( used > max_used )) && max_used="$used"
    printf '%s\t%s\t%s\n' "$(date +%s)" "$used" "$util" >> "$RESOURCE_LOG"
    sample_count=$((sample_count + 1))
    sleep 1
  done
  printf 'peak_gpu_mib\t%s\nsamples\t%s\n' "$max_used" "$sample_count" \
    >> "$METADATA_FILE"
}

common=(
  "+model=vcip"
  "exp.seed=$SEED"
  "exp.use_mlflow=false"
  "exp.test=false"
  "+exp.em_outer_iters=$OUTER_ITERS"
  "+exp.em_m_steps_per_outer=$M_STEPS"
  "+exp.iql_actor_update=awr"
  "+exp.iql_beta=2.0"
  "+exp.iql_adv_max=100.0"
  "exp.ct_use_weight_net=true"
  "exp.ct_align_loss=sinkhorn"
  "+exp.em_ckpt_dir=$CKPT_DIR"
)

if [[ "$DATASET" == "tumor" ]]; then
  dataset_args=(
    "+dataset=cancer_sim_cont"
    "dataset.seed=$SEED"
    "dataset.coeff=4"
    "exp.load_data=false"
    "+exp.em_e_epochs=5"
    "+exp.em_e_w_lr=0.01"
  )
elif [[ "$DATASET" == "mimic" ]]; then
  dataset_args=(
    "+dataset=mimic3_synthetic_gift"
    "dataset.max_number=500"
    "dataset.data_seed=10"
    "exp.batch_size_val=512"
    "+model.inference.local_conv_layers=1"
    "+exp.em_e_epochs=5"
    "+exp.em_e_w_lr=0.01"
  )
else
  printf 'unsupported dataset=%s\n' "$DATASET" >&2
  exit 2
fi

printf 'RUNNING\n' > "$STATUS_FILE"
start_ns="$(date +%s%N)"
set +e
CUDA_VISIBLE_DEVICES="$GPU" \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
HYDRA_FULL_ERROR=1 \
python -u runnables/train_ct_iql_em.py \
  "${dataset_args[@]}" "${common[@]}" \
  > "$TRAIN_LOG" 2>&1 &
child_pid=$!
monitor_gpu "$child_pid" &
monitor_pid=$!
wait "$child_pid"
exit_code=$?
wait "$monitor_pid" || true
set -e
end_ns="$(date +%s%N)"

elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
printf 'elapsed_ms\t%s\nexit_code\t%s\n' "$elapsed_ms" "$exit_code" \
  >> "$METADATA_FILE"

if [[ "$exit_code" -ne 0 ]]; then
  printf 'FAILED exit_code=%s\n' "$exit_code" > "$STATUS_FILE"
  exit "$exit_code"
fi

if [[ ! -s "$CKPT_DIR/ct_iql_em_best.pt" ]]; then
  printf 'FAILED missing_checkpoint\n' > "$STATUS_FILE"
  exit 3
fi

if rg -n 'Traceback|RuntimeError|CUDA out of memory|Error executing job|Killed' \
  "$TRAIN_LOG" > "$LOG_DIR/error_signatures.txt"; then
  printf 'FAILED error_signature\n' > "$STATUS_FILE"
  exit 4
fi

printf 'COMPLETED\n' > "$STATUS_FILE"
