#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
GIFT_ROOT="$PROJECT_ROOT/external_repos/GIFT"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/efficiency_kdd26/formal_20260726}"
TIMING_PATCH="${TIMING_PATCH:-$PROJECT_ROOT/scripts/efficiency/baseline_timing}"
CODE_COMMIT="${CODE_COMMIT:-b29df9d36edb1e391472f3f13dcf86e29e8ac3a9}"
HARNESS_COMMIT="${HARNESS_COMMIT:-5a72cd64d09a2d1a0f93503575e927f699384c7d}"
DATASET="${1:?dataset: tumor or mimic}"
MODEL="${2:?model}"
SEED="${3:?seed}"
GPU="${4:?gpu index}"

if [[ -z "${CONDA_ENV:-}" ]]; then
  case "$DATASET:$MODEL" in
    tumor:crn|*:ct) CONDA_ENV="pytorch-lightning" ;;
    *) CONDA_ENV="vcip" ;;
  esac
fi

TASK_ROOT="$RUN_ROOT/$DATASET/$MODEL/seed_$SEED"
LOG_DIR="$TASK_ROOT/logs"
TRAIN_LOG="$LOG_DIR/train.log"
RESOURCE_LOG="$LOG_DIR/resources.tsv"
TIMING_LOG="$TASK_ROOT/inference_timing.jsonl"
STATUS_FILE="$TASK_ROOT/status.txt"
METADATA_FILE="$TASK_ROOT/metadata.tsv"
EXP_NAME="efficiency_kdd26_${DATASET}_${MODEL}_${SEED}"

mkdir -p "$LOG_DIR"
source /home/liam/anaconda3/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "$CONDA_ENV"
cd "$GIFT_ROOT"

printf 'dataset\t%s\nmodel\t%s\nseed\t%s\ngpu\t%s\nexp_name\t%s\nconda_env\t%s\ngit_commit\t%s\nharness_commit\t%s\n' \
  "$DATASET" "$MODEL" "$SEED" "$GPU" "$EXP_NAME" "$CONDA_ENV" \
  "$CODE_COMMIT" "$HARNESS_COMMIT" > "$METADATA_FILE"
: > "$TIMING_LOG"

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
    used="${row%%,*}"; util="${row##*,}"
    used="${used// /}"; util="${util// /}"
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
  "exp.exp_name=$EXP_NAME"
  "exp.seed=$SEED"
  "exp.logging=False"
  "model.name=$MODEL"
  "exp.test=True"
  "exp.load_model=False"
  "exp.load_data=False"
  "+exp.eval_max_tau=12"
  "+exp.skip_case_plot=True"
)

if [[ "$DATASET" == "tumor" ]]; then
  dataset_args=(
    "+dataset=tumor"
    "+model=$MODEL"
    "+hparam/$MODEL/tumor=4*"
    "dataset.num_patients.train=1000"
    "dataset.coeff=4"
  )
elif [[ "$DATASET" == "mimic" ]]; then
  dataset_args=(
    "+dataset=mimic"
    "+model=$MODEL"
    "+hparam/$MODEL=mimic"
    "dataset.max_number=500"
    "dataset.projection_horizon=11"
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
PYTHONPATH="$TIMING_PATCH${PYTHONPATH:+:$PYTHONPATH}" \
EFFICIENCY_TIMING_JSONL="$TIMING_LOG" \
HYDRA_FULL_ERROR=1 \
python -u experiments/train.py "${dataset_args[@]}" "${common[@]}" \
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
printf 'wall_elapsed_ms\t%s\nexit_code\t%s\n' "$elapsed_ms" "$exit_code" \
  >> "$METADATA_FILE"

complexity_csv="$(find "results/$EXP_NAME" -type f -name complexity_info.csv \
  -print -quit 2>/dev/null || true)"
if [[ -n "$complexity_csv" ]]; then
  cp "$complexity_csv" "$TASK_ROOT/complexity_info.csv"
  printf 'complexity_source\t%s\n' "$GIFT_ROOT/$complexity_csv" >> "$METADATA_FILE"
fi

if [[ "$exit_code" -ne 0 ]]; then
  printf 'FAILED exit_code=%s\n' "$exit_code" > "$STATUS_FILE"
  exit "$exit_code"
fi
if [[ ! -s "$TASK_ROOT/complexity_info.csv" ]]; then
  printf 'FAILED missing_complexity\n' > "$STATUS_FILE"
  exit 3
fi
if [[ ! -s "$TIMING_LOG" ]]; then
  printf 'FAILED missing_timing\n' > "$STATUS_FILE"
  exit 4
fi
if rg -n 'Traceback|CUDA out of memory|Error executing job|Killed' \
  "$TRAIN_LOG" > "$LOG_DIR/error_signatures.txt"; then
  printf 'FAILED error_signature\n' > "$STATUS_FILE"
  exit 5
fi
printf 'COMPLETED\n' > "$STATUS_FILE"
