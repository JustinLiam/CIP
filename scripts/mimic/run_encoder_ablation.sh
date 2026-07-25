#!/usr/bin/env bash
set -uo pipefail

if [[ -f /home/liam/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/liam/anaconda3/etc/profile.d/conda.sh
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate vcip

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-grid_results/mimic_encoder_ablation_$(date +%Y%m%d_%H%M%S)}"
SEEDS=(${ABLATION_SEEDS:-10 101 1010 10101 101010})
GPU_SLOTS=(${GPU_SLOTS:-0 0 1 1})
MAX_NUMBER="${MAX_NUMBER:-500}"
OUTER_ITERS="${OUTER_ITERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-512}"
FORCE="${FORCE:-0}"

VARIANTS=(full conv1d_only attention_only)

mkdir -p "${RUN_ROOT}"/{ckpts,logs,done}
printf '%s\n' "${RUN_ROOT}" > "${RUN_ROOT}/run_root.txt"

variant_num_layers() {
  case "$1" in
    full) echo 2 ;;
    conv1d_only|attention_only) echo 1 ;;
    *) return 2 ;;
  esac
}

variant_local_layers() {
  case "$1" in
    full|conv1d_only) echo 1 ;;
    attention_only) echo 0 ;;
    *) return 2 ;;
  esac
}

run_one() {
  local variant="$1"
  local seed="$2"
  local gpu="$3"
  local num_layers local_layers global_layers tag ckpt_dir ckpt train_log
  num_layers="$(variant_num_layers "${variant}")"
  local_layers="$(variant_local_layers "${variant}")"
  global_layers=$((num_layers - local_layers))
  tag="${variant}_seed${seed}"
  ckpt_dir="${RUN_ROOT}/ckpts/${tag}"
  ckpt="${ckpt_dir}/ct_iql_em_best.pt"
  train_log="${RUN_ROOT}/logs/${tag}_train.log"

  if [[ "${FORCE}" != "1" && -f "${RUN_ROOT}/done/${tag}.done" ]]; then
    echo "[skip] ${tag}"
    return 0
  fi
  mkdir -p "${ckpt_dir}"

  echo "[train] variant=${variant} seed=${seed} data_seed=${seed} gpu=${gpu} layers=${num_layers}/${local_layers}/${global_layers} started=$(date -Is)"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" HYDRA_FULL_ERROR=1 python -u runnables/train_ct_iql_em.py \
      +dataset=mimic3_synthetic_gift \
      +model=vcip \
      exp.seed="${seed}" \
      dataset.seed="${seed}" \
      dataset.data_seed="${seed}" \
      dataset.max_number="${MAX_NUMBER}" \
      exp.use_mlflow=false \
      exp.tau=6 \
      exp.batch_size="${BATCH_SIZE}" \
      exp.batch_size_val="${BATCH_SIZE_VAL}" \
      +model.inference.num_layers="${num_layers}" \
      +model.inference.local_conv_layers="${local_layers}" \
      +exp.em_outer_iters="${OUTER_ITERS}" \
      exp.em_val_metric=rmse_norm \
      'exp.em_val_tau_list=[1,2,3,4,5,6]' \
      +exp.em_val_tau_agg=max \
      +exp.em_ckpt_dir="${ckpt_dir}" \
      > "${train_log}" 2>&1; then
    echo "[error] training failed: ${tag}; see ${train_log}" >&2
    return 1
  fi

  if [[ ! -f "${ckpt}" ]]; then
    echo "[error] checkpoint missing: ${ckpt}" >&2
    return 1
  fi

  local eval_flag split eval_log
  for eval_flag in false true; do
    if [[ "${eval_flag}" == "true" ]]; then split=test; else split=val; fi
    eval_log="${RUN_ROOT}/logs/${tag}_${split}_eval.log"
    echo "[eval] variant=${variant} seed=${seed} split=${split} gpu=${gpu} started=$(date -Is)"
    if ! CUDA_VISIBLE_DEVICES="${gpu}" HYDRA_FULL_ERROR=1 python -u runnables/eval_iql_planner.py \
        +dataset=mimic3_synthetic_gift \
        +model=vcip \
        exp.seed="${seed}" \
        dataset.seed="${seed}" \
        dataset.data_seed="${seed}" \
        dataset.max_number="${MAX_NUMBER}" \
        exp.use_mlflow=false \
        exp.test="${eval_flag}" \
        exp.tau=6 \
        exp.batch_size_val="${BATCH_SIZE_VAL}" \
        +model.inference.num_layers="${num_layers}" \
        +model.inference.local_conv_layers="${local_layers}" \
        +exp.em_eval_ckpt="${ckpt}" \
        'exp.iql_eval_tau_list=[1,2,3,4,5,6,7,8,9,10,11,12]' \
        > "${eval_log}" 2>&1; then
      echo "[error] ${split} evaluation failed: ${tag}; see ${eval_log}" >&2
      return 1
    fi
  done

  {
    echo "variant=${variant}"
    echo "seed=${seed}"
    echo "dataset_seed=${seed}"
    echo "dataset_data_seed=${seed}"
    echo "num_layers=${num_layers}"
    echo "local_conv_layers=${local_layers}"
    echo "global_attention_layers=${global_layers}"
    echo "checkpoint=${ckpt}"
    echo "finished_at=$(date -Is)"
  } > "${RUN_ROOT}/done/${tag}.done"
  echo "[done] ${tag} finished=$(date -Is)"
}

TASK_VARIANTS=()
TASK_SEEDS=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    TASK_VARIANTS+=("${variant}")
    TASK_SEEDS+=("${seed}")
  done
done

worker() {
  local slot="$1"
  local gpu="$2"
  local i failures=0
  for ((i=slot; i<${#TASK_VARIANTS[@]}; i+=${#GPU_SLOTS[@]})); do
    run_one "${TASK_VARIANTS[$i]}" "${TASK_SEEDS[$i]}" "${gpu}" || failures=$((failures + 1))
  done
  return "${failures}"
}

echo "[protocol] run_root=${RUN_ROOT}"
echo "[protocol] seeds=${SEEDS[*]} max_number=${MAX_NUMBER} outer_iters=${OUTER_ITERS}"
echo "[protocol] selection=min validation max(rmse_norm[tau=1..6])"
echo "[protocol] evaluation=validation,test tau=1..12 repeats=MIMIC-code-default"
echo "[protocol] gpu_slots=${GPU_SLOTS[*]} concurrent_workers=${#GPU_SLOTS[@]}"

pids=()
for ((slot=0; slot<${#GPU_SLOTS[@]}; slot++)); do
  worker "${slot}" "${GPU_SLOTS[$slot]}" > "${RUN_ROOT}/worker_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

python - "${RUN_ROOT}" <<'PY'
import csv
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
detail = []
detail_uns = []
selection = []

for train_log in sorted((root / "logs").glob("*_seed*_train.log")):
    m = re.fullmatch(r"(.+)_seed(\d+)_train\.log", train_log.name)
    if not m:
        continue
    variant, seed = m.group(1), int(m.group(2))
    text = train_log.read_text(errors="replace")
    done = list(re.finditer(
        r"EM training done\. best_outer=(\d+) best_([^=]+)=([0-9.eE+-]+)", text
    ))
    if done:
        last = done[-1]
        selection.append({
            "variant": variant,
            "seed": seed,
            "best_outer": int(last.group(1)),
            "selection_metric": last.group(2),
            "selection_score": float(last.group(3)),
        })

for eval_log in sorted((root / "logs").glob("*_seed*_*_eval.log")):
    m = re.fullmatch(r"(.+)_seed(\d+)_(val|test)_eval\.log", eval_log.name)
    if not m:
        continue
    variant, seed, split = m.group(1), int(m.group(2)), m.group(3)
    current_tau = None
    for line in eval_log.read_text(errors="replace").splitlines():
        mt = re.search(r"IQL eval unified closed-loop rollout: .*?\(tau=(\d+),", line)
        if mt:
            current_tau = int(mt.group(1))
            continue
        mr = re.search(r"Global RMSE on stacked batches \(normalized space\): ([0-9.eE+-]+)", line)
        if mr and current_tau is not None:
            detail.append({
                "variant": variant,
                "seed": seed,
                "split": split,
                "tau": current_tau,
                "rmse_norm": float(mr.group(1)),
            })
            continue
        mu = re.search(r"RMSE (?:unscaled|scaled): ([0-9.eE+-]+)", line)
        if mu and current_tau is not None:
            detail_uns.append({
                "variant": variant,
                "seed": seed,
                "split": split,
                "tau": current_tau,
                "rmse_uns": float(mu.group(1)),
            })
            current_tau = None

with (root / "checkpoint_selection.csv").open("w", newline="") as f:
    fields = ["variant", "seed", "best_outer", "selection_metric", "selection_score"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(selection, key=lambda x: (x["variant"], x["seed"])))

with (root / "rmse_norm_by_seed_tau.csv").open("w", newline="") as f:
    fields = ["variant", "seed", "split", "tau", "rmse_norm"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(detail, key=lambda x: (x["split"], x["variant"], x["tau"], x["seed"])))

with (root / "rmse_uns_by_seed_tau.csv").open("w", newline="") as f:
    fields = ["variant", "seed", "split", "tau", "rmse_uns"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(detail_uns, key=lambda x: (x["split"], x["variant"], x["tau"], x["seed"])))

summary = []
for split in ("val", "test"):
    for variant in ("full", "conv1d_only", "attention_only"):
        for tau in range(1, 13):
            vals = [x["rmse_norm"] for x in detail
                    if x["split"] == split and x["variant"] == variant and x["tau"] == tau]
            if vals:
                summary.append({
                    "variant": variant,
                    "split": split,
                    "tau": tau,
                    "n": len(vals),
                    "mean_rmse_norm": statistics.fmean(vals),
                    "std_rmse_norm": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                })

with (root / "rmse_norm_summary.csv").open("w", newline="") as f:
    fields = ["variant", "split", "tau", "n", "mean_rmse_norm", "std_rmse_norm"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

summary_uns = []
for split in ("val", "test"):
    for variant in ("full", "conv1d_only", "attention_only"):
        for tau in range(1, 13):
            vals = [x["rmse_uns"] for x in detail_uns
                    if x["split"] == split and x["variant"] == variant and x["tau"] == tau]
            if vals:
                summary_uns.append({
                    "variant": variant,
                    "split": split,
                    "tau": tau,
                    "n": len(vals),
                    "mean_rmse_uns": statistics.fmean(vals),
                    "std_rmse_uns": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                })

with (root / "rmse_uns_summary.csv").open("w", newline="") as f:
    fields = ["variant", "split", "tau", "n", "mean_rmse_uns", "std_rmse_uns"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary_uns)

print(
    f"[summary] selection={len(selection)}/15 "
    f"rmse_norm_rows={len(detail)}/360 rmse_uns_rows={len(detail_uns)}/360"
)
PY

if [[ "${failed}" != "0" ]]; then
  echo "[error] one or more workers failed; inspect ${RUN_ROOT}/worker_*.log" >&2
  exit 1
fi
echo "[complete] ${RUN_ROOT}"
