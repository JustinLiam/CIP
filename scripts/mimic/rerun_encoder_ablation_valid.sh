#!/usr/bin/env bash
set -uo pipefail

source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

CKPT_ROOT="${CKPT_ROOT:-grid_results/mimic_encoder_ablation_normmax_20260716}"
OUT_ROOT="${OUT_ROOT:-grid_results/mimic_encoder_ablation_valid_rerun_$(date +%Y%m%d_%H%M%S)}"
GPU_SLOTS=(${GPU_SLOTS:-0 0 1 1})
SEEDS=(10 101 1010 10101 101010)
VARIANTS=(full conv1d_only attention_only)
mkdir -p "${OUT_ROOT}/logs"

num_layers() {
  case "$1" in full) echo 2 ;; *) echo 1 ;; esac
}

local_layers() {
  case "$1" in attention_only) echo 0 ;; *) echo 1 ;; esac
}

TASK_VARIANTS=()
TASK_SEEDS=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    TASK_VARIANTS+=("${variant}")
    TASK_SEEDS+=("${seed}")
  done
done

run_one() {
  local variant="$1" seed="$2" gpu="$3"
  local ckpt="${CKPT_ROOT}/ckpts/${variant}_seed${seed}/ct_iql_em_best.pt"
  local log="${OUT_ROOT}/logs/${variant}_seed${seed}_val_eval.log"
  echo "[eval] variant=${variant} seed=${seed} gpu=${gpu} started=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpu}" HYDRA_FULL_ERROR=1 python -u runnables/eval_iql_planner.py \
    +dataset=mimic3_synthetic_gift \
    +model=vcip \
    exp.seed="${seed}" \
    dataset.seed="${seed}" \
    dataset.data_seed="${seed}" \
    dataset.max_number=500 \
    exp.use_mlflow=false \
    exp.test=false \
    exp.tau=6 \
    exp.batch_size_val=512 \
    +model.inference.num_layers="$(num_layers "${variant}")" \
    +model.inference.local_conv_layers="$(local_layers "${variant}")" \
    +exp.em_eval_ckpt="${ckpt}" \
    'exp.iql_eval_tau_list=[1,2,3,4,5,6,7,8,9,10,11,12]' \
    > "${log}" 2>&1
}

worker() {
  local slot="$1" gpu="$2" i failures=0
  for ((i=slot; i<${#TASK_VARIANTS[@]}; i+=${#GPU_SLOTS[@]})); do
    run_one "${TASK_VARIANTS[$i]}" "${TASK_SEEDS[$i]}" "${gpu}" || failures=$((failures + 1))
  done
  return "${failures}"
}

pids=()
for ((slot=0; slot<${#GPU_SLOTS[@]}; slot++)); do
  worker "${slot}" "${GPU_SLOTS[$slot]}" > "${OUT_ROOT}/worker_${slot}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if [[ "${failed}" != "0" ]]; then
  echo "One or more validation evaluations failed." >&2
  exit 1
fi

python - "${OUT_ROOT}" "${CKPT_ROOT}/rmse_uns_by_seed_tau.csv" <<'PY'
import csv
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
reference_path = pathlib.Path(sys.argv[2])
rows = []
for path in sorted((root / "logs").glob("*_seed*_val_eval.log")):
    match = re.fullmatch(r"(.+)_seed(\d+)_val_eval\.log", path.name)
    if not match:
        continue
    variant, seed = match.group(1), int(match.group(2))
    tau = None
    for line in path.read_text(errors="replace").splitlines():
        tau_match = re.search(r"IQL eval unified closed-loop rollout: .*?\(tau=(\d+),", line)
        if tau_match:
            tau = int(tau_match.group(1))
            continue
        rmse_match = re.search(r"RMSE (?:unscaled|scaled): ([0-9.eE+-]+)", line)
        if rmse_match and tau is not None:
            rows.append({
                "variant": variant,
                "seed": seed,
                "split": "val",
                "tau": tau,
                "rmse_uns": float(rmse_match.group(1)),
            })
            tau = None

assert len(rows) == 180, f"Expected 180 rows, got {len(rows)}"
rows.sort(key=lambda x: (x["variant"], x["tau"], x["seed"]))
with (root / "rmse_uns_by_seed_tau.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["variant", "seed", "split", "tau", "rmse_uns"])
    writer.writeheader()
    writer.writerows(rows)

summary = []
for variant in ("full", "conv1d_only", "attention_only"):
    for tau in range(1, 13):
        values = [row["rmse_uns"] for row in rows if row["variant"] == variant and row["tau"] == tau]
        assert len(values) == 5
        summary.append({
            "variant": variant,
            "split": "val",
            "tau": tau,
            "n": len(values),
            "mean_rmse_uns": statistics.fmean(values),
            "std_rmse_uns": statistics.stdev(values),
        })
with (root / "rmse_uns_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["variant", "split", "tau", "n", "mean_rmse_uns", "std_rmse_uns"],
    )
    writer.writeheader()
    writer.writerows(summary)

reference = {
    (row["variant"], int(row["seed"]), int(row["tau"])): float(row["rmse_uns"])
    for row in csv.DictReader(reference_path.open()) if row["split"] == "val"
}
differences = []
for row in rows:
    key = row["variant"], row["seed"], row["tau"]
    differences.append(abs(row["rmse_uns"] - reference[key]))
max_difference = max(differences)
(root / "reproducibility.txt").write_text(
    f"rows=180\nmax_abs_difference={max_difference:.12g}\nexact_at_6_decimals={max_difference < 0.5e-6}\n"
)
print(f"rows=180 max_abs_difference={max_difference:.12g}")
PY

echo "[complete] ${OUT_ROOT}"
