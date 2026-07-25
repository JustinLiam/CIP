#!/usr/bin/env bash
set -euo pipefail

source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-10}"
REPEATS="${REPEATS:-5}"
CKPT_ROOT="${CKPT_ROOT:-grid_results/mimic_encoder_ablation_normmax_20260716}"
OUT_ROOT="${OUT_ROOT:-grid_results/mimic_encoder_repeats${REPEATS}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
VARIANTS=(full conv1d_only attention_only)
GPUS=(${GPUS:-0 1 0})
mkdir -p "${OUT_ROOT}/logs"

num_layers() { if [[ "$1" == "full" ]]; then echo 2; else echo 1; fi; }
local_layers() { if [[ "$1" == "attention_only" ]]; then echo 0; else echo 1; fi; }

pids=()
for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"
  gpu="${GPUS[$i]}"
  ckpt="${CKPT_ROOT}/ckpts/${variant}_seed${SEED}/ct_iql_em_best.pt"
  log="${OUT_ROOT}/logs/${variant}_seed${SEED}_val_eval.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" HYDRA_FULL_ERROR=1 python -u runnables/eval_iql_planner.py \
      +dataset=mimic3_synthetic_gift \
      +model=vcip \
      exp.seed="${SEED}" \
      dataset.seed="${SEED}" \
      dataset.data_seed="${SEED}" \
      dataset.max_number=500 \
      exp.use_mlflow=false \
      exp.test=false \
      exp.tau=6 \
      exp.batch_size_val=512 \
      +model.inference.num_layers="$(num_layers "${variant}")" \
      +model.inference.local_conv_layers="$(local_layers "${variant}")" \
      +exp.iql_eval_repeats_override="${REPEATS}" \
      +exp.em_eval_ckpt="${ckpt}" \
      'exp.iql_eval_tau_list=[1,2,3,4,5,6,7,8,9,10,11,12]' \
      > "${log}" 2>&1
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "${pid}"; done

python - "${OUT_ROOT}" "${CKPT_ROOT}/rmse_uns_by_seed_tau.csv" "${SEED}" "${REPEATS}" <<'PY'
import csv
import pathlib
import re
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
reference_path = pathlib.Path(sys.argv[2])
seed = int(sys.argv[3])
requested_repeats = int(sys.argv[4])
rows = []
for path in sorted((root / "logs").glob("*_val_eval.log")):
    match = re.fullmatch(r"(.+)_seed(\d+)_val_eval\.log", path.name)
    if not match:
        continue
    variant = match.group(1)
    tau = None
    for line in path.read_text(errors="replace").splitlines():
        tau_match = re.search(r"IQL eval unified closed-loop rollout: .*?\(tau=(\d+),", line)
        if tau_match:
            tau = int(tau_match.group(1))
            continue
        rmse_match = re.search(r"RMSE (?:unscaled|scaled): ([0-9.eE+-]+)", line)
        if rmse_match and tau is not None:
            rng = np.random.RandomState(seed)
            histories = np.unique(rng.randint(20, 60 - tau, requested_repeats))
            rows.append({
                "variant": variant,
                "seed": seed,
                "split": "val",
                "tau": tau,
                "requested_repeats": requested_repeats,
                "effective_repeats": len(histories),
                "history_lengths": ";".join(str(x) for x in histories.tolist()),
                "rmse_uns": float(rmse_match.group(1)),
            })
            tau = None

assert len(rows) == 36, f"Expected 36 rows, got {len(rows)}"
reference = {
    (row["variant"], int(row["tau"])): float(row["rmse_uns"])
    for row in csv.DictReader(reference_path.open())
    if row["split"] == "val" and int(row["seed"]) == seed
}
for row in rows:
    baseline = reference[row["variant"], row["tau"]]
    row["rmse_uns_repeats3"] = baseline
    row["delta_repeats5_minus_repeats3"] = row["rmse_uns"] - baseline
    row["relative_delta_percent"] = 100.0 * (row["rmse_uns"] - baseline) / baseline

fields = [
    "variant", "seed", "split", "tau", "requested_repeats", "effective_repeats",
    "history_lengths", "rmse_uns", "rmse_uns_repeats3",
    "delta_repeats5_minus_repeats3", "relative_delta_percent",
]
with (root / "rmse_uns_comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: (row["variant"], row["tau"])))
print(f"rows={len(rows)}")
PY

echo "[complete] ${OUT_ROOT}"
