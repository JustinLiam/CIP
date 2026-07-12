#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/liam/pythonProject/VCIP-ICML-main
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

RUN_ROOT="${1:?Usage: $0 <run_root>}"
RUN_ROOT="${RUN_ROOT%/}"
CONFIG="${CONFIG:-$RUN_ROOT/configs/epi_abm_multi_daily_seed100.yaml}"
EVAL_PARALLEL="${EVAL_PARALLEL:-8}"
VAL_ROWS="${VAL_ROWS:-23}"
TEST_ROWS="${TEST_ROWS:-23}"
SEEDS=(${SEEDS:-10 101 1010 10101 101010})
TAUS=(7 14 21)
ERRPAT='Traceback|RuntimeError|ValueError|IndexError|OutOfMemoryError|CUDA out|Error executing job|Killed'

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/aggregate"
log(){ printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$RUN_ROOT/logs/parallel_eval.log"; }
fail(){ log "FAILED: $*"; exit 1; }

read_manifest_field(){
  python - "$RUN_ROOT" "$1" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
key = sys.argv[2]
manifest = json.loads((run / "manifest.json").read_text())
print(manifest[key])
PY
}

CACHE_DIR="${CACHE_DIR:-$(read_manifest_field cache_dir)}"
CACHE_VERSION="${CACHE_VERSION:-$(read_manifest_field cache_version)}"
OUTCOME_TRANSFORM="${OUTCOME_TRANSFORM:-$(read_manifest_field outcome_transform)}"

[[ -f "$CONFIG" ]] || fail "missing config: $CONFIG"
[[ "$(read_manifest_field epi_root)" == "data_generation/epi_diff_abm" ]] || fail "wrong epi_root in manifest"
[[ "$(read_manifest_field action_hold_days)" == "1" ]] || fail "wrong action_hold_days in manifest"
[[ "$OUTCOME_TRANSFORM" == "raw_cases_zscore" || "$OUTCOME_TRANSFORM" == "per10k_cases_zscore" ]] || fail "bad outcome transform: $OUTCOME_TRANSFORM"

for seed in "${SEEDS[@]}"; do
  for outer in $(seq -w 1 12); do
    label="outer00${outer}"
    ckpt="$RUN_ROOT/train/seed_${seed}/em_ckpt/ct_iql_em_${label}.pt"
    [[ -f "$ckpt" ]] || fail "missing checkpoint seed=$seed label=$label"
  done
done

running_jobs=0
wait_pool(){
  while [[ "$running_jobs" -gt 0 ]]; do
    wait -n || fail "one or more parallel eval workers failed"
    running_jobs=$((running_jobs - 1))
  done
}

run_limited(){
  local _label="$1"
  shift
  "$@" &
  running_jobs=$((running_jobs + 1))
  if [[ "$running_jobs" -ge "$EVAL_PARALLEL" ]]; then
    wait -n || fail "parallel eval worker failed: $_label"
    running_jobs=$((running_jobs - 1))
  fi
}

val_shard(){
  local seed="$1" row="$2"
  local runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  local out="$RUN_ROOT/val_selection_parallel/seed_${seed}/row_$(printf '%02d' "$row")"
  local expected=36
  if [[ -f "$out/county_metrics.jsonl" ]] && [[ "$(wc -l < "$out/county_metrics.jsonl")" -eq "$expected" ]]; then
    return 0
  fi
  rm -rf "$out"
  mkdir -p "$out"
  local ckpt_args=()
  for outer in $(seq -w 1 12); do
    local label="outer00${outer}"
    ckpt_args+=(--ckpt "${label}=${RUN_ROOT}/train/seed_${seed}/em_ckpt/ct_iql_em_${label}.pt")
  done
  python -u scripts/epi_abm/evaluate_county_last_window_iql.py \
    --config "$CONFIG" \
    "${ckpt_args[@]}" \
    --out-dir "$out" \
    --splits val \
    --taus 7 14 21 \
    --window-mode fixed-start \
    --decision-day 161 \
    --target-mode factual_final \
    --target-scale 1.0 \
    --selector q_sample \
    --candidate-actions 64 \
    --candidate-noise-std 0.25 \
    --q-bc-penalty 1.0 \
    --eval-seed 20260708 \
    --model-device cpu \
    --abm-device cpu \
    --epi-root "$runtime" \
    --processed-data-dir "$CACHE_DIR" \
    --cache-version "$CACHE_VERSION" \
    --dataset-seed 100 \
    --outcome-transform "$OUTCOME_TRANSFORM" \
    --row-start "$row" \
    --row-end "$((row + 1))" \
    > "$out/eval.log" 2>&1
  if rg -n "$ERRPAT|external_repos/epi-diff-abm" "$out/eval.log" > "$out/eval_errors.txt"; then
    return 1
  fi
  [[ -f "$out/county_metrics.jsonl" ]] || return 1
  [[ "$(wc -l < "$out/county_metrics.jsonl")" -eq "$expected" ]] || return 1
}

test_shard(){
  local seed="$1" row="$2"
  local runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  local ckpt="$RUN_ROOT/train/seed_${seed}/em_ckpt/selected_best_by_val_target_rmse.pt"
  local out="$RUN_ROOT/eval_parallel/seed_${seed}/row_$(printf '%02d' "$row")"
  local expected=3
  [[ -f "$ckpt" ]] || return 1
  if [[ -f "$out/county_metrics.jsonl" ]] && [[ "$(wc -l < "$out/county_metrics.jsonl")" -eq "$expected" ]]; then
    return 0
  fi
  rm -rf "$out"
  mkdir -p "$out"
  python -u scripts/epi_abm/evaluate_county_last_window_iql.py \
    --config "$CONFIG" \
    --ckpt "selected=${ckpt}" \
    --out-dir "$out" \
    --splits test \
    --taus 7 14 21 \
    --window-mode fixed-start \
    --decision-day 161 \
    --target-mode factual_final \
    --target-scale 1.0 \
    --selector q_sample \
    --candidate-actions 64 \
    --candidate-noise-std 0.25 \
    --q-bc-penalty 1.0 \
    --eval-seed 20260708 \
    --model-device cpu \
    --abm-device cpu \
    --epi-root "$runtime" \
    --processed-data-dir "$CACHE_DIR" \
    --cache-version "$CACHE_VERSION" \
    --dataset-seed 100 \
    --outcome-transform "$OUTCOME_TRANSFORM" \
    --row-start "$row" \
    --row-end "$((row + 1))" \
    > "$out/eval.log" 2>&1
  if rg -n "$ERRPAT|external_repos/epi-diff-abm" "$out/eval.log" > "$out/eval_errors.txt"; then
    return 1
  fi
  [[ -f "$out/county_metrics.jsonl" ]] || return 1
  [[ "$(wc -l < "$out/county_metrics.jsonl")" -eq "$expected" ]] || return 1
}

aggregate_val(){
  python - "$RUN_ROOT" "$VAL_ROWS" "${SEEDS[@]}" <<'PY'
from pathlib import Path
import csv
import json
import math
import shutil
import sys

run = Path(sys.argv[1])
val_rows = int(sys.argv[2])
seeds = [int(x) for x in sys.argv[3:]]
taus = [7, 14, 21]

def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None

for seed in seeds:
    rows = []
    shard_root = run / "val_selection_parallel" / f"seed_{seed}"
    for row_idx in range(val_rows):
        path = shard_root / f"row_{row_idx:02d}" / "county_metrics.jsonl"
        if not path.exists():
            raise SystemExit(f"missing val shard seed={seed} row={row_idx}: {path}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected = val_rows * 3 * 12
    if len(rows) != expected:
        raise SystemExit(f"seed {seed} val rows {len(rows)} != {expected}")
    merged = run / "val_selection" / f"seed_{seed}" / "parallel_merged"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "county_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = []
    for label in sorted({r["label"] for r in rows}):
        item = {"seed": seed, "label": label}
        vals_all, fact_vals_all = [], []
        for tau in taus:
            group = [r for r in rows if r["label"] == label and int(r["tau"]) == tau]
            vals = [float(r["target_distance_per_10k"]) for r in group]
            fact_vals = [float(r["factual_target_distance_per_10k"]) for r in group]
            rmse = rms(vals)
            fact_rmse = rms(fact_vals)
            item[f"target_RMSE_per_10k_tau{tau}"] = rmse
            item[f"factual_target_RMSE_per_10k_tau{tau}"] = fact_rmse
            item[f"target_MAE_per_10k_tau{tau}"] = sum(abs(v) for v in vals) / len(vals)
            vals_all.append(rmse)
            fact_vals_all.append(fact_rmse)
        item["mean_target_RMSE_per_10k"] = sum(vals_all) / len(vals_all)
        item["mean_factual_target_RMSE_per_10k"] = sum(fact_vals_all) / len(fact_vals_all)
        item["RMSE_improvement_per_10k"] = item["mean_factual_target_RMSE_per_10k"] - item["mean_target_RMSE_per_10k"]
        summary.append(item)
    summary.sort(key=lambda x: x["mean_target_RMSE_per_10k"])
    fields = list(summary[0].keys())
    with (merged / "outer_target_rmse_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    (merged / "outer_target_rmse_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    best = summary[0]
    sel = run / "val_selection" / f"seed_{seed}"
    sel.mkdir(parents=True, exist_ok=True)
    (sel / "selected_best_by_val_target_rmse.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    src = run / "train" / f"seed_{seed}" / "em_ckpt" / f"ct_iql_em_{best['label']}.pt"
    dst = run / "train" / f"seed_{seed}" / "em_ckpt" / "selected_best_by_val_target_rmse.pt"
    shutil.copy2(src, dst)
    print(json.dumps({"seed": seed, "best": best}, sort_keys=True))
PY
}

aggregate_test(){
  python - "$RUN_ROOT" "$VAL_ROWS" "$TEST_ROWS" "${SEEDS[@]}" <<'PY'
from pathlib import Path
import csv
import json
import math
import statistics
import sys

run = Path(sys.argv[1])
val_rows_count = int(sys.argv[2])
test_rows = int(sys.argv[3])
seeds = [int(x) for x in sys.argv[4:]]
taus = [7, 14, 21]
agg = run / "aggregate"
agg.mkdir(exist_ok=True)

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None

for seed in seeds:
    rows = []
    shard_root = run / "eval_parallel" / f"seed_{seed}"
    for row_idx in range(test_rows):
        path = shard_root / f"row_{row_idx:02d}" / "county_metrics.jsonl"
        if not path.exists():
            raise SystemExit(f"missing test shard seed={seed} row={row_idx}: {path}")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected = test_rows * 3
    if len(rows) != expected:
        raise SystemExit(f"seed {seed} test rows {len(rows)} != {expected}")
    merged = run / "eval" / f"seed_{seed}" / "test_selected_best"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "county_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

seed_tau = []
for seed in seeds:
    selected = json.loads((run / "val_selection" / f"seed_{seed}" / "selected_best_by_val_target_rmse.json").read_text())
    rows = [json.loads(x) for x in (run / "eval" / f"seed_{seed}" / "test_selected_best" / "county_metrics.jsonl").read_text().splitlines() if x.strip()]
    for tau in taus:
        group = [r for r in rows if int(r["tau"]) == tau]
        td = [float(r["target_distance_per_10k"]) for r in group]
        ftd = [float(r["factual_target_distance_per_10k"]) for r in group]
        pred = [float(r["pred_final_per_10k"]) for r in group]
        target = [float(r["target_final_per_10k"]) for r in group]
        viol = [max(p - t, 0.0) for p, t in zip(pred, target)]
        row = {
            "seed": seed,
            "tau": tau,
            "n_counties": len(group),
            "selected_outer": selected["label"],
            "target_RMSE_per_10k": rms(td),
            "factual_target_RMSE_per_10k": rms(ftd),
            "RMSE_improvement_per_10k": rms(ftd) - rms(td),
            "RMSE_improvement_pct": ((rms(ftd) - rms(td)) / rms(ftd) * 100.0) if rms(ftd) and rms(ftd) > 0 else None,
            "target_MAE_per_10k": mean([abs(v) for v in td]),
            "factual_target_MAE_per_10k": mean([abs(v) for v in ftd]),
            "one_sided_violation_RMSE_per_10k": rms(viol),
            "pred_final_per_10k_mean": mean(pred),
            "target_final_per_10k_mean": mean(target),
            "factual_final_per_10k_mean": mean([float(r["factual_final_per_10k"]) for r in group]),
            "policy_vs_factual_final_improvement_per_10k_mean": mean([float(r["policy_vs_factual_final_improvement_per_10k"]) for r in group]),
            "policy_vs_factual_cumulative_improvement_per_10k_mean": mean([float(r["policy_vs_factual_cumulative_improvement_per_10k"]) for r in group]),
            "action_mean": mean([float(r["action_mean"]) for r in group if r.get("action_mean") is not None]),
            "action_std": mean([float(r["action_std"]) for r in group if r.get("action_std") is not None]),
        }
        seed_tau.append(row)
fields = list(seed_tau[0].keys())
with (agg / "test_target_metrics_by_seed_tau.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(seed_tau)
(agg / "test_target_metrics_by_seed_tau.json").write_text(json.dumps(seed_tau, indent=2, sort_keys=True) + "\n")
across = []
for tau in taus:
    group = [r for r in seed_tau if r["tau"] == tau]
    out = {"tau": tau, "n_seeds": len(group)}
    for key in fields:
        if key in {"seed", "tau", "selected_outer"}:
            continue
        vals = [float(r[key]) for r in group if r.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = mean(vals)
            out[f"{key}_std"] = sd(vals)
            out[f"{key}_median"] = statistics.median(vals)
    across.append(out)
fields2 = sorted({k for r in across for k in r})
with (agg / "test_target_metrics_across_seeds.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields2)
    writer.writeheader()
    writer.writerows(across)
(agg / "test_target_metrics_across_seeds.json").write_text(json.dumps(across, indent=2, sort_keys=True) + "\n")
val_rows = [json.loads((run / "val_selection" / f"seed_{seed}" / "selected_best_by_val_target_rmse.json").read_text()) for seed in seeds]
fields3 = sorted({k for r in val_rows for k in r})
with (agg / "val_selected_checkpoints.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields3)
    writer.writeheader()
    writer.writerows(val_rows)
(agg / "val_selected_checkpoints.json").write_text(json.dumps(val_rows, indent=2, sort_keys=True) + "\n")
lines = ["# Daily-treatment factual-target results", "", "## Selected checkpoints", ""]
lines.append("| seed | selected outer | val mean target RMSE /10k |")
lines.append("| ---: | --- | ---: |")
for row in val_rows:
    lines.append(f"| {row['seed']} | {row['label']} | {row['mean_target_RMSE_per_10k']:.4f} |")
lines += ["", "## Test across seeds", "", "| tau | target RMSE /10k | factual target RMSE /10k | final improvement /10k | action mean |"]
lines.append("| ---: | ---: | ---: | ---: | ---: |")
for row in across:
    lines.append(
        f"| {row['tau']} | {row.get('target_RMSE_per_10k_mean', 0):.4f} +/- {row.get('target_RMSE_per_10k_std', 0):.4f} | "
        f"{row.get('factual_target_RMSE_per_10k_mean', 0):.4f} | "
        f"{row.get('policy_vs_factual_final_improvement_per_10k_mean_mean', 0):.4f} | "
        f"{row.get('action_mean_mean', 0):.4f} |"
    )
(agg / "RESULTS.md").write_text("\n".join(lines) + "\n")
manifest = json.loads((run / "manifest.json").read_text())
manifest["status"] = "parallel_eval_complete"
manifest["aggregate_dir"] = str(agg)
manifest["parallel_eval"] = {"val_rows_per_seed": val_rows_count, "test_rows_per_seed": test_rows}
(run / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "complete", "aggregate_dir": str(agg)}, indent=2))
PY
}

log "parallel val eval start run_root=$RUN_ROOT workers=$EVAL_PARALLEL"
for seed in "${SEEDS[@]}"; do
  for row in $(seq 0 "$((VAL_ROWS - 1))"); do
    run_limited "val seed=$seed row=$row" val_shard "$seed" "$row"
  done
done
wait_pool
log "parallel val eval shards done; aggregating selection"
aggregate_val
log "parallel val selection done"

log "parallel test eval start"
for seed in "${SEEDS[@]}"; do
  for row in $(seq 0 "$((TEST_ROWS - 1))"); do
    run_limited "test seed=$seed row=$row" test_shard "$seed" "$row"
  done
done
wait_pool
log "parallel test eval shards done; aggregating results"
aggregate_test
log "parallel eval complete"
