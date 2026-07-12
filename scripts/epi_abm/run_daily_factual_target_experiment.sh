#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/liam/pythonProject/VCIP-ICML-main
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

OUTCOME_TRANSFORM="${OUTCOME_TRANSFORM:-${1:-per10k_cases_zscore}}"
case "$OUTCOME_TRANSFORM" in
  raw_cases_zscore|per10k_cases_zscore) ;;
  *) echo "Unknown OUTCOME_TRANSFORM=$OUTCOME_TRANSFORM" >&2; exit 2 ;;
esac

TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/epi_abm/daily_treatment_seed100_factual_target_${OUTCOME_TRANSFORM}_${TS}}"
CACHE_DIR="${CACHE_DIR:-data/processed/epi_abm/full_factual_daily_seed100_20260709_223702}"
CACHE_VERSION="${CACHE_VERSION:-current_runtime_factual_daily_seed100}"
CONFIG="$RUN_ROOT/configs/epi_abm_multi_daily_seed100.yaml"
ERRPAT='Traceback|RuntimeError|ValueError|IndexError|OutOfMemoryError|CUDA out|Error executing job|Killed'
SEEDS=(${SEEDS:-10 101 1010 10101 101010})
TAUS=(7 14 21)
TRAIN_PARALLEL="${TRAIN_PARALLEL:-2}"

mkdir -p "$RUN_ROOT"/{configs,logs,train,val_selection,eval,aggregate,runtime}
log(){ printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$RUN_ROOT/logs/driver.log"; }
fail(){ log "FAILED: $*"; exit 1; }

python - "$RUN_ROOT" "$CONFIG" "$CACHE_DIR" "$CACHE_VERSION" "$OUTCOME_TRANSFORM" <<'PY'
from pathlib import Path
import json
import sys
from omegaconf import OmegaConf

run = Path(sys.argv[1])
config_path = Path(sys.argv[2])
cache_dir = sys.argv[3]
cache_version = sys.argv[4]
outcome_transform = sys.argv[5]

cfg = OmegaConf.merge(
    OmegaConf.load("configs/config.yaml"),
    OmegaConf.load("configs/dataset/epi_abm_multi.yaml"),
    OmegaConf.load("configs/model/vcip.yaml"),
)
OmegaConf.set_struct(cfg, False)
cfg.dataset.epi_root = "data_generation/epi_diff_abm"
cfg.dataset.processed_data_dir = cache_dir
cfg.dataset.cache_version = cache_version
cfg.dataset.seed = 100
cfg.dataset.device = "cpu"
cfg.dataset.action_hold_days = 1
cfg.dataset.generate_if_missing = False
cfg.dataset.force_regenerate = False
cfg.dataset.behavior_policy_subset = "factual_only"
cfg.dataset.split_by = "county"
cfg.dataset.outcome_transform = outcome_transform
cfg.exp.iql_decision_interval_days = 1
cfg.exp.log_dir = str(run / "hydra_preview")
OmegaConf.save(cfg, config_path)

manifest = {
    "run_root": str(run),
    "created_at": run.name.rsplit("_", 1)[-1],
    "epi_root": "data_generation/epi_diff_abm",
    "forbidden_epi_root": "external_repos/epi-diff-abm",
    "cache_dir": cache_dir,
    "cache_version": cache_version,
    "dataset_seed": 100,
    "action_hold_days": 1,
    "target_mode": "factual_final",
    "target_scale": 1.0,
    "outcome_transform": outcome_transform,
    "exp_seeds": [10, 101, 1010, 10101, 101010],
    "taus": [7, 14, 21],
    "status": "created",
}
(run / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

log "run_root=$RUN_ROOT outcome_transform=$OUTCOME_TRANSFORM cache_dir=$CACHE_DIR"
[[ -f "$CACHE_DIR/multi_01009_01031_01039_plus150_202010-202104_${CACHE_VERSION}.pkl" ]] || fail "missing cache pkl in $CACHE_DIR"

python - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
if manifest["epi_root"] != "data_generation/epi_diff_abm":
    raise SystemExit("wrong epi_root")
if manifest["action_hold_days"] != 1:
    raise SystemExit("wrong action_hold_days")
PY

log "quick factual replay sanity start"
mkdir -p "$RUN_ROOT/sanity/quick_factual_replay"
python -u scripts/epi_abm/evaluate_county_last_window_iql.py \
  --config "$CONFIG" \
  --out-dir "$RUN_ROOT/sanity/quick_factual_replay" \
  --splits val \
  --taus 7 \
  --window-mode fixed-start \
  --decision-day 161 \
  --target-mode factual_final \
  --target-scale 1.0 \
  --factual-only \
  --model-device cpu \
  --abm-device cpu \
  --epi-root data_generation/epi_diff_abm \
  --processed-data-dir "$CACHE_DIR" \
  --cache-version "$CACHE_VERSION" \
  --dataset-seed 100 \
  --outcome-transform "$OUTCOME_TRANSFORM" \
  --max-counties 2 \
  > "$RUN_ROOT/sanity/quick_factual_replay.log" 2>&1
python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json
import sys
run = Path(sys.argv[1])
rows = json.loads((run / "sanity/quick_factual_replay/summary.json").read_text())
if not rows:
    raise SystemExit("empty quick sanity summary")
row = rows[0]
if float(row["rmse_factual_uns"]) != 0.0 or float(row["mean_target_distance_uns"]) != 0.0:
    raise SystemExit(f"quick sanity failed: {row}")
manifest = json.loads((run / "manifest.json").read_text())
manifest["quick_sanity"] = row
manifest["status"] = "quick_sanity_passed"
(run / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
log "quick factual replay sanity passed"

create_runtime(){
  local seed="$1"
  local runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  mkdir -p "$RUN_ROOT/train/seed_${seed}/logs"
  python scripts/epi_abm/create_isolated_runtime.py \
    --source data_generation/epi_diff_abm \
    --dest "$runtime" \
    --force \
    > "$RUN_ROOT/train/seed_${seed}/logs/runtime_create.log" 2>&1
  if rg -n 'external_repos/epi-diff-abm' "$RUN_ROOT/train/seed_${seed}/logs/runtime_create.log"; then
    fail "seed $seed runtime points to external_repos"
  fi
}

launch_train(){
  local seed="$1" gpu="$2"
  local sdir="$RUN_ROOT/train/seed_${seed}"
  local runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  mkdir -p "$sdir/em_ckpt" "$sdir/hydra" "$sdir/logs"
  log "train start seed=$seed gpu=$gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u runnables/train_ct_iql_em.py \
      +dataset=epi_abm_multi +model=vcip \
      dataset.epi_root="$runtime" \
      dataset.processed_data_dir="$CACHE_DIR" \
      dataset.cache_version="$CACHE_VERSION" \
      dataset.seed=100 \
      dataset.device=cpu \
      dataset.action_hold_days=1 \
      dataset.generate_if_missing=false \
      dataset.force_regenerate=false \
      dataset.behavior_policy_subset=factual_only \
      dataset.split_by=county \
      dataset.outcome_transform="$OUTCOME_TRANSFORM" \
      exp.seed="$seed" \
      exp.log_dir="$sdir/hydra" \
      +exp.em_ckpt_dir="$sdir/em_ckpt" \
      +exp.max_tau=21 \
      '+exp.iql_target_horizons=[7,14,21]' \
      '+exp.em_val_tau_list=[7,14,21]' \
      '+exp.iql_eval_tau_list=[7,14,21]' \
      exp.iql_decision_interval_days=1 \
      exp.use_mlflow=false \
      +exp.em_outer_iters=12 \
      +exp.em_m_steps_per_outer=1000 \
      +exp.em_e_epochs=3 \
      +exp.em_val_every=0 \
      +exp.em_val_repeats=1 \
      +exp.em_val_tau_agg=mean \
      +exp.em_warmup_outer_iters=1 \
      +exp.em_log_m_every=50 \
      +exp.em_save_every_outer_checkpoint=true \
      +exp.em_save_every_eval_checkpoint=true \
      +exp.iql_batch_size=256 \
      +exp.ct_batch_size=512 \
      +exp.iql_actor_lr=5e-5 +exp.iql_qf_lr=5e-5 +exp.iql_vf_lr=5e-5 +exp.em_encoder_lr=1e-5 \
      +exp.iql_eval_action_selector=q_sample \
      +exp.iql_eval_candidate_actions=64 \
      +exp.iql_eval_candidate_noise_std=0.25 \
      +exp.iql_eval_q_bc_penalty=1.0 \
      +exp.iql_actor_update=td3bc \
      +exp.iql_beta=1.0 \
      +exp.iql_td3bc_q_alpha=0.8 \
      +exp.iql_td3bc_bc_alpha=0.3 \
      +exp.iql_td3bc_action_penalty_alpha=0.02 \
      > "$sdir/logs/train.log" 2>&1
  )
}

for seed in "${SEEDS[@]}"; do
  create_runtime "$seed"
done
log "isolated runtimes ready"

idx=0
pids=()
pid_labels=()
for seed in "${SEEDS[@]}"; do
  gpu=$((idx % TRAIN_PARALLEL))
  launch_train "$seed" "$gpu" &
  pids+=("$!")
  pid_labels+=("$seed")
  idx=$((idx + 1))
  if [[ "${#pids[@]}" -ge "$TRAIN_PARALLEL" ]]; then
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || fail "train process failed for seed=${pid_labels[$i]}"
    done
    pids=()
    pid_labels=()
  fi
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || fail "train process failed for seed=${pid_labels[$i]}"
done

log "training finished; validating checkpoints"
for seed in "${SEEDS[@]}"; do
  sdir="$RUN_ROOT/train/seed_${seed}"
  if rg -n "$ERRPAT" "$sdir/logs/train.log" > "$sdir/logs/train_errors.txt"; then
    fail "train log contains errors for seed=$seed"
  fi
  for outer in $(seq -w 1 12); do
    label="outer00${outer}"
    [[ -f "$sdir/em_ckpt/ct_iql_em_${label}.pt" ]] || fail "missing $label for seed=$seed"
  done
  rg -n 'EM outer 12/12|EM training done' "$sdir/logs/train.log" >/dev/null || fail "seed=$seed did not complete 12 outers"
done
python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json, sys
run = Path(sys.argv[1])
m = json.loads((run / "manifest.json").read_text())
m["status"] = "training_done"
(run / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
PY

for seed in "${SEEDS[@]}"; do
  sdir="$RUN_ROOT/train/seed_${seed}"
  runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  out="$RUN_ROOT/val_selection/seed_${seed}/all_outers"
  mkdir -p "$out"
  ckpt_args=()
  for outer in $(seq -w 1 12); do
    label="outer00${outer}"
    ckpt_args+=(--ckpt "${label}=${sdir}/em_ckpt/ct_iql_em_${label}.pt")
  done
  log "val factual-target eval start seed=$seed"
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
    > "$out/eval.log" 2>&1
  if rg -n "$ERRPAT|external_repos/epi-diff-abm" "$out/eval.log" > "$out/eval_errors.txt"; then
    fail "val eval log contains errors or external root for seed=$seed"
  fi
  python - "$RUN_ROOT" "$seed" <<'PY'
from pathlib import Path
import csv, json, math, shutil, sys
run = Path(sys.argv[1])
seed = sys.argv[2]
out = run / "val_selection" / f"seed_{seed}" / "all_outers"
rows = [json.loads(x) for x in (out / "county_metrics.jsonl").read_text().splitlines() if x.strip()]
expected = 12 * 23 * 3
if len(rows) != expected:
    raise SystemExit(f"seed {seed} val rows {len(rows)} != {expected}")
summary = []
for label in sorted({r["label"] for r in rows}):
    item = {"seed": int(seed), "label": label}
    vals_all, fact_vals_all = [], []
    for tau in [7, 14, 21]:
        group = [r for r in rows if r["label"] == label and int(r["tau"]) == tau]
        vals = [float(r["target_distance_per_10k"]) for r in group]
        fact_vals = [float(r["factual_target_distance_per_10k"]) for r in group]
        rmse = math.sqrt(sum(v * v for v in vals) / len(vals))
        fact_rmse = math.sqrt(sum(v * v for v in fact_vals) / len(fact_vals))
        item[f"target_RMSE_per_10k_tau{tau}"] = rmse
        item[f"factual_target_RMSE_per_10k_tau{tau}"] = fact_rmse
        vals_all.append(rmse)
        fact_vals_all.append(fact_rmse)
    item["mean_target_RMSE_per_10k"] = sum(vals_all) / len(vals_all)
    item["mean_factual_target_RMSE_per_10k"] = sum(fact_vals_all) / len(fact_vals_all)
    item["RMSE_improvement_per_10k"] = item["mean_factual_target_RMSE_per_10k"] - item["mean_target_RMSE_per_10k"]
    summary.append(item)
summary.sort(key=lambda x: x["mean_target_RMSE_per_10k"])
fields = list(summary[0].keys())
with (out / "outer_target_rmse_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)
(out / "outer_target_rmse_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
best = summary[0]
sel = run / "val_selection" / f"seed_{seed}"
(sel / "selected_best_by_val_target_rmse.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
src = run / "train" / f"seed_{seed}" / "em_ckpt" / f"ct_iql_em_{best['label']}.pt"
dst = run / "train" / f"seed_{seed}" / "em_ckpt" / "selected_best_by_val_target_rmse.pt"
shutil.copy2(src, dst)
print(json.dumps({"seed": seed, "best": best}, indent=2, sort_keys=True))
PY
  log "val factual-target selection done seed=$seed"
done

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json, sys
run = Path(sys.argv[1])
m = json.loads((run / "manifest.json").read_text())
m["status"] = "val_selection_done"
(run / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
PY

for seed in "${SEEDS[@]}"; do
  runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  ckpt="$RUN_ROOT/train/seed_${seed}/em_ckpt/selected_best_by_val_target_rmse.pt"
  out="$RUN_ROOT/eval/seed_${seed}/test_selected_best"
  mkdir -p "$out"
  log "test factual-target eval start seed=$seed"
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
    > "$out/eval.log" 2>&1
  if rg -n "$ERRPAT|external_repos/epi-diff-abm" "$out/eval.log" > "$out/eval_errors.txt"; then
    fail "test eval log contains errors or external root for seed=$seed"
  fi
  rows=$(wc -l < "$out/county_metrics.jsonl")
  [[ "$rows" -eq 69 ]] || fail "seed=$seed test rows $rows != 69"
done

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import csv, json, math, statistics, sys
run = Path(sys.argv[1])
seeds = [10, 101, 1010, 10101, 101010]
taus = [7, 14, 21]
agg = run / "aggregate"
agg.mkdir(exist_ok=True)
def mean(xs): return sum(xs) / len(xs) if xs else None
def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
def rms(xs): return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else None
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
            "target_MAE_per_10k": mean(td),
            "factual_target_MAE_per_10k": mean(ftd),
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
        f"| {row['tau']} | {row.get('target_RMSE_per_10k_mean', 0):.4f} ± {row.get('target_RMSE_per_10k_std', 0):.4f} | "
        f"{row.get('factual_target_RMSE_per_10k_mean', 0):.4f} | "
        f"{row.get('policy_vs_factual_final_improvement_per_10k_mean_mean', 0):.4f} | "
        f"{row.get('action_mean_mean', 0):.4f} |"
    )
(agg / "RESULTS.md").write_text("\n".join(lines) + "\n")
m = json.loads((run / "manifest.json").read_text())
m["status"] = "complete"
m["aggregate_dir"] = str(agg)
(run / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "complete", "aggregate_dir": str(agg)}, indent=2))
PY
log "all done"
