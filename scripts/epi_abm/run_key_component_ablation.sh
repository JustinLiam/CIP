#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/liam/pythonProject/VCIP-ICML-main
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/ablation_studies/epi_abm/key_components_${TS}}"
CACHE_DIR="${CACHE_DIR:-data/processed/epi_abm/full_factual_daily_seed100_20260709_223702}"
CACHE_VERSION="${CACHE_VERSION:-current_runtime_factual_daily_seed100}"
OUTCOME_TRANSFORM="${OUTCOME_TRANSFORM:-per10k_cases_zscore}"
SEEDS=(${SEEDS:-10 101 1010 10101 101010})
VAL_TAUS="${VAL_TAUS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21}"
TEST_TAUS="${TEST_TAUS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21}"
EVAL_THREADS="${EVAL_THREADS:-6}"
ERRPAT='Traceback|RuntimeError|ValueError|IndexError|OutOfMemoryError|CUDA out|Error executing job|Killed'
VARIANTS=(conv1d_only attention_only without_weightnet)

mkdir -p "$RUN_ROOT"/{configs,logs,aggregate}
log(){ printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$RUN_ROOT/logs/driver.log"; }
fail(){ log "FAILED: $*"; exit 1; }

variant_num_layers(){
  case "$1" in
    conv1d_only|attention_only) echo 1 ;;
    without_weightnet) echo 2 ;;
    *) return 2 ;;
  esac
}

variant_local_layers(){
  case "$1" in
    conv1d_only|without_weightnet) echo 1 ;;
    attention_only) echo 0 ;;
    *) return 2 ;;
  esac
}

variant_use_weightnet(){
  case "$1" in
    conv1d_only|attention_only) echo true ;;
    without_weightnet) echo false ;;
    *) return 2 ;;
  esac
}

update_status(){
  local path="$1" status="$2"
  python - "$path" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
manifest = json.loads(path.read_text())
manifest["status"] = status
manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

log "preflight start run_root=$RUN_ROOT"
python - "$RUN_ROOT" "$CACHE_DIR" "$CACHE_VERSION" "$OUTCOME_TRANSFORM" "${SEEDS[@]}" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

run = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
cache_version = sys.argv[3]
outcome_transform = sys.argv[4]
seeds = [int(seed) for seed in sys.argv[5:]]
epi_root = Path("data_generation/epi_diff_abm")

calibrated = sorted(epi_root.glob("result_graphs/*/*/*/calibrated_params.txt"))
if len(calibrated) != 153:
    raise SystemExit(f"expected 153 calibrated assets, found {len(calibrated)}")
cache_files = sorted(cache_dir.glob(f"*_{cache_version}.pkl"))
if len(cache_files) != 1:
    raise SystemExit(f"expected one cache pkl for {cache_version}, found {cache_files}")
if outcome_transform != "per10k_cases_zscore":
    raise SystemExit(f"unexpected outcome transform: {outcome_transform}")

git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
common = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_sha": git_sha,
    "epi_root": str(epi_root),
    "forbidden_epi_root": "external_repos/epi-diff-abm",
    "calibrated_asset_count": len(calibrated),
    "cache_dir": str(cache_dir),
    "cache_file": str(cache_files[0]),
    "cache_version": cache_version,
    "dataset_seed": 100,
    "action_hold_days": 1,
    "behavior_policy_subset": "factual_only",
    "split_by": "county",
    "split_counts": {"train": 107, "val": 23, "test": 23},
    "outcome_transform": outcome_transform,
    "target_mode": "factual_final",
    "target_scale": 1.0,
    "exp_seeds": seeds,
    "em_outer_iters": 12,
    "em_e_epochs": 3,
    "em_m_steps_per_outer": 1000,
    "em_warmup_outer_iters": 1,
    "validation_taus": list(range(1, 22)),
    "validation_selection_taus": [7, 14, 21],
    "test_taus": list(range(1, 22)),
    "status": "preflight_passed",
}
variants = {
    "conv1d_only": {
        "num_layers": 1,
        "local_conv_layers": 1,
        "global_attention_layers": 0,
        "ct_use_weight_net": True,
    },
    "attention_only": {
        "num_layers": 1,
        "local_conv_layers": 0,
        "global_attention_layers": 1,
        "ct_use_weight_net": True,
    },
    "without_weightnet": {
        "num_layers": 2,
        "local_conv_layers": 1,
        "global_attention_layers": 1,
        "ct_use_weight_net": False,
        "weight_definition": "exact w=1; E-step skipped; M-step receives uniform unit weights",
    },
}
(run / "manifest.json").write_text(
    json.dumps({**common, "variants": variants}, indent=2, sort_keys=True) + "\n"
)
for name, overrides in variants.items():
    variant_root = run / name
    for subdir in ("configs", "logs", "train", "aggregate", "runtime"):
        (variant_root / subdir).mkdir(parents=True, exist_ok=True)
    manifest = {**common, "run_root": str(variant_root), "variant": name, "overrides": overrides}
    (variant_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

for variant in "${VARIANTS[@]}"; do
  variant_root="$RUN_ROOT/$variant"
  config="$variant_root/configs/epi_abm_multi_daily_seed100.yaml"
  num_layers="$(variant_num_layers "$variant")"
  local_layers="$(variant_local_layers "$variant")"
  use_weightnet="$(variant_use_weightnet "$variant")"
  python - "$variant_root" "$config" "$CACHE_DIR" "$CACHE_VERSION" "$OUTCOME_TRANSFORM" \
    "$num_layers" "$local_layers" "$use_weightnet" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

run = Path(sys.argv[1])
config_path = Path(sys.argv[2])
cfg = OmegaConf.merge(
    OmegaConf.load("configs/config.yaml"),
    OmegaConf.load("configs/dataset/epi_abm_multi.yaml"),
    OmegaConf.load("configs/model/vcip.yaml"),
)
OmegaConf.set_struct(cfg, False)
cfg.dataset.epi_root = "data_generation/epi_diff_abm"
cfg.dataset.processed_data_dir = sys.argv[3]
cfg.dataset.cache_version = sys.argv[4]
cfg.dataset.seed = 100
cfg.dataset.device = "cpu"
cfg.dataset.action_hold_days = 1
cfg.dataset.generate_if_missing = False
cfg.dataset.force_regenerate = False
cfg.dataset.behavior_policy_subset = "factual_only"
cfg.dataset.split_by = "county"
cfg.dataset.outcome_transform = sys.argv[5]
cfg.model.inference.num_layers = int(sys.argv[6])
cfg.model.inference.local_conv_layers = int(sys.argv[7])
cfg.exp.ct_use_weight_net = sys.argv[8].lower() == "true"
cfg.exp.iql_decision_interval_days = 1
cfg.exp.log_dir = str(run / "hydra_preview")
OmegaConf.save(cfg, config_path)
PY
done

log "quick factual replay sanity start"
sanity_root="$RUN_ROOT/sanity/quick_factual_replay"
mkdir -p "$sanity_root"
python -u scripts/epi_abm/evaluate_county_last_window_iql.py \
  --config "$RUN_ROOT/conv1d_only/configs/epi_abm_multi_daily_seed100.yaml" \
  --out-dir "$sanity_root" \
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
  > "$RUN_ROOT/logs/quick_factual_replay.log" 2>&1
python - "$sanity_root/summary.json" <<'PY'
import json
import sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())
if not rows:
    raise SystemExit("empty quick sanity summary")
row = rows[0]
if float(row["rmse_factual_uns"]) != 0.0 or float(row["mean_target_distance_uns"]) != 0.0:
    raise SystemExit(f"quick sanity failed: {row}")
PY
log "quick factual replay sanity passed"

create_runtime(){
  local variant="$1" seed="$2"
  local variant_root="$RUN_ROOT/$variant"
  local runtime="$variant_root/runtime/seed_${seed}_epi_diff_abm"
  local log_path="$variant_root/train/seed_${seed}/logs/runtime_create.log"
  mkdir -p "$(dirname "$log_path")"
  python scripts/epi_abm/create_isolated_runtime.py \
    --source data_generation/epi_diff_abm \
    --dest "$runtime" \
    --force \
    > "$log_path" 2>&1
  if rg -n 'external_repos/epi-diff-abm' "$log_path"; then
    fail "$variant seed=$seed runtime points to external_repos"
  fi
}

train_seed(){
  local variant="$1" seed="$2" gpu="$3"
  local variant_root="$RUN_ROOT/$variant"
  local sdir="$variant_root/train/seed_${seed}"
  local runtime="$variant_root/runtime/seed_${seed}_epi_diff_abm"
  local num_layers local_layers use_weightnet
  num_layers="$(variant_num_layers "$variant")"
  local_layers="$(variant_local_layers "$variant")"
  use_weightnet="$(variant_use_weightnet "$variant")"
  mkdir -p "$sdir"/{em_ckpt,hydra,logs}
  create_runtime "$variant" "$seed"
  log "train start variant=$variant seed=$seed gpu=$gpu layers=$num_layers/$local_layers/$((num_layers-local_layers)) weightnet=$use_weightnet"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export HYDRA_FULL_ERROR=1
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
      +model.inference.num_layers="$num_layers" \
      +model.inference.local_conv_layers="$local_layers" \
      exp.ct_use_weight_net="$use_weightnet" \
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

verify_variant_training(){
  local variant="$1"
  local variant_root="$RUN_ROOT/$variant"
  for seed in "${SEEDS[@]}"; do
    local log_path="$variant_root/train/seed_${seed}/logs/train.log"
    [[ -f "$log_path" ]] || fail "missing train log variant=$variant seed=$seed"
    if rg -n "$ERRPAT" "$log_path" > "$variant_root/train/seed_${seed}/logs/train_errors.txt"; then
      fail "training error variant=$variant seed=$seed"
    fi
    rg -n 'EM outer 12/12|EM training done' "$log_path" >/dev/null \
      || fail "incomplete 12 outer variant=$variant seed=$seed"
    for outer in $(seq -w 1 12); do
      [[ -f "$variant_root/train/seed_${seed}/em_ckpt/ct_iql_em_outer00${outer}.pt" ]] \
        || fail "missing outer00${outer} variant=$variant seed=$seed"
    done
    case "$variant" in
      conv1d_only)
        rg -n 'CTHistoryEncoder layers \| local=1 global=0 total=1' "$log_path" >/dev/null \
          || fail "encoder audit failed variant=$variant seed=$seed"
        ;;
      attention_only)
        rg -n 'CTHistoryEncoder layers \| local=0 global=1 total=1' "$log_path" >/dev/null \
          || fail "encoder audit failed variant=$variant seed=$seed"
        ;;
      without_weightnet)
        rg -n 'WeightNet mode \| enabled=False align_loss=none \(uniform weights\)' "$log_path" >/dev/null \
          || fail "uniform WeightNet audit failed variant=$variant seed=$seed"
        ;;
    esac
  done
  update_status "$variant_root/manifest.json" training_done
  log "training verified variant=$variant"
}

train_variant_serial(){
  local variant="$1" gpu="$2"
  for seed in "${SEEDS[@]}"; do
    train_seed "$variant" "$seed" "$gpu" \
      || fail "train process failed variant=$variant seed=$seed gpu=$gpu"
  done
  verify_variant_training "$variant"
}

train_without_weightnet_parallel(){
  local variant=without_weightnet
  local idx=0
  while [[ "$idx" -lt "${#SEEDS[@]}" ]]; do
    local pids=() labels=()
    for gpu in 0 1; do
      [[ "$idx" -lt "${#SEEDS[@]}" ]] || break
      local seed="${SEEDS[$idx]}"
      train_seed "$variant" "$seed" "$gpu" &
      pids+=("$!")
      labels+=("$seed")
      idx=$((idx + 1))
    done
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || fail "train process failed variant=$variant seed=${labels[$i]}"
    done
  done
  verify_variant_training "$variant"
}

start_eval(){
  local variant="$1"
  local variant_root="$RUN_ROOT/$variant"
  log "CPU validation/test start variant=$variant taus=1..21 selection=7,14,21 threads=$EVAL_THREADS"
  (
    EVAL_PARALLEL=1 \
    EVAL_SHARDS=1 \
    THREADS_PER_WORKER="$EVAL_THREADS" \
    VAL_TAUS="$VAL_TAUS" \
    TEST_TAUS="$TEST_TAUS" \
    scripts/epi_abm/resume_daily_factual_target_parallel_eval.sh "$variant_root"
  ) > "$variant_root/logs/eval_driver.log" 2>&1
}

update_status "$RUN_ROOT/manifest.json" training_encoder_ablations
train_variant_serial conv1d_only 0 &
conv_pid=$!
train_variant_serial attention_only 1 &
attention_pid=$!
wait "$conv_pid" || fail "conv1d_only training worker failed"
wait "$attention_pid" || fail "attention_only training worker failed"

update_status "$RUN_ROOT/manifest.json" evaluating_encoders_training_without_weightnet
start_eval conv1d_only &
conv_eval_pid=$!
start_eval attention_only &
attention_eval_pid=$!
train_without_weightnet_parallel
start_eval without_weightnet &
weight_eval_pid=$!

wait "$conv_eval_pid" || fail "conv1d_only evaluation failed"
wait "$attention_eval_pid" || fail "attention_only evaluation failed"
wait "$weight_eval_pid" || fail "without_weightnet evaluation failed"

python - "$RUN_ROOT" "${VARIANTS[@]}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
variants = sys.argv[2:]
rows = []
for variant in variants:
    path = run / variant / "aggregate" / "test_target_metrics_across_seeds.csv"
    with path.open(newline="") as handle:
        tau21 = next(row for row in csv.DictReader(handle) if int(row["tau"]) == 21)
    rows.append({
        "variant": variant,
        "tau": 21,
        "n_seeds": int(tau21["n_seeds"]),
        "target_RMSE_per_10k_mean": float(tau21["target_RMSE_per_10k_mean"]),
        "target_RMSE_per_10k_std": float(tau21["target_RMSE_per_10k_std"]),
    })
with (run / "aggregate" / "epi_abm_tau21_ablation.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(run / "aggregate" / "epi_abm_tau21_ablation.json").write_text(
    json.dumps(rows, indent=2, sort_keys=True) + "\n"
)
manifest_path = run / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["status"] = "complete"
manifest["tau21_results"] = rows
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
log "all requested EpiABM key-component ablations complete"
