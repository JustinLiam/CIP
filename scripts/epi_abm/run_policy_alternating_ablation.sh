#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/liam/pythonProject/VCIP-ICML-main
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/ablation_studies/epi_abm/policy_alternating_${TS}}"
CACHE_DIR="${CACHE_DIR:-data/processed/epi_abm/full_factual_daily_seed100_20260709_223702}"
CACHE_VERSION="${CACHE_VERSION:-current_runtime_factual_daily_seed100}"
OUTCOME_TRANSFORM="${OUTCOME_TRANSFORM:-per10k_cases_zscore}"
SEEDS=(${SEEDS:-10 101 1010 10101 101010})
VAL_TAUS="${VAL_TAUS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21}"
TEST_TAUS="${TEST_TAUS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21}"
EVAL_THREADS="${EVAL_THREADS:-6}"
ERRPAT='Traceback|RuntimeError|ValueError|IndexError|OutOfMemoryError|CUDA out|Error executing job|Killed'
VARIANTS=(behavior_cloning without_alternating)

mkdir -p "$RUN_ROOT"/{configs,logs,aggregate}
log(){ printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$RUN_ROOT/logs/driver.log"; }
fail(){ log "FAILED: $*"; exit 1; }

variant_num_layers(){
  case "$1" in
    behavior_cloning|without_alternating) echo 2 ;;
    *) return 2 ;;
  esac
}

variant_local_layers(){
  case "$1" in
    behavior_cloning|without_alternating) echo 1 ;;
    *) return 2 ;;
  esac
}

variant_use_weightnet(){
  case "$1" in
    behavior_cloning|without_alternating) echo true ;;
    *) return 2 ;;
  esac
}

variant_actor_update(){
  case "$1" in
    behavior_cloning) echo bc ;;
    without_alternating) echo td3bc ;;
    *) return 2 ;;
  esac
}

variant_e_refresh_every(){
  case "$1" in
    behavior_cloning) echo 1 ;;
    without_alternating) echo 21 ;;
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
    "behavior_cloning": {
        "num_layers": 2,
        "local_conv_layers": 1,
        "global_attention_layers": 1,
        "ct_use_weight_net": True,
        "iql_actor_update": "bc",
        "em_e_refresh_every": 1,
        "definition": "Actor uses behavior-cloning loss; all other CRIPO settings are unchanged.",
    },
    "without_alternating": {
        "num_layers": 2,
        "local_conv_layers": 1,
        "global_attention_layers": 1,
        "ct_use_weight_net": True,
        "iql_actor_update": "td3bc",
        "em_e_refresh_every": 21,
        "definition": "WeightNet is refreshed at outer 1 and reused for the remaining 12-outer run.",
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

log "factual replay sanity reused from the active key-components run; no extra CPU ABM job launched"

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
  local num_layers local_layers use_weightnet actor_update e_refresh_every
  num_layers="$(variant_num_layers "$variant")"
  local_layers="$(variant_local_layers "$variant")"
  use_weightnet="$(variant_use_weightnet "$variant")"
  actor_update="$(variant_actor_update "$variant")"
  e_refresh_every="$(variant_e_refresh_every "$variant")"
  mkdir -p "$sdir"/{em_ckpt,hydra,logs}
  create_runtime "$variant" "$seed"
  log "train start variant=$variant seed=$seed gpu=$gpu layers=$num_layers/$local_layers/$((num_layers-local_layers)) weightnet=$use_weightnet actor_update=$actor_update e_refresh_every=$e_refresh_every"
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
      +exp.em_e_refresh_every="$e_refresh_every" \
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
      +exp.iql_actor_update="$actor_update" \
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
    rg -n 'CTHistoryEncoder layers \| local=1 global=1 total=2' "$log_path" >/dev/null \
      || fail "encoder audit failed variant=$variant seed=$seed"
    rg -n 'WeightNet mode \| enabled=True align_loss=sinkhorn' "$log_path" >/dev/null \
      || fail "WeightNet audit failed variant=$variant seed=$seed"
    case "$variant" in
      behavior_cloning)
        rg -n 'iql_actor_update: bc' "$log_path" >/dev/null \
          || fail "BC actor audit failed variant=$variant seed=$seed"
        ;;
      without_alternating)
        rg -n 'em_e_refresh_every: 21' "$log_path" >/dev/null \
          || fail "E-refresh audit failed variant=$variant seed=$seed"
        [[ "$(rg -c 'E-step full fit outer=' "$log_path")" -eq 1 ]] \
          || fail "expected exactly one WeightNet refresh variant=$variant seed=$seed"
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

update_status "$RUN_ROOT/manifest.json" training_policy_ablations
train_variant_serial behavior_cloning 0 &
bc_pid=$!
train_variant_serial without_alternating 1 &
no_alt_pid=$!
wait "$bc_pid" || fail "behavior_cloning training worker failed"
wait "$no_alt_pid" || fail "without_alternating training worker failed"

update_status "$RUN_ROOT/behavior_cloning/manifest.json" training_done_waiting_for_cpu_evaluation
update_status "$RUN_ROOT/without_alternating/manifest.json" training_done_waiting_for_cpu_evaluation
update_status "$RUN_ROOT/manifest.json" training_done_waiting_for_cpu_evaluation
log "policy ablation training complete; CPU validation/test intentionally deferred until the active key-components evaluators release capacity"
