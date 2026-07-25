#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/liam/pythonProject/VCIP-ICML-main}"
BASELINE_ROOT="${BASELINE_ROOT:-/home/liam/pythonProject/VCIP-ICML-main-qiu-eval}"
PROTOCOL_ID="${PROTOCOL_ID:-factual_daily_seed100_fixedstart161_h5-21_target1_v2}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-gpu_factual_target_20260722_g0_ct_e0300}"
BASELINE_FINAL="$BASELINE_ROOT/results/epi_abm/baselines/aggregate/$PROTOCOL_ID/$BASELINE_RUN_ID/final/target_rmse_per_10k.json"

OUTCOME_TRANSFORM="${OUTCOME_TRANSFORM:-per10k_cases_zscore}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/results/epi_abm/daily_treatment_seed100_factual_target_${OUTCOME_TRANSFORM}_h5-21_mimic_lr_${RUN_TS}}"
CACHE_DIR="${CACHE_DIR:-data/processed/epi_abm/full_factual_daily_seed100_20260709_223702}"
CACHE_VERSION="${CACHE_VERSION:-current_runtime_factual_daily_seed100}"
CONFIG="$RUN_ROOT/configs/epi_abm_multi_daily_seed100.yaml"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-30}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-1024}"
GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-2}"
THREADS_PER_SEED="${THREADS_PER_SEED:-6}"

SEEDS=(10 101 1010 10101 101010)
GPU_ASSIGNMENTS=(0 1 0 1 0)
ERRPAT='Traceback|RuntimeError|ValueError|IndexError|OutOfMemoryError|CUDA out|Error executing job|Killed'

mkdir -p "$RUN_ROOT/logs"
QUEUE_LOG="$RUN_ROOT/logs/queue_supervisor.log"
log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$QUEUE_LOG"; }
fail() { log "FAILED: $*"; exit 1; }

baseline_processes_active() {
  ps -eo pid=,comm=,args= | awk -v root="$BASELINE_ROOT" '
    $2 != "awk" && index($0, root) && ($0 ~ /evaluate_baseline_county_major_gpu.py/ || $0 ~ /run_baseline_gpu_protocol_queue.sh/) { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

gpus_are_idle() {
  local used
  while IFS= read -r used; do
    used="${used//[[:space:]]/}"
    [[ "$used" =~ ^[0-9]+$ ]] || return 1
    (( used <= GPU_IDLE_MEMORY_MIB )) || return 1
  done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
}

log "queue created run_root=$RUN_ROOT"
log "waiting for baseline final=$BASELINE_FINAL"
stable=0
while (( stable < GPU_STABLE_CHECKS )); do
  if [[ -f "$BASELINE_FINAL" ]] && ! baseline_processes_active && gpus_are_idle; then
    stable=$((stable + 1))
    log "baseline complete and GPUs idle check=$stable/$GPU_STABLE_CHECKS"
  else
    stable=0
    if [[ -f "$BASELINE_FINAL" ]]; then
      log "baseline final exists; waiting for evaluator exit and GPU memory <= ${GPU_IDLE_MEMORY_MIB} MiB"
    else
      log "baseline still running; CRIPO training remains queued"
    fi
  fi
  if (( stable < GPU_STABLE_CHECKS )); then
    sleep "$WAIT_INTERVAL_SECONDS"
  fi
done

cd "$PROJECT_ROOT"
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

[[ -f "$CACHE_DIR/multi_01009_01031_01039_plus150_202010-202104_${CACHE_VERSION}.pkl" ]] \
  || fail "missing cache pkl in $CACHE_DIR"
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ]] \
  || fail "two CUDA GPUs are required"

mkdir -p "$RUN_ROOT"/{configs,train,runtime,logs}
git status --short > "$RUN_ROOT/logs/git_status_at_start.txt"
git rev-parse HEAD > "$RUN_ROOT/logs/git_commit_at_start.txt"

python - "$RUN_ROOT" "$CONFIG" "$CACHE_DIR" "$CACHE_VERSION" "$OUTCOME_TRANSFORM" <<'PY'
from pathlib import Path
import json
import sys
from datetime import datetime
from omegaconf import OmegaConf

run = Path(sys.argv[1])
config_path = Path(sys.argv[2])
cache_dir = sys.argv[3]
cache_version = sys.argv[4]
outcome_transform = sys.argv[5]
horizons = [5, 7, 9, 11, 13, 15, 17, 19, 21]
seeds = [10, 101, 1010, 10101, 101010]

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
    "created_at": datetime.now().astimezone().isoformat(),
    "status": "queued_after_baselines",
    "dataset": {
        "epi_root": "data_generation/epi_diff_abm",
        "cache_dir": cache_dir,
        "cache_version": cache_version,
        "dataset_seed": 100,
        "action_hold_days": 1,
        "behavior_policy_subset": "factual_only",
        "split_by": "county",
        "outcome_transform": outcome_transform,
    },
    "experiment_seeds": seeds,
    "gpu_assignments": {str(seed): gpu for seed, gpu in zip(seeds, [0, 1, 0, 1, 0])},
    "changed_hyperparameters": {
        "iql_target_horizons": horizons,
        "em_val_tau_list": horizons,
        "iql_eval_tau_list": horizons,
        "em_outer_iters": 20,
        "em_m_steps_per_outer": 1000,
        "em_e_epochs": 3,
        "iql_actor_lr": 3e-4,
        "iql_qf_lr": 3e-4,
        "iql_vf_lr": 3e-4,
        "iql_actor_update": "awr",
        "iql_beta": 2.0,
        "iql_adv_max": 100.0,
        "iql_actor_bc_loss": "expectile",
        "iql_actor_bc_expectile": 0.8,
        "iql_tau": 0.7,
        "iql_weight_max": 3.0,
        "iql_discount": 0.99,
        "iql_target_tau": 0.005,
        "iql_max_grad_norm": 5.0,
        "iql_hidden_dim": 256,
        "iql_n_hidden": 2,
        "iql_target_sampling": "horizon_aligned",
        "iql_horizon_terminal_done": True,
        "em_encoder_lr": 5e-5,
        "em_encoder_max_grad_norm": 1.0,
        "em_e_w_lr": 1e-2,
        "ct_w_lr": 1e-2,
        "ct_use_weight_net": True,
        "ct_align_loss": "sinkhorn",
        "ct_sinkhorn_blur": 0.01,
        "ct_weight_hidden": 64,
        "ct_weight_decay": 1e-5,
        "ct_w_clip": 5.0,
    },
    "preserved_from_previous_episc_run": {
        "em_val_every": 0,
        "em_val_repeats": 1,
        "em_val_tau_agg": "mean",
        "em_warmup_outer_iters": 1,
        "iql_batch_size": 256,
        "ct_batch_size": 512,
        "iql_eval_action_selector": "q_sample",
        "iql_eval_candidate_actions": 64,
        "iql_eval_candidate_noise_std": 0.25,
        "iql_eval_q_bc_penalty": 1.0,
    },
    "inactive_td3bc_parameters_preserved_for_reproducibility": {
        "iql_td3bc_q_alpha": 0.8,
        "iql_td3bc_bc_alpha": 0.3,
        "iql_td3bc_action_penalty_alpha": 0.02,
    },
}
(run / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json
import sys
from datetime import datetime

path = Path(sys.argv[1]) / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "preparing_runtimes"
manifest["baseline_gate_passed_at"] = datetime.now().astimezone().isoformat()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

create_runtime() {
  local seed="$1"
  local runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  mkdir -p "$RUN_ROOT/train/seed_${seed}/logs"
  python scripts/epi_abm/create_isolated_runtime.py \
    --source data_generation/epi_diff_abm \
    --dest "$runtime" \
    --force \
    > "$RUN_ROOT/train/seed_${seed}/logs/runtime_create.log" 2>&1
  if rg -n 'external_repos/epi-diff-abm' "$RUN_ROOT/train/seed_${seed}/logs/runtime_create.log"; then
    fail "seed $seed runtime points to forbidden external_repos/epi-diff-abm"
  fi
}

for seed in "${SEEDS[@]}"; do
  create_runtime "$seed"
done
log "isolated runtimes ready"

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json
import sys
from datetime import datetime

path = Path(sys.argv[1]) / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "training"
manifest["training_started_at"] = datetime.now().astimezone().isoformat()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

launch_train() {
  local seed="$1"
  local gpu="$2"
  local sdir="$RUN_ROOT/train/seed_${seed}"
  local runtime="$RUN_ROOT/runtime/seed_${seed}_epi_diff_abm"
  mkdir -p "$sdir/em_ckpt" "$sdir/hydra" "$sdir/logs"
  log "train start seed=$seed gpu=$gpu"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="$THREADS_PER_SEED"
    export MKL_NUM_THREADS="$THREADS_PER_SEED"
    export OPENBLAS_NUM_THREADS="$THREADS_PER_SEED"
    export NUMEXPR_NUM_THREADS="$THREADS_PER_SEED"
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
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
      '+exp.iql_target_horizons=[5,7,9,11,13,15,17,19,21]' \
      '+exp.em_val_tau_list=[5,7,9,11,13,15,17,19,21]' \
      '+exp.iql_eval_tau_list=[5,7,9,11,13,15,17,19,21]' \
      exp.iql_decision_interval_days=1 \
      exp.use_mlflow=false \
      +exp.em_outer_iters=20 \
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
      +exp.iql_actor_lr=3e-4 \
      +exp.iql_qf_lr=3e-4 \
      +exp.iql_vf_lr=3e-4 \
      +exp.iql_hidden_dim=256 \
      +exp.iql_n_hidden=2 \
      +exp.iql_tau=0.7 \
      +exp.iql_weight_max=3.0 \
      +exp.iql_discount=0.99 \
      +exp.iql_target_tau=0.005 \
      +exp.iql_max_grad_norm=5.0 \
      +exp.iql_target_sampling=horizon_aligned \
      +exp.iql_horizon_terminal_done=true \
      +exp.em_encoder_lr=5e-5 \
      +exp.em_encoder_max_grad_norm=1.0 \
      +exp.em_e_w_lr=1e-2 \
      +exp.ct_w_lr=1e-2 \
      exp.ct_use_weight_net=true \
      exp.ct_align_loss=sinkhorn \
      +exp.ct_sinkhorn_blur=0.01 \
      +exp.ct_weight_hidden=64 \
      +exp.ct_weight_decay=1e-5 \
      +exp.ct_w_clip=5.0 \
      +exp.iql_eval_action_selector=q_sample \
      +exp.iql_eval_candidate_actions=64 \
      +exp.iql_eval_candidate_noise_std=0.25 \
      +exp.iql_eval_q_bc_penalty=1.0 \
      +exp.iql_actor_update=awr \
      +exp.iql_beta=2.0 \
      +exp.iql_adv_max=100.0 \
      +exp.iql_actor_bc_loss=expectile \
      +exp.iql_actor_bc_expectile=0.8 \
      +exp.iql_td3bc_q_alpha=0.8 \
      +exp.iql_td3bc_bc_alpha=0.3 \
      +exp.iql_td3bc_action_penalty_alpha=0.02 \
      > "$sdir/logs/train.log" 2>&1
  )
}

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPU_ASSIGNMENTS[$i]}"
  launch_train "$seed" "$gpu" &
  pids+=("$!")
  printf '%s %s %s\n' "$!" "$seed" "$gpu" >> "$RUN_ROOT/logs/training_pids.tsv"
done

audit_hydra_configs() {
  python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json
import math
import sys
import time

from omegaconf import OmegaConf

run = Path(sys.argv[1])
seeds = [10, 101, 1010, 10101, 101010]
deadline = time.time() + 180
config_paths = {
    seed: run / "train" / f"seed_{seed}" / "hydra" / ".hydra" / "config.yaml"
    for seed in seeds
}
while time.time() < deadline and not all(path.is_file() for path in config_paths.values()):
    time.sleep(2)

expected = {
    "dataset.seed": 100,
    "dataset.device": "cpu",
    "dataset.action_hold_days": 1,
    "dataset.generate_if_missing": False,
    "dataset.force_regenerate": False,
    "dataset.behavior_policy_subset": "factual_only",
    "dataset.split_by": "county",
    "dataset.outcome_transform": "per10k_cases_zscore",
    "exp.max_tau": 21,
    "exp.iql_target_horizons": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "exp.em_val_tau_list": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "exp.iql_eval_tau_list": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "exp.iql_decision_interval_days": 1,
    "exp.em_outer_iters": 20,
    "exp.em_m_steps_per_outer": 1000,
    "exp.em_e_epochs": 3,
    "exp.em_val_every": 0,
    "exp.em_val_repeats": 1,
    "exp.em_val_tau_agg": "mean",
    "exp.em_warmup_outer_iters": 1,
    "exp.iql_batch_size": 256,
    "exp.ct_batch_size": 512,
    "exp.iql_actor_lr": 3e-4,
    "exp.iql_qf_lr": 3e-4,
    "exp.iql_vf_lr": 3e-4,
    "exp.iql_hidden_dim": 256,
    "exp.iql_n_hidden": 2,
    "exp.iql_tau": 0.7,
    "exp.iql_beta": 2.0,
    "exp.iql_adv_max": 100.0,
    "exp.iql_weight_max": 3.0,
    "exp.iql_actor_update": "awr",
    "exp.iql_actor_bc_loss": "expectile",
    "exp.iql_actor_bc_expectile": 0.8,
    "exp.iql_discount": 0.99,
    "exp.iql_target_tau": 0.005,
    "exp.iql_max_grad_norm": 5.0,
    "exp.iql_target_sampling": "horizon_aligned",
    "exp.iql_horizon_terminal_done": True,
    "exp.em_encoder_lr": 5e-5,
    "exp.em_encoder_max_grad_norm": 1.0,
    "exp.em_e_w_lr": 1e-2,
    "exp.ct_w_lr": 1e-2,
    "exp.ct_use_weight_net": True,
    "exp.ct_align_loss": "sinkhorn",
    "exp.ct_sinkhorn_blur": 0.01,
    "exp.ct_weight_hidden": 64,
    "exp.ct_weight_decay": 1e-5,
    "exp.ct_w_clip": 5.0,
    "exp.iql_eval_action_selector": "q_sample",
    "exp.iql_eval_candidate_actions": 64,
    "exp.iql_eval_candidate_noise_std": 0.25,
    "exp.iql_eval_q_bc_penalty": 1.0,
}

def normalize(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value

def equal(actual, wanted):
    actual = normalize(actual)
    if isinstance(wanted, float):
        try:
            return math.isclose(float(actual), wanted, rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == wanted

report = {"status": "pass", "seeds": {}}
errors = []
for seed, path in config_paths.items():
    seed_report = {"config": str(path), "checks": {}}
    report["seeds"][str(seed)] = seed_report
    if not path.is_file():
        errors.append(f"seed={seed}: missing {path}")
        seed_report["status"] = "missing_config"
        continue
    cfg = OmegaConf.load(path)
    seed_expected = dict(expected)
    seed_expected["exp.seed"] = seed
    for key, wanted in seed_expected.items():
        actual = OmegaConf.select(cfg, key, default="__MISSING__")
        ok = equal(actual, wanted)
        seed_report["checks"][key] = {
            "actual": normalize(actual),
            "expected": wanted,
            "ok": ok,
        }
        if not ok:
            errors.append(f"seed={seed} {key}: actual={actual!r} expected={wanted!r}")
    seed_report["status"] = "pass" if not any(
        not item["ok"] for item in seed_report["checks"].values()
    ) else "fail"

if errors:
    report["status"] = "fail"
    report["errors"] = errors
audit_path = run / "logs" / "post_launch_parameter_audit.json"
audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    raise SystemExit("; ".join(errors))
print(f"parameter audit passed: {audit_path}")
PY
}

if ! audit_hydra_configs >> "$QUEUE_LOG" 2>&1; then
  log "post-launch parameter audit failed; terminating the five new training processes"
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  fail "resolved Hydra parameters do not match the approved protocol"
fi
log "post-launch Hydra parameter audit passed for all five seeds"

failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    log "train process complete seed=${SEEDS[$i]} gpu=${GPU_ASSIGNMENTS[$i]}"
  else
    log "train process failed seed=${SEEDS[$i]} gpu=${GPU_ASSIGNMENTS[$i]}"
    failed=1
  fi
done
(( failed == 0 )) || fail "one or more seed training processes failed; logs preserved"

log "all training processes exited; validating checkpoints"
for seed in "${SEEDS[@]}"; do
  sdir="$RUN_ROOT/train/seed_${seed}"
  if rg -n "$ERRPAT" "$sdir/logs/train.log" > "$sdir/logs/train_errors.txt"; then
    fail "train log contains an error signature for seed=$seed"
  fi
  for outer in $(seq 1 20); do
    label="$(printf 'outer%04d' "$outer")"
    [[ -f "$sdir/em_ckpt/ct_iql_em_${label}.pt" ]] || fail "missing $label for seed=$seed"
  done
  rg -n 'EM outer 20/20|EM training done' "$sdir/logs/train.log" >/dev/null \
    || fail "seed=$seed did not complete 20 outer iterations"
done

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json
import math
import sys

from omegaconf import OmegaConf
import torch

run = Path(sys.argv[1])
seeds = [10, 101, 1010, 10101, 101010]
expected = {
    "exp.iql_target_horizons": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "exp.em_val_tau_list": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "exp.iql_eval_tau_list": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "exp.em_outer_iters": 20,
    "exp.em_m_steps_per_outer": 1000,
    "exp.em_e_epochs": 3,
    "exp.em_val_every": 0,
    "exp.iql_actor_lr": 3e-4,
    "exp.iql_qf_lr": 3e-4,
    "exp.iql_vf_lr": 3e-4,
    "exp.iql_actor_update": "awr",
    "exp.iql_beta": 2.0,
    "exp.iql_adv_max": 100.0,
    "exp.iql_actor_bc_loss": "expectile",
    "exp.iql_actor_bc_expectile": 0.8,
    "exp.em_encoder_lr": 5e-5,
    "exp.em_e_w_lr": 1e-2,
    "exp.ct_w_lr": 1e-2,
}

def normalize(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value

def equal(actual, wanted):
    actual = normalize(actual)
    if isinstance(wanted, float):
        try:
            return math.isclose(float(actual), wanted, rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == wanted

report = {"status": "pass", "seeds": {}}
errors = []
for seed in seeds:
    path = run / "train" / f"seed_{seed}" / "em_ckpt" / "ct_iql_em_outer0020.pt"
    seed_report = {"checkpoint": str(path), "checks": {}}
    report["seeds"][str(seed)] = seed_report
    payload = torch.load(path, map_location="cpu")
    cfg = OmegaConf.create(payload["config"])
    seed_expected = dict(expected)
    seed_expected["exp.seed"] = seed
    for key, wanted in seed_expected.items():
        actual = OmegaConf.select(cfg, key, default="__MISSING__")
        ok = equal(actual, wanted)
        seed_report["checks"][key] = {
            "actual": normalize(actual),
            "expected": wanted,
            "ok": ok,
        }
        if not ok:
            errors.append(f"seed={seed} {key}: actual={actual!r} expected={wanted!r}")
    seed_report["status"] = "pass" if not any(
        not item["ok"] for item in seed_report["checks"].values()
    ) else "fail"

if errors:
    report["status"] = "fail"
    report["errors"] = errors
audit_path = run / "logs" / "final_checkpoint_parameter_audit.json"
audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    raise SystemExit("; ".join(errors))
print(f"checkpoint parameter audit passed: {audit_path}")
PY

python - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json
import sys
from datetime import datetime

path = Path(sys.argv[1]) / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "training_done"
manifest["training_completed_at"] = datetime.now().astimezone().isoformat()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
log "CRIPO EpiSCTA training complete for all five seeds"
