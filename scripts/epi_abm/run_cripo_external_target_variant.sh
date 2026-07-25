#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/liam/pythonProject/VCIP-ICML-main
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

TRAIN_RUN_ROOT="${1:?Usage: $0 TRAIN_RUN_ROOT EVAL_ROOT VARIANT TARGET_MODE TARGET_SCALE GPU TARGET_FILE}"
EVAL_ROOT="${2:?missing EVAL_ROOT}"
VARIANT="${3:?missing VARIANT}"
TARGET_MODE="${4:?missing TARGET_MODE}"
TARGET_SCALE="${5:?missing TARGET_SCALE}"
GPU="${6:?missing GPU}"
TARGET_FILE="${7:?missing TARGET_FILE}"

SEEDS=(10 101 1010 10101 101010)
SELECTION_TAUS=(5 7 9 11 13 15 17 19 21)
REPORT_TAUS=($(seq 1 21))
WORKER_IDS=("${VARIANT}_g${GPU}a" "${VARIANT}_g${GPU}b")
ROW_GROUPS=("0 2 4 6 8 10 12 14 16 18 20 22" "1 3 5 7 9 11 13 15 17 19 21")
THREADS_PER_WORKER="${THREADS_PER_WORKER:-6}"
EXPECTED_TARGET_SHA256="600d2b0decd9f154ec978541ea48556f6905cd0331b2863aba74f1bc3a350ede"
ERRPAT='Traceback|RuntimeError|ValueError|IndexError|OutOfMemoryError|CUDA out|Missing external target|Duplicate external target mismatch|Error executing job|Killed'

VARIANT_ROOT="$EVAL_ROOT/$VARIANT"
LOG_DIR="$VARIANT_ROOT/logs"
VAL_MANIFEST="$VARIANT_ROOT/manifests/val_all_checkpoints.json"
SELECTION="$VARIANT_ROOT/selection/selected_best.json"
TEST_MANIFEST="$VARIANT_ROOT/manifests/test_selected.json"
VALID_SELECTED_MANIFEST="$VARIANT_ROOT/manifests/valid_selected.json"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
mkdir -p "$LOG_DIR" "$VARIANT_ROOT/manifests" "$VARIANT_ROOT/selection"

log(){ printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$SUPERVISOR_LOG"; }
fail(){ log "FAILED: $*"; exit 1; }

[[ -f "$TARGET_FILE" ]] || fail "missing target file: $TARGET_FILE"
actual_sha="$(sha256sum "$TARGET_FILE" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_TARGET_SHA256" ]] \
  || fail "target SHA256 mismatch: $actual_sha"
[[ "$TARGET_MODE" == "factual_final" || "$TARGET_MODE" == "half_factual_final" ]] \
  || fail "unsupported target mode: $TARGET_MODE"

python - "$TRAIN_RUN_ROOT" "$VARIANT_ROOT" "$VAL_MANIFEST" "${SEEDS[@]}" <<'PY'
from pathlib import Path
import json
import sys

train = Path(sys.argv[1]).resolve()
variant = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
seeds = [int(seed) for seed in sys.argv[4:]]
config = train / "configs" / "epi_abm_multi_daily_seed100.yaml"
if not config.is_file():
    raise SystemExit(f"missing config: {config}")
jobs = []
for seed in seeds:
    checkpoints = {}
    for outer in range(1, 21):
        label = f"outer{outer:04d}"
        checkpoint = train / "train" / f"seed_{seed}" / "em_ckpt" / f"ct_iql_em_{label}.pt"
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint: {checkpoint}")
        checkpoints[label] = str(checkpoint)
    jobs.append({
        "id": f"cripo_seed_{seed}",
        "method": "cripo",
        "seed": seed,
        "config": str(config),
        "out_dir": str(variant / "val_all_checkpoints" / f"seed_{seed}"),
        "ckpts": checkpoints,
        "selector": "q_sample",
        "candidate_actions": 64,
        "q_bc_penalty": 1.0,
        "candidate_noise_std": 0.25,
        "eval_seed": 20260708,
    })
output.write_text(json.dumps({
    "schema": "epi_abm_county_major_jobs_v2",
    "purpose": "validation checkpoint selection against immutable GPU replay target",
    "jobs": jobs,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

prepare_runtimes(){
  local idx worker runtime
  for idx in 0 1; do
    worker="${WORKER_IDS[$idx]}"
    runtime="$VARIANT_ROOT/runtime/$worker"
    if [[ ! -d "$runtime" ]]; then
      mkdir -p "$(dirname "$runtime")"
      python scripts/epi_abm/create_isolated_runtime.py \
        --source data_generation/epi_diff_abm \
        --dest "$runtime" \
        --force \
        > "$LOG_DIR/runtime_${worker}.log" 2>&1
    fi
  done
}

run_workers(){
  local phase="$1" manifest="$2" split="$3"
  shift 3
  local taus=("$@")
  local pids=() logs=() idx worker runtime log_path rows
  for idx in 0 1; do
    worker="${WORKER_IDS[$idx]}"
    runtime="$VARIANT_ROOT/runtime/$worker"
    log_path="$LOG_DIR/${phase}_${worker}.log"
    rows="${ROW_GROUPS[$idx]}"
    logs+=("$log_path")
    log "phase=$phase worker=$worker gpu=$GPU split=$split rows=[$rows] start"
    env OMP_NUM_THREADS="$THREADS_PER_WORKER" \
      MKL_NUM_THREADS="$THREADS_PER_WORKER" \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER" \
      EPI_ABM_REQUIRE_NETWORK_ARCHIVE=1 \
      python -u scripts/epi_abm/evaluate_county_major_iql.py \
        --manifest "$manifest" \
        --worker-id "$worker" \
        --split "$split" \
        --taus "${taus[@]}" \
        --decision-day 161 \
        --window-mode fixed-start \
        --target-mode "$TARGET_MODE" \
        --target-scale "$TARGET_SCALE" \
        --external-target-file "$TARGET_FILE" \
        --selector q_sample \
        --candidate-actions 64 \
        --candidate-noise-std 0.25 \
        --q-bc-penalty 1.0 \
        --eval-seed 20260708 \
        --model-device "cuda:$GPU" \
        --abm-device "cuda:$GPU" \
        --epi-root "$runtime" \
        --processed-data-dir data/processed/epi_abm/full_factual_daily_seed100_20260709_223702 \
        --cache-version current_runtime_factual_daily_seed100 \
        --dataset-seed 100 \
        --outcome-transform per10k_cases_zscore \
        --row-indices $rows \
        > "$log_path" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for idx in 0 1; do
    if ! wait "${pids[$idx]}"; then
      failed=1
      log "phase=$phase worker=${WORKER_IDS[$idx]} exited nonzero"
    fi
    if rg -n "$ERRPAT" "${logs[$idx]}" > "${logs[$idx]%.log}_errors.txt"; then
      failed=1
      log "phase=$phase worker=${WORKER_IDS[$idx]} has error signatures"
    fi
  done
  (( failed == 0 )) || fail "phase=$phase failed"
  python scripts/epi_abm/merge_county_major_results.py \
    --manifest "$manifest" \
    --worker-ids "${WORKER_IDS[@]}" \
    --output-name parallel_merged \
    >> "$SUPERVISOR_LOG" 2>&1
  log "phase=$phase merged"
}

prepare_runtimes
log "variant start target_mode=$TARGET_MODE target_scale=$TARGET_SCALE gpu=$GPU target_sha256=$actual_sha"

run_workers val_all_checkpoints "$VAL_MANIFEST" val "${SELECTION_TAUS[@]}"
python scripts/epi_abm/select_county_major_best.py \
  --val-manifest "$VAL_MANIFEST" \
  --selection-taus "${SELECTION_TAUS[@]}" \
  --expected-counties 23 \
  --merged-name parallel_merged \
  --target-reference external_target_file \
  --test-dir-name test_selected \
  --selection-output "$SELECTION" \
  --test-manifest-output "$TEST_MANIFEST" \
  >> "$SUPERVISOR_LOG" 2>&1
log "validation checkpoint selection complete"

python - "$TEST_MANIFEST" "$VALID_SELECTED_MANIFEST" "$VARIANT_ROOT" <<'PY'
from pathlib import Path
import json
import sys

test_manifest = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
variant = Path(sys.argv[3]).resolve()
payload = json.loads(test_manifest.read_text(encoding="utf-8"))
jobs = []
for job in payload["jobs"]:
    updated = dict(job)
    updated["out_dir"] = str(variant / "valid_selected" / f"seed_{int(job['seed'])}")
    jobs.append(updated)
output.write_text(json.dumps({
    "schema": "epi_abm_county_major_jobs_v2",
    "source_selection": payload.get("source_selection"),
    "target_reference": "external_target_file",
    "jobs": jobs,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

run_workers valid_selected "$VALID_SELECTED_MANIFEST" val "${REPORT_TAUS[@]}"
run_workers test_selected "$TEST_MANIFEST" test "${REPORT_TAUS[@]}"

python scripts/epi_abm/aggregate_county_major_target.py \
  --manifest "$VALID_SELECTED_MANIFEST" \
  --split val \
  --output-dir "$VARIANT_ROOT/final/valid" \
  --expected-counties 23 \
  --expected-seeds 5 \
  --taus "${REPORT_TAUS[@]}" \
  >> "$SUPERVISOR_LOG" 2>&1
python scripts/epi_abm/aggregate_county_major_target.py \
  --manifest "$TEST_MANIFEST" \
  --split test \
  --output-dir "$VARIANT_ROOT/final/test" \
  --expected-counties 23 \
  --expected-seeds 5 \
  --taus "${REPORT_TAUS[@]}" \
  >> "$SUPERVISOR_LOG" 2>&1

python - "$VARIANT_ROOT" "$TRAIN_RUN_ROOT" "$TARGET_FILE" "$actual_sha" "$TARGET_MODE" "$TARGET_SCALE" "$GPU" <<'PY'
from datetime import datetime
from pathlib import Path
import json
import sys

root = Path(sys.argv[1]).resolve()
payload = {
    "schema": "epi_abm_cripo_external_gpu_replay_target_variant_v1",
    "status": "complete",
    "completed_at": datetime.now().astimezone().isoformat(),
    "training_run": str(Path(sys.argv[2]).resolve()),
    "external_target_file": str(Path(sys.argv[3]).resolve()),
    "external_target_sha256": sys.argv[4],
    "target_mode": sys.argv[5],
    "target_scale": float(sys.argv[6]),
    "gpu": int(sys.argv[7]),
    "selection_taus": [5, 7, 9, 11, 13, 15, 17, 19, 21],
    "report_taus": list(range(1, 22)),
    "selection": str(root / "selection" / "selected_best.json"),
    "valid_results": str(root / "final" / "valid"),
    "test_results": str(root / "final" / "test"),
}
(root / "manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
log "variant complete"
