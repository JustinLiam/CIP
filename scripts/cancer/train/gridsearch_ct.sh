#!/usr/bin/env bash
# Grid search over CT-stage hyperparameters, scored by downstream IQL val MAE (unscaled).
#
# Sweeps (edit arrays below):
#   exp.ct_lr            — CT weight_net learning rate (theta lr is hardcoded 1e-4 in train_ct.py)
#   exp.ct_multi_k_max   — multi-horizon teacher depth {1, 2, 3}
#   exp.ct_multi_eta     — geometric weight ratio across k
#   exp.ct_es_metric     — early-stop / best-ckpt metric: l1 | weighted
#
# Pipeline per combo × seed:
#   train_ct.py  -> copy ct_best_encoder.pt to combo-specific file
#   train_iql_planner.py  (exp.iql_inference_ckpt = that file)
#                           -> copy iql_planner.pt to combo-specific file
#   eval_iql_planner.py  (exp.iql_eval_ckpt = that file, exp.test=${TEST_SPLIT:-false})
#
# Outputs (ALL under grid_results/; canonical ct_checkpoints/ is NEVER touched):
#   grid_results/ct/gamma_${gamma}/summary.csv            — one row per (combo, seed)
#   grid_results/ct/gamma_${gamma}/logs/${combo_id}__seed_${seed}.log
#   grid_results/ct/gamma_${gamma}/ct_ckpts_work/${combo_id}__seed_${seed}/ct_best_encoder.pt
#                                                         — train_ct writes here (via exp.ct_ckpt_dir)
#   grid_results/ct/gamma_${gamma}/ct_ckpts/${combo_id}__seed_${seed}.pt
#                                                         — flat archive copy (what IQL reads)
#   grid_results/ct/gamma_${gamma}/iql_ckpts/${combo_id}__seed_${seed}.pt
#
# Isolation from main pipeline:
#   This script passes exp.ct_ckpt_dir=<grid sandbox>, so train_ct.py never
#   overwrites ct_checkpoints/seed_${s}_gamma_${g}/ct_best_encoder.pt. You can
#   safely keep running train_ct_iql.sh (or any downstream IQL dev work that
#   reads the canonical path) in parallel on a different GPU.
#
# Usage (from repo root):
#   bash scripts/cancer/train/gridsearch_ct.sh [gamma] [gpu] [test_split]
#     gamma       - dataset.coeff, default 4
#     gpu         - CUDA_VISIBLE_DEVICES, default 0
#     test_split  - exp.test for eval: false (val, default) | true (test)
#
# Optional env vars:
#   GRID_SKIP_IQL=1          — skip IQL train+eval; score by CT val_L1 only (fast proxy)
#   GRID_SEEDS="10 20"       — override SEEDS array at runtime
#   GRID_WORKER_ID=0         — this worker's index in [0, GRID_NUM_WORKERS-1]
#   GRID_NUM_WORKERS=1       — total number of parallel workers sharing summary.csv
#
# Multi-GPU parallel grid search (e.g. 2x H800):
#   # Terminal A (GPU 0, worker 0/2)
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_ct.sh 4 0
#   # Terminal B (GPU 1, worker 1/2)
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_ct.sh 4 1
# Workers auto-split combos by hash, share one summary.csv (flock-protected),
# and each writes into its own per-tag CT/IQL sandbox directory.
#
# Safe to re-run: existing (combo, seed) rows in summary.csv are skipped.

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate vcip

gamma=${1:-4}
gpu=${2:-0}
TEST_SPLIT=${3:-false}

# Sharding across parallel workers (e.g. multi-GPU). Default: single worker.
GRID_WORKER_ID=${GRID_WORKER_ID:-0}
GRID_NUM_WORKERS=${GRID_NUM_WORKERS:-1}
if (( GRID_NUM_WORKERS < 1 )) || (( GRID_WORKER_ID < 0 )) || (( GRID_WORKER_ID >= GRID_NUM_WORKERS )); then
  echo "ERROR: invalid sharding: GRID_WORKER_ID=${GRID_WORKER_ID} GRID_NUM_WORKERS=${GRID_NUM_WORKERS}" >&2
  exit 1
fi
echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${gamma} gpu=${gpu}"

# ---------- grid definition ----------
CT_LR_LIST=("5e-4" "1e-3" "3e-3")
CT_MULTI_K_MAX_LIST=("1" "2" "3")
CT_MULTI_ETA_LIST=("0.3" "0.5" "0.7")
CT_ES_METRIC_LIST=("l1" "weighted")

# Seeds: override at runtime with `GRID_SEEDS="10 20" bash ...`
if [[ -n "${GRID_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "${GRID_SEEDS}"
else
  SEEDS=(10)
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

grid_root="${ROOT}/grid_results/ct/gamma_${gamma}"
mkdir -p "${grid_root}/logs" "${grid_root}/ct_ckpts" "${grid_root}/iql_ckpts" \
         "${grid_root}/ct_ckpts_work" "${grid_root}/iql_ckpts_work"
summary_csv="${grid_root}/summary.csv"
summary_lock="${grid_root}/.summary.lock"

# Cross-worker lock helpers: flock() serializes summary.csv reads/writes so two
# GPUs don't (a) pick the same combo (skip-check + append must be atomic as a
# unit for correctness) and (b) interleave their CSV rows.
with_lock() {
  # usage: with_lock <command...>
  # Holds an exclusive file lock on $summary_lock for the duration of the cmd.
  ( flock -x 9; "$@" ) 9> "${summary_lock}"
}

# Header: only worker 0 initializes (others wait a moment if needed).
if [[ ! -f "${summary_csv}" ]]; then
  if (( GRID_WORKER_ID == 0 )); then
    with_lock bash -c "[[ -f '${summary_csv}' ]] || echo 'combo_id,seed,ct_lr,ct_multi_k_max,ct_multi_eta,ct_es_metric,ct_val_L1,iql_best_val_mae_uns,iql_best_step,mae_uns_eval_split,eval_split' > '${summary_csv}'"
  else
    # Give worker 0 up to 10s to create the header.
    for _ in $(seq 1 20); do
      [[ -f "${summary_csv}" ]] && break
      sleep 0.5
    done
    [[ -f "${summary_csv}" ]] || { echo "ERROR: worker ${GRID_WORKER_ID} timed out waiting for summary.csv header"; exit 1; }
  fi
fi

parse_last() {
  # $1 = log file, $2 = perl-compat regex with \K marking captured group
  local file="$1" pat="$2"
  grep -oP "${pat}" "${file}" 2>/dev/null | tail -n1 || true
}

# Deterministic shard assignment: hash(tag) % GRID_NUM_WORKERS.
# Bash has no builtin hash; cksum (CRC32) is universal and stable across runs.
shard_owner() {
  # echo integer in [0, GRID_NUM_WORKERS) assigned to tag $1
  local tag="$1"
  local h
  h=$(printf '%s' "${tag}" | cksum | awk '{print $1}')
  echo $(( h % GRID_NUM_WORKERS ))
}

run_combo_seed() {
  local lr="$1" kmax="$2" eta="$3" esm="$4" seed="$5"
  local combo_id="lr-${lr}_k-${kmax}_eta-${eta}_esm-${esm}"
  local tag="${combo_id}__seed_${seed}"
  local log_file="${grid_root}/logs/${tag}.log"
  # CT writes directly into this sandbox dir; the canonical
  # ct_checkpoints/seed_${s}_gamma_${g}/ is NEVER touched.
  local ct_work_dir="${grid_root}/ct_ckpts_work/${tag}"
  local ct_ckpt_src="${ct_work_dir}/ct_best_encoder.pt"
  # IQL writes directly into this sandbox dir; the canonical
  # iql_models/seed_${s}/gamma_${g}/ is NEVER touched. This also eliminates
  # the race when two GPU workers train IQL concurrently.
  local iql_work_dir="${grid_root}/iql_ckpts_work/${tag}"
  local iql_ckpt_src="${iql_work_dir}/iql_planner.pt"
  # Flat per-combo archives (stable user-facing paths, same layout as before).
  local ct_ckpt_grid="${grid_root}/ct_ckpts/${tag}.pt"
  local iql_ckpt_grid="${grid_root}/iql_ckpts/${tag}.pt"

  # Shard: this worker only owns combos whose hash matches.
  local owner
  owner=$(shard_owner "${tag}")
  if (( owner != GRID_WORKER_ID )); then
    return 0  # someone else's job
  fi

  # Skip-check must be inside the lock, otherwise two workers could both
  # decide to run the same tag after a simultaneous summary.csv read.
  if with_lock grep -q "^${combo_id},${seed}," "${summary_csv}"; then
    echo "[w${GRID_WORKER_ID}][skip] ${tag} already recorded in ${summary_csv}"
    return 0
  fi

  mkdir -p "${ct_work_dir}" "${iql_work_dir}"

  echo "================================================================"
  echo "=== ${tag} | train_ct ==="
  echo "================================================================"
  CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_ct.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    exp.ct_lr="${lr}" \
    exp.ct_multi_k_max="${kmax}" \
    exp.ct_multi_eta="${eta}" \
    exp.ct_es_metric="${esm}" \
    "+exp.ct_ckpt_dir=${ct_work_dir}" \
    2>&1 | tee "${log_file}"

  if [[ ! -f "${ct_ckpt_src}" ]]; then
    echo "ERROR: CT ckpt missing for ${tag} at ${ct_ckpt_src}" >&2
    with_lock bash -c "echo '${combo_id},${seed},${lr},${kmax},${eta},${esm},NA,NA,NA,NA,${TEST_SPLIT}' >> '${summary_csv}'"
    return 0
  fi
  cp "${ct_ckpt_src}" "${ct_ckpt_grid}"

  # train_ct logs "Saved encoder checkpoint to ... val_L1=X" every time a new best is saved.
  # tail -n1 gives the best epoch's val_L1 (uniform across ct_es_metric=l1|weighted).
  local ct_val_l1
  ct_val_l1=$(parse_last "${log_file}" 'Saved encoder checkpoint to .*val_L1=\K[0-9.eE+-]+')
  [[ -z "${ct_val_l1}" ]] && ct_val_l1="NA"

  local iql_best_mae="NA" iql_best_step="NA" mae_eval="NA"

  if [[ "${GRID_SKIP_IQL:-0}" != "1" ]]; then
    echo "================================================================"
    echo "=== ${tag} | train_iql_planner ==="
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.iql_inference_ckpt="${ct_ckpt_grid}" \
      "+exp.iql_save_dir=${iql_work_dir}" \
      2>&1 | tee -a "${log_file}"

    if [[ ! -f "${iql_ckpt_src}" ]]; then
      echo "ERROR: IQL ckpt missing for ${tag} at ${iql_ckpt_src}" >&2
      with_lock bash -c "echo '${combo_id},${seed},${lr},${kmax},${eta},${esm},${ct_val_l1:-NA},NA,NA,NA,${TEST_SPLIT}' >> '${summary_csv}'"
      return 0
    fi
    cp "${iql_ckpt_src}" "${iql_ckpt_grid}"

    # Matches both old format ("Saved BEST IQL planner to /...pt (mae_uns=X at step Y)")
    # and new format ("Saved BEST IQL planner (selection_world='sim') to /...pt (mae_uns=X at step Y)").
    iql_best_mae=$(parse_last "${log_file}" 'Saved BEST IQL planner.*\(mae_uns=\K[0-9.eE+-]+')
    iql_best_step=$(parse_last "${log_file}" 'Saved BEST IQL planner.* at step \K[0-9]+')
    [[ -z "${iql_best_mae}" ]] && iql_best_mae="NA"
    [[ -z "${iql_best_step}" ]] && iql_best_step="NA"

    echo "================================================================"
    echo "=== ${tag} | eval_iql_planner (exp.test=${TEST_SPLIT}) ==="
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/eval_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.test="${TEST_SPLIT}" \
      exp.iql_inference_ckpt="${ct_ckpt_grid}" \
      exp.iql_eval_ckpt="${iql_ckpt_grid}" \
      2>&1 | tee -a "${log_file}"

    mae_eval=$(parse_last "${log_file}" 'MAE unscaled: \K[0-9.eE+-]+')
    [[ -z "${mae_eval}" ]] && mae_eval="NA"
  fi

  with_lock bash -c "echo '${combo_id},${seed},${lr},${kmax},${eta},${esm},${ct_val_l1:-NA},${iql_best_mae},${iql_best_step},${mae_eval},${TEST_SPLIT}' >> '${summary_csv}'"
  echo "[w${GRID_WORKER_ID}][${tag}] ct_val_L1=${ct_val_l1:-NA} iql_best_val_mae=${iql_best_mae} mae_eval=${mae_eval}"
}

for lr in "${CT_LR_LIST[@]}"; do
  for kmax in "${CT_MULTI_K_MAX_LIST[@]}"; do
    # When ct_multi_k_max==1 the multi-horizon loss collapses to loss_pred1, so
    # ct_multi_eta (only used as horiz_w[1:] weights) has NO effect, and
    # ct_es_metric ('l1' vs 'weighted') both reduce to mean_l1. Iterating all
    # eta/esm combos here trains an identical model multiple times -- wasted
    # compute. Collapse to a single canonical (eta, esm) pair in that case.
    if [[ "${kmax}" == "1" ]]; then
      eta_iter=("${CT_MULTI_ETA_LIST[0]}")
      esm_iter=("${CT_ES_METRIC_LIST[0]}")
    else
      eta_iter=("${CT_MULTI_ETA_LIST[@]}")
      esm_iter=("${CT_ES_METRIC_LIST[@]}")
    fi
    for eta in "${eta_iter[@]}"; do
      for esm in "${esm_iter[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_combo_seed "${lr}" "${kmax}" "${eta}" "${esm}" "${seed}"
        done
      done
    done
  done
done

echo
echo "================================================================"
echo "=== Grid done. Summary CSV: ${summary_csv}"
echo "================================================================"

python - <<PY
import csv
from collections import defaultdict
from statistics import median

path = "${summary_csv}"
skip_iql = "${GRID_SKIP_IQL:-0}" == "1"

# Primary score: mae_uns_eval_split (downstream eval); fallback: iql_best_val_mae_uns; fallback: ct_val_L1.
# Aggregate across seeds by median (robust to occasional noisy runs).
def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

rows = list(csv.DictReader(open(path)))
groups = defaultdict(list)  # combo_id -> list[(score_primary, score_iql, score_ct, row)]
for r in rows:
    primary = to_float(r["mae_uns_eval_split"])
    iql = to_float(r["iql_best_val_mae_uns"])
    ctv = to_float(r["ct_val_L1"])
    groups[r["combo_id"]].append((primary, iql, ctv, r))

def pick_score(triples):
    """Return (primary_median, secondary_median, tertiary_median, n_seeds)."""
    p = [t[0] for t in triples if t[0] is not None]
    q = [t[1] for t in triples if t[1] is not None]
    c = [t[2] for t in triples if t[2] is not None]
    return (
        median(p) if p else None,
        median(q) if q else None,
        median(c) if c else None,
        len(triples),
    )

scored = []
for combo_id, triples in groups.items():
    p, q, c, n = pick_score(triples)
    sample = triples[0][3]
    if skip_iql:
        sort_key = c if c is not None else float("inf")
    else:
        sort_key = p if p is not None else (q if q is not None else float("inf"))
    scored.append((sort_key, combo_id, p, q, c, n, sample))

scored.sort(key=lambda x: x[0])
print(f"\n{len(scored)} unique combos; score column = {'ct_val_L1 (GRID_SKIP_IQL)' if skip_iql else 'mae_uns_eval_split (or iql_best_val_mae_uns)'}\n")
print(f"{'rank':<4}  {'score':<10}  {'iql_best':<10}  {'ct_L1':<10}  {'n':<2}  lr        k  eta   esm       combo_id")
for i, (sk, combo_id, p, q, c, n, row) in enumerate(scored[:10], 1):
    lr = row["ct_lr"]
    kmax = row["ct_multi_k_max"]
    eta = row["ct_multi_eta"]
    esm = row["ct_es_metric"]
    def fmt(x): return f"{x:.6f}" if isinstance(x, float) else "NA"
    print(f"{i:<4}  {fmt(p):<10}  {fmt(q):<10}  {fmt(c):<10}  {n:<2}  {lr:<9} {kmax}  {eta:<4}  {esm:<8}  {combo_id}")

if scored:
    best_combo = scored[0][1]
    print(f"\nBest combo (by lowest median score): {best_combo}")
    print(f"Ckpt files (one per seed) under: ${grid_root}/{{ct_ckpts,iql_ckpts}}/{best_combo}__seed_*.pt")
PY
