#!/usr/bin/env bash
# Grid search over CT-stage hyperparameters, scored by downstream IQL val MAE (unscaled).
#
# Sweeps (edit arrays below; defaults match configs/model/vcip.yaml):
#   exp.ct_lr                      - encoder + outcome-predictor LR
#   exp.ct_w_lr                    - WeightNet LR (see vcip.yaml; often >= ct_lr)
#   exp.ct_multi_k_max             - multi-horizon depth {1, 2, 3}
#   exp.ct_multi_eta               - geometric horizon weights
#   exp.ct_es_metric               - early-stop metric: l1 | weighted | mae_uw
#   exp.ct_anchor_weight           - Plan-A anchor blend vs WeightNet MSE
#   exp.ct_dyn_hidden              - width of latent dynamics g(z,a); unused if dyn loss off
#   exp.ct_dyn_consistency_weight  - latent dynamics regularizer (0 = off)
#
# summary.csv schema (20 columns):
#   combo_id, seed,
#   ct_lr, ct_w_lr, ct_multi_k_max, ct_multi_eta, ct_es_metric, ct_anchor_weight,
#   ct_dyn_hidden, ct_dyn_con_w,
#   ct_val_L1w, ct_val_L1anc, ct_val_MAE_uw_norm, ct_val_MAE_uw_uns,
#   ct_val_align, ct_best_epoch,
#   iql_best_val_mae_uns, iql_best_step,
#   mae_uns_eval_split, eval_split
#
# Metric cheat-sheet (all at best-saved epoch, measured on val):
#   ct_val_L1w        : k=1 WeightNet-reweighted MSE (LEGACY, BIASED; kept for diagnosis)
#   ct_val_L1anc      : k=1 UN-weighted MSE (normalized y) -- offset-bias diagnostic
#   ct_val_MAE_uw_norm: k=1 UN-weighted MAE (normalized y) -- primary CT selection metric
#   ct_val_MAE_uw_uns : k=1 UN-weighted MAE (unscaled cancer_volume) -- human-readable
#   ct_val_align      : Sinkhorn / MMD alignment loss (deconfounding activity)
#   iql_best_val_mae_uns: in-loop IQL validation MAE at iql_planner.pt save time
#   mae_uns_eval_split : FINAL downstream IQL MAE on val (TEST_SPLIT=false) or test (true)
#
# Expected relationship (sanity):
#   ct_val_L1anc / ct_val_L1w > 1  => WeightNet concentrates mass (expected).
#   If ratio >> 2, the legacy 'l1' metric is badly biased; prefer 'mae_uw' for ES.
#
# Pipeline per (combo, seed):
#   train_ct.py           -> grid_results/.../ct_ckpts_work/<tag>/ct_best_encoder.pt
#     (read metrics from this .pt via python; never from log)
#     -> archive to grid_results/.../ct_ckpts/<tag>.pt
#   train_iql_planner.py  -> grid_results/.../iql_ckpts_work/<tag>/iql_planner.pt
#     -> archive to grid_results/.../iql_ckpts/<tag>.pt
#   eval_iql_planner.py   -> stdout "MAE unscaled: X" (parsed from log)
#
# Isolation: canonical ct_checkpoints/ and iql_models/ are NEVER touched.
#
# Usage (from repo root):
#   bash scripts/cancer/train/gridsearch_ct.sh [gamma] [gpu] [test_split]
#
# Optional env vars:
#   GRID_SKIP_IQL=1          - skip IQL; score by ct_val_MAE_uw_norm only (fast proxy)
#   GRID_SEEDS="10 20"       - override SEEDS array at runtime
#   GRID_WORKER_ID=0         - this worker index in [0, GRID_NUM_WORKERS-1]
#   GRID_NUM_WORKERS=1       - total parallel workers sharing summary.csv
#
# Multi-GPU parallel (e.g. 2x H800):
#   # Terminal A (GPU 0)
#   GRID_WORKER_ID=0 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_ct.sh 4 0
#   # Terminal B (GPU 1)
#   GRID_WORKER_ID=1 GRID_NUM_WORKERS=2 bash scripts/cancer/train/gridsearch_ct.sh 4 1
#
# Schema migration: if an existing summary.csv has a different header, it is
# auto-archived to summary.csv.<unixtime>.bak and a fresh file is started.
#
# Safe to re-run: existing (combo, seed) rows in summary.csv are skipped.

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate vcip

gamma=${1:-4}
gpu=${2:-0}
TEST_SPLIT=${3:-false}

GRID_WORKER_ID=${GRID_WORKER_ID:-0}
GRID_NUM_WORKERS=${GRID_NUM_WORKERS:-1}
if (( GRID_NUM_WORKERS < 1 )) || (( GRID_WORKER_ID < 0 )) || (( GRID_WORKER_ID >= GRID_NUM_WORKERS )); then
  echo "ERROR: invalid sharding: GRID_WORKER_ID=${GRID_WORKER_ID} GRID_NUM_WORKERS=${GRID_NUM_WORKERS}" >&2
  exit 1
fi
echo "[worker ${GRID_WORKER_ID}/${GRID_NUM_WORKERS}] gamma=${gamma} gpu=${gpu}"

# ---------- grid definition (aligned with configs/model/vcip.yaml) ----------
# ct_lr default in yaml: 1e-4
CT_LR_LIST=("1e-4" "5e-4" "1e-3")
# ct_w_lr default: 1e-2; comments suggest trying 5e-3–1e-2 when ct_k_inner>1
CT_W_LR_LIST=("5e-3" "1e-2" "3e-2")
CT_MULTI_K_MAX_LIST=("1" "2" "3")
CT_MULTI_ETA_LIST=("0.3" "0.5" "0.7")
CT_ANCHOR_WEIGHT_LIST=("0.3" "0.5" "0.7")
CT_DYN_HIDDEN_LIST=("64" "128")
# ct_dyn_consistency_weight default 0.0; yaml suggests {1e-2, 5e-2, 1e-1} when on
CT_DYN_CONSISTENCY_WEIGHT_LIST=("0.0" "0.01" "0.05" "0.1")
CT_ES_METRIC_LIST=("mae_uw")


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

# Authoritative header (keep in sync with run_combo_seed append / error rows).
EXPECTED_HEADER="combo_id,seed,ct_lr,ct_w_lr,ct_multi_k_max,ct_multi_eta,ct_es_metric,ct_anchor_weight,ct_dyn_hidden,ct_dyn_con_w,ct_val_L1w,ct_val_L1anc,ct_val_MAE_uw_norm,ct_val_MAE_uw_uns,ct_val_align,ct_best_epoch,iql_best_val_mae_uns,iql_best_step,mae_uns_eval_split,eval_split"
with_lock() {
  ( flock -x 9; "$@" ) 9> "${summary_lock}"
}

# ---------- header bootstrap + schema-migration ----------
# If existing summary has a different header (e.g. pre-2026-04 layout), archive
# it instead of silently appending into the wrong columns.
if [[ -f "${summary_csv}" ]]; then
  current_header=$(head -n1 "${summary_csv}" || true)
  if [[ "${current_header}" != "${EXPECTED_HEADER}" ]]; then
    ts=$(date +%s)
    bak="${summary_csv}.${ts}.bak"
    if (( GRID_WORKER_ID == 0 )); then
      with_lock bash -c "
        current=\$(head -n1 '${summary_csv}' 2>/dev/null || true)
        if [[ \"\$current\" != \"${EXPECTED_HEADER}\" ]]; then
          mv '${summary_csv}' '${bak}'
          echo '${EXPECTED_HEADER}' > '${summary_csv}'
        fi
      "
      echo "[w${GRID_WORKER_ID}] NOTE: summary.csv schema differs; archived old file to ${bak}"
    else
      for _ in $(seq 1 20); do
        cur=$(head -n1 "${summary_csv}" 2>/dev/null || true)
        [[ "${cur}" == "${EXPECTED_HEADER}" ]] && break
        sleep 0.5
      done
    fi
  fi
fi

if [[ ! -f "${summary_csv}" ]]; then
  if (( GRID_WORKER_ID == 0 )); then
    with_lock bash -c "[[ -f '${summary_csv}' ]] || echo '${EXPECTED_HEADER}' > '${summary_csv}'"
  else
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
shard_owner() {
  local tag="$1"
  local h
  h=$(printf '%s' "${tag}" | cksum | awk '{print $1}')
  echo $(( h % GRID_NUM_WORKERS ))
}

# Extract CT metrics directly from the saved ckpt (.pt). Returns a single
# space-separated line:
#   val_L1w val_L1anc val_MAE_uw_norm val_MAE_uw_uns val_align best_epoch
# All numeric; missing keys -> 'NA' (useful for older ckpts without full metadata).
read_ct_metrics() {
  local ckpt_path="$1"
  python - "${ckpt_path}" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
except Exception as e:
    print("NA NA NA NA NA NA", flush=True)
    sys.exit(0)
def g(k):
    v = c.get(k, None)
    if v is None: return "NA"
    try:
        return f"{float(v):.8g}"
    except Exception:
        return str(v)
print(" ".join([
    g("val_loss_l1"),
    g("val_loss_l1_anchor"),
    g("val_mae_uw_norm"),
    g("val_mae_uw_uns"),
    g("val_loss_align"),
    g("epoch"),
]))
PY
}



run_combo_seed() {
  local lr="$1" w_lr="$2" kmax="$3" eta="$4" esm="$5" aw="$6" dyn_h="$7" dyn_cw="$8" seed="$9"
  local combo_id="lr-${lr}_wlr-${w_lr}_k-${kmax}_eta-${eta}_esm-${esm}_aw-${aw}_dh-${dyn_h}_dcw-${dyn_cw}"
  local tag="${combo_id}__seed_${seed}"
  local log_file="${grid_root}/logs/${tag}.log"
  local ct_work_dir="${grid_root}/ct_ckpts_work/${tag}"
  local ct_ckpt_src="${ct_work_dir}/ct_best_encoder.pt"
  local iql_work_dir="${grid_root}/iql_ckpts_work/${tag}"
  local iql_ckpt_src="${iql_work_dir}/iql_planner.pt"
  local ct_ckpt_grid="${grid_root}/ct_ckpts/${tag}.pt"
  local iql_ckpt_grid="${grid_root}/iql_ckpts/${tag}.pt"

  local owner
  owner=$(shard_owner "${tag}")
  if (( owner != GRID_WORKER_ID )); then
    return 0
  fi

  if with_lock grep -q "^${combo_id},${seed}," "${summary_csv}"; then
    echo "[w${GRID_WORKER_ID}][skip] ${tag} already recorded in ${summary_csv}"
    return 0
  fi

  mkdir -p "${ct_work_dir}" "${iql_work_dir}"

  echo "================================================================"
  echo "=== ${tag} | train_ct ==="
  echo "================================================================"
  CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/train_ct.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    exp.ct_lr="${lr}" \
    exp.ct_w_lr="${w_lr}" \
    exp.ct_multi_k_max="${kmax}" \
    exp.ct_multi_eta="${eta}" \
    exp.ct_es_metric="${esm}" \
    exp.ct_anchor_weight="${aw}" \
    exp.ct_dyn_hidden="${dyn_h}" \
    exp.ct_dyn_consistency_weight="${dyn_cw}" \
    "+exp.ct_ckpt_dir=${ct_work_dir}" \
    2>&1 | tee "${log_file}"

  # Default all CT metrics to NA (in case the training crashed before saving).
  local ct_l1w="NA" ct_l1anc="NA" ct_mae_norm="NA" ct_mae_uns="NA" ct_align="NA" ct_best_ep="NA"

  if [[ ! -f "${ct_ckpt_src}" ]]; then
    echo "ERROR: CT ckpt missing for ${tag} at ${ct_ckpt_src}" >&2
    with_lock bash -c "echo '${combo_id},${seed},${lr},${w_lr},${kmax},${eta},${esm},${aw},${dyn_h},${dyn_cw},NA,NA,NA,NA,NA,NA,NA,NA,NA,${TEST_SPLIT}' >> '${summary_csv}'"
    return 0
  fi
  cp "${ct_ckpt_src}" "${ct_ckpt_grid}"

  # Read all CT metrics from the archived ckpt in one python call.
  if read -r ct_l1w ct_l1anc ct_mae_norm ct_mae_uns ct_align ct_best_ep <<< "$(read_ct_metrics "${ct_ckpt_grid}")"; then
    :
  else
    echo "[w${GRID_WORKER_ID}] warning: failed to read metrics from ${ct_ckpt_grid}"
  fi
  [[ -z "${ct_l1w:-}" ]]     && ct_l1w="NA"
  [[ -z "${ct_l1anc:-}" ]]   && ct_l1anc="NA"
  [[ -z "${ct_mae_norm:-}" ]] && ct_mae_norm="NA"
  [[ -z "${ct_mae_uns:-}" ]] && ct_mae_uns="NA"
  [[ -z "${ct_align:-}" ]]   && ct_align="NA"
  [[ -z "${ct_best_ep:-}" ]] && ct_best_ep="NA"

  local iql_best_mae="NA" iql_best_step="NA" mae_eval="NA"

  if [[ "${GRID_SKIP_IQL:-0}" != "1" ]]; then
    echo "================================================================"
    echo "=== ${tag} | train_iql_planner ==="
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/train_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.iql_inference_ckpt="${ct_ckpt_grid}" \
      "+exp.iql_save_dir=${iql_work_dir}" \
      2>&1 | tee -a "${log_file}"

    if [[ ! -f "${iql_ckpt_src}" ]]; then
      echo "ERROR: IQL ckpt missing for ${tag} at ${iql_ckpt_src}" >&2
      with_lock bash -c "echo '${combo_id},${seed},${lr},${w_lr},${kmax},${eta},${esm},${aw},${dyn_h},${dyn_cw},${ct_l1w},${ct_l1anc},${ct_mae_norm},${ct_mae_uns},${ct_align},${ct_best_ep},NA,NA,NA,${TEST_SPLIT}' >> '${summary_csv}'"
      return 0
    fi
    cp "${iql_ckpt_src}" "${iql_ckpt_grid}"

    iql_best_mae=$(parse_last "${log_file}" 'Saved BEST IQL planner.*\(mae_uns=\K[0-9.eE+-]+')
    iql_best_step=$(parse_last "${log_file}" 'Saved BEST IQL planner.* at step \K[0-9]+')
    [[ -z "${iql_best_mae}" ]] && iql_best_mae="NA"
    [[ -z "${iql_best_step}" ]] && iql_best_step="NA"

    echo "================================================================"
    echo "=== ${tag} | eval_iql_planner (exp.test=${TEST_SPLIT}) ==="
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/eval_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.test="${TEST_SPLIT}" \
      exp.iql_inference_ckpt="${ct_ckpt_grid}" \
      exp.iql_eval_ckpt="${iql_ckpt_grid}" \
      2>&1 | tee -a "${log_file}"

    mae_eval=$(parse_last "${log_file}" 'MAE unscaled: \K[0-9.eE+-]+')
    [[ -z "${mae_eval}" ]] && mae_eval="NA"
  fi

  with_lock bash -c "echo '${combo_id},${seed},${lr},${w_lr},${kmax},${eta},${esm},${aw},${dyn_h},${dyn_cw},${ct_l1w},${ct_l1anc},${ct_mae_norm},${ct_mae_uns},${ct_align},${ct_best_ep},${iql_best_mae},${iql_best_step},${mae_eval},${TEST_SPLIT}' >> '${summary_csv}'"
  echo "[w${GRID_WORKER_ID}][${tag}] L1w=${ct_l1w} L1anc=${ct_l1anc} MAE_uw_uns=${ct_mae_uns} | iql_val=${iql_best_mae} mae_eval=${mae_eval}"
}

for lr in "${CT_LR_LIST[@]}"; do
  for w_lr in "${CT_W_LR_LIST[@]}"; do
    for kmax in "${CT_MULTI_K_MAX_LIST[@]}"; do
      # When kmax==1 the multi-horizon loss collapses to loss_pred1, so eta has
      # no effect and ct_es_metric in {l1, weighted} reduce to mean_l1. Collapse
      # eta+esm to their first entries to avoid training identical models.
      # Note: mae_uw is independent from weighted/l1 even at kmax=1, so we keep
      # the full esm list in that case. anchor_weight always varies.
      if [[ "${kmax}" == "1" ]]; then
        eta_iter=("${CT_MULTI_ETA_LIST[0]}")
      else
        eta_iter=("${CT_MULTI_ETA_LIST[@]}")
      fi
      for eta in "${eta_iter[@]}"; do
        for esm in "${CT_ES_METRIC_LIST[@]}"; do
          for aw in "${CT_ANCHOR_WEIGHT_LIST[@]}"; do
            for dyn_cw in "${CT_DYN_CONSISTENCY_WEIGHT_LIST[@]}"; do
              # Latent dynamics width is irrelevant when consistency loss is off (train_ct skips loss_dyn).
              if python -c "import sys; sys.exit(0 if float('${dyn_cw}') > 0 else 1)"; then
                dyn_h_iter=("${CT_DYN_HIDDEN_LIST[@]}")
              else
                dyn_h_iter=("${CT_DYN_HIDDEN_LIST[0]}")
              fi
              for dyn_h in "${dyn_h_iter[@]}"; do
                for seed in "${SEEDS[@]}"; do
                  run_combo_seed "${lr}" "${w_lr}" "${kmax}" "${eta}" "${esm}" "${aw}" "${dyn_h}" "${dyn_cw}" "${seed}"
                done
              done
            done
          done
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

# Selection policy:
#   skip_iql        -> rank by ct_val_MAE_uw_norm (un-biased CT-only signal)
#   full pipeline   -> primary: mae_uns_eval_split (downstream MAE on eval split)
#                       fallback: iql_best_val_mae_uns (in-loop IQL val MAE)
#                       fallback: ct_val_MAE_uw_norm  (CT predictor proxy)
# Bias diagnostic:
#   ratio_L1anc_over_L1w = ct_val_L1anc / ct_val_L1w
#   >> 1 means legacy 'l1' ES was heavily biased by WeightNet concentration.

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def safe_ratio(num, den):
    if num is None or den is None or den == 0: return None
    return num / den

rows = list(csv.DictReader(open(path)))
groups = defaultdict(list)
for r in rows:
    mae_eval    = to_float(r.get("mae_uns_eval_split"))
    iql_valmae  = to_float(r.get("iql_best_val_mae_uns"))
    ct_l1w      = to_float(r.get("ct_val_L1w"))
    ct_l1anc    = to_float(r.get("ct_val_L1anc"))
    ct_mae_norm = to_float(r.get("ct_val_MAE_uw_norm"))
    ct_mae_uns  = to_float(r.get("ct_val_MAE_uw_uns"))
    ct_align    = to_float(r.get("ct_val_align"))
    groups[r["combo_id"]].append({
        "row":        r,
        "mae_eval":   mae_eval,
        "iql_valmae": iql_valmae,
        "ct_l1w":     ct_l1w,
        "ct_l1anc":   ct_l1anc,
        "ct_mae_norm":ct_mae_norm,
        "ct_mae_uns": ct_mae_uns,
        "ct_align":   ct_align,
        "ratio":      safe_ratio(ct_l1anc, ct_l1w),
    })

def med(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None

scored = []
for combo_id, trials in groups.items():
    sample = trials[0]["row"]
    mae_eval_med    = med([t["mae_eval"]    for t in trials])
    iql_valmae_med  = med([t["iql_valmae"]  for t in trials])
    ct_mae_norm_med = med([t["ct_mae_norm"] for t in trials])
    ct_mae_uns_med  = med([t["ct_mae_uns"]  for t in trials])
    ct_l1w_med      = med([t["ct_l1w"]      for t in trials])
    ct_l1anc_med    = med([t["ct_l1anc"]    for t in trials])
    ct_align_med    = med([t["ct_align"]    for t in trials])
    ratio_med       = med([t["ratio"]       for t in trials])
    n = len(trials)
    if skip_iql:
        sort_key = ct_mae_norm_med if ct_mae_norm_med is not None else float("inf")
    else:
        sort_key = mae_eval_med if mae_eval_med is not None else (
                   iql_valmae_med if iql_valmae_med is not None else (
                   ct_mae_norm_med if ct_mae_norm_med is not None else float("inf")))
    scored.append({
        "combo_id":   combo_id,
        "sort_key":   sort_key,
        "row":        sample,
        "n":          n,
        "mae_eval":   mae_eval_med,
        "iql_valmae": iql_valmae_med,
        "ct_mae_uns": ct_mae_uns_med,
        "ct_l1w":     ct_l1w_med,
        "ct_l1anc":   ct_l1anc_med,
        "ct_align":   ct_align_med,
        "ratio":      ratio_med,
    })

scored.sort(key=lambda x: x["sort_key"])

print(f"\n{len(scored)} unique combos")
print(f"Score column = {'ct_val_MAE_uw_norm (SKIP_IQL)' if skip_iql else 'mae_uns_eval_split (or iql_best_val_mae_uns)'}\n")

hdr = (
    f"{'rank':<4}  {'score':<9}  {'iql_val':<9}  {'MAE_uns':<9}  "
    f"{'L1w':<10}  {'L1anc':<10}  {'ratio':<6}  {'align':<9}  "
    f"{'n':<2}  lr        w_lr      k  eta   esm      aw    dyn_w  dh  combo_id"
)
print(hdr)
def fmt(x, digits=6):
    if x is None: return "NA"
    return f"{x:.{digits}f}"
for i, s in enumerate(scored[:15], 1):
    r = s["row"]
    print(
        f"{i:<4}  {fmt(s['sort_key']):<9}  {fmt(s['iql_valmae']):<9}  "
        f"{fmt(s['ct_mae_uns'],4):<9}  {fmt(s['ct_l1w'],3):<10}  "
        f"{fmt(s['ct_l1anc'],3):<10}  {fmt(s['ratio'],2):<6}  "
        f"{fmt(s['ct_align'],3):<9}  {s['n']:<2}  "
        f"{r['ct_lr']:<9} {r['ct_w_lr']:<9} {r['ct_multi_k_max']}  "
        f"{r['ct_multi_eta']:<4}  {r['ct_es_metric']:<7}  "
        f"{r['ct_anchor_weight']:<5} {r['ct_dyn_con_w']:<5} {r['ct_dyn_hidden']:<3} "
        f"{s['combo_id']}"
    )

if scored:
    best = scored[0]
    print(f"\nBest combo (lowest median score): {best['combo_id']}")
    print(f"  anchor_weight={best['row']['ct_anchor_weight']}  "
          f"ratio L1anc/L1w (val) = {fmt(best['ratio'],2)}  "
          f"(>>1 means legacy 'l1' ES is biased)")
    print(f"  CT ckpt(s): ${grid_root}/ct_ckpts/{best['combo_id']}__seed_*.pt")
    print(f"  IQL ckpt(s): ${grid_root}/iql_ckpts/{best['combo_id']}__seed_*.pt")
PY
