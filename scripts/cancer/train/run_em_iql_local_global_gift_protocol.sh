#!/usr/bin/env bash
# One-stage local/global CT+IQL EM run under the Tumor comparison protocol.
#
# Defaults are read from configs/model/vcip.yaml, configs/dataset/cancer_sim_cont.yaml,
# and fixed code defaults in src/utils/stable_iql_em_defaults.py. This script only
# controls protocol-level choices: seeds, split, dataset size/length overrides, and output paths.
#
# Usage from repo root:
#   GRID_SEEDS="10 101 1010 10101 101010" TEST_SPLIT=true bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4
#   GRID_SEEDS="10 101 1010 10101 101010" TEST_SPLIT=both bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4
#   DATASET_SEED_MODE=exp_seed GRID_SEEDS="10 101 1010 10101 101010" TEST_SPLIT=true bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4

set -euo pipefail

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f /home/liam/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/liam/anaconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda is not available" >&2
  exit 1
fi
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"
TEST_SPLIT="${TEST_SPLIT:-true}"
SEEDS_RAW="${GRID_SEEDS:-20 202 2020 20202 202020}"
read -r -a SEEDS <<< "${SEEDS_RAW}"

DATASET_SEED="${DATASET_SEED:-10101}"
DATASET_SEED_MODE="${DATASET_SEED_MODE:-fixed}"  # fixed | exp_seed
DATASET_TRAIN="${DATASET_TRAIN:-1000}"
DATASET_VAL="${DATASET_VAL:-200}"
DATASET_TEST="${DATASET_TEST:-200}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-60}"
MIN_SEQ_LENGTH="${MIN_SEQ_LENGTH:-${MAX_SEQ_LENGTH}}"  # summary/tag only for Tumor

GPU_WAIT_MEMORY_MB="${GPU_WAIT_MEMORY_MB:-5000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-em_iql_local_global_gift_protocol}"
MLFLOW_URI="${MLFLOW_URI:-}"
USE_MLFLOW="${USE_MLFLOW:-true}"
FORCE="${FORCE:-0}"
CT_USE_WEIGHT_NET="${CT_USE_WEIGHT_NET:-true}"
CT_ALIGN_LOSS="${CT_ALIGN_LOSS:-sinkhorn}"
IQL_BETA="${IQL_BETA:-2.0}"
IQL_ADV_MAX="${IQL_ADV_MAX:-100.0}"
IQL_ACTOR_UPDATE="${IQL_ACTOR_UPDATE:-awr}"
EM_E_EPOCHS="${EM_E_EPOCHS:-}"
EM_E_W_LR="${EM_E_W_LR:-}"
EVAL_TAU_LIST="${EVAL_TAU_LIST:-1 2 3 4 5 6}"

if [[ "${TEST_SPLIT}" != "true" && "${TEST_SPLIT}" != "false" && "${TEST_SPLIT}" != "both" ]]; then
  echo "ERROR: TEST_SPLIT must be true, false, or both, got ${TEST_SPLIT}" >&2
  exit 2
fi

if [[ "${CT_USE_WEIGHT_NET}" != "true" && "${CT_USE_WEIGHT_NET}" != "false" ]]; then
  echo "ERROR: CT_USE_WEIGHT_NET must be true or false, got ${CT_USE_WEIGHT_NET}" >&2
  exit 2
fi
if [[ "${CT_ALIGN_LOSS}" != "sinkhorn" && "${CT_ALIGN_LOSS}" != "mmd" ]]; then
  echo "ERROR: CT_ALIGN_LOSS must be sinkhorn or mmd, got ${CT_ALIGN_LOSS}" >&2
  exit 2
fi
if [[ "${IQL_ACTOR_UPDATE}" != "awr" && "${IQL_ACTOR_UPDATE}" != "bc" && "${IQL_ACTOR_UPDATE}" != "td3bc" && "${IQL_ACTOR_UPDATE}" != "awr_td3bc" ]]; then
  echo "ERROR: IQL_ACTOR_UPDATE must be awr, bc, td3bc, or awr_td3bc, got ${IQL_ACTOR_UPDATE}" >&2
  exit 2
fi
if [[ "${USE_MLFLOW}" != "true" && "${USE_MLFLOW}" != "false" ]]; then
  echo "ERROR: USE_MLFLOW must be true or false, got ${USE_MLFLOW}" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${GRID_ROOT:-${ROOT}/results/tumor/em_iql_local_global_gift_protocol/gamma_${GAMMA}}"
mkdir -p "${GRID_ROOT}/logs" "${GRID_ROOT}/ckpts" "${GRID_ROOT}/done"

SUMMARY="${GRID_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "combo_id,seed,dataset_seed,dataset_seed_mode,split,dataset_train,dataset_val,dataset_test,max_seq_length,min_seq_length,ct_use_weight_net,ct_align_loss,iql_beta,iql_adv_max,iql_actor_update,em_e_epochs,em_e_w_lr,best_outer,best_val_metric,best_val_score,eval_tau,mae_norm,mae_uns,rmse_norm,rmse_norm_x_std,rmse_uns,em_ckpt,train_log,eval_log,finished_at" > "${SUMMARY}"
fi

MLFLOW_URI_ARGS=()
if [[ -n "${MLFLOW_URI}" ]]; then
  MLFLOW_URI_ARGS=("exp.mlflow_uri=${MLFLOW_URI}")
fi
EM_E_ARGS=()
if [[ -n "${EM_E_EPOCHS}" ]]; then
  EM_E_ARGS+=("+exp.em_e_epochs=${EM_E_EPOCHS}")
fi
if [[ -n "${EM_E_W_LR}" ]]; then
  EM_E_ARGS+=("+exp.em_e_w_lr=${EM_E_W_LR}")
fi
EVAL_TAU_CSV="${EVAL_TAU_LIST// /,}"
EVAL_ARGS=("+exp.iql_eval_tau_list=[${EVAL_TAU_CSV}]")

wait_for_gpu() {
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | head -n1 | tr -d ' ')"
    if [[ -z "${used}" || "${used}" -lt "${GPU_WAIT_MEMORY_MB}" ]]; then
      return 0
    fi
    echo "[wait] GPU ${GPU} memory.used=${used}MB >= ${GPU_WAIT_MEMORY_MB}MB; sleeping ${GPU_WAIT_SECONDS}s"
    sleep "${GPU_WAIT_SECONDS}"
  done
}

resolve_dataset_seed() {
  local seed="$1"
  case "${DATASET_SEED_MODE}" in
    fixed) echo "${DATASET_SEED}" ;;
    exp_seed) echo "${seed}" ;;
    *) echo "ERROR: DATASET_SEED_MODE must be fixed or exp_seed, got ${DATASET_SEED_MODE}" >&2; return 1 ;;
  esac
}

hydra_data_overrides() {
  local -n out_ref=$1
  out_ref=()
  [[ "${DATASET_TRAIN}" != "1000" ]] && out_ref+=("dataset.num_patients.train=${DATASET_TRAIN}")
  [[ "${DATASET_VAL}" != "200" ]] && out_ref+=("dataset.num_patients.val=${DATASET_VAL}")
  [[ "${DATASET_TEST}" != "200" ]] && out_ref+=("dataset.num_patients.test=${DATASET_TEST}")
  [[ "${MAX_SEQ_LENGTH}" != "60" ]] && out_ref+=("dataset.max_seq_length=${MAX_SEQ_LENGTH}")
  return 0
}

read_em_metrics() {
  local ckpt_path="$1"
  python - "${ckpt_path}" <<'PY2'
import sys, torch
p = sys.argv[1]
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    print("NA NA NA")
    raise SystemExit(0)
extra = c.get("extra") or {}
outer = c.get("outer_iter", extra.get("outer_iter", "NA"))
metric = c.get("val_metric", extra.get("val_metric", "NA"))
val = c.get("val_score", extra.get("val_score", extra.get("best_val_mae_uns", extra.get("val_mae_uns", "NA"))))
def f(v):
    try:
        return f"{float(v):.8g}"
    except Exception:
        return str(v)
print(f(outer), str(metric), f(val))
PY2
}

append_eval_rows() {
  local combo_id="$1"
  local seed="$2"
  local dataset_seed="$3"
  local split="$4"
  local best_outer="$5"
  local best_val_metric="$6"
  local best_val_score="$7"
  local em_ckpt="$8"
  local train_log="$9"
  local eval_log="${10}"
  local finished_at="${11}"

  python - "${SUMMARY}" "${combo_id}" "${seed}" "${dataset_seed}" "${DATASET_SEED_MODE}" "${split}"     "${DATASET_TRAIN}" "${DATASET_VAL}" "${DATASET_TEST}" "${MAX_SEQ_LENGTH}" "${MIN_SEQ_LENGTH}"     "${CT_USE_WEIGHT_NET}" "${CT_ALIGN_LOSS}" "${IQL_BETA}" "${IQL_ADV_MAX}" "${IQL_ACTOR_UPDATE}"     "${EM_E_EPOCHS:-code_default}" "${EM_E_W_LR:-code_default}" "${best_outer}" "${best_val_metric}" "${best_val_score}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}" <<'PY2'
import csv
import os
import re
import sys

(
    summary_path, combo_id, seed, dataset_seed, dataset_seed_mode, split,
    dataset_train, dataset_val, dataset_test, max_seq_length, min_seq_length,
    ct_use_weight_net, ct_align_loss, iql_beta, iql_adv_max, iql_actor_update,
    em_e_epochs, em_e_w_lr,
    best_outer, best_val_metric, best_val_score, em_ckpt, train_log, eval_log, finished_at,
) = sys.argv[1:]

def empty_metrics():
    return {
        "eval_tau": "NA",
        "mae_norm": "NA",
        "mae_uns": "NA",
        "rmse_norm": "NA",
        "rmse_norm_x_std": "NA",
        "rmse_uns": "NA",
    }

rows = []
current = None
with open(eval_log, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = re.search(r"IQL eval (?:autoregressive action rollout|unified closed-loop rollout): .*?\(tau=(\d+),", line)
        if m:
            if current is not None:
                rows.append(current)
            current = empty_metrics()
            current["eval_tau"] = m.group(1)
            continue
        if current is None:
            continue
        m = re.search(r"Global RMSE on stacked batches \(normalized space\): ([0-9.eE+-]+)", line)
        if m:
            current["rmse_norm"] = m.group(1)
            continue
        m = re.search(r"Global RMSE .*\): ([0-9.eE+-]+)", line)
        if m:
            current["rmse_norm_x_std"] = m.group(1)
            continue
        m = re.search(r"MAE normalized: ([0-9.eE+-]+) \| MAE (?:unscaled|scaled): ([0-9.eE+-]+) \| RMSE (?:unscaled|scaled): ([0-9.eE+-]+)", line)
        if m:
            current["mae_norm"] = m.group(1)
            current["mae_uns"] = m.group(2)
            current["rmse_uns"] = m.group(3)
            continue
if current is not None:
    rows.append(current)
if not rows:
    rows = [empty_metrics()]

fieldnames = [
    "combo_id", "seed", "dataset_seed", "dataset_seed_mode", "split",
    "dataset_train", "dataset_val", "dataset_test", "max_seq_length", "min_seq_length",
    "ct_use_weight_net", "ct_align_loss", "iql_beta", "iql_adv_max", "iql_actor_update",
    "em_e_epochs", "em_e_w_lr",
    "best_outer", "best_val_metric", "best_val_score", "eval_tau",
    "mae_norm", "mae_uns", "rmse_norm", "rmse_norm_x_std", "rmse_uns",
    "em_ckpt", "train_log", "eval_log", "finished_at",
]
with open(summary_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for metrics in rows:
        row = {
            "combo_id": combo_id,
            "seed": seed,
            "dataset_seed": dataset_seed,
            "dataset_seed_mode": dataset_seed_mode,
            "split": split,
            "dataset_train": dataset_train,
            "dataset_val": dataset_val,
            "dataset_test": dataset_test,
            "max_seq_length": max_seq_length,
            "min_seq_length": min_seq_length,
            "ct_use_weight_net": ct_use_weight_net,
            "ct_align_loss": ct_align_loss,
            "iql_beta": iql_beta,
            "iql_adv_max": iql_adv_max,
            "iql_actor_update": iql_actor_update,
            "em_e_epochs": em_e_epochs,
            "em_e_w_lr": em_e_w_lr,
            "best_outer": best_outer,
            "best_val_metric": best_val_metric,
            "best_val_score": best_val_score,
            "em_ckpt": em_ckpt,
            "train_log": train_log,
            "eval_log": eval_log,
            "finished_at": finished_at,
        }
        row.update(metrics)
        writer.writerow(row)
print(f"appended {len(rows)} eval row(s) from {os.path.basename(eval_log)}")
PY2
}

run_one() {
  local seed="$1"
  local dataset_seed
  dataset_seed="$(resolve_dataset_seed "${seed}")"
  local beta_id="${IQL_BETA//./p}"
  local adv_id="${IQL_ADV_MAX//./p}"
  local e_id="${EM_E_EPOCHS:-default}"
  local wlr_id="${EM_E_W_LR:-default}"
  wlr_id="${wlr_id//./p}"
  local combo_id="seq${MAX_SEQ_LENGTH}_localglobal_dseed${dataset_seed}_wn${CT_USE_WEIGHT_NET}_align${CT_ALIGN_LOSS}_e${e_id}_wlr${wlr_id}_beta${beta_id}_adv${adv_id}_actor${IQL_ACTOR_UPDATE}"
  local tag="${combo_id}_seed${seed}"
  local em_dir="${GRID_ROOT}/ckpts/${tag}"
  local em_ckpt="${em_dir}/ct_iql_em_best.pt"
  local log_dir="${GRID_ROOT}/logs/${tag}"
  local train_log="${log_dir}/train.log"
  local done_flag="${GRID_ROOT}/done/${tag}.done"
  local data_overrides=()
  hydra_data_overrides data_overrides

  if [[ "${FORCE}" != "1" ]]; then
    if [[ -f "${done_flag}" ]]; then
      echo "[skip] ${tag} (done marker exists)"
      return 0
    fi
    if grep -q "^${combo_id},${seed}," "${SUMMARY}" 2>/dev/null; then
      echo "[skip] ${tag} already recorded in ${SUMMARY}"
      return 0
    fi
  fi

  mkdir -p "${em_dir}" "${log_dir}"

  echo "========== ${tag} =========="
  echo "  data train/val/test=${DATASET_TRAIN}/${DATASET_VAL}/${DATASET_TEST}, max_seq_length=${MAX_SEQ_LENGTH}"
  echo "  seed=${seed} dataset_seed=${dataset_seed} dataset_seed_mode=${DATASET_SEED_MODE} gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/train_ct_iql_em.py \
    +dataset=cancer_sim_cont +model=vcip \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" "dataset.seed=${dataset_seed}" \
    "${data_overrides[@]}" \
    exp.load_data=false \
    "exp.ct_use_weight_net=${CT_USE_WEIGHT_NET}" \
    "exp.ct_align_loss=${CT_ALIGN_LOSS}" \
    "+exp.iql_beta=${IQL_BETA}" \
    "+exp.iql_adv_max=${IQL_ADV_MAX}" \
    "+exp.iql_actor_update=${IQL_ACTOR_UPDATE}" \
    "${EM_E_ARGS[@]}" \
    "+exp.em_ckpt_dir=${em_dir}" \
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.use_mlflow=${USE_MLFLOW}" \
    "exp.mlflow_combo_id=${combo_id}" \
    "${MLFLOW_URI_ARGS[@]}" \
    2>&1 | tee "${train_log}"

  if [[ ! -f "${em_ckpt}" ]]; then
    echo "ERROR: EM checkpoint missing after train: ${em_ckpt}" >&2
    return 1
  fi

  local best_outer best_val_metric best_val_score finished_at split eval_flag eval_log
  local eval_flags=()
  read -r best_outer best_val_metric best_val_score <<< "$(read_em_metrics "${em_ckpt}")"
  if [[ "${TEST_SPLIT}" == "both" ]]; then
    eval_flags=(false true)
  else
    eval_flags=("${TEST_SPLIT}")
  fi

  for eval_flag in "${eval_flags[@]}"; do
    if [[ "${eval_flag}" == "true" ]]; then
      split="test"
    else
      split="val"
    fi
    if [[ "${TEST_SPLIT}" == "both" ]]; then
      eval_log="${log_dir}/eval_${split}.log"
    else
      eval_log="${log_dir}/eval.log"
    fi

    wait_for_gpu
    CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/eval_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip \
      exp.seed="${seed}" dataset.coeff="${GAMMA}" "dataset.seed=${dataset_seed}" \
      exp.test="${eval_flag}" \
      "${data_overrides[@]}" \
      exp.load_data=false \
      "exp.ct_use_weight_net=${CT_USE_WEIGHT_NET}" \
      "exp.ct_align_loss=${CT_ALIGN_LOSS}" \
      "+exp.iql_beta=${IQL_BETA}" \
      "+exp.iql_adv_max=${IQL_ADV_MAX}" \
      "+exp.iql_actor_update=${IQL_ACTOR_UPDATE}" \
      "${EM_E_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "+exp.em_eval_ckpt=${em_ckpt}" \
      "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
      "exp.use_mlflow=${USE_MLFLOW}" \
      "exp.mlflow_combo_id=${combo_id}" \
      "${MLFLOW_URI_ARGS[@]}" \
      2>&1 | tee "${eval_log}"

    finished_at="$(date -Iseconds)"
    append_eval_rows "${combo_id}" "${seed}" "${dataset_seed}" "${split}" "${best_outer}" "${best_val_metric}" "${best_val_score}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}"
  done

  {
    echo "finished_at=${finished_at}"
    echo "combo_id=${combo_id}"
    echo "seed=${seed}"
    echo "dataset_seed=${dataset_seed}"
    echo "dataset_seed_mode=${DATASET_SEED_MODE}"
    echo "split=${TEST_SPLIT}"
    echo "dataset_train=${DATASET_TRAIN}"
    echo "dataset_val=${DATASET_VAL}"
    echo "dataset_test=${DATASET_TEST}"
    echo "max_seq_length=${MAX_SEQ_LENGTH}"
    echo "ct_use_weight_net=${CT_USE_WEIGHT_NET}"
    echo "ct_align_loss=${CT_ALIGN_LOSS}"
    echo "iql_beta=${IQL_BETA}"
    echo "iql_adv_max=${IQL_ADV_MAX}"
    echo "iql_actor_update=${IQL_ACTOR_UPDATE}"
    echo "em_e_epochs=${EM_E_EPOCHS:-code_default}"
    echo "em_e_w_lr=${EM_E_W_LR:-code_default}"
    echo "eval_tau_list=${EVAL_TAU_LIST}"
    echo "use_mlflow=${USE_MLFLOW}"
    echo "em_ckpt=${em_ckpt}"
    echo "train_log=${train_log}"
    echo "eval_logs=${log_dir}/eval*.log"
  } > "${done_flag}"
  echo "[done] ${tag}"
}

echo "[tumor-protocol] one-stage local/global CT+IQL EM"
echo "[tumor-protocol] gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
echo "[tumor-protocol] seeds=(${SEEDS[*]}) dataset_seed_mode=${DATASET_SEED_MODE} dataset_seed=${DATASET_SEED}"
echo "[tumor-protocol] data train/val/test=${DATASET_TRAIN}/${DATASET_VAL}/${DATASET_TEST} max_seq_length=${MAX_SEQ_LENGTH}"
echo "[tumor-protocol] weight_net=${CT_USE_WEIGHT_NET} align_loss=${CT_ALIGN_LOSS}"
echo "[tumor-protocol] iql_beta=${IQL_BETA} iql_adv_max=${IQL_ADV_MAX} actor_update=${IQL_ACTOR_UPDATE}"
echo "[tumor-protocol] em_e_epochs=${EM_E_EPOCHS:-code_default} em_e_w_lr=${EM_E_W_LR:-code_default}"
echo "[tumor-protocol] eval_tau_list=${EVAL_TAU_LIST}"
echo "[tumor-protocol] use_mlflow=${USE_MLFLOW}"
echo "[tumor-protocol] grid_root=${GRID_ROOT}"

for seed in "${SEEDS[@]}"; do
  run_one "${seed}"
done

python - "${SUMMARY}" <<'PY2'
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    if row.get("rmse_uns") in ("", "NA", None):
        continue
    groups[row["eval_tau"]].append(float(row["rmse_uns"]))

print()
print(f"Summary: {path}")
for tau, vals in sorted(groups.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999):
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(
        f"tau={tau}: n={len(vals)} rmse_mean={statistics.mean(vals):.6f} "
        f"rmse_std={std:.6f} vals={[round(v, 6) for v in vals]}"
    )
PY2
