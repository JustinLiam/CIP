#!/usr/bin/env bash
# One-stage EM+IQL local-global run under the GIFT Tumor data protocol.
#
# Data protocol aligned to external_repos/GIFT/configs/dataset/tumor.yaml:
#   train/val/test = 1000/200/200
#   max_seq_length = 40
#   min_seq_length = 40 (accepted as a config field for parity; the local simulator
#   uses fixed seq_length through max_seq_length)
#
# Metric protocol:
#   existing MAE/RMSE metrics are preserved, and eval_iql_planner logs:
#     gift_rmse         = unscaled cancer-volume RMSE
#     gift_rmse_percent = gift_rmse / 1150 * 100
#
# Usage from repo root:
#   MAX_TAU=6 bash scripts/cancer/train/run_em_iql_local_global_gift_protocol_maxtau.sh [GPU] [GAMMA]
#   MAX_TAU=12 bash scripts/cancer/train/run_em_iql_local_global_gift_protocol_maxtau.sh [GPU] [GAMMA]

set -euo pipefail

if [[ -f /home/liam/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/liam/anaconda3/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)"
fi
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"

TEST_SPLIT="${TEST_SPLIT:-false}"
SEEDS_RAW="${GRID_SEEDS:-20 2020 202020}"
read -r -a SEEDS <<< "${SEEDS_RAW}"

IQL_BETA="${IQL_BETA:-2.0}"
IQL_TAU="${IQL_TAU:-0.7}"
IQL_ACTOR_LR="${IQL_ACTOR_LR:-3e-4}"
IQL_QF_LR="${IQL_QF_LR:-3e-4}"
IQL_VF_LR="${IQL_VF_LR:-${IQL_ACTOR_LR}}"
IQL_MAX_GRAD="${IQL_MAX_GRAD:-5.0}"
IQL_REWARD_TYPE="${IQL_REWARD_TYPE:-negative_outcome}"
IQL_REWARD_HUBER_DELTA="${IQL_REWARD_HUBER_DELTA:-1.0}"
IQL_ADV_MAX="${IQL_ADV_MAX:-100}"
EM_M_STEPS="${EM_M_STEPS:-1000}"
EM_HER_SAMPLES="${EM_HER_SAMPLES:-1}"
EM_HER_REFRESH="${EM_HER_REFRESH:-0}"
EM_SAVE_EVAL_CKPTS="${EM_SAVE_EVAL_CKPTS:-false}"
EM_ENCODER_DIAGNOSTICS="${EM_ENCODER_DIAGNOSTICS:-false}"
EM_ENCODER_DIAGNOSTICS_EVERY="${EM_ENCODER_DIAGNOSTICS_EVERY:-50}"
IQL_GOAL_ADAPTER="${IQL_GOAL_ADAPTER:-false}"
IQL_GOAL_ADAPTER_HIDDEN="${IQL_GOAL_ADAPTER_HIDDEN:-64}"
IQL_GOAL_ADAPTER_INIT_SCALE="${IQL_GOAL_ADAPTER_INIT_SCALE:-1e-3}"
EVAL_TAU_LIST="${EVAL_TAU_LIST:-[1,2,3,4,5,6]}"
VAL_METRIC="${VAL_METRIC:-mae_uns}"
MAX_TAU="${MAX_TAU:-12}"
GPU_WAIT_MEMORY_MB="${GPU_WAIT_MEMORY_MB:-1000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-em_iql_local_global_gift_protocol_maxtau}"
MLFLOW_URI="${MLFLOW_URI:-}"
FORCE="${FORCE:-0}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${GRID_ROOT:-${ROOT}/grid_results/em_iql_local_global_gift_protocol_maxtau/gamma_${GAMMA}/max_tau_${MAX_TAU}}"
mkdir -p "${GRID_ROOT}/logs" "${GRID_ROOT}/ckpts" "${GRID_ROOT}/done"

SUMMARY="${GRID_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "combo_id,seed,split,local_conv_layers,dataset_train,dataset_val,dataset_test,max_seq_length,min_seq_length,iql_tau,iql_actor_lr,iql_qf_lr,iql_vf_lr,iql_beta,iql_max_grad_norm,em_m_steps_per_outer,em_her_samples_per_transition,em_her_refresh_every,best_outer,best_val_metric,best_val_score,eval_tau,eval_mae_norm,eval_mae_uns,eval_rmse_norm,eval_rmse_norm_x_std,eval_rmse_uns,gift_rmse,gift_rmse_percent,gift_mae_percent,em_ckpt,train_log,eval_log,finished_at" > "${SUMMARY}"
fi

MLFLOW_URI_ARGS=()
if [[ -n "${MLFLOW_URI}" ]]; then
  MLFLOW_URI_ARGS=("exp.mlflow_uri=${MLFLOW_URI}")
fi

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

read_em_metrics() {
  local ckpt_path="$1"
  python - "${ckpt_path}" <<'PY'
import sys, torch
p = sys.argv[1]
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    print("NA NA")
    raise SystemExit(0)
extra = c.get("extra") or {}
outer = c.get("outer_iter", extra.get("outer_iter", "NA"))
metric = c.get("val_metric", extra.get("val_metric", "NA"))
val = c.get(
    "val_score",
    extra.get("val_score", extra.get("best_val_mae_uns", extra.get("val_mae_uns", "NA"))),
)
def f(v):
    try:
        return f"{float(v):.8g}"
    except Exception:
        return str(v)
print(f(outer), str(metric), f(val))
PY
}

append_eval_rows() {
  local combo_id="$1"
  local seed="$2"
  local split="$3"
  local best_outer="$4"
  local best_val_metric="$5"
  local best_val_score="$6"
  local em_ckpt="$7"
  local train_log="$8"
  local eval_log="$9"
  local finished_at="${10}"

  python - "${SUMMARY}" "${combo_id}" "${seed}" "${split}" "${IQL_TAU}" "${IQL_ACTOR_LR}" "${IQL_QF_LR}" "${IQL_VF_LR}" \
    "${IQL_BETA}" "${IQL_MAX_GRAD}" "${EM_M_STEPS}" "${EM_HER_SAMPLES}" "${EM_HER_REFRESH}" \
    "${best_outer}" "${best_val_metric}" "${best_val_score}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}" <<'PY'
import csv
import os
import re
import sys

(
    summary_path, combo_id, seed, split, iql_tau, actor_lr, qf_lr, vf_lr,
    iql_beta, iql_max_grad, em_m_steps, em_her_samples, em_her_refresh,
    best_outer, best_val_metric, best_val_score, em_ckpt, train_log, eval_log, finished_at
) = sys.argv[1:]

def empty_metrics():
    return {
        "eval_tau": "NA",
        "eval_mae_norm": "NA",
        "eval_mae_uns": "NA",
        "eval_rmse_norm": "NA",
        "eval_rmse_norm_x_std": "NA",
        "eval_rmse_uns": "NA",
        "gift_rmse": "NA",
        "gift_rmse_percent": "NA",
        "gift_mae_percent": "NA",
    }

rows = []
current = None
with open(eval_log, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = re.search(r"IQL eval autoregressive action rollout: .*?\(tau=(\d+),", line)
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
            current["eval_rmse_norm"] = m.group(1)
            continue
        m = re.search(r"Global RMSE .*VCIP-style\): ([0-9.eE+-]+)", line)
        if m:
            current["eval_rmse_norm_x_std"] = m.group(1)
            continue
        m = re.search(
            r"MAE normalized: ([0-9.eE+-]+) \| MAE unscaled: ([0-9.eE+-]+) \| RMSE unscaled: ([0-9.eE+-]+)",
            line,
        )
        if m:
            current["eval_mae_norm"] = m.group(1)
            current["eval_mae_uns"] = m.group(2)
            current["eval_rmse_uns"] = m.group(3)
            continue
        m = re.search(
            r"GIFT-style tumor RMSE unscaled: ([0-9.eE+-]+) \| GIFT-style tumor RMSE \(% of [0-9.eE+-]+\): ([0-9.eE+-]+) \| GIFT-style tumor MAE .*: ([0-9.eE+-]+)",
            line,
        )
        if m:
            current["gift_rmse"] = m.group(1)
            current["gift_rmse_percent"] = m.group(2)
            current["gift_mae_percent"] = m.group(3)
            continue
if current is not None:
    rows.append(current)
if not rows:
    rows = [empty_metrics()]

fieldnames = [
    "combo_id", "seed", "split", "local_conv_layers", "dataset_train", "dataset_val", "dataset_test",
    "max_seq_length", "min_seq_length", "iql_tau", "iql_actor_lr", "iql_qf_lr", "iql_vf_lr",
    "iql_beta", "iql_max_grad_norm", "em_m_steps_per_outer", "em_her_samples_per_transition",
    "em_her_refresh_every", "best_outer", "best_val_metric", "best_val_score", "eval_tau", "eval_mae_norm",
    "eval_mae_uns", "eval_rmse_norm", "eval_rmse_norm_x_std", "eval_rmse_uns", "gift_rmse",
    "gift_rmse_percent", "gift_mae_percent", "em_ckpt", "train_log", "eval_log", "finished_at",
]
with open(summary_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for metrics in rows:
        row = {
            "combo_id": combo_id,
            "seed": seed,
            "split": split,
            "local_conv_layers": "1",
            "dataset_train": "1000",
            "dataset_val": "200",
            "dataset_test": "200",
            "max_seq_length": "40",
            "min_seq_length": "40",
            "iql_tau": iql_tau,
            "iql_actor_lr": actor_lr,
            "iql_qf_lr": qf_lr,
            "iql_vf_lr": vf_lr,
            "iql_beta": iql_beta,
            "iql_max_grad_norm": iql_max_grad,
            "em_m_steps_per_outer": em_m_steps,
            "em_her_samples_per_transition": em_her_samples,
            "em_her_refresh_every": em_her_refresh,
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
PY
}

run_one() {
  local seed="$1"
  local local_layers=1
  local reward_id="${IQL_REWARD_TYPE//[^A-Za-z0-9]/}"
  local beta_id="${IQL_BETA//./p}"
  beta_id="${beta_id//-/m}"
  beta_id="${beta_id//+/p}"
  beta_id="${beta_id//[^A-Za-z0-9pm]/}"
  local adv_id="${IQL_ADV_MAX//./p}"
  adv_id="${adv_id//-/m}"
  adv_id="${adv_id//+/p}"
  adv_id="${adv_id//[^A-Za-z0-9pm]/}"
  if [[ "${IQL_REWARD_TYPE}" == "negative_outcome_huber" || "${IQL_REWARD_TYPE}" == "huber" || "${IQL_REWARD_TYPE}" == "smooth_l1" ]]; then
    reward_id="${reward_id}_d${IQL_REWARD_HUBER_DELTA//[^A-Za-z0-9]/}"
  fi
  local combo_id="gift40_lg_tau07_lr3e4_grad5_m1k_b${beta_id}_adv${adv_id}_val${VAL_METRIC}_${reward_id}_maxtau${MAX_TAU}"
  if [[ "${EM_ENCODER_DIAGNOSTICS}" == "true" ]]; then
    combo_id="${combo_id}_encdiag${EM_ENCODER_DIAGNOSTICS_EVERY}"
  fi
  if [[ "${IQL_GOAL_ADAPTER}" == "true" ]]; then
    combo_id="${combo_id}_goaladapter"
  fi
  local tag="${combo_id}_seed${seed}"
  local em_dir="${GRID_ROOT}/ckpts/${tag}"
  local em_ckpt="${em_dir}/ct_iql_em_best.pt"
  local log_dir="${GRID_ROOT}/logs/${tag}"
  local train_log="${log_dir}/train.log"
  local eval_log="${log_dir}/eval.log"
  local done_flag="${GRID_ROOT}/done/${tag}.done"

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
  echo "  GIFT data protocol: train/val/test=1000/200/200, max_seq_length=40"
  echo "  seed=${seed} gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
  echo "  local_conv_layers=${local_layers}"
  echo "  iql_tau=${IQL_TAU} actor_lr=${IQL_ACTOR_LR} qf_lr=${IQL_QF_LR} vf_lr=${IQL_VF_LR}"
  echo "  val_metric=${VAL_METRIC}"
  echo "  max_tau=${MAX_TAU}"
  echo "  goal_adapter=${IQL_GOAL_ADAPTER} hidden=${IQL_GOAL_ADAPTER_HIDDEN} init_scale=${IQL_GOAL_ADAPTER_INIT_SCALE}"
  echo "  encoder_diagnostics=${EM_ENCODER_DIAGNOSTICS} every=${EM_ENCODER_DIAGNOSTICS_EVERY}"

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/train_ct_iql_em.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" \
    "dataset.num_patients.train=1000" \
    "dataset.num_patients.val=200" \
    "dataset.num_patients.test=200" \
    "dataset.max_seq_length=40" \
    "+dataset.min_seq_length=40" \
    "exp.load_data=false" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_her_refresh_every=${EM_HER_REFRESH}" \
    "exp.em_her_samples_per_transition=${EM_HER_SAMPLES}" \
    "exp.em_save_every_eval_checkpoint=${EM_SAVE_EVAL_CKPTS}" \
    "exp.ct_num_workers=0" \
    "exp.iql_reward_type=${IQL_REWARD_TYPE}" \
    "exp.iql_reward_huber_delta=${IQL_REWARD_HUBER_DELTA}" \
    "exp.iql_beta=${IQL_BETA}" \
    "+exp.iql_adv_max=${IQL_ADV_MAX}" \
    "exp.iql_tau=${IQL_TAU}" \
    "exp.iql_actor_lr=${IQL_ACTOR_LR}" \
    "exp.iql_qf_lr=${IQL_QF_LR}" \
    "exp.iql_vf_lr=${IQL_VF_LR}" \
    "exp.iql_max_grad_norm=${IQL_MAX_GRAD}" \
    "exp.em_m_steps_per_outer=${EM_M_STEPS}" \
    "exp.max_tau=${MAX_TAU}" \
    "+exp.em_encoder_diagnostics=${EM_ENCODER_DIAGNOSTICS}" \
    "+exp.em_encoder_diagnostics_every=${EM_ENCODER_DIAGNOSTICS_EVERY}" \
    "+exp.iql_goal_adapter_enabled=${IQL_GOAL_ADAPTER}" \
    "+exp.iql_goal_adapter_hidden_dim=${IQL_GOAL_ADAPTER_HIDDEN}" \
    "+exp.iql_goal_adapter_init_scale=${IQL_GOAL_ADAPTER_INIT_SCALE}" \
    "exp.em_val_metric=${VAL_METRIC}" \
    "exp.iql_val_metric=${VAL_METRIC}" \
    "+exp.em_ckpt_dir=${em_dir}" \
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.mlflow_combo_id=${combo_id}" \
    "${MLFLOW_URI_ARGS[@]}" \
    2>&1 | tee "${train_log}"

  if [[ ! -f "${em_ckpt}" ]]; then
    echo "ERROR: EM checkpoint missing after train: ${em_ckpt}" >&2
    return 1
  fi

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/eval_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" \
    exp.test="${TEST_SPLIT}" \
    "dataset.num_patients.train=1000" \
    "dataset.num_patients.val=200" \
    "dataset.num_patients.test=200" \
    "dataset.max_seq_length=40" \
    "+dataset.min_seq_length=40" \
    "exp.load_data=false" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_eval_ckpt=${em_ckpt}" \
    "exp.max_tau=${MAX_TAU}" \
    "exp.iql_eval_tau_list=${EVAL_TAU_LIST}" \
    "exp.mlflow_experiment=${MLFLOW_EXPERIMENT}" \
    "exp.mlflow_combo_id=${combo_id}" \
    "${MLFLOW_URI_ARGS[@]}" \
    2>&1 | tee "${eval_log}"

  local best_outer best_val_metric best_val_score finished_at split
  read -r best_outer best_val_metric best_val_score <<< "$(read_em_metrics "${em_ckpt}")"
  finished_at="$(date -Iseconds)"
  if [[ "${TEST_SPLIT}" == "true" ]]; then
    split="test"
  else
    split="val"
  fi
  append_eval_rows "${combo_id}" "${seed}" "${split}" "${best_outer}" "${best_val_metric}" "${best_val_score}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}"

  {
    echo "finished_at=${finished_at}"
    echo "combo_id=${combo_id}"
    echo "seed=${seed}"
    echo "split=${split}"
    echo "gift_protocol=true"
    echo "dataset_train=1000"
    echo "dataset_val=200"
    echo "dataset_test=200"
    echo "max_seq_length=40"
    echo "min_seq_length=40"
    echo "iql_beta=${IQL_BETA}"
    echo "iql_adv_max=${IQL_ADV_MAX}"
    echo "em_ckpt=${em_ckpt}"
    echo "train_log=${train_log}"
    echo "eval_log=${eval_log}"
  } > "${done_flag}"
  echo "[done] ${tag}"
}

echo "[gift-protocol] one-stage EM+IQL local-global"
echo "[gift-protocol] gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
echo "[gift-protocol] seeds=(${SEEDS[*]})"
echo "[gift-protocol] data train/val/test=1000/200/200 max_seq_length=40"
echo "[gift-protocol] iql_tau=${IQL_TAU} actor_lr=${IQL_ACTOR_LR} qf_lr=${IQL_QF_LR} vf_lr=${IQL_VF_LR}"
echo "[gift-protocol] beta=${IQL_BETA} adv_max=${IQL_ADV_MAX} grad=${IQL_MAX_GRAD} m_steps=${EM_M_STEPS} eval_tau_list=${EVAL_TAU_LIST}"
echo "[gift-protocol] max_tau=${MAX_TAU} val_metric=${VAL_METRIC}"
echo "[gift-protocol] grid_root=${GRID_ROOT}"

for seed in "${SEEDS[@]}"; do
  run_one "${seed}"
done

python - "${SUMMARY}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    if row.get("gift_rmse_percent") in ("", "NA", None):
        continue
    groups[row["eval_tau"]].append(float(row["gift_rmse_percent"]))

print(f"\nSummary: {path}")
for tau, vals in sorted(groups.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999):
    print(
        f"tau={tau}: n={len(vals)} "
        f"gift_rmse_percent_mean={statistics.mean(vals):.6f} "
        f"vals={[round(v, 6) for v in vals]}"
    )
PY
