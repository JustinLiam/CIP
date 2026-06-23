#!/usr/bin/env bash
# One-stage EM+IQL local-global run under the GIFT Tumor data protocol.
#
# Default data protocol for the current tumor comparison:
#   train/val/test = 1000/200/200
#   max_seq_length = 60
#   min_seq_length = 60
# Override MAX_SEQ_LENGTH/MIN_SEQ_LENGTH to run alternate-history variants.
#
# Metric protocol:
#   existing MAE/RMSE metrics are preserved, and eval_iql_planner logs:
#     gift_rmse         = unscaled cancer-volume RMSE
#     gift_rmse_percent = gift_rmse / 1150 * 100
#
# Usage from repo root:
#   bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh [GPU] [GAMMA]

set -euo pipefail

if [[ -f /home/liam/anaconda3/etc/profile.d/conda.sh ]]; then
  source /home/liam/anaconda3/etc/profile.d/conda.sh
else
  eval "$(conda shell.bash hook)"
fi
conda activate vcip

GPU="${1:-0}"
GAMMA="${2:-4}"

TEST_SPLIT="${TEST_SPLIT:-true}"
SEEDS_RAW="${GRID_SEEDS:-20 202 2020 20202 202020}"
read -r -a SEEDS <<< "${SEEDS_RAW}"
DATASET_TRAIN="${DATASET_TRAIN:-1000}"
DATASET_VAL="${DATASET_VAL:-200}"
DATASET_TEST="${DATASET_TEST:-200}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-60}"
MIN_SEQ_LENGTH="${MIN_SEQ_LENGTH:-${MAX_SEQ_LENGTH}}"

IQL_BETA="${IQL_BETA:-2.0}"
IQL_TAU="${IQL_TAU:-0.7}"
IQL_ACTOR_LR="${IQL_ACTOR_LR:-3e-4}"
IQL_QF_LR="${IQL_QF_LR:-3e-4}"
IQL_VF_LR="${IQL_VF_LR:-${IQL_ACTOR_LR}}"
IQL_MAX_GRAD="${IQL_MAX_GRAD:-5.0}"
IQL_REWARD_TYPE="${IQL_REWARD_TYPE:-negative_outcome}"
IQL_REWARD_HUBER_DELTA="${IQL_REWARD_HUBER_DELTA:-1.0}"
IQL_ADV_MAX="${IQL_ADV_MAX:-100}"
IQL_WEIGHT_MAX="${IQL_WEIGHT_MAX:-1.0}"
IQL_ACTOR_UPDATE="${IQL_ACTOR_UPDATE:-awr}"
IQL_ACTOR_BC_LOSS="${IQL_ACTOR_BC_LOSS:-expectile}"
IQL_ACTOR_BC_EXPECTILE="${IQL_ACTOR_BC_EXPECTILE:-0.8}"
IQL_TD3BC_Q_ALPHA="${IQL_TD3BC_Q_ALPHA:-0.0}"
IQL_TD3BC_BC_ALPHA="${IQL_TD3BC_BC_ALPHA:-1.0}"
IQL_CQL_ALPHA="${IQL_CQL_ALPHA:-0.0}"
IQL_CQL_N_ACTIONS="${IQL_CQL_N_ACTIONS:-10}"
EM_OUTER_ITERS="${EM_OUTER_ITERS:-20}"
EM_M_STEPS="${EM_M_STEPS:-1000}"
EM_ENCODER_LR="${EM_ENCODER_LR:-5e-5}"
EM_ENCODER_MAX_GRAD="${EM_ENCODER_MAX_GRAD:-1.0}"
EM_WARMUP_OUTER_ITERS="${EM_WARMUP_OUTER_ITERS:-2}"
EM_VAL_EVERY="${EM_VAL_EVERY:-2}"
EM_HER_SAMPLES="${EM_HER_SAMPLES:-1}"
EM_HER_REFRESH="${EM_HER_REFRESH:-0}"
EM_SAVE_EVAL_CKPTS="${EM_SAVE_EVAL_CKPTS:-true}"
EM_SAVE_OUTER_CKPTS="${EM_SAVE_OUTER_CKPTS:-false}"
EM_ENCODER_DIAGNOSTICS="${EM_ENCODER_DIAGNOSTICS:-false}"
EM_ENCODER_DIAGNOSTICS_EVERY="${EM_ENCODER_DIAGNOSTICS_EVERY:-50}"
IQL_GOAL_ADAPTER="${IQL_GOAL_ADAPTER:-false}"
IQL_GOAL_ADAPTER_HIDDEN="${IQL_GOAL_ADAPTER_HIDDEN:-64}"
IQL_GOAL_ADAPTER_INIT_SCALE="${IQL_GOAL_ADAPTER_INIT_SCALE:-1e-3}"
IQL_TARGET_SAMPLING="${IQL_TARGET_SAMPLING:-horizon_aligned}"
IQL_TARGET_HORIZONS="${IQL_TARGET_HORIZONS:-[1,2,3,4,5,6]}"
IQL_HORIZON_TERMINAL_DONE="${IQL_HORIZON_TERMINAL_DONE:-true}"
EVAL_TAU_LIST="${EVAL_TAU_LIST:-[1,2,3,4,5,6]}"
VAL_METRIC="${VAL_METRIC:-rmse_uns}"
EM_VAL_TAU_AGG="${EM_VAL_TAU_AGG:-max}"
GPU_WAIT_MEMORY_MB="${GPU_WAIT_MEMORY_MB:-1000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-em_iql_local_global_gift_protocol}"
MLFLOW_URI="${MLFLOW_URI:-}"
FORCE="${FORCE:-0}"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

GRID_ROOT="${GRID_ROOT:-${ROOT}/grid_results/em_iql_local_global_gift_protocol/gamma_${GAMMA}}"
mkdir -p "${GRID_ROOT}/logs" "${GRID_ROOT}/ckpts" "${GRID_ROOT}/done"

SUMMARY="${GRID_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "combo_id,seed,split,local_conv_layers,dataset_train,dataset_val,dataset_test,max_seq_length,min_seq_length,iql_tau,iql_actor_lr,iql_qf_lr,iql_vf_lr,iql_beta,iql_weight_max,iql_max_grad_norm,em_m_steps_per_outer,em_her_samples_per_transition,em_her_refresh_every,em_warmup_outer_iters,best_outer,best_val_metric,best_val_score,eval_tau,eval_mae_norm,eval_mae_uns,eval_rmse_norm,eval_rmse_norm_x_std,eval_rmse_uns,gift_rmse,gift_rmse_percent,gift_mae_percent,em_ckpt,train_log,eval_log,finished_at" > "${SUMMARY}"
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

  python - "${SUMMARY}" "${combo_id}" "${seed}" "${split}" "${DATASET_TRAIN}" "${DATASET_VAL}" "${DATASET_TEST}" "${MAX_SEQ_LENGTH}" "${MIN_SEQ_LENGTH}" "${IQL_TAU}" "${IQL_ACTOR_LR}" "${IQL_QF_LR}" "${IQL_VF_LR}" \
    "${IQL_BETA}" "${IQL_WEIGHT_MAX}" "${IQL_MAX_GRAD}" "${EM_M_STEPS}" "${EM_HER_SAMPLES}" "${EM_HER_REFRESH}" "${EM_WARMUP_OUTER_ITERS}" \
    "${best_outer}" "${best_val_metric}" "${best_val_score}" "${em_ckpt}" "${train_log}" "${eval_log}" "${finished_at}" <<'PY'
import csv
import os
import re
import sys

(
    summary_path, combo_id, seed, split, dataset_train, dataset_val, dataset_test, max_seq_length, min_seq_length,
    iql_tau, actor_lr, qf_lr, vf_lr,
    iql_beta, iql_weight_max, iql_max_grad, em_m_steps, em_her_samples, em_her_refresh, em_warmup_outer_iters,
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
    "iql_beta", "iql_weight_max", "iql_max_grad_norm", "em_m_steps_per_outer", "em_her_samples_per_transition",
    "em_her_refresh_every", "em_warmup_outer_iters", "best_outer", "best_val_metric", "best_val_score", "eval_tau", "eval_mae_norm",
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
            "dataset_train": dataset_train,
            "dataset_val": dataset_val,
            "dataset_test": dataset_test,
            "max_seq_length": max_seq_length,
            "min_seq_length": min_seq_length,
            "iql_tau": iql_tau,
            "iql_actor_lr": actor_lr,
            "iql_qf_lr": qf_lr,
            "iql_vf_lr": vf_lr,
            "iql_beta": iql_beta,
            "iql_weight_max": iql_weight_max,
            "iql_max_grad_norm": iql_max_grad,
            "em_m_steps_per_outer": em_m_steps,
            "em_her_samples_per_transition": em_her_samples,
            "em_her_refresh_every": em_her_refresh,
            "em_warmup_outer_iters": em_warmup_outer_iters,
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
  local sampling_id="${IQL_TARGET_SAMPLING//[^A-Za-z0-9]/}"
  local reward_id="${IQL_REWARD_TYPE//[^A-Za-z0-9]/}"
  local beta_id="${IQL_BETA//./p}"
  beta_id="${beta_id//-/m}"
  beta_id="${beta_id//+/p}"
  beta_id="${beta_id//[^A-Za-z0-9pm]/}"
  local adv_id="${IQL_ADV_MAX//./p}"
  adv_id="${adv_id//-/m}"
  adv_id="${adv_id//+/p}"
  adv_id="${adv_id//[^A-Za-z0-9pm]/}"
  local wmax_id="${IQL_WEIGHT_MAX//./p}"
  wmax_id="${wmax_id//-/m}"
  wmax_id="${wmax_id//+/p}"
  wmax_id="${wmax_id//[^A-Za-z0-9pm]/}"
  local enc_lr_id="${EM_ENCODER_LR//./p}"
  enc_lr_id="${enc_lr_id//-/m}"
  enc_lr_id="${enc_lr_id//+/p}"
  enc_lr_id="${enc_lr_id//[^A-Za-z0-9pm]/}"
  local enc_grad_id="${EM_ENCODER_MAX_GRAD//./p}"
  enc_grad_id="${enc_grad_id//-/m}"
  enc_grad_id="${enc_grad_id//+/p}"
  enc_grad_id="${enc_grad_id//[^A-Za-z0-9pm]/}"
  if [[ "${IQL_REWARD_TYPE}" == "negative_outcome_huber" || "${IQL_REWARD_TYPE}" == "huber" || "${IQL_REWARD_TYPE}" == "smooth_l1" ]]; then
    reward_id="${reward_id}_d${IQL_REWARD_HUBER_DELTA//[^A-Za-z0-9]/}"
  fi
  local actor_id="${IQL_ACTOR_UPDATE//[^A-Za-z0-9]/}"
  local qalpha_id="${IQL_TD3BC_Q_ALPHA//./p}"
  qalpha_id="${qalpha_id//-/m}"
  qalpha_id="${qalpha_id//+/p}"
  qalpha_id="${qalpha_id//[^A-Za-z0-9pm]/}"
  local bcalpha_id="${IQL_TD3BC_BC_ALPHA//./p}"
  bcalpha_id="${bcalpha_id//-/m}"
  bcalpha_id="${bcalpha_id//+/p}"
  bcalpha_id="${bcalpha_id//[^A-Za-z0-9pm]/}"
  local combo_id="seq${MAX_SEQ_LENGTH}_lg_tau07_lr3e4_grad5_m1k_b${beta_id}_adv${adv_id}_w${wmax_id}_actor${actor_id}_qa${qalpha_id}_bc${bcalpha_id}_enc${enc_lr_id}_eg${enc_grad_id}_val${VAL_METRIC}_${sampling_id}_${reward_id}_her${EM_HER_SAMPLES}"
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
  echo "  Tumor data protocol: train/val/test=${DATASET_TRAIN}/${DATASET_VAL}/${DATASET_TEST}, max_seq_length=${MAX_SEQ_LENGTH}, min_seq_length=${MIN_SEQ_LENGTH}"
  echo "  seed=${seed} gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
  echo "  local_conv_layers=${local_layers}"
  echo "  iql_tau=${IQL_TAU} actor_lr=${IQL_ACTOR_LR} qf_lr=${IQL_QF_LR} vf_lr=${IQL_VF_LR}"
  echo "  iql_beta=${IQL_BETA} iql_adv_max=${IQL_ADV_MAX} iql_weight_max=${IQL_WEIGHT_MAX}"
  echo "  iql_actor_update=${IQL_ACTOR_UPDATE} td3bc_q_alpha=${IQL_TD3BC_Q_ALPHA} td3bc_bc_alpha=${IQL_TD3BC_BC_ALPHA}"
  echo "  iql_cql_alpha=${IQL_CQL_ALPHA} iql_cql_n_actions=${IQL_CQL_N_ACTIONS}"
  echo "  iql_actor_bc_loss=${IQL_ACTOR_BC_LOSS} iql_actor_bc_expectile=${IQL_ACTOR_BC_EXPECTILE}"
  echo "  val_metric=${VAL_METRIC} em_val_tau_agg=${EM_VAL_TAU_AGG} val_tau_list=${EVAL_TAU_LIST}"
  echo "  em_encoder_lr=${EM_ENCODER_LR} em_encoder_max_grad=${EM_ENCODER_MAX_GRAD} em_m_steps=${EM_M_STEPS} warmup_outer_iters=${EM_WARMUP_OUTER_ITERS}"
  echo "  goal_adapter=${IQL_GOAL_ADAPTER} hidden=${IQL_GOAL_ADAPTER_HIDDEN} init_scale=${IQL_GOAL_ADAPTER_INIT_SCALE}"
  echo "  encoder_diagnostics=${EM_ENCODER_DIAGNOSTICS} every=${EM_ENCODER_DIAGNOSTICS_EVERY}"
  echo "  reward_type=${IQL_REWARD_TYPE} reward_huber_delta=${IQL_REWARD_HUBER_DELTA}"
  echo "  target_sampling=${IQL_TARGET_SAMPLING} target_horizons=${IQL_TARGET_HORIZONS} horizon_terminal_done=${IQL_HORIZON_TERMINAL_DONE}"

  wait_for_gpu
  CUDA_VISIBLE_DEVICES="${GPU}" python -u runnables/train_ct_iql_em.py \
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${GAMMA}*" \
    exp.seed="${seed}" dataset.coeff="${GAMMA}" \
    "dataset.num_patients.train=${DATASET_TRAIN}" \
    "dataset.num_patients.val=${DATASET_VAL}" \
    "dataset.num_patients.test=${DATASET_TEST}" \
    "dataset.max_seq_length=${MAX_SEQ_LENGTH}" \
    "+dataset.min_seq_length=${MIN_SEQ_LENGTH}" \
    "exp.load_data=false" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_her_refresh_every=${EM_HER_REFRESH}" \
    "exp.em_her_samples_per_transition=${EM_HER_SAMPLES}" \
    "exp.em_save_every_eval_checkpoint=${EM_SAVE_EVAL_CKPTS}" \
    "exp.iql_target_sampling=${IQL_TARGET_SAMPLING}" \
    "exp.iql_target_horizons=${IQL_TARGET_HORIZONS}" \
    "exp.iql_horizon_terminal_done=${IQL_HORIZON_TERMINAL_DONE}" \
    "exp.ct_num_workers=0" \
    "exp.iql_reward_type=${IQL_REWARD_TYPE}" \
    "exp.iql_reward_huber_delta=${IQL_REWARD_HUBER_DELTA}" \
    "exp.iql_beta=${IQL_BETA}" \
    "+exp.iql_adv_max=${IQL_ADV_MAX}" \
    "exp.iql_weight_max=${IQL_WEIGHT_MAX}" \
    "exp.iql_actor_update=${IQL_ACTOR_UPDATE}" \
    "exp.iql_actor_bc_loss=${IQL_ACTOR_BC_LOSS}" \
    "exp.iql_actor_bc_expectile=${IQL_ACTOR_BC_EXPECTILE}" \
    "exp.iql_td3bc_q_alpha=${IQL_TD3BC_Q_ALPHA}" \
    "exp.iql_td3bc_bc_alpha=${IQL_TD3BC_BC_ALPHA}" \
    "exp.iql_cql_alpha=${IQL_CQL_ALPHA}" \
    "exp.iql_cql_n_actions=${IQL_CQL_N_ACTIONS}" \
    "exp.iql_val_worlds=[sim]" \
    "exp.iql_val_selection_world=sim" \
    "exp.iql_tau=${IQL_TAU}" \
    "exp.iql_actor_lr=${IQL_ACTOR_LR}" \
    "exp.iql_qf_lr=${IQL_QF_LR}" \
    "exp.iql_vf_lr=${IQL_VF_LR}" \
    "exp.iql_max_grad_norm=${IQL_MAX_GRAD}" \
    "exp.em_outer_iters=${EM_OUTER_ITERS}" \
    "exp.em_m_steps_per_outer=${EM_M_STEPS}" \
    "exp.em_encoder_lr=${EM_ENCODER_LR}" \
    "exp.em_encoder_max_grad_norm=${EM_ENCODER_MAX_GRAD}" \
    "exp.em_warmup_outer_iters=${EM_WARMUP_OUTER_ITERS}" \
    "exp.em_val_every=${EM_VAL_EVERY}" \
    "exp.em_val_tau_list=${EVAL_TAU_LIST}" \
    "exp.em_save_every_outer_checkpoint=${EM_SAVE_OUTER_CKPTS}" \
    "+exp.em_encoder_diagnostics=${EM_ENCODER_DIAGNOSTICS}" \
    "+exp.em_encoder_diagnostics_every=${EM_ENCODER_DIAGNOSTICS_EVERY}" \
    "+exp.iql_goal_adapter_enabled=${IQL_GOAL_ADAPTER}" \
    "+exp.iql_goal_adapter_hidden_dim=${IQL_GOAL_ADAPTER_HIDDEN}" \
    "+exp.iql_goal_adapter_init_scale=${IQL_GOAL_ADAPTER_INIT_SCALE}" \
    "exp.em_val_metric=${VAL_METRIC}" \
    "exp.em_val_tau_agg=${EM_VAL_TAU_AGG}" \
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
    "dataset.num_patients.train=${DATASET_TRAIN}" \
    "dataset.num_patients.val=${DATASET_VAL}" \
    "dataset.num_patients.test=${DATASET_TEST}" \
    "dataset.max_seq_length=${MAX_SEQ_LENGTH}" \
    "+dataset.min_seq_length=${MIN_SEQ_LENGTH}" \
    "exp.load_data=false" \
    "model.inference.local_conv_layers=${local_layers}" \
    "exp.em_eval_ckpt=${em_ckpt}" \
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
    echo "dataset_train=${DATASET_TRAIN}"
    echo "dataset_val=${DATASET_VAL}"
    echo "dataset_test=${DATASET_TEST}"
    echo "max_seq_length=${MAX_SEQ_LENGTH}"
    echo "min_seq_length=${MIN_SEQ_LENGTH}"
    echo "iql_target_sampling=${IQL_TARGET_SAMPLING}"
    echo "iql_target_horizons=${IQL_TARGET_HORIZONS}"
    echo "iql_horizon_terminal_done=${IQL_HORIZON_TERMINAL_DONE}"
    echo "iql_beta=${IQL_BETA}"
    echo "iql_adv_max=${IQL_ADV_MAX}"
    echo "iql_weight_max=${IQL_WEIGHT_MAX}"
    echo "iql_actor_update=${IQL_ACTOR_UPDATE}"
    echo "iql_td3bc_q_alpha=${IQL_TD3BC_Q_ALPHA}"
    echo "iql_td3bc_bc_alpha=${IQL_TD3BC_BC_ALPHA}"
    echo "em_ckpt=${em_ckpt}"
    echo "train_log=${train_log}"
    echo "eval_log=${eval_log}"
  } > "${done_flag}"
  echo "[done] ${tag}"
}

echo "[gift-protocol] one-stage EM+IQL local-global"
echo "[gift-protocol] gamma=${GAMMA} gpu=${GPU} test_split=${TEST_SPLIT}"
echo "[gift-protocol] seeds=(${SEEDS[*]})"
echo "[gift-protocol] data train/val/test=${DATASET_TRAIN}/${DATASET_VAL}/${DATASET_TEST} max_seq_length=${MAX_SEQ_LENGTH} min_seq_length=${MIN_SEQ_LENGTH}"
echo "[gift-protocol] iql_tau=${IQL_TAU} actor_lr=${IQL_ACTOR_LR} qf_lr=${IQL_QF_LR} vf_lr=${IQL_VF_LR}"
echo "[gift-protocol] beta=${IQL_BETA} adv_max=${IQL_ADV_MAX} weight_max=${IQL_WEIGHT_MAX} actor_update=${IQL_ACTOR_UPDATE} td3bc_q_alpha=${IQL_TD3BC_Q_ALPHA} td3bc_bc_alpha=${IQL_TD3BC_BC_ALPHA} grad=${IQL_MAX_GRAD} outer_iters=${EM_OUTER_ITERS} m_steps=${EM_M_STEPS} em_encoder_lr=${EM_ENCODER_LR} em_encoder_max_grad=${EM_ENCODER_MAX_GRAD} warmup_outer_iters=${EM_WARMUP_OUTER_ITERS} val_every=${EM_VAL_EVERY} eval_tau_list=${EVAL_TAU_LIST}"
echo "[gift-protocol] cql_alpha=${IQL_CQL_ALPHA} cql_n_actions=${IQL_CQL_N_ACTIONS}"
echo "[gift-protocol] save_outer_ckpts=${EM_SAVE_OUTER_CKPTS} save_eval_ckpts=${EM_SAVE_EVAL_CKPTS}"
echo "[gift-protocol] goal_adapter=${IQL_GOAL_ADAPTER} hidden=${IQL_GOAL_ADAPTER_HIDDEN} init_scale=${IQL_GOAL_ADAPTER_INIT_SCALE}"
echo "[gift-protocol] target_sampling=${IQL_TARGET_SAMPLING} target_horizons=${IQL_TARGET_HORIZONS} horizon_terminal_done=${IQL_HORIZON_TERMINAL_DONE} her_samples=${EM_HER_SAMPLES}"
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
