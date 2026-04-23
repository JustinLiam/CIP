#!/usr/bin/env bash
# CT standalone (train_ct) -> IQL (train_iql_planner) -> eval (eval_iql_planner).
# Usage (from repo root recommended):
#   bash scripts/cancer/train/train_ct_iql.sh [test] [gamma] [gpu] [eval_tau]
#     test     - passed as exp.test to eval only (false=val, true=test), default false
#     gamma    - dataset.coeff / confounding level, default 4
#     gpu      - CUDA_VISIBLE_DEVICES, default 0
#     eval_tau - optional; if set, passed as exp.tau to eval_iql_planner only (rollout horizon + CIPDataset target window).
#                CT/IQL training do not use this; same checkpoints work for any eval_tau (e.g. 6 then 4 without retraining).
#
# Eval-only (reuse existing checkpoints — each seed must have been fully trained once WITHOUT this flag):
#   CT_IQL_SKIP_TRAIN=1 bash scripts/cancer/train/train_ct_iql.sh false 4 0 4
# New seeds: run WITHOUT CT_IQL_SKIP_TRAIN first so train_ct creates ct_best_encoder.pt and train_iql_planner creates iql_planner.pt.
#
# Checkpoints (tau is NOT part of these paths):
#   ct_checkpoints/seed_${seed}_gamma_${gamma}/ct_best_encoder.pt
#   iql_models/seed_${seed}/gamma_${gamma}/iql_planner.pt

# set -euo pipefail 的作用是：
# -e: 遇到命令执行出错时立即退出脚本
# -u: 访问未定义变量时立即退出
# -o pipefail: 只要管道（|）中的任何一个命令出错，则整个管道被视为失败（返回非零值）
# 这样可以让 bash 脚本在遇到错误时及时终止，避免隐式 bug 并提高脚本的健壮性
set -euo pipefail

# 该语句的作用是：让当前 Bash 脚本能够使用 conda activate 命令切换环境。
# 具体地，eval "$(conda shell.bash hook)" 会在当前 shell 会话中注册 conda 的 shell 插件，
# 这样就能直接用 conda activate xxx 激活指定的 conda 虚拟环境。
eval "$(conda shell.bash hook)"
conda activate vcip

test=${1:-false}
gamma=${2:-4}
gpu=${3:-0}
eval_tau=${4:-}

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

# Shared CSV (append-only) for all runs; safe under concurrent bash processes.
CSV_DIR="${ROOT}/results"
CSV_PATH="${CSV_DIR}/ct_iql_eval_metrics.csv"
CSV_LOCK="${CSV_PATH}.lock"
mkdir -p "${CSV_DIR}"
touch "${CSV_LOCK}"

ensure_csv_header() {
  if [[ ! -f "${CSV_PATH}" ]]; then
    printf "timestamp,gamma,seed,tau,test,mae_norm,mae_uns,rmse_uns,rmse_norm,rmse_x_std,rmse_factual_x_std\n" > "${CSV_PATH}"
  fi
}

append_csv_row() {
  local row="$1"
  (
    flock -x 200
    ensure_csv_header
    printf "%s\n" "${row}" >> "${CSV_PATH}"
  ) 200>"${CSV_LOCK}"
}

seeds=(10 101 1010 10101 101010)
# seeds=(20 202 2020 20202 202020)
# seeds=(10)

for seed in "${seeds[@]}"; do
  CT_CKPT="${ROOT}/ct_checkpoints/seed_${seed}_gamma_${gamma}/ct_best_encoder.pt"
  IQL_CKPT="${ROOT}/iql_models/seed_${seed}/gamma_${gamma}/iql_planner.pt"

  tau_msg=""
  [[ -n "${eval_tau}" ]] && tau_msg=" exp.tau=${eval_tau}"
  echo "=== seed=${seed} gamma=${gamma} | CT -> IQL -> eval (exp.test=${test})${tau_msg} ==="

  if [[ "${CT_IQL_SKIP_TRAIN:-0}" != "1" ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_ct.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}"

    if [[ ! -f "${CT_CKPT}" ]]; then
      echo "ERROR: CT checkpoint missing after train_ct: ${CT_CKPT}" >&2
      exit 1
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_iql_planner.py \
      +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*" \
      exp.seed="${seed}" dataset.coeff="${gamma}" \
      exp.iql_inference_ckpt="${CT_CKPT}"
  else
    if [[ ! -f "${CT_CKPT}" ]]; then
      echo "ERROR: CT checkpoint missing (CT_IQL_SKIP_TRAIN=1): ${CT_CKPT}" >&2
      echo "       New seed? Unset CT_IQL_SKIP_TRAIN and run the full script once (train_ct -> train_iql -> eval)." >&2
      exit 1
    fi
    if [[ ! -f "${IQL_CKPT}" ]]; then
      echo "ERROR: IQL checkpoint missing (CT_IQL_SKIP_TRAIN=1): ${IQL_CKPT}" >&2
      echo "       New seed? Unset CT_IQL_SKIP_TRAIN and run the full script once." >&2
      exit 1
    fi
  fi

  eval_args=(
    +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=${gamma}*"
    exp.seed="${seed}" dataset.coeff="${gamma}"
    exp.test="${test}"
    exp.iql_inference_ckpt="${CT_CKPT}"
  )
  [[ -n "${eval_tau}" ]] && eval_args+=( "exp.tau=${eval_tau}" )

  eval_log="$(mktemp)"
  CUDA_VISIBLE_DEVICES=${gpu} python runnables/eval_iql_planner.py "${eval_args[@]}" 2>&1 | tee "${eval_log}"

  # Parse aggregate metrics from eval log (use Python: default awk is often mawk and does not
  # support gawk-only match(..., ..., array) capture groups).
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  read -r tau_used mae_norm mae_uns rmse_uns rmse_norm rmse_x_std rmse_factual_x_std < <(
    python -c '
import re
import sys

def last_num(pat, text):
    ms = list(re.finditer(pat, text))
    return ms[-1].group(1) if ms else "NA"

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
tau_m = re.search(r"\(tau=(\d+),", text)
tau = tau_m.group(1) if tau_m else "NA"
m3 = None
for m in re.finditer(
    r"MAE normalized:\s*([0-9.]+)\s*\|\s*MAE unscaled:\s*([0-9.]+)\s*\|\s*RMSE unscaled:\s*([0-9.]+)",
    text,
):
    m3 = m
if m3:
    mae_norm, mae_uns, rmse_uns = m3.groups()
else:
    mae_norm = mae_uns = rmse_uns = "NA"
rmse_norm = last_num(r"Global RMSE on stacked batches \(normalized space\):\s*([0-9.]+)", text)
rmse_x_std = last_num(r"VCIP-style\):\s*([0-9.]+)", text)
rmse_factual_x_std = last_num(r"Factual global RMSE[^\n:]*:\s*([0-9.]+)", text)
print(tau, mae_norm, mae_uns, rmse_uns, rmse_norm, rmse_x_std, rmse_factual_x_std)
' "${eval_log}"
  )

  # Fallbacks if parsing misses a field for any reason.
  [[ -z "${mae_norm}" ]] && mae_norm="NA"
  [[ -z "${mae_uns}" ]] && mae_uns="NA"
  [[ -z "${rmse_uns}" ]] && rmse_uns="NA"
  [[ -z "${rmse_norm}" ]] && rmse_norm="NA"
  [[ -z "${rmse_x_std}" ]] && rmse_x_std="NA"
  [[ -z "${rmse_factual_x_std}" ]] && rmse_factual_x_std="NA"

  append_csv_row "${timestamp},${gamma},${seed},${tau_used},${test},${mae_norm},${mae_uns},${rmse_uns},${rmse_norm},${rmse_x_std},${rmse_factual_x_std}"
  rm -f "${eval_log}"
done
