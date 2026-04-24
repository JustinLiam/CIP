#!/usr/bin/env bash
# IQL Planner 超参数网格搜索脚本
# 自动加载预训练好的 CT 模型并跑完 10 个 seed

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate vcip

gamma=${1:-4}
gpu=${2:-0}
TEST_SPLIT=${3:-false}

GRID_WORKER_ID=${GRID_WORKER_ID:-0}
GRID_NUM_WORKERS=${GRID_NUM_WORKERS:-1}

# ---------- IQL 搜索空间定义 ----------
IQL_TAU_LIST=("0.5" "0.7" "0.8" "0.9")
IQL_BETA_LIST=("1.0" "3.0" "5.0" "10.0")
IQL_TARGET_TAU_LIST=("0.001" "0.005" "0.01")
IQL_LR_LIST=("1e-4" "3e-4" "5e-4")
IQL_DISCOUNT_LIST=("0.9" "0.95" "0.99")

# 指定的 10 个 Seed
SEEDS=(20 202 2020 20202 202020 10 101 1010 10101 101010)

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT}"

grid_root="${ROOT}/grid_results/iql_search/gamma_${gamma}"
mkdir -p "${grid_root}/logs" "${grid_root}/iql_ckpts" "${grid_root}/iql_ckpts_work"
summary_csv="${grid_root}/summary.csv"
summary_lock="${grid_root}/.summary.lock"

# 定义 CSV 表头
EXPECTED_HEADER="combo_id,seed,iql_tau,iql_beta,iql_target_tau,iql_lr,iql_discount,iql_best_val_mae_uns,iql_best_step,mae_uns_eval_split,eval_split"

with_lock() {
  ( flock -x 9; "$@" ) 9> "${summary_lock}"
}

# 初始化表头
if [[ ! -f "${summary_csv}" ]]; then
  echo "${EXPECTED_HEADER}" > "${summary_csv}"
fi

parse_last() {
  local file="$1" pat="$2"
  grep -oP "${pat}" "${file}" 2>/dev/null | tail -n1 || true
}

shard_owner() {
  local tag="$1"
  local h
  h=$(printf '%s' "${tag}" | cksum | awk '{print $1}')
  echo $(( h % GRID_NUM_WORKERS ))
}

run_iql_combo() {
  local tau="$1" beta="$2" t_tau="$3" lr="$4" disc="$5" seed="$6"
  local combo_id="tau-${tau}_beta-${beta}_ttau-${t_tau}_lr-${lr}_disc-${disc}"
  local tag="${combo_id}__seed_${seed}"
  local log_file="${grid_root}/logs/${tag}.log"
  local iql_work_dir="${grid_root}/iql_ckpts_work/${tag}"
  local iql_ckpt_src="${iql_work_dir}/iql_planner.pt"
  local iql_ckpt_grid="${grid_root}/iql_ckpts/${tag}.pt"

  # 根据您提供的路径模板拼接 CT 模型路径
  local ct_ckpt="/home/liam/pythonProject/VCIP-ICML-main/ct_checkpoints/seed_${seed}_gamma_4/kmax3_dyn01/ct_best_encoder.pt"

  # 检查任务归属及是否已运行
  local owner
  owner=$(shard_owner "${tag}")
  if (( owner != GRID_WORKER_ID )); then return 0; fi

  if grep -q "^${combo_id},${seed}," "${summary_csv}" 2>/dev/null; then
    echo "[w${GRID_WORKER_ID}][skip] ${tag} 已存在"
    return 0
  fi

  if [[ ! -f "${ct_ckpt}" ]]; then
    echo "警告: 找不到预训练的 CT 模型: ${ct_ckpt}，跳过此 seed。"
    return 0
  fi

  mkdir -p "${iql_work_dir}"

  echo "=== [Worker ${GRID_WORKER_ID}] 正在运行: ${tag} ==="
  
  # 1. 训练 IQL Planner
  CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/train_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    exp.iql_tau="${tau}" \
    exp.iql_beta="${beta}" \
    exp.iql_target_tau="${t_tau}" \
    exp.iql_actor_lr="${lr}" exp.iql_qf_lr="${lr}" exp.iql_vf_lr="${lr}" \
    exp.iql_discount="${disc}" \
    exp.iql_inference_ckpt="${ct_ckpt}" \
    "+exp.iql_save_dir=${iql_work_dir}" \
    2>&1 | tee "${log_file}"

  if [[ ! -f "${iql_ckpt_src}" ]]; then
    echo "错误: IQL 训练未生成模型 ${tag}"
    return 0
  fi
  cp "${iql_ckpt_src}" "${iql_ckpt_grid}"

  # 解析训练日志中的最佳指标
  local iql_best_mae=$(parse_last "${log_file}" 'Saved BEST IQL planner.*\(mae_uns=\K[0-9.eE+-]+')
  local iql_best_step=$(parse_last "${log_file}" 'Saved BEST IQL planner.* at step \K[0-9]+')
  [[ -z "${iql_best_mae}" ]] && iql_best_mae="NA"
  [[ -z "${iql_best_step}" ]] && iql_best_step="NA"

  # 2. 运行闭环评估 (tau=12)
  echo "=== 正在评估: ${tag} ==="
  CUDA_VISIBLE_DEVICES=${gpu} python -u runnables/eval_iql_planner.py \
    +dataset=cancer_sim_cont +model=vcip \
    exp.seed="${seed}" dataset.coeff="${gamma}" \
    exp.test="${TEST_SPLIT}" \
    exp.iql_inference_ckpt="${ct_ckpt}" \
    exp.iql_eval_ckpt="${iql_ckpt_grid}" \
    2>&1 | tee -a "${log_file}"

  local mae_eval=$(parse_last "${log_file}" 'MAE unscaled: \K[0-9.eE+-]+')
  [[ -z "${mae_eval}" ]] && mae_eval="NA"

  # 写入 CSV
  with_lock bash -c "echo '${combo_id},${seed},${tau},${beta},${t_tau},${lr},${disc},${iql_best_mae},${iql_best_step},${mae_eval},${TEST_SPLIT}' >> '${summary_csv}'"
}

# 嵌套循环执行网格搜索
for tau in "${IQL_TAU_LIST[@]}"; do
  for beta in "${IQL_BETA_LIST[@]}"; do
    for t_tau in "${IQL_TARGET_TAU_LIST[@]}"; do
      for lr in "${IQL_LR_LIST[@]}"; do
        for disc in "${IQL_DISCOUNT_LIST[@]}"; do
          for seed in "${SEEDS[@]}"; do
            run_iql_combo "${tau}" "${beta}" "${t_tau}" "${lr}" "${disc}" "${seed}"
          done
        done
      done
    done
  done
done

# ---------------- 统计部分 ----------------
echo "搜索完成。正在汇总结果..."
python - <<PY
import csv
from collections import defaultdict
import numpy as np

path = "${summary_csv}"
def to_f(x):
    try: return float(x)
    except: return None

groups = defaultdict(list)
with open(path, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        groups[r["combo_id"]].append(to_f(r["mae_uns_eval_split"]))

results = []
for cid, maes in groups.items():
    maes = [m for m in maes if m is not None]
    if len(maes) > 0:
        results.append({
            "combo_id": cid,
            "mean": np.mean(maes),
            "median": np.median(maes),
            "count": len(maes)
        })

results.sort(key=lambda x: x["mean"])
print("\n--- 最优组合排序 (按 10 个 Seed 平均 MAE 排序) ---")
print(f"{'Rank':<4} {'Mean MAE':<10} {'Median':<10} {'Seeds':<6} {'Combo_ID'}")
for i, r in enumerate(results[:20], 1):
    print(f"{i:<4} {r['mean']:<10.6f} {r['median']:<10.6f} {r['count']:<6} {r['combo_id']}")
PY