# README



### Prerequisites

- Anaconda or Miniconda installed on your system
- Git for version control
- Python 3.8 or higher

### Setting Up the Environments

1. First, create and configure the VCIP environment:

```bash
# Create and activate VCIP environment
conda create -n vcip python=3.8
conda activate vcip
pip install -r requirements_vcip.txt
```

2. Then, create and configure the baseline environment:

```bash
# Create and activate baseline environment
conda create -n baseline python=3.8
conda activate baseline
pip install -r requirements_ct.txt
```

3. prepare the MLFlow enviroment:
```bash
mlflow server \
  --backend-store-uri ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

## Running Experiments

### One-stage CT+IQL EM under the GIFT tumor protocol

Use this command for the current VCIP one-stage CT+IQL EM experiment. It uses the synthetic tumor comparison protocol (`train/val/test=1000/200/200`, `max_seq_length=60`) and evaluates `tau=1,...,6`.

```bash
ssh thinkstation
cd /home/liam/pythonProject/VCIP-ICML-main
source /home/liam/anaconda3/etc/profile.d/conda.sh
conda activate vcip

GRID_ROOT=grid_results/stable_gift_protocol_best_$(date +%Y%m%d) \
GRID_SEEDS="20 202 2020 20202 202020" \
TEST_SPLIT=true \
FORCE=1 \
bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4
```

Current default hyperparameters are stored in `configs/model/vcip.yaml`: AWR actor update, expectile actor BC (`iql_actor_bc_loss=expectile`, `iql_actor_bc_expectile=0.8`), `iql_weight_max=1.0`, `em_warmup_outer_iters=2`, `em_val_metric=rmse_uns`, and max-over-`tau=1,...,6` validation selection. Keep `iql_eval_action_shift=0.0` for main experiments.

For a single diagnostic seed, run:

```bash
GRID_ROOT=grid_results/stable_gift_protocol_seed2_$(date +%Y%m%d) \
GRID_SEEDS="2" \
TEST_SPLIT=true \
IQL_WEIGHT_MAX=5.0 \
FORCE=1 \
bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4
```

Do not choose the final seed set based on test-set performance. If seed 2 is useful, report it as an additional seed or predefine the replacement before reading test metrics.

To train the CT model:

1. Run:
```bash
python runnables/train_ct.py +dataset=cancer_sim_cont +model=vcip
```

CUDA_VISIBLE_DEVICES=0 python runnables/train_ct.py +dataset=cancer_sim_cont +model=vcip "+model/hparams/cancer=4*" exp.seed=10 dataset.coeff=4 exp.ct_epochs=2 exp.ct_weight_log_every=1
  "+exp.ct_ckpt_dir=/tmp/ct_smoke_$$"

2. Run the IQL model using the CT checkpoint:
```bash
python runnables/train_iql_planner.py +dataset=cancer_sim_cont +model=vcip   exp.iql_inference_ckpt=/home/liam/pythonProject/VCIP-ICML-main/ct_checkpoints/seed_10_gamma_4/ct_best_encoder.pt
```

```bash
CUDA_VISIBLE_DEVICES=1 python runnables/train_iql_planner.py +dataset=cancer_sim_cont +model=vcip \
  exp.iql_inference_ckpt=/home/liam/pythonProject/VCIP-ICML-main/ct_checkpoints/seed_10_gamma_4/kmax1_dyn005/ct_best_encoder.pt \
  +exp.iql_save_dir=/home/liam/pythonProject/VCIP-ICML-main/iql_runs/kmax3_seed10_g4
```

CUDA_VISIBLE_DEVICES=1 python runnables/eval_iql_planner.py +dataset=cancer_sim_cont +model=vcip   exp.seed=10 dataset.coeff=4 exp.tau=12 exp.max_tau=12.0   exp.test=True   exp.iql_inference_ckpt=/home/liam/pythonProject/VCIP-ICML-main/ct_checkpoints/seed_10_gamma_4/kmax1_dyn005/ct_best_encoder.pt exp.iql_eval_ckpt=/home/liam/pythonProject/VCIP-ICML-main/iql_runs/kmax3_seed10_g4/iql_planner_best_predictor.pt

3. Run the IQL model validation:
```bash
python runnables/eval_iql_planner.py +dataset=cancer_sim_cont +model=vcip exp.test=True  exp.iql_inference_ckpt=/home/liam/pythonProject/VCIP-ICML-main/ct_checkpoints/seed_10_gamma_4/ct_best_encoder.pt
```

开了 CT_IQL_SKIP_TRAIN=1，脚本不会跑 train_ct 和 train_iql_planner，只会做 eval。

第一次为这个 seed 跑完整流程（不要设 CT_IQL_SKIP_TRAIN）：

```bash 
scripts/cancer/train/train_ct_iql.sh false 4 0 12
```
（把 4 0 12 换成你的 gamma、GPU、eval_tau；若不需要改 tau，第 4 个参数可省略。）
这样会依次：train_ct → 写出 ct_best_encoder.pt → train_iql_planner（用该 CT）→ eval。

之后若只想换 exp.tau 做 eval、不重训，再用：
```bash
CT_IQL_SKIP_TRAIN=1 bash scripts/cancer/train/train_ct_iql.sh false 4 0 12
```


The results will be saved in the `results/all/` directory, matching the experimental results presented in the paper.

###Experimental Platform

To ensure consistency and fairness in all experimental comparisons, both VCIP and all baseline models are tested on the same computational setup:

**Hardware Specifications**

- **Processor (CPU)**: AMD Ryzen 9 5900X 12-Core Processor
- **Graphics Processing Units (GPUs)**: 4x NVIDIA GeForce RTX 4080 Ti
