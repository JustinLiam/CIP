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

## Running Experiments

To train the CT model:

1. Run:
```bash
python runnables/train_ct.py +dataset=cancer_sim_cont +model=vcip
```

2. Run the IQL model using the CT checkpoint:
```bash
python runnables/train_iql_planner.py +dataset=cancer_sim_cont +model=vcip   exp.iql_inference_ckpt=/home/liam/pythonProject/VCIP-ICML-main/ct_checkpoints/seed_10_gamma_4/ct_best_encoder.pt
```

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