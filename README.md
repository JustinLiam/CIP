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
cd /path/to/VCIP-ICML-main
source /path/to/conda/etc/profile.d/conda.sh
conda activate vcip

GRID_ROOT=results/tumor/stable_gift_protocol_best_$(date +%Y%m%d) \
GRID_SEEDS="20 202 2020 20202 202020" \
TEST_SPLIT=true \
FORCE=1 \
bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4
```

Current default hyperparameters are stored in `configs/model/vcip.yaml`: AWR actor update, expectile actor BC (`iql_actor_bc_loss=expectile`, `iql_actor_bc_expectile=0.8`), `iql_weight_max=1.0`, `em_warmup_outer_iters=2`, `em_val_metric=rmse_uns`, and max-over-`tau=1,...,6` validation selection. Keep `iql_eval_action_shift=0.0` for main experiments.

For a single diagnostic seed, run:

```bash
GRID_ROOT=results/tumor/stable_gift_protocol_seed2_$(date +%Y%m%d) \
GRID_SEEDS="2" \
TEST_SPLIT=true \
IQL_WEIGHT_MAX=5.0 \
FORCE=1 \
bash scripts/cancer/train/run_em_iql_local_global_gift_protocol.sh 0 4
```

Do not choose the final seed set based on test-set performance. If seed 2 is useful, report it as an additional seed or predefine the replacement before reading test metrics.

For a single current-method smoke run:

```bash
bash scripts/cancer/train/train_ct_iql_em.sh 0 4 10
```

For MIMIC synthetic experiments, use `+dataset=mimic3_synthetic_gift`, which is the GIFT-aligned semi-synthetic MIMIC data path. The legacy `mimic3_real` and two-stage `train_iql_planner.py` workflows have been removed from the supported experiment surface because they do not implement the current one-stage CT+IQL EM method.

### EpiABM / EpiCF Data Generation

The modified EpiABM simulator used by the CRIPO experiments is bundled under
`data_generation/epi_diff_abm`; reviewers do not need `external_repos`, git
submodules, or patch application. See
`data_generation/epi_diff_abm/README.md` for upstream asset preparation,
calibration, cache generation, and data-placement rules.

Generated EpiABM assets remain outside version control. The canonical layout is:

```text
data_generation/epi_diff_abm/data/           raw and upstream-prepared assets
data_generation/epi_diff_abm/populations/    generated population packages
data_generation/epi_diff_abm/result_graphs/  calibrated_params.txt and diagnostics
data/processed/epi_abm/                      CRIPO-ready dataset caches
results/epi_abm/                             logs, smoke tests, and evaluations
```

Reviewer-facing reproduction flow:

```bash
# 1. Place downloaded EpiCF/EpiABM assets under data_generation/epi_diff_abm/.
# 2. If upstream assets are absent, generate them with API keys in the environment.
bash scripts/epi_abm/prepare_epicf_counties_upstream.sh prep

# 3. Calibrate counties; outputs stay under data_generation/epi_diff_abm/result_graphs/.
python scripts/epi_abm/run_parallel_calibration_pool.py

# 4. Build the CRIPO-ready factual-only multi-county cache.
python scripts/epi_abm/build_multi_county_cache.py

# 5. Run a minimal but end-to-end release smoke test.
python scripts/epi_abm/smoke_test_release.py --device cuda
```

For a faster 01045-only smoke check after assets are already present:

```bash
python scripts/epi_abm/smoke_test_release.py --device cuda --rollout-days 14 --cache-days 14
```

The non-EpiABM results will be saved in the configured `results/` directory,
matching the experimental results presented in the paper.

###Experimental Platform

To ensure consistency and fairness in all experimental comparisons, both VCIP and all baseline models are tested on the same computational setup:

**Hardware Specifications**

- **Processor (CPU)**: AMD Ryzen 9 5900X 12-Core Processor
- **Graphics Processing Units (GPUs)**: 4x NVIDIA GeForce RTX 4080 Ti
