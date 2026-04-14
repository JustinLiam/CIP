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

The results will be saved in the `results/all/` directory, matching the experimental results presented in the paper.

###Experimental Platform

To ensure consistency and fairness in all experimental comparisons, both VCIP and all baseline models are tested on the same computational setup:

**Hardware Specifications**

- **Processor (CPU)**: AMD Ryzen 9 5900X 12-Core Processor
- **Graphics Processing Units (GPUs)**: 4x NVIDIA GeForce RTX 4080 Ti