# README

## Project Description

<img src="./imgs/comparison.png" alt="comparison" style="zoom:150%;" />

This project develops a novel framework called Variational Counterfactual Intervention Planning (VCIP) to address temporal counterfactual reasoning in dynamic systems. Unlike traditional approaches that focus on predicting future outcomes, VCIP aims to identify optimal intervention sequences that guide systems toward specific target outcomes. The framework employs variational inference to directly model the conditional likelihood of achieving target outcomes, avoiding error accumulation issues common in explicit prediction methods. By leveraging the g-formula, VCIP establishes theoretical connections between observational and interventional distributions, enabling reliable training on real-world observational data. The methodology has significant applications in healthcare, economics, and other domains where precise intervention planning is crucial for achieving specific target states.

## Installation Guide

We recommend using conda virtual environments for a cleaner and more controlled development environment. This project requires two separate environments: one for the VCIP implementation and another for the baseline methods.

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

To reproduce the experimental results from the paper:

1. Run the training script:
```bash
./train_all.sh
```

2. Generate results:
```bash
python results/all/read_data.py
```

The results will be saved in the `results/all/` directory, matching the experimental results presented in the paper.

###Experimental Platform

To ensure consistency and fairness in all experimental comparisons, both VCIP and all baseline models are tested on the same computational setup:

**Hardware Specifications**

- **Processor (CPU)**: AMD Ryzen 9 5900X 12-Core Processor
- **Graphics Processing Units (GPUs)**: 4x NVIDIA GeForce RTX 4080 Ti