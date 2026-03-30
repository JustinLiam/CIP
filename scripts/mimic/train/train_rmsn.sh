#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ct
alpha=${1:-0.01}
gpu=${2:-0}


seeds=(10 101 1010 10101 101010 20 202 2020 20202 202020)
seeds=(10 101 1010 10101 101010)
seeds=(10)

for seed in "${seeds[@]}"
do
    python runnables/train_rmsn.py +dataset=mimic3_real +baselines=rmsn +baselines/rmsn_hparams/mimic3_real=diastolic_blood_pressure exp.seed=${seed} exp.logging=False exp.max_epochs=100 exp.lambda_entropy=0.1 exp.max=100 model.name=RMSN
done