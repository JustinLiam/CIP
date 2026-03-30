#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate vcip
test=${1:-false}
gamma=${2:-4}
lambda_D=${3:-0.01}
gpu=${4:-2}

seeds=(10 101 1010 10101 101010 20 202 2020 20202 202020)
seeds=(10 101 1010 10101 101010)
# seeds=(10)
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_actin.py +dataset=cancer_sim_cont +baselines=actin +baselines/actin_hparams/cancer=${gamma}* exp.seed=${seed} exp.logging=False exp.lambda_D=${lambda_D} model.name=ACTIN/${lambda_D} exp.epochs=100 exp.test=${test} 
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_actin.py +dataset=cancer_sim_cont +baselines=actin +baselines/actin_hparams/cancer=${gamma}* exp.seed=${seed} exp.logging=False exp.lambda_D=${lambda_D} model.name=ACTIN/${lambda_D} exp.epochs=100 exp.test=${test} exp.rank=False
done