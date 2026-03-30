#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate vcip
lambda_D=${1:-0.01}
gpu=${2:-1}
seeds=(10 101 1010 10101 101010 20 202 2020 20202 202020)
seeds=(10 101 1010 10101 101010)
seeds=(10)
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_actin.py +dataset=mimic3_real +baselines=actin +baselines/actin_hparams/mimic3_real=diastolic_blood_pressure exp.seed=${seed} exp.logging=False exp.lambda_D=${lambda_D} model.name=ACTIN/${lambda_D} exp.epochs=100
done