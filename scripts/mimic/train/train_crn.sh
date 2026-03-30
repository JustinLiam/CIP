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
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_enc_dec.py +dataset=mimic3_real +baselines=crn +baselines/crn_hparams/mimic3_real=diastolic_blood_pressure exp.seed=${seed} exp.logging=False exp.max_epochs=100 exp.alpha=${alpha} model.name=CRN/${alpha}
done