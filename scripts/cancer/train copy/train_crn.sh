#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ct
test=${1:-false}
gamma=${2:-4}
alpha=${3:-0.01}
gpu=${4:-3}

seeds=(10 101 1010 10101 101010 20 202 2020 20202 202020)
seeds=(10 101 1010 10101 101010)
# seeds=(10)
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_enc_dec.py +dataset=cancer_sim_cont +baselines=crn +baselines/crn_hparams/cancer_sim_domain_conf=${gamma}* exp.seed=${seed} exp.logging=False exp.alpha=${alpha} model.name=CRN/${alpha} exp.test=${test} exp.max_epochs=150
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_enc_dec.py +dataset=cancer_sim_cont +baselines=crn +baselines/crn_hparams/cancer_sim_domain_conf=${gamma}* exp.seed=${seed} exp.logging=False exp.alpha=${alpha} model.name=CRN/${alpha} exp.test=${test} exp.max_epochs=150 exp.rank=False
done
# exp.max_epochs=150