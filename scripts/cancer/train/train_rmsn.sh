#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ct
test=${1:-false}
gamma=${2:-4}
lambda_entropy=${3:-1}
max=${4:-10}
gpu=${5:-0}

seeds=(10 101 1010 10101 101010 20 202 2020 20202 202020)
seeds=(10 101 1010 10101 101010)
# seeds=(101)
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_rmsn.py +dataset=cancer_sim_cont +baselines=rmsn +baselines/rmsn_hparams/cancer_sim=${gamma}* exp.seed=${seed} exp.logging=False exp.max_epochs=100 model.name=RMSN exp.test=${test} 
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_rmsn.py +dataset=cancer_sim_cont +baselines=rmsn +baselines/rmsn_hparams/cancer_sim=${gamma}* exp.seed=${seed} exp.logging=False exp.max_epochs=100 model.name=RMSN exp.test=${test} exp.rank=False
    if [ "$gamma" -eq 4 ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_rmsn.py +dataset=cancer_sim_cont +baselines=rmsn +baselines/rmsn_hparams/cancer_sim=${gamma}* exp.seed=${seed} exp.logging=False exp.max_epochs=100 exp.lambda_entropy=1 exp.max=0.1 model.name=RMSN_ab exp.test=${test} 
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_rmsn.py +dataset=cancer_sim_cont +baselines=rmsn +baselines/rmsn_hparams/cancer_sim=${gamma}* exp.seed=${seed} exp.logging=False exp.max_epochs=100 exp.lambda_entropy=1 exp.max=0.1 model.name=RMSN_ab exp.test=${test} exp.rank=False
    fi
done