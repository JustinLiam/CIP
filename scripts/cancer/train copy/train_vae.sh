eval "$(conda shell.bash hook)"
conda activate vcip
test=${1:-false}
gamma=${2:-4}
gpu=${3:-1}
seeds=(10 101 1010 10101 101010 20 202 2020 20202 202020)
seeds=(10 101 1010 10101 101010)
# seeds=(10)
for seed in "${seeds[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_vae.py +dataset=cancer_sim_cont +model=vcip +model/hparams/cancer=${gamma}* exp.seed=${seed} exp.epochs=100 dataset.coeff=${gamma} model.name=VCIP exp.test=${test}
    CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_vae.py +dataset=cancer_sim_cont +model=vcip +model/hparams/cancer=${gamma}* exp.seed=${seed} exp.epochs=100 dataset.coeff=${gamma} model.name=VCIP exp.test=${test} exp.rank=False
    if [ "$gamma" -eq 4 ]; then
        CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_vae.py +dataset=cancer_sim_cont +model=vcip +model/hparams/cancer=${gamma}* exp.seed=${seed} exp.epochs=100 dataset.coeff=${gamma} exp.lambda_step=0 exp.lambda_action=0 model.name=VCIP_ablation exp.test=${test}
        
        CUDA_VISIBLE_DEVICES=${gpu} python runnables/train_vae.py +dataset=cancer_sim_cont +model=vcip +model/hparams/cancer=${gamma}* exp.seed=${seed} exp.epochs=100 dataset.coeff=${gamma} exp.lambda_step=0 exp.lambda_action=0 model.name=VCIP_ablation exp.test=${test} exp.rank=False
    fi
done