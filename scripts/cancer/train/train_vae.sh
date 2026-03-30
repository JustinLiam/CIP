eval "$(conda shell.bash hook)"
conda activate vcip
test=${1:-false}  # 在./scripts/cancer/train/train_${model}.sh false ${gamma}中的参数false 或true 赋值给这个test， 如果没有传入这个参数，则默认为false
gamma=${2:-4}  # 运行脚本时如果传了第 (2) 个参数（例如 ./train_vae.sh false 8），则 gamma=8，如果没有传入第（2）个参数，则默认为4
#gpu=${3:-3}
gpu=${3:-0}
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