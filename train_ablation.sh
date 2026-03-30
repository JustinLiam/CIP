gammas=(1 2 3 4)

for gamma in "${gammas[@]}"
do
    ./scripts/cancer/train/train_vae.sh ${gamma}
    ./scripts/cancer/train/train_rmsn.sh ${gamma}
    ./scripts/cancer/train/train_ct.sh ${gamma}
    ./scripts/cancer/train/train_actin.sh ${gamma}
    ./scripts/cancer/train/train_crn.sh ${gamma}
done
