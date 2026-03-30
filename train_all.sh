gammas=(1 2 3 4)
model=vae

if [[ ! $model =~ ^(vae|rmsn|crn|ct|actin)$ ]]; then
    echo "Error: Invalid model parameter. Please use vae, rmsn, crn, ct, or actin"
    exit 1
fi


for gamma in "${gammas[@]}"
do
    echo "Running ${model} training with gamma = ${gamma}"
    ./scripts/cancer/train/train_${model}.sh false ${gamma}
    ./scripts/cancer/train/train_${model}.sh true ${gamma}
    echo "Completed ${model} training with gamma = ${gamma}"
    echo "----------------------------------------"
done

echo "All ${model} training completed for all gamma values"