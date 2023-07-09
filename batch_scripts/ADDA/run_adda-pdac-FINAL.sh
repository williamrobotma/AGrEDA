#!/usr/bin/env bash

start=`date +%s`

CONFIG_FILE="adda-final-pdac-ht.yml"

mkdir -p logs/ADDA

echo "ADDA config file: ${CONFIG_FILE}"

ps_seeds=(3679 343 25 234 98098)

model_seeds=(2353 24385 284 86322 98237)

./prep_data.py -s standard \
    --dset pdac \
    --st_id GSE111672 \
    --sc_id CA001063 \
    --nmarkers 20 \
    --nmix 50 \
    --one_model

python -u adda.py \
    -f "${CONFIG_FILE}" \
    -l "log.txt" \
    -cdir "configs" \
    --model_dir="model_FINAL" \
    -c 0

echo "Evaluating"
./eval_config.py \
    -n ADDA \
    -f "${CONFIG_FILE}" \
    -cdir "configs" \
    --early_stopping -t \
    --model_dir="model_FINAL" \
    --results_dir="results_FINAL" \
    --njobs 16 \
    -c 0

for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    model_seed=${model_seeds[$i]}
    
    echo ps_seed: $ps_seed model_seed: $model_seed
    ./prep_data.py -s standard \
        --dset pdac \
        --st_id GSE111672 \
        --sc_id CA001063 \
        --nmarkers 20 \
        --nmix 50 \
        --one_model \
        --ps_seed=$ps_seed

    python -u adda.py \
        -f "${CONFIG_FILE}" \
        -l "log.txt" \
        -cdir "configs" \
        --model_dir="model_FINAL/std" \
        --seed_override=$model_seed \
        --ps_seed=$ps_seed \
        -c 0

    echo "Evaluating"
    ./eval_config.py \
        -n ADDA \
        -f "${CONFIG_FILE}" \
        -cdir "configs" \
        --early_stopping -t \
        --model_dir="model_FINAL/std" \
        --seed_override=$model_seed \
        --ps_seed=$ps_seed \
        --results_dir="results_FINAL/std" \
        --njobs 16 \
        -c 0
done

end=`date +%s`
echo "script time: $(($end-$start))"
