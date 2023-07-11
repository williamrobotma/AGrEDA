#!/usr/bin/env bash

start=`date +%s`

CONFIG_FILE="coral-final-spotless-ht.yml"

mkdir -p logs/CORAL

echo "CORAL config file: ${CONFIG_FILE}"

ps_seeds=(3679 343 25 234 98098)

model_seeds=(2353 24385 284 86322 98237)

./prep_data.py -s standard \
    --dset mouse_cortex \
    --st_id spotless_mouse_cortex \
    --sc_id GSE115746 \
    --nmarkers 40 \
    --nmix 5 \
    --samp_split

python -u coral.py \
    -f "${CONFIG_FILE}" \
    -l "log.txt" \
    -cdir "configs" \
    --model_dir="model_FINAL" \
    -c 2

echo "Evaluating"
./eval_config.py \
    -n CORAL \
    -f "${CONFIG_FILE}" \
    -cdir "configs" \
    -t \
    --model_dir="model_FINAL" \
    --results_dir="results_FINAL" \
    --njobs 16 \
    -c 2

for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    model_seed=${model_seeds[$i]}
    
    echo ps_seed: $ps_seed model_seed: $model_seed
    ./prep_data.py -s standard \
        --dset mouse_cortex \
        --st_id spotless_mouse_cortex \
        --sc_id GSE115746 \
        --nmarkers 40 \
        --nmix 5 \
        --samp_split \
        --ps_seed=$ps_seed


    python -u coral.py \
        -f "${CONFIG_FILE}" \
        -l "log.txt" \
        -cdir "configs" \
        --model_dir="model_FINAL/std" \
        --seed_override=$model_seed \
        --ps_seed=$ps_seed \
        -c 2

    echo "Evaluating"
    ./eval_config.py \
        -n CORAL \
        -f "${CONFIG_FILE}" \
        -cdir "configs" \
        -t \
        --model_dir="model_FINAL/std" \
        --seed_override=$model_seed \
        --ps_seed=$ps_seed \
        --results_dir="results_FINAL/std" \
        --njobs 16 \
        -c 2
done

end=`date +%s`
echo "script time: $(($end-$start))"
