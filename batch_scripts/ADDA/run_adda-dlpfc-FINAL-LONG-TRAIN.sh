#!/usr/bin/env bash

start=`date +%s`

CONFIG_FILE="adda-final-dlpfc-ht-long-train.yml"

mkdir -p logs/ADDA

echo "ADDA config file: ${CONFIG_FILE}"

ps_seeds=(3679 343 25 234 98098)

model_seeds=(2353 24385 284 86322 98237)

# ./prep_data.py -s minmax \
#     --dset dlpfc \
#     --st_id spatialLIBD \
#     --sc_id GSE144136 \
#     --nmarkers 40 \
#     --nmix 3 \
#     --samp_split \
#     --val_samp

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
    -e \
    -t \
    --model_dir="model_FINAL" \
    --results_dir="results_FINAL" \
    --njobs 16 \
    -c 0

echo "Evaluating again"
./eval_config.py \
    -n ADDA \
    -f "${CONFIG_FILE}" \
    -cdir "configs" \
    -t \
    --model_dir="model_FINAL" \
    --results_dir="results_FINAL" \
    --njobs 16 \
    -c 0

