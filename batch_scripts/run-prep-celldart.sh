#!/usr/bin/env bash

set -x

ps_seeds=(3679 343 25 234 98098)

for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    
    echo ps_seed: $ps_seed
    ./prep_data.py -s minmax \
        --dset dlpfc \
        --st_id spatialLIBD \
        --sc_id GSE144136 \
        --nmarkers 20 \
        --nmix 8 \
        --samp_split \
        --ps_seed=$ps_seed

done