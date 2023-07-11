#!/usr/bin/env bash

set -x

ps_seeds=(3679 343 25 234 98098)

# original paper
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

for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    
    echo ps_seed: $ps_seed
    ./prep_data.py -s minmax \
        --dset mouse_cortex \
        --st_id spotless_mouse_cortex \
        --sc_id GSE115746 \
        --nmarkers 40 \
        --nmix 5 \
        --samp_split \
        --ps_seed=$ps_seed

done