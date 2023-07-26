#!/usr/bin/env bash

set -x

ps_seeds=(3679 343 25 234 98098)

echo "Original paper"
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
        --val_samp \
        --ps_seed=$ps_seed

done

echo "dlpfc"
# dlpfc
for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    
    echo ps_seed: $ps_seed
    ./prep_data.py -s minmax \
        --dset dlpfc \
        --st_id spatialLIBD \
        --sc_id GSE144136 \
        --nmarkers 40 \
        --nmix 3 \
        --samp_split \
        --val_samp \
        --ps_seed=$ps_seed

done

echo "spotless"
# spotless
for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    
    echo ps_seed: $ps_seed
    ./prep_data.py -s minmax \
        --dset mouse_cortex \
        --st_id spotless_mouse_cortex \
        --sc_id GSE115746 \
        --nmarkers 80 \
        --nmix 10 \
        --samp_split \
        --val_samp \
        --ps_seed=$ps_seed

done

echo "pdac"
# pdac
for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    
    echo ps_seed: $ps_seed
    ./prep_data.py -s minmax \
        --dset pdac \
        --st_id GSE111672 \
        --sc_id CA001063 \
        --nmarkers 80 \
        --nmix 50 \
        --one_model \
        --ps_seed=$ps_seed

done