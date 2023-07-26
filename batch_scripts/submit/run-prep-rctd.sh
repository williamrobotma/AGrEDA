#!/usr/bin/env bash

# ps_seeds=(3679 343 25 234 98098)
# for i in "${!ps_seeds[@]}"; do
    # ps_seed=${ps_seeds[$i]}
./prep_data.py -s unscaled --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --samp_split --val_samp --allgenes --no_process
./prep_data.py -s unscaled --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --samp_split --val_samp --allgenes --no_process
./prep_data.py -s unscaled --dset pdac --st_id GSE111672 --sc_id CA001063 --one_model --allgenes --no_process