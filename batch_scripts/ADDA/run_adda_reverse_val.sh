#!/bin/bash

# python prep_data.py -s minmax --dset dlpfc --st_id "spatialLIBD" --sc_id "GSE144136" --nmix 8 --nmarkers 20 --samp_split --njobs -1
# python prep_data.py -s standard --dset dlpfc --st_id "spatialLIBD" --sc_id "GSE144136" --nmix 8 --nmarkers 20 --samp_split --njobs -1

# ./adda.py -f "standard_bnfix_adam_beta1_5_samp_split.yml" -l "log.txt"
./eval_config.py -n "ADDA" -f  "basic_config.yml" --test --reverse_val --njobs -1

# ./reproduce_celldart.py -f "bnfix_minmax_samp_split.yml" -l "log.txt"
# ./eval_config.py -n "CellDART" -f  "bnfix_minmax_samp_split.yml" --test --njobs -1

# ./dann.py -f "dann.yml" -l "log.txt"
# ./eval_config.py -n "DANN" -f  "dann.yml" --test --njobs -1

./coral.py -f "coral.yml" -l "log.txt"
./eval_config.py -n "CORAL" -f  "coral.yml" --test --njobs -1