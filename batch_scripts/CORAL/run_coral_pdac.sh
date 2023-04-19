#!/usr/bin/env bash

# ./scripts/data/preprocess_pdac_GSE111672.py

# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50

# ./coral.py -f coral_pdac.yml -l "log.txt"
./eval_config.py -n CORAL -f coral_pdac.yml --njobs -1 

./coral.py -f coral_pdac_peng.yml -l "log.txt"
./eval_config.py -n CORAL -f coral_pdac_peng.yml --njobs -1 
