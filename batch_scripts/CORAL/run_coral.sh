#!/usr/bin/env bash

./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
./coral.py -f coral_pdac.yml
./eval_config.py --njobs -1 -f coral_pdac.yml -n CORAL