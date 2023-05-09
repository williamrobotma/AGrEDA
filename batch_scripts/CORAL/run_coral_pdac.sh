#!/usr/bin/env bash

# ./scripts/data/preprocess_pdac_GSE111672.py

# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50

# ./prep_data.py -s standard --dset dlpfc
# ./prep_data.py -s minmax --dset dlpfc

# ./prep_data.py -s standard --dset dlpfc --samp_split
# ./prep_data.py -s minmax --dset dlpfc --samp_split

# ./coral.py -f coral.yml -l "log.txt"
# ./eval_config.py --njobs -1 -f coral.yml -n CORAL

# ./coral.py -f coral_pdac.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral_pdac.yml --njobs -1  -t

# ./coral.py -f coral_pdac_peng.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral_pdac_peng.yml --njobs -1  -t

./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --samp_split --nmix 10

./reproduce_celldart.py -f bnfix_minmax_spotless.yml -l "log.txt"
./eval_config.py -n CellDART -f bnfix_minmax_spotless.yml --njobs -1 -t

# ./adda.py -f standard_bnfix_adam_beta1_5.yml -l "log.txt"
# ./eval_config.py -n ADDA -f standard_bnfix_adam_beta1_5.yml --njobs -1 -t

# ./dann.py -f dann.yml -l "log.txt"
# ./eval_config.py -n DANN -f dann.yml --njobs -1 -t

# ./coral.py -f coral.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral.yml --njobs -1 -t
