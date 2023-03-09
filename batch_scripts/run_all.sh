#!/usr/bin/env bash

# ./prep_data.py --njobs -1
# ./prep_data.py -s standard  --njobs -1
# ./prep_data.py -s standard -a  --njobs -1
# ./prep_data.py -s standard --stsplit  --njobs -1
# ./prep_data.py -s standard -a --stsplit  --njobs -1
# ./prep_data.py --njobs -1 --nspots 100000
# ./prep_data.py -s standard  --njobs -1 --nspots 100000
# ./prep_data.py -s standard -a  --njobs -1 --nspots 100000
# ./prep_data.py -s standard --stsplit  --njobs -1 --nspots 100000
# ./prep_data.py -s standard -a --stsplit --njobs -1 --nspots 100000

# nohup ./run_dann.sh > run_dann.log &
# nohup ./run_dann_celldart.sh > run_dann_celldart.log &
# nohup ./run_dann_celldart_nobnfix.sh > run_dann_celldart_nobnfix.log &

# ./coral.py -f coral.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral.yml --njobs -1 

# ./coral.py -f coral_pdac.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral_pdac.yml --njobs -1

# python -u adda.py -f "standard_bnfix_adam_beta1_5.yml" -l "log.txt"
# python -u eval_config.py -n "ADDA" -f "standard_bnfix_adam_beta1_5.yml" --njobs -1

# python -u reproduce_celldart.py -f "bnfix_minmax.yml" -l "log.txt"
# python -u eval_config.py -n "CellDART" -f "bnfix_minmax.yml" --njobs -1

python -u dann.py -f "dann_legacy.yml" -l "log.txt"
python -u eval_config.py -n "DANN" -f "dann_legacy.yml" --njobs -1 
