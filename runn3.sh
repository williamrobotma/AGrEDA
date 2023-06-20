#!/usr/bin/env bash

# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
# ./prep_data.py -s standard
# ./prep_data.py -s minmax

# ./coral.py -f coral_pdac.yml
# ./eval_config.py --njobs 32 -f coral_pdac.yml -n CORAL

# ./coral.py -f coral.yml
# ./eval_config.py --njobs 32 -f coral.yml -n CORAL

# ./reproduce_celldart.py -f bnfix_minmax.yml
# ./eval_config.py --njobs 32 -f bnfix_minmax.yml -n CellDART

# ./reproduce_celldart.py -f nobnfix_minmax.yml
# ./eval_config.py --njobs 32 -f nobnfix_minmax.yml -n CellDART

./adda.py -f minmax_bnfix_adam_beta1_5.yml -c 2
./eval_config.py --njobs 32 -f minmax_bnfix_adam_beta1_5.yml -n ADDA -c 2

./adda.py -f standard_bnfix_adam_beta1_5.yml -c 2
./eval_config.py --njobs 32 -f standard_bnfix_adam_beta1_5.yml -n ADDA -c 2

# ./dann.py -f dann_legacy.yml
# ./eval_config.py --njobs 32 -f dann_legacy.yml -n DANN

# ./dann.py -f dann.yml
# ./eval_config.py --njobs 32 -f dann.yml -n DANN

