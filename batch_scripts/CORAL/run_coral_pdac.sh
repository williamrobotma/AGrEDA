#!/usr/bin/env bash

# ./scripts/data/preprocess_pdac_GSE111672.py

# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
# ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50

# ./prep_data.py -s standard --dset dlpfc
# ./prep_data.py -s minmax --dset dlpfc

# ./prep_data.py -s standard --dset dlpfc --samp_split
# ./prep_data.py -s minmax --dset dlpfc --samp_split
# python -m src.da_utils.scripts.data.preprocessing_spotless
# python -m src.da_utils.scripts.data.preprocessing_mouse_GSE115746

./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --samp_split --nmix 10
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --samp_split --nmix 10
./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --samp_split --nmix 10
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --samp_split --nmix 10

# ./coral.py -f coral.yml -l "log.txt"
# ./eval_config.py --njobs -1 -f coral.yml -n CORAL

# ./coral.py -f coral_pdac.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral_pdac.yml --njobs -1  -t

# ./coral.py -f coral_pdac_peng.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral_pdac_peng.yml --njobs -1  -t





# ./reproduce_celldart.py -f bnfix_minmax_samp_split.yml -l "log.txt"
# ./eval_config.py -n CellDART -f bnfix_minmax_samp_split.yml --njobs -1 -t

# ./adda.py -f standard_bnfix_adam_beta1_5.yml -l "log.txt"
# ./eval_config.py -n ADDA -f standard_bnfix_adam_beta1_5.yml --njobs -1 -t

# ./dann.py -f dann.yml -l "log.txt"
# ./eval_config.py -n DANN -f dann.yml --njobs -1 -t


# ./dann.py -f standard_batchnorm_sameasdann20000spots.yml -l "log.txt"
# ./eval_config.py -n DANN -f standard_batchnorm_sameasdann20000spots.yml --njobs -1 -t

# ./reproduce_celldart.py -f bnfix_minmax_spotless_sc.yml -l "log.txt"
# ./eval_config.py -n CellDART -f bnfix_minmax_spotless_sc.yml --njobs -1 -t

# ./coral.py -f coral_spotless_sc.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral_spotless_sc.yml --njobs -1 -t

# ./adda.py -f standard_bnfix_adam_beta1_5_spotless_sc.yml -l "log.txt"
# ./eval_config.py -n ADDA -f standard_bnfix_adam_beta1_5_spotless_sc.yml --njobs -1 -t

# ./dann.py -f dann_spotless_sc.yml -l "log.txt"
# ./eval_config.py -n DANN -f dann_spotless_sc.yml --njobs -1 -t