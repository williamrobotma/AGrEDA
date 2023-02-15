#!/usr/bin/env bash
# ./prep_data.py
# ./prep_data.py -s standard
# ./prep_data.py -s standard -a
# ./prep_data.py -s standard --stsplit
# ./prep_data.py -s standard -a --stsplit
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
# ./eval.py -d "data/preprocessed_markers_celldart" -n "CellDART" -v "bn_fix"
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute adda.ipynb
# ./eval.py -d "data/preprocessed_markers_standard" -n "ADDA" -v "Standard1"
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute dann.ipynb
# ./eval.py -d "data/preprocessed_markers_standard" -n "DANN" -v "Standard1" -p
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc_allgenes.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc_st.ipynb

python reproduce_celldart.py -f "bnfix_minmax.yml"
python eval_config.py -n "CellDART" -f "bnfix_minmax.yml"

python reproduce_celldart.py -f "nobnfix_minmax.yml"
python eval_config.py -n "CellDART" -f "nobnfix_minmax.yml"

python adda.py -f "minmax_bnfix_adam_beta1_5.yml"
python eval_config.py -n "ADDA" -f "minmax_bnfix_adam_beta1_5.yml"

python adda.py -f "minmax_bnfix_adam_beta1_9.yml"
python eval_config.py -n "ADDA" -f "minmax_bnfix_adam_beta1_9.yml"

python adda.py -f "minmax_nobnfix_adam_beta1_5.yml"
python eval_config.py -n "ADDA" -f "minmax_nobnfix_adam_beta1_5.yml"

python adda.py -f "minmax_nobnfix_adam_beta1_9.yml"
python eval_config.py -n "ADDA" -f "minmax_nobnfix_adam_beta1_9.yml"


# python -u adda.py -f "standard_nobnfix_adam_beta1_9.yml" -c 0  --njobs 16 > "logs/adda-standard_nobnfix_adam_beta1_9.out"
# python -u eval_config.py -n "ADDA" -f "standard_nobnfix_adam_beta1_9.yml" -c 0  --njobs 32

# python -u adda.py -f "unscaled_bnfix_adam_beta1_5.yml" -c 0  --njobs 16 > "logs/adda-unscaled_bnfix_adam_beta1_5.out"
# python -u eval_config.py -n "ADDA" -f "unscaled_bnfix_adam_beta1_5.yml" -c 0  --njobs 32

# python -u adda.py -f "unscaled_bnfix_adam_beta1_9.yml" -c 0  --njobs 16 > "logs/adda-unscaled_bnfix_adam_beta1_9.out"
# python -u eval_config.py -n "ADDA" -f "unscaled_bnfix_adam_beta1_9.yml" -c 0  --njobs 32

# python -u adda.py -f "unscaled_nobnfix_adam_beta1_5.yml" -c 0  --njobs 16 > "logs/adda-unscaled_nobnfix_adam_beta1_5.out"
# python -u eval_config.py -n "ADDA" -f "unscaled_nobnfix_adam_beta1_5.yml" -c 0  --njobs 32

# python -u adda.py -f "unscaled_nobnfix_adam_beta1_9.yml" -c 0  --njobs 16 > "logs/adda-unscaled_nobnfix_adam_beta1_9.out"
# python -u eval_config.py -n "ADDA" -f "unscaled_nobnfix_adam_beta1_9.yml" -c 0  --njobs 32