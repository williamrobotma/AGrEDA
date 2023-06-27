#!/usr/bin/env bash

set -x

# python -m src.da_utils.scripts.data.preprocessing_spotless
# python -m src.da_utils.scripts.data.preprocessing_mouse_GSE115746

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_celldart.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_adda.ipynb

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_dann.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_coral.ipynb

# ./batch_scripts/run_prep.sh

nohup ./batch_scripts/CellDART/run_CellDART-spotless.sh &> logs/CellDART/generated_spotless/run.out &
nohup ./batch_scripts/ADDA/run_adda-spotless.sh &> logs/ADDA/generated_spotless/run.out &
# nohup ./batch_scripts/CORAL/run_coral-spotless.sh &> logs/CORAL/generated_spotless/run.out &
nohup ./batch_scripts/DANN/run_dann-spotless.sh &> logs/DANN/generated_spotless/run.out &
