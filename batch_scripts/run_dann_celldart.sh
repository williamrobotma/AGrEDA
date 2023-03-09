#!/usr/bin/env bash

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
# python -u eval.py -n "CellDART" -v "celldart1_nobnfix" -s celldart -c 0  --njobs 32 --seed 1205
python -u reproduce_celldart.py -f "bnfix_minmax.yml" -l "log.txt"
python -u eval_config.py -n "CellDART" -f "bnfix_minmax.yml" --njobs -1

