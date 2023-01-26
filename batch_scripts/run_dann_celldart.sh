#!/usr/bin/env bash

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
# python -u eval.py -n "CellDART" -v "celldart1_nobnfix" -s celldart -c 0  --njobs 32 --seed 1205
python -u reproduce_celldart.py -f "celldart1_nobnfix.yml" -c 0  --njobs 16 > "logs/reproduce_celldart_nobnfix.out"
python -u eval_config.py -n "CellDART" -f "celldart1_nobnfix.yml" -c 0  --njobs 32
python -u reproduce_celldart.py -f "celldart1_bnfix.yml" -c 0  --njobs 16 > "logs/reproduce_celldart_bnfix.out"
python -u eval_config.py -n "CellDART" -f "celldart1_bnfix.yml" -c 0  --njobs 32
