#!/usr/bin/env bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
python -u eval.py -n "CellDART" -v "celldart1_bnfix" -s celldart -c 0  --njobs 32 --seed 1205
