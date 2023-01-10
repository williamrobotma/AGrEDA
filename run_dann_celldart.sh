#!/usr/bin/env bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
./eval.py -n "CellDART" -v "celldart1_bnfix" -s celldart -c 2  --njobs 16
