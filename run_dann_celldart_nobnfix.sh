#!/usr/bin/env bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart_nobnfix.ipynb
./eval.py -n "CellDART" -v "celldart1_nobnfix" -s celldart -c 3  --njobs 16
