#!/usr/bin/env bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute dann.ipynb
./eval.py -n "DANN" -v "Standard1" -p -s standard -c 0 --njobs 16
