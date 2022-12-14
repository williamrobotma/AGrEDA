#!/usr/bin/env bash
# ./prep_data.py
# ./prep_data.py -s standard
./prep_data.py -s standard -a
./prep_data.py -s standard --stsplit
./prep_data.py -s standard -a --stsplit
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
./eval.py -d "data/preprocessed_markers_celldart" -n "CellDART" -v "bn_fix"
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute adda.ipynb
./eval.py -d "data/preprocessed_markers_standard" -n "ADDA" -v "Standard1"
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute dann.ipynb
./eval.py -d "data/preprocessed_markers_standard" -n "DANN" -v "Standard1" -p
