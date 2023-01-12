#!/usr/bin/env bash

# ./prep_data.py --njobs 64
# ./prep_data.py -s standard  --njobs 64
# ./prep_data.py -s standard -a  --njobs 64
# ./prep_data.py -s standard --stsplit  --njobs 64
# ./prep_data.py -s standard -a --stsplit  --njobs 64
# ./prep_data.py --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard  --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard -a  --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard --stsplit  --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard -a --stsplit --njobs 64 --nmarkers 100000
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
# ./eval.py -d "data/preprocessed_markers_celldart" -n "CellDART" -v "bn_fix"
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute adda.ipynb
python -u eval.py -n "ADDA" -v "celldart" -s "celldart" --njobs 32  -c 0 --seed 72
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute dann.ipynb
# ./eval.py -d "data/preprocessed_markers_standard" -n "DANN" -v "Standard1" -p
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc_allgenes.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc_st.ipynb
