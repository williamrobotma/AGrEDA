#!/usr/bin/env bash
# python -u prep_data.py -s standard --njobs 20 --nspots 100000
python -u dann.py -f "standard_batchnorm_100000spots.yml" --njobs 20
python -u eval_config.py -n "DANN" -f "standard_batchnorm_100000spots.yml" -p --njobs 20
# python -u eval.py -n "DANN" -v "Standard1" -p -s standard -c 2 --njobs 32 --seed 25098
