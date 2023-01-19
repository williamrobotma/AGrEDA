#!/usr/bin/env bash

python -u dann.py -f "dann.yml" -c 3  --njobs 16 > "logs/dann_dann.out"
python -u eval_config.py -n "DANN" -f "dann.yml" -c 3  -p --njobs 32
# python -u eval.py -n "DANN" -v "Standard1" -p -s standard -c 2 --njobs 32 --seed 25098
