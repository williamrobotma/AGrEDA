#!/usr/bin/env bash

./prep_data.py --njobs -1
./prep_data.py -s standard  --njobs -1
./prep_data.py -s standard -a  --njobs -1
./prep_data.py -s standard --stsplit  --njobs -1
./prep_data.py -s standard -a --stsplit  --njobs -1
./prep_data.py --njobs -1 --nspots 100000
./prep_data.py -s standard  --njobs -1 --nspots 100000
./prep_data.py -s standard -a  --njobs -1 --nspots 100000
./prep_data.py -s standard --stsplit  --njobs -1 --nspots 100000
./prep_data.py -s standard -a --stsplit --njobs -1 --nspots 100000

nohup ./run_dann.sh > run_dann.log &
nohup ./run_dann_celldart.sh > run_dann_celldart.log &
nohup ./run_dann_celldart_nobnfix.sh > run_dann_celldart_nobnfix.log &
