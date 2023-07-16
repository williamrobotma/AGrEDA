#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA
mkdir -p logs/CellDART
mkdir -p logs/CORAL
mkdir -p logs/DANN

# nohup ./batch_scripts/ADDA/run_adda-pdac-FINAL.sh 1> logs/ADDA/run-pdac-FINAL.out 2> logs/ADDA/run-pdac-FINAL.err &
# nohup ./batch_scripts/CellDART/run_CellDART-pdac-FINAL.sh 1> logs/CellDART/run-pdac-FINAL.out 2> logs/CellDART/run-pdac-FINAL.err &
# nohup ./batch_scripts/CORAL/run_coral-pdac-FINAL.sh 1> logs/CORAL/run-pdac-FINAL.out 2> logs/CORAL/run-pdac-FINAL.err &
# nohup ./batch_scripts/DANN/run_dann-pdac-FINAL.sh 1> logs/DANN/run-pdac-FINAL.out 2> logs/DANN/run-pdac-FINAL.err &

# ./batch_scripts/ADDA/run_adda-pdac-FINAL.sh 1> logs/ADDA/run-pdac-FINAL.out 2> logs/ADDA/run-pdac-FINAL.err
# ./batch_scripts/CellDART/run_CellDART-pdac-FINAL.sh 1> logs/CellDART/run-pdac-FINAL.out 2> logs/CellDART/run-pdac-FINAL.err
# ./batch_scripts/CORAL/run_coral-pdac-FINAL.sh 1> logs/CORAL/run-pdac-FINAL.out 2> logs/CORAL/run-pdac-FINAL.err
./batch_scripts/DANN/run_dann-pdac-FINAL.sh 1> logs/DANN/run-pdac-FINAL.out 2> logs/DANN/run-pdac-FINAL.err
./batch_scripts/DANN/run_dann-pdac-FINAL-MINMAX.sh 1> logs/DANN/run-pdac-FINAL-MINMAX.out 2> logs/DANN/run-pdac-FINAL-MINMAX.err
./batch_scripts/DANN/run_dann-pdac-FINAL-MINMAX-2.sh 1> logs/DANN/run-pdac-FINAL-MINMAX-2.out 2> logs/DANN/run-pdac-FINAL-MINMAX-2.err
