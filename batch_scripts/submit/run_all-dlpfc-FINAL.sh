#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA
mkdir -p logs/CellDART
mkdir -p logs/CORAL
mkdir -p logs/DANN

nohup ./batch_scripts/ADDA/run_adda-dlpfc-FINAL.sh 1> logs/ADDA/run-dlpfc-FINAL.out 2> logs/ADDA/run-dlpfc-FINAL.err &
nohup ./batch_scripts/CellDART/run_CellDART-dlpfc-FINAL.sh 1> logs/CellDART/run-dlpfc-FINAL.out 2> logs/CellDART/run-dlpfc-FINAL.err &
nohup ./batch_scripts/CORAL/run_coral-dlpfc-FINAL.sh 1> logs/CORAL/run-dlpfc-FINAL.out 2> logs/CORAL/run-dlpfc-FINAL.err &
nohup ./batch_scripts/DANN/run_dann-dlpfc-FINAL.sh 1> logs/DANN/run-dlpfc-FINAL.out 2> logs/DANN/run-dlpfc-FINAL.err &
