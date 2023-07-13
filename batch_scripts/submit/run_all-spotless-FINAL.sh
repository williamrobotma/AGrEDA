#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA
mkdir -p logs/CellDART
mkdir -p logs/CORAL
mkdir -p logs/DANN

nohup ./batch_scripts/ADDA/run_adda-spotless-FINAL.sh 1> logs/ADDA/run-spotless-FINAL.out 2> logs/ADDA/run-spotless-FINAL.err &
nohup ./batch_scripts/CellDART/run_CellDART-spotless-FINAL.sh 1> logs/CellDART/run-spotless-FINAL.out 2> logs/CellDART/run-spotless-FINAL.err &
nohup ./batch_scripts/CORAL/run_coral-spotless-FINAL.sh 1> logs/CORAL/run-spotless-FINAL.out 2> logs/CORAL/run-spotless-FINAL.err &
nohup ./batch_scripts/DANN/run_dann-spotless-FINAL.sh 1> logs/DANN/run-spotless-FINAL.out 2> logs/DANN/run-spotless-FINAL.err &
