#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA
mkdir -p logs/CellDART
mkdir -p logs/CORAL
mkdir -p logs/DANN

./batch_scripts/ADDA/run_adda-spotless-FINAL.sh > logs/ADDA/run.out
./batch_scripts/CellDART/run_CellDART-spotless-FINAL.sh > logs/CellDART/run.out
./batch_scripts/CORAL/run_coral-spotless-FINAL.sh > logs/CORAL/run.out
./batch_scripts/DANN/run_dann-spotless-FINAL.sh > logs/DANN/run.out
