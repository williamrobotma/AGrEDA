#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA/generated_pdac
mkdir -p logs/CellDART/generated_pdac
mkdir -p logs/CORAL/generated_pdac
mkdir -p logs/DANN/generated_pdac

nohup ./batch_scripts/ADDA/run_adda-pdac.sh &> logs/ADDA/generated_pdac/run.out &
nohup ./batch_scripts/CellDART/run_CellDART-pdac.sh &> logs/CellDART/generated_pdac/run.out &
nohup ./batch_scripts/CORAL/run_coral-pdac.sh &> logs/CORAL/generated_pdac/run.out &
nohup ./batch_scripts/DANN/run_dann-pdac.sh &> logs/DANN/generated_pdac/run.out &
