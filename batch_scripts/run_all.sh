#!/usr/bin/env bash

set -x

nohup ./batch_scripts/ADDA/run_adda-dlpfc.sh &> logs/ADDA/generated_dlpfc/run.out &
nohup ./batch_scripts/CellDART/run_CellDART-dlpfc.sh &> logs/CellDART/generated_dlpfc/run.out &
nohup ./batch_scripts/CORAL/run_coral-dlpfc.sh &> logs/CORAL/generated_dlpfc/run.out &
nohup ./batch_scripts/DANN/run_dann-dlpfc.sh &> logs/DANN/generated_dlpfc/run.out &
