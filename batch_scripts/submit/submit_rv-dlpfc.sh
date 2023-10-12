#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA/generated_dlpfc
mkdir -p logs/CellDART/generated_dlpfc
mkdir -p logs/CORAL/generated_dlpfc
mkdir -p logs/DANN/generated_dlpfc

GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs-dlpfc.sh" | tr -dc '0-9')
sleep 5
PREP_JOBID=$(sbatch "batch_scripts/sbatch-prep-rv-dlpfc.sh" | tr -dc '0-9')
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart_rv-dlpfc.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CellDART/sbatch-celldart_rv-dlpfc-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda_rv-dlpfc.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/ADDA/sbatch-adda_rv-dlpfc-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann_rv-dlpfc.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/DANN/sbatch-dann_rv-dlpfc-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral_rv-dlpfc.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CORAL/sbatch-coral_rv-dlpfc-eval.sh"
sleep 5