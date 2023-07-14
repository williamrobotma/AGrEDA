#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA/generated_dlpfc
mkdir -p logs/CellDART/generated_dlpfc
mkdir -p logs/CORAL/generated_dlpfc
mkdir -p logs/DANN/generated_dlpfc

GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs-dlpfc.sh" | tr -dc '0-9')
sleep 5
PREP_JOBID=$(sbatch "batch_scripts/sbatch-prep-dlpfc.sh" | tr -dc '0-9')
sleep 5

ADDA_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda-ht-dlpfc.sh" | tr -dc '0-9')
sleep 5

CELLDART_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart-ht-dlpfc.sh" | tr -dc '0-9')
sleep 5

CORAL_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral-ht-dlpfc.sh" | tr -dc '0-9')
sleep 5

DANN_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann-ht-dlpfc.sh" | tr -dc '0-9')
sleep 5

sbatch -d afterok:$ADDA_JOBID,afterok:$CELLDART_JOBID,afterok:$CORAL_JOBID,afterok:$DANN_JOBID "batch_scripts/sbatch-final_hyperval-dlpfc.sh"
