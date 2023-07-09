#!/usr/bin/env bash

set -x


mkdir -p logs/ADDA/generated_pdac
mkdir -p logs/CellDART/generated_pdac
mkdir -p logs/CORAL/generated_pdac
mkdir -p logs/DANN/generated_pdac

GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs.sh" | tr -dc '0-9')
sleep 5
PREP_JOBID=$(sbatch "batch_scripts/sbatch-prep.sh" | tr -dc '0-9')
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda-ht-pdac.sh"
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart-ht-pdac.sh"
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral-ht-pdac.sh"
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann-ht-pdac.sh"
sleep 5
