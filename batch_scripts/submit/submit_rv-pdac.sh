#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA/generated_pdac
mkdir -p logs/CellDART/generated_pdac
mkdir -p logs/CORAL/generated_pdac
mkdir -p logs/DANN/generated_pdac

GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs-pdac.sh" | tr -dc '0-9')
sleep 5
PREP_JOBID=$(sbatch "batch_scripts/sbatch-prep-pdac.sh" | tr -dc '0-9')
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart_rv-pdac.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CellDART/sbatch-celldart_rv-pdac-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda_rv-pdac.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/ADDA/sbatch-adda_rv-pdac-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann_rv-pdac.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/DANN/sbatch-dann_rv-pdac-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral_rv-pdac.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CORAL/sbatch-coral_rv-pdac-eval.sh"
sleep 5