#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA/generated_spotless
mkdir -p logs/CellDART/generated_spotless
mkdir -p logs/CORAL/generated_spotless
mkdir -p logs/DANN/generated_spotless

GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs-spotless.sh" | tr -dc '0-9')
sleep 5
PREP_JOBID=$(sbatch "batch_scripts/sbatch-prep-spotless.sh" | tr -dc '0-9')
sleep 5


JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart_rv-spotless.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CellDART/sbatch-celldart_rv-spotless-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda_rv-spotless.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/ADDA/sbatch-adda_rv-spotless-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann_rv-spotless.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/DANN/sbatch-dann_rv-spotless-eval.sh"
sleep 5

JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral_rv-spotless.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CORAL/sbatch-coral_rv-spotless-eval.sh"
sleep 5