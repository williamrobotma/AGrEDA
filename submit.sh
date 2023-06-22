#!/usr/bin/env bash

set -x

JOBID=$(sbatch "batch_scripts/CellDART/sbatch-celldart_rv.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CellDART/sbatch-celldart_rv-eval.sh"
sleep 5

JOBID=$(sbatch "batch_scripts/ADDA/sbatch-adda_rv.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/ADDA/sbatch-adda_rv-eval.sh"
sleep 5

JOBID=$(sbatch "batch_scripts/DANN/sbatch-dann_rv.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/DANN/sbatch-dann_rv-eval.sh"
sleep 5

JOBID=$(sbatch "batch_scripts/CORAL/sbatch-coral_rv.sh" | tr -dc '0-9')
sleep 5
sbatch --depend="afterok:${JOBID}" "batch_scripts/CORAL/sbatch-coral_rv-eval.sh"
sleep 5