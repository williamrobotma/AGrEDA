#!/usr/bin/env bash

set -x

mkdir -p logs/CORAL/generated_spotless

GENCONFIG_JOBID=$(sbatch "batch_scripts/CORAL/sbatch-gen-configs-spotless-coral.sh" | tr -dc '0-9')
sleep 5

CORAL_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID "batch_scripts/CORAL/sbatch-coral-ht-spotless.sh" | tr -dc '0-9')
sleep 5

sbatch -d afterok:$CORAL_JOBID "batch_scripts/CORAL/sbatch-final_hyperval-spotless-coral.sh"
