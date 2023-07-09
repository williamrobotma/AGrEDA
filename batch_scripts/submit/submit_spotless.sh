#!/usr/bin/env bash

set -x


GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs.sh" | tr -dc '0-9')
sleep 5
PREP_JOBID=$(sbatch "batch_scripts/sbatch-prep.sh" | tr -dc '0-9')
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda-ht-spotless.sh"
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart-ht-spotless.sh"
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral-ht-spotless.sh"
sleep 5

sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann-ht-spotless.sh"
sleep 5