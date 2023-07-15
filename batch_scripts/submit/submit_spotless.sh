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

ADDA_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/ADDA/sbatch-adda-ht-spotless.sh" | tr -dc '0-9')
sleep 5

CELLDART_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CellDART/sbatch-celldart-ht-spotless.sh" | tr -dc '0-9')
sleep 5

CORAL_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/CORAL/sbatch-coral-ht-spotless.sh" | tr -dc '0-9')
sleep 5

DANN_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID,afterok:$PREP_JOBID "batch_scripts/DANN/sbatch-dann-ht-spotless.sh" | tr -dc '0-9')
sleep 5

# sbatch -d afterok:$ADDA_JOBID,afterok:$CELLDART_JOBID,afterok:$CORAL_JOBID,afterok:$DANN_JOBID "batch_scripts/sbatch-final_hyperval-spotless.sh"

# GENCONFIG_JOBID=$(sbatch "batch_scripts/sbatch-gen-configs-spotless.sh" | tr -dc '0-9')
# sleep 5

# ADDA_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID "batch_scripts/ADDA/sbatch-adda-ht-spotless.sh" | tr -dc '0-9')
# sleep 5

# CELLDART_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID "batch_scripts/CellDART/sbatch-celldart-ht-spotless.sh" | tr -dc '0-9')
# sleep 5

# CORAL_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID "batch_scripts/CORAL/sbatch-coral-ht-spotless.sh" | tr -dc '0-9')
# sleep 5

# DANN_JOBID=$(sbatch -d afterok:$GENCONFIG_JOBID "batch_scripts/DANN/sbatch-dann-ht-spotless.sh" | tr -dc '0-9')
# sleep 5

sbatch -d afterok:$ADDA_JOBID,afterok:$CELLDART_JOBID,afterok:$CORAL_JOBID,afterok:$DANN_JOBID "batch_scripts/sbatch-final_hyperval-spotless.sh"
