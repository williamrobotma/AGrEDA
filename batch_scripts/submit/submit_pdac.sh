#!/usr/bin/env bash

sbatch --depend=afterok:38459386 batch_scripts/ADDA/sbatch-adda-ht-pdac.sh
sbatch --depend=afterok:38459386 batch_scripts/CellDART/sbatch-celldart-ht-pdac.sh
sbatch --depend=afterok:38459386 batch_scripts/CORAL/sbatch-coral-ht-pdac.sh
sbatch --depend=afterok:38459386 batch_scripts/DANN/sbatch-dann-ht-pdac.sh
