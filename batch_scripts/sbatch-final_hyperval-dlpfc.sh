#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G
#SBATCH --time=1:00:00

#SBATCH --output=logs/final_hyperval-dlpfc-%N-%j.out

set -x
source ~/.venv-agreda/bin/activate

./hyperval_generated.py -n ADDA -cdir configs/generated_dlpfc --output_dir logs/ADDA/generated_dlpfc
./hyperval_generated.py -n CellDART -cdir configs/generated_dlpfc --output_dir logs/CellDART/generated_dlpfc
./hyperval_generated.py -n DANN -cdir configs/generated_dlpfc --output_dir logs/DANN/generated_dlpfc
./hyperval_generated.py -n CORAL -cdir configs/generated_dlpfc --output_dir logs/CORAL/generated_dlpfc
