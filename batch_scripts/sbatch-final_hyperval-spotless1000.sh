#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G
#SBATCH --time=30:00

#SBATCH --output=logs/final_hyperval-%N-%j.out

set -x
source ~/.venv-agreda/bin/activate

./hyperval_generated.py -n CellDART -cdir configs/generated_spotless1000 --output_dir logs/CellDART/generated_spotless1000
./hyperval_generated.py -n DANN -cdir configs/generated_spotless1000 --output_dir logs/DANN/generated_spotless1000
