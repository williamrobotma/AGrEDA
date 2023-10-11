#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G
#SBATCH --time=0:15:00

#SBATCH --output=logs/final_hyperval-%N-%j.out

set -x
source ~/.venv-agreda/bin/activate

# ./hyperval_generated.py -n ADDA -cdir configs/generated_spotless --output_dir logs/ADDA/generated_spotless
# ./hyperval_generated.py -n CellDART -cdir configs/generated_spotless --output_dir logs/CellDART/generated_spotless
# ./hyperval_generated.py -n DANN -cdir configs/generated_spotless --output_dir logs/DANN/generated_spotless
./hyperval_generated.py -n CORAL -cdir configs/generated_spotless --output_dir logs/CORAL/generated_spotless
