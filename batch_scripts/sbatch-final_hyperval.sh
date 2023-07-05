#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G
#SBATCH --time=1:00:00

#SBATCH --output=logs/final_hyperval-%N-%j.out

set -x
source .venv/bin/activate

./hyperval_generated.py -n ADDA -cdir configs/generated_spotless -e > logs/ADDA/generated_spotless/final_hyperval.txt
./hyperval_generated.py -n CellDART -cdir configs/generated_spotless -e > logs/CellDART/generated_spotless/final_hyperval.txt
./hyperval_generated.py -n DANN -cdir configs/generated_spotless -e > logs/DANN/generated_spotless/final_hyperval.txt
# ./hyperval_generated.py -n CORAL -cdir configs/generated_spotless > logs/CORAL/generated_spotless/final_hyperval.txt
