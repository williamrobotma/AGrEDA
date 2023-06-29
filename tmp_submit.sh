#!/bin/bash

#SBATCH --account=rrg-aminemad

# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1 
# #SBATCH --gpus=1 
#SBATCH --cpus-per-task=24  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G        
#SBATCH --time=0:00:10
#SBATCH --array=5,10,20,40,80

#SBATCH --output=logs/prep-nmarkers%a-%N-%A.out
# #SBATCH --error=logs/prep%N-%j.err

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_adda.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_celldart.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_coral.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_dann.ipynb