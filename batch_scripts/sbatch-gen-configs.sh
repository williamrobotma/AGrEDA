#!/bin/bash

#SBATCH --account=rrg-aminemad

# #SBATCH --nodes=1
# #SBATCH --tasks-per-node=1 
# #SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G
#SBATCH --time=0:10:00
# #SBATCH --array=5,10,20,40,80

#SBATCH --output=logs/prep-genconfigs2-%N-%j.out
# #SBATCH --error=logs/prep%N-%j.err

set -x

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
source .venv/bin/activate

# if ["$SLURM_ARRAY_TASK_ID" == "5"]; then

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_adda.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_celldart.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_coral.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_dann.ipynb
