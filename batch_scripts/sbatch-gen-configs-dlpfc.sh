#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=4G
#SBATCH --time=00:15:00

#SBATCH --output=logs/prep-genconfigs-dlpfc-%N-%j.out

set -x

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
# ./gen_venv_cc.sh
source ~/.venv-agreda/bin/activate

# if ["$SLURM_ARRAY_TASK_ID" == "5"]; then

python gen_configs_adda.py
python gen_configs_celldart.py
python gen_configs_coral.py
python gen_configs_dann.py
