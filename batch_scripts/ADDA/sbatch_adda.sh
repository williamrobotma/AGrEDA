#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      
#SBATCH --time=0-01:00:00
#SBATCH --array=1-1000

#SBATCH --output=logs/CellDART/generated/gen_v1%a-%N-%A.out
#SBATCH --error=logs/CellDART/generated/gen_v1%a-%N-%A.err

CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/generated/a_list.txt)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
source .venv/bin/activate

echo "CellDART config file: ${CONFIG_FILE}"
python -u eval_config.py -n "ADDA" -f "${CONFIG_FILE}" --njobs $SLURM_CPUS_PER_TASK -p

