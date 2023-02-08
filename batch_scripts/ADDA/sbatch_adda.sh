#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1 
#SBATCH --cpus-per-task=40  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem-per-cpu=256M
#SBATCH --time=0-01:00:00
#SBATCH --array=1-12

#SBATCH --output=logs/ADDA/configs_list_evalonly%a-%N-%A.out
##SBATCH --error=logs/ADDA/configs_list_evalonly%a-%N-%A.err

CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/ADDA/configs_list.txt)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
num_workers=$(($SLURM_CPUS_PER_TASK/2))

# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
source .venv/bin/activate

echo "ADDA config file: ${CONFIG_FILE}"
# python -u adda.py -f "${CONFIG_FILE}"  --njobs $num_workers

echo "$SLURM_CPUS_PER_TASK"
python -u eval_config.py -n "ADDA" -f "${CONFIG_FILE}" --njobs $SLURM_CPUS_PER_TASK -p

