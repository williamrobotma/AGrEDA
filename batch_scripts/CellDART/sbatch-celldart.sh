#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      
#SBATCH --time=0-06:00:00
#SBATCH --array=1-991:10

#SBATCH --output=logs/CellDART/generated/gen_v1-%a-%N-%A.out
#SBATCH --error=logs/CellDART/generated/gen_v1-%a-%N-%A.err

set -x

start=`date +%s`

CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+9))p" configs/generated/a_list.txt)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
num_workers=$(($SLURM_CPUS_PER_TASK/2))

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source .venv/bin/activate

endbuild=`date +%s`
echo "build time: $(($endbuild-$start))"

for config_file in $CONFIG_FILES;
do
    echo "CellDART config file: ${config_file}"
    ./reproduce_celldart.py -f "${config_file}" -l "log.txt" -cdir "configs/generated" -d "$SLURM_TMPDIR/tmp_model"
done

echo "running eval"
sbatch --output="./logs/CellDART/generated/gen_v1-${SLURM_ARRAY_TASK_ID}-eval.out" --export=SLURM_ARRAY_TASK_ID ./batch_scripts/CellDART/sbatch-celldart-eval.sh

end=`date +%s`
echo "script time: $(($end-$start))" 