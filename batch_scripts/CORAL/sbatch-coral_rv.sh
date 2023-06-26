#!/bin/bash

#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
# #SBATCH --ntasks-per-node=1
#SBATCH --mem=8G             
#SBATCH --time=0-03:00:00
#SBATCH --array=1-991:10

#SBATCH --output=logs/CORAL/generated_test/gen_v1-%a-%N-%A.out
#SBATCH --error=logs/CORAL/generated_test/gen_v1-%a-%N-%A.err

set -x

start=`date +%s`

CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+9))p" configs/generated/CORAL/a_list.txt)
export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source .venv/bin/activate

endbuild=`date +%s`
echo "build time: $(($endbuild-$start))"

for CONFIG_FILE in $CONFIG_FILES; do
    echo "CORAL config file: ${CONFIG_FILE}"
    ./coral.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated" -d "$SLURM_TMPDIR/tmp_model" -t $SLURM_CPUS_PER_TASK --num_workers 1
done

end=`date +%s`
echo "script time: $(($end-$start))" 