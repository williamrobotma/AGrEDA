#!/bin/bash

#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G       

#SBATCH --time=0-20:00:00
#SBATCH --array=1-1000:10

#SBATCH --output=logs/DANN/generated_spotless1000/gen_v1-%a-%N-%A.out
#SBATCH --error=logs/DANN/generated_spotless1000/gen_v1-%a-%N-%A.err

# #SBATCH --time=0-1:00:00
# #SBATCH --array=1000-1000:1

# #SBATCH --output=logs/DANN/generated_spotless1000/TESTgen_v1-%a-%N-%A.out
# #SBATCH --error=logs/DANN/generated_spotless1000/TESTgen_v1-%a-%N-%A.err

set -x

start=`date +%s`

CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+9))p" configs/generated_spotless1000/DANN/a_list.txt)

# CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+0))p" configs/generated_spotless1000/DANN/a_list.txt)

# export BLIS_NUM_THREADS=1
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source ~/.venv-agreda/bin/activate

endbuild=`date +%s`
echo "build time: $(($endbuild-$start))"

for config_file in $CONFIG_FILES;
do
    echo "DANN config file no. ${n}: ${config_file}"
    ./dann.py -f "${config_file}" -l "log.txt" -cdir "configs/generated_spotless1000" -d "$SLURM_TMPDIR/tmp_model"
done

end=`date +%s`
echo "script time: $(($end-$start))" 