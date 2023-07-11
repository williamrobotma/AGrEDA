#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G      
#SBATCH --time=0-00:40:00
#SBATCH --array=1-200:25

#SBATCH --output=logs/CORAL/generated_pdac/gen_v1-%a-%N-%A.out
#SBATCH --error=logs/CORAL/generated_pdac/gen_v1-%a-%N-%A.err

set -x

start=`date +%s`

CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+24))p" configs/generated_pdac/CORAL/a_list.txt)

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

for config_file in $CONFIG_FILES;
do
    echo "CORAL config file no. ${n}: ${config_file}"
    ./coral.py -f "${config_file}" -l "log.txt" -cdir "configs/generated_pdac" -d "$SLURM_TMPDIR/tmp_model"
    ./eval_config.py -n CORAL -f "${config_file}" -cdir "configs/generated_pdac" -m --njobs=$SLURM_CPUS_PER_TASK -d "$SLURM_TMPDIR/tmp_results"
done

# echo "running eval"
# sbatch --output="./logs/CORAL/generated_pdac/gen_v1-${SLURM_ARRAY_TASK_ID}-eval.out" --export=SLURM_ARRAY_TASK_ID ./batch_scripts/CORAL/sbatch-coral-ht-pdac-eval.sh

end=`date +%s`
echo "script time: $(($end-$start))" 