#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G      
#SBATCH --time=0-01:30:00

# #SBATCH --output=logs/CORAL/generated_pdac/gen_v1-%a-eval-%N-%A.out
# #SBATCH --error=logs/CORAL/generated_pdac/gen_v1-%a-%N-%A.out

set -x

start=`date +%s`

CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+99))p" configs/generated_pdac/CORAL/a_list.txt)

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
    ./eval_config.py -n CORAL -f "${config_file}" -cdir "configs/generated_pdac" -m --njobs=$SLURM_CPUS_PER_TASK -d "$SLURM_TMPDIR/tmp_results"
done

end=`date +%s`
echo "script time: $(($end-$start))"