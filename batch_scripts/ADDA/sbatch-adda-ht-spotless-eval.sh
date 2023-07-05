#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16G      
#SBATCH --time=0-06:00:00

# #SBATCH --output=logs/ADDA/generated_spotless/gen_v1-%a-eval-%N-%A.out
# #SBATCH --error=logs/ADDA/generated_spotless/gen_v1-%a-%N-%A.out

set -x

start=`date +%s`

CONFIG_FILES=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+99))p" configs/generated_spotless/ADDA/a_list.txt)
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

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
    echo "ADDA config file no. ${n}: ${config_file}"
    ./eval_config.py -n ADDA -f "${config_file}" -cdir "configs/generated_spotless" --early_stopping  -m --njobs=$SLURM_CPUS_PER_TASK -d "$SLURM_TMPDIR/tmp_results"
done

end=`date +%s`
echo "script time: $(($end-$start))"