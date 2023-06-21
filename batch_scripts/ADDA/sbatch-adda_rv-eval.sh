#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --cpus-per-task=32  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=127500M      
#SBATCH --time=0-03:00:00

#SBATCH --output=logs/ADDA/generated/gen_v1-eval-%N-%j.out
# #SBATCH --error=logs/ADDA/generated/gen_v1-%a-%N-%A.out

set -x

start=`date +%s`

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source .venv/bin/activate

endbuild=`date +%s`
echo "build time: $(($endbuild-$start))"

./eval_config.py -n "ADDA" -f  "basic_config.yml" -l "log.txt" -cdir "configs/generated" -d "$SLURM_TMPDIR/tmp_model" --test --reverse_val --njobs -1

end=`date +%s`
echo "script time: $(($end-$start))"