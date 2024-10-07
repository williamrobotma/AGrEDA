#!/bin/bash

#SBATCH --cpus-per-task=32  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G     
#SBATCH --time=0-00:30:00

#SBATCH --output=logs/ADDA/generated_pdac/gen_v1-eval-%N-%j.out
# #SBATCH --error=logs/ADDA/generated_pdac/gen_v1-%a-%N-%A.out

set -x

start=`date +%s`

export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
source ~/.venv-agreda/bin/activate

endbuild=`date +%s`
echo "build time: $(($endbuild-$start))"

./eval_config.py -n "ADDA" -f  "basic_config.yml" -cdir "configs/generated_pdac" --njobs=$SLURM_CPUS_PER_TASK -d "$SLURM_TMPDIR/tmp_results" --test --reverse_val

end=`date +%s`
echo "script time: $(($end-$start))"