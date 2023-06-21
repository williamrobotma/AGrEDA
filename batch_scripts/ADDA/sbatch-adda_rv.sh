#!/bin/bash

#SBATCH --account=rrg-aminemad
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      
#SBATCH --time=0-06:00:00
#SBATCH --array=1-1000

#SBATCH --output=logs/ADDA/generated/gen_v1-%a-%N-%A.out
#SBATCH --error=logs/ADDA/generated/gen_v1-%a-%N-%A.err

set -x

start=`date +%s`

CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/generated/ADDA/a_list.txt)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source .venv/bin/activate

endbuild=`date +%s`
echo "build time: $(($endbuild-$start))"
echo "ADDA config file: ${CONFIG_FILE}"
./adda.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated" -d "$SLURM_TMPDIR/tmp_model" -r

end=`date +%s`
echo "script time: $(($end-$start))" 