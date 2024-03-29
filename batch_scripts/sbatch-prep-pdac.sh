#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=32  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G   
#SBATCH --time=0:20:00
# #SBATCH --time=0:20:00
#SBATCH --array=20,40,80

#SBATCH --output=logs/prep-%a-%N-%A.out
# #SBATCH --error=logs/prep%N-%j.err

set -x

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# num_workers=$(($SLURM_CPUS_PER_TASK/2))

# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt

source ~/.venv-agreda/bin/activate

# for n in 20 40 80
# do
for m in 30 50 70
do
    ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix $m --one_model --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s minmax --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix $m --one_model --nmarkers $SLURM_ARRAY_TASK_ID
done
# for m in 3 5 8 10; do
#     ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --val_samp --nmarkers $SLURM_ARRAY_TASK_ID
#     ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --val_samp --nmarkers $SLURM_ARRAY_TASK_ID
# done
# for m in 5 8 10 15; do
#     ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --val_samp --nmarkers $SLURM_ARRAY_TASK_ID
#     ./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --val_samp --nmarkers $SLURM_ARRAY_TASK_ID
# done
# done

