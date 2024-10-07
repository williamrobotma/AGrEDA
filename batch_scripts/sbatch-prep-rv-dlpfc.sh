#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1
#SBATCH --mem=16G        
#SBATCH --time=0:20:00
#SBATCH --array=20,40,80

#SBATCH --output=logs/prep-rv-dlpfc-%a-%N-%A.out

set -x

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/.venv-agreda/bin/activate

# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt



# for n in 20 40 80
# do
# for m in 30 50 70
# do
#     ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix $m --one_model --nmarkers $SLURM_ARRAY_TASK_ID
#     ./prep_data.py -s minmax --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix $m --one_model --nmarkers $SLURM_ARRAY_TASK_ID
# done
for m in 3 5 8 10; do
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --val_samp --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --val_samp --nmarkers $SLURM_ARRAY_TASK_ID
done
# for m in 5 8 10 15; do
#     ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --nmarkers $SLURM_ARRAY_TASK_ID
#     ./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --nmarkers $SLURM_ARRAY_TASK_ID
# done
# done

