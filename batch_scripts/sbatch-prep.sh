#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
# #SBATCH --gpus=1 
#SBATCH --cpus-per-task=64  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=128G      
#SBATCH --time=1:00:00
#SBATCH --array=160

#SBATCH --output=logs/prep-nmarkers%a-%N-%A.out
# #SBATCH --error=logs/prep%N-%j.err

set -x

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
num_workers=$(($SLURM_CPUS_PER_TASK/2))

module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
source .venv/bin/activate



./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50 --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50 --nmarkers $SLURM_ARRAY_TASK_ID

./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50 --stsplit  --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50 --stsplit --nmarkers $SLURM_ARRAY_TASK_ID

./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50 --stsplit --one_model --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50 --stsplit --one_model --nmarkers $SLURM_ARRAY_TASK_ID

for m in 5 8 10 15 20
do
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --one_model --nmarkers $SLURM_ARRAY_TASK_ID

    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --nmarkers $SLURM_ARRAY_TASK_ID
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --one_model --nmarkers $SLURM_ARRAY_TASK_ID
done

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --stsplit --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --stsplit --one_model --nmarkers $SLURM_ARRAY_TASK_ID

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --stsplit --nmarkers $SLURM_ARRAY_TASK_ID
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --stsplit --one_model --nmarkers $SLURM_ARRAY_TASK_ID
