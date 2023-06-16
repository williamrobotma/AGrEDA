#!/bin/bash

#SBATCH --account=def-aminemad

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      
#SBATCH --time=2:00:00

#SBATCH --output=logs/all%N-%j.out
#SBATCH --error=logs/all%N-%j.err


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
num_workers=$(($SLURM_CPUS_PER_TASK/2))

# module load python/3.8
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# pip install --no-index -r requirements_cc.txt
source /home/williamm/.venv-agreda/bin/activate

python prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50

# python prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
# ./coral.py -f coral.yml -l "log.txt"
# ./eval_config.py -n CORAL -f coral.yml --njobs -1 

./coral.py -f coral_pdac.yml -l "log.txt"
./eval_config.py -n CORAL -f coral_pdac.yml --njobs -1 -d "$SLURM_TMPDIR/compute_node_temp"

# python adda.py -f "standard_bnfix_adam_beta1_5.yml" -l "log.txt"
# python eval_config.py -n "ADDA" -f "standard_bnfix_adam_beta1_5.yml" --njobs -1

# python reproduce_celldart.py -f "bnfix_minmax.yml" -l "log.txt"
# python eval_config.py -n "CellDART" -f "bnfix_minmax.yml" --njobs -1

# python dann.py -f "dann_legacy.yml" -l "log.txt"
python eval_config.py -n "DANN" -f "dann_legacy.yml" --njobs -1 

# python -u dann.py -f "dann_legacy.yml" --njobs $num_workers
# python -u eval_config.py -n "DANN" -f "dann_legacy.yml" --njobs $SLURM_CPUS_PER_TASK
# python -u eval.py -n "DANN" -v "Standard1" -p -s standard -c 2 --njobs 32 --seed 25098
