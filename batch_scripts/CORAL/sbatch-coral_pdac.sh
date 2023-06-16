#!/bin/bash

#SBATCH --account=def-aminemad

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=48  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=498G      
#SBATCH --time=1:00:00

#SBATCH --output=logs/CORAL/coral_pdac%N-%j.out
#SBATCH --error=logs/CORAL/coral_pdac%N-%j.err


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
num_workers=$(($SLURM_CPUS_PER_TASK/2))

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source .venv/bin/activate

# ./prep_data.py --njobs -1
# ./prep_data.py -s standard  --njobs -1
# ./prep_data.py -s standard -a  --njobs -1
# ./prep_data.py -s standard --stsplit  --njobs -1
# ./prep_data.py -s standard -a --stsplit  --njobs -1
# ./prep_data.py --njobs -1 --nspots 100000
# ./prep_data.py -s standard  --njobs -1 --nspots 100000
# ./prep_data.py -s standard -a  --njobs -1 --nspots 100000
# ./prep_data.py -s standard --stsplit  --njobs -1 --nspots 100000
# ./prep_data.py -s standard -a --stsplit --njobs -1 --nspots 100000

python download_pdac.py
python preprocess_pdac.py

python prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50
python coral.py -f coral_pdac.yml
python eval_config.py --njobs -1 -f coral_pdac.yml -n CORAL

# python -u dann.py -f "dann_legacy.yml" --njobs $num_workers
# python -u eval_config.py -n "DANN" -f "dann_legacy.yml" --njobs $SLURM_CPUS_PER_TASK
# python -u eval.py -n "DANN" -v "Standard1" -p -s standard -c 2 --njobs 32 --seed 25098
