#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
# #SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=40  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G      
#SBATCH --time=24:00:00

#SBATCH --output=logs/DANN/standard_batchnorm_100000spots%N-%j.out
#SBATCH --error=logs/DANN/standard_batchnorm_100000spots%N-%j.err


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt

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

python -u dann.py -f "standard_batchnorm_100000spots.yml" --njobs 40
python -u eval_config.py -n "DANN" -f "standard_batchnorm_100000spots.yml" -p --njobs -1
# python -u eval.py -n "DANN" -v "Standard1" -p -s standard -c 2 --njobs 32 --seed 25098
