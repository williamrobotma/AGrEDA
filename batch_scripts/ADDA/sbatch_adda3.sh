#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G      
#SBATCH --time=24:00:00

#SBATCH --output=logs/dann-standard_batchnorm_sameasdann%N-%j.out
#SBATCH --error=logs/dann-standard_batchnorm_sameasdann%N-%j.err


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt

# ./prep_data.py --njobs 64
# ./prep_data.py -s standard  --njobs 64
# ./prep_data.py -s standard -a  --njobs 64
# ./prep_data.py -s standard --stsplit  --njobs 64
# ./prep_data.py -s standard -a --stsplit  --njobs 64
# ./prep_data.py --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard  --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard -a  --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard --stsplit  --njobs 64 --nmarkers 100000
# ./prep_data.py -s standard -a --stsplit --njobs 64 --nmarkers 100000
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute reproduce_celldart.ipynb
# ./eval.py -d "data/preprocessed_markers_celldart" -n "CellDART" -v "bn_fix"
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute adda.ipynb
# python -u eval.py -n "ADDA" -v "celldart" -s "celldart" --njobs 16   --seed 72
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute dann.ipynb
# ./eval.py -d "data/preprocessed_markers_standard" -n "DANN" -v "Standard1" -p
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc_allgenes.ipynb
# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute autoenc_st.ipynb
# python -u adda.py -f "celldart_bnfix_adam_beta1_5.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "celldart_bnfix_adam_beta1_5.yml"   --njobs 16

# python -u adda.py -f "celldart_bnfix_adam_beta1_9.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "celldart_bnfix_adam_beta1_9.yml"   --njobs 16

python -u adda.py -f "celldart_nobnfix_adam_beta1_5.yml"   --njobs 16
python -u eval_config.py -n "ADDA" -f "celldart_nobnfix_adam_beta1_5.yml"   --njobs 16

# python -u adda.py -f "celldart_nobnfix_adam_beta1_9.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "celldart_nobnfix_adam_beta1_9.yml"   --njobs 16

# python -u adda.py -f "standard_bnfix_adam_beta1_5.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "standard_bnfix_adam_beta1_5.yml"   --njobs 16

# python -u adda.py -f "standard_bnfix_adam_beta1_9.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "standard_bnfix_adam_beta1_9.yml"   --njobs 16

# python -u adda.py -f "standard_nobnfix_adam_beta1_5.yml" --njobs 16
# python -u eval_config.py -n "ADDA" -f "standard_nobnfix_adam_beta1_5.yml" --njobs 16

# python -u adda.py -f "standard_nobnfix_adam_beta1_9.yml" --njobs 16
# python -u eval_config.py -n "ADDA" -f "standard_nobnfix_adam_beta1_9.yml" --njobs 16

# python -u adda.py -f "unscaled_bnfix_adam_beta1_5.yml" --njobs 16
# python -u eval_config.py -n "ADDA" -f "unscaled_bnfix_adam_beta1_5.yml"   --njobs 16

# python -u adda.py -f "unscaled_bnfix_adam_beta1_9.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "unscaled_bnfix_adam_beta1_9.yml"   --njobs 16

# python -u adda.py -f "unscaled_nobnfix_adam_beta1_5.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "unscaled_nobnfix_adam_beta1_5.yml"   --njobs 16

# python -u adda.py -f "unscaled_nobnfix_adam_beta1_9.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "unscaled_nobnfix_adam_beta1_9.yml"   --njobs 16
