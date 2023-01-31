#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1 
#SBATCH --cpus-per-task=40  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
##SBATCH --mem-per-cpu=256M
#SBATCH --time=0-04:00:00
#SBATCH --array=1-12

#SBATCH --output=logs/ADDA/configs_list_evalonly%a-%N-%A.out
##SBATCH --error=logs/ADDA/configs_list_evalonly%a-%N-%A.err

CONFIG_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" configs/ADDA/configs_list.txt)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements_cc.txt
# source .venv/bin/activate

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

echo "ADDA config file: ${CONFIG_FILE}"
# python -u adda.py -f "${CONFIG_FILE}"  --njobs $SLURM_CPUS_PER_TASK

echo "$SLURM_CPUS_PER_TASK"
python -u eval_config.py -n "ADDA" -f "${CONFIG_FILE}" --njobs $SLURM_CPUS_PER_TASK

# python -u adda.py -f "celldart_bnfix_adam_beta1_9.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "celldart_bnfix_adam_beta1_9.yml"   --njobs 16

# python -u adda.py -f "celldart_nobnfix_adam_beta1_5.yml"   --njobs 16
# python -u eval_config.py -n "ADDA" -f "celldart_nobnfix_adam_beta1_5.yml"   --njobs 16

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
