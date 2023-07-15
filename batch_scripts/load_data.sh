#!/bin/bash

#SBATCH --account=rrg-aminemad

#SBATCH --cpus-per-task=1
#SBATCH --mem=64G        
#SBATCH --time=0:30:00

#SBATCH --output=logs/load_data%N-%j.out
# #SBATCH --error=logs/prep%N-%j.err

set -x
if ! [ -z "${SLURM_JOB_ID}" ]; then
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    # num_workers=$(($SLURM_CPUS_PER_TASK/2))

    # module load python/3.8
    # virtualenv --no-download $SLURM_TMPDIR/env
    # source $SLURM_TMPDIR/env/bin/activate
    # pip install --no-index --upgrade pip
    # pip install --no-index -r requirements_cc.txt

    source ~/.venv-agreda/bin/activate
fi

python src/download_pdac.py
python src/preprocess_pdac_GSE111672.py
python src/preprocess_pdac_zenodo6024273.py
python src/preprocessing_libd.py.py
python src/preprocessing_dlpfc.py
python src/preprocessing_spotless.py
python src/preprocessing_mouse_GSE115746.py
