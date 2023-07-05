#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80
rm -rf configs/generated_spotless/CORAL
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_coral.ipynb


CONFIG_FILE="gen_spotless_oracle-124973.yml"


echo "CORAL config file: ${CONFIG_FILE}"

SLURM_TMPDIR="tmp"
mkdir -p $SLURM_TMPDIR
python -u coral.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" -c 2 -d "$SLURM_TMPDIR/tmp_model" 2> logs/CORAL/generated_spotless/training_FINAL.err 1> logs/CORAL/generated_spotless/training_FINAL.out
rm -rf "$SLURM_TMPDIR"

echo "Evaluating"
mkdir -p $SLURM_TMPDIR
./eval_config.py -n CORAL -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --njobs 32 -c 2 -d "$SLURM_TMPDIR/tmp_results"
rm -rf "$SLURM_TMPDIR"


end=`date +%s`
echo "script time: $(($end-$start))"