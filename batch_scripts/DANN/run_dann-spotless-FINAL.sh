#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80
rm -rf configs/generated_spotless/DANN
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_spotless_dann.ipynb


CONFIG_FILE="gen_spotless_oracle-124973.yml"

mkdir -p logs/DANN/generated_spotless

echo "DANN config file: ${CONFIG_FILE}"
python -u dann.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" 2> logs/DANN/generated_spotless/training_FINAL.err 1> logs/DANN/generated_spotless/training_FINAL.out
echo "Evaluating"
./eval_config.py -n DANN -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --njobs 16 -te > logs/DANN/generated_spotless/eval_FINAL.out


end=`date +%s`
echo "script time: $(($end-$start))"