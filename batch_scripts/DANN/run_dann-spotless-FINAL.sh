#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80

CONFIG_FILE="dann-final-spotless-ht.yml"

mkdir -p logs/DANN

echo "DANN config file: ${CONFIG_FILE}"
python -u dann.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs" 2> logs/DANN/training_FINAL.err 1> logs/DANN/training_FINAL.out
echo "Evaluating"
./eval_config.py -n DANN -f "${CONFIG_FILE}" -cdir "configs" --njobs 16 -t > logs/DANN/eval_FINAL.out


end=`date +%s`
echo "script time: $(($end-$start))"