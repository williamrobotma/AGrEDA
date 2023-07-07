#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80
CONFIG_FILE="celldart-final-spotless-ht.yml"

mkdir -p logs/CellDART

echo "CellDART config file: ${CONFIG_FILE}"
python -u reproduce_celldart.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs" 2> logs/CellDART/training_FINAL.err 1> logs/CellDART/training_FINAL.out
echo "Evaluating"
./eval_config.py -n CellDART -f "${CONFIG_FILE}" -cdir "configs" --njobs 16 -t > logs/CellDART/eval_FINAL.out


end=`date +%s`
echo "script time: $(($end-$start))"