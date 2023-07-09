#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80

CONFIG_FILE="celldart-final-spotless-ht.yml"

mkdir -p logs/CellDART

for i in 838465 3453934 3546 294 98237; do
    echo random seed: $i
    python -u reproduce_celldart.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs" --model_dir="model_FINAL" --seed_override=$i 2>> logs/CellDART/training_FINAL.err 1>> logs/CellDART/training_FINAL.out
    echo "Evaluating"
    ./eval_config.py -n CellDART -f "${CONFIG_FILE}" -cdir "configs" --njobs 16 --early_stopping -t --model_dir="model_FINAL" --seed_override=$i --results_dir="results_final" >> logs/CellDART/eval_FINAL.out
done

end=`date +%s`
echo "script time: $(($end-$start))"