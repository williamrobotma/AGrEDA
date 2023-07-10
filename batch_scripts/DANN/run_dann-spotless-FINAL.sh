#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80

CONFIG_FILE="dann-final-spotless-ht.yml"

mkdir -p logs/DANN

for i in 34958 3546 373737 4543512 744; do
    echo random seed: $i
    python -u dann.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs" --model_dir="model_FINAL" --seed_override=$i 2>> logs/DANN/training_FINAL.err 1>> logs/DANN/training_FINAL.out
    echo "Evaluating"
    ./eval_config.py -n DANN -f "${CONFIG_FILE}" -cdir "configs" --njobs 16 --early_stopping -t --model_dir="model_FINAL" --seed_override=$i --results_dir="results_final" >> logs/DANN/eval_FINAL.out
done


end=`date +%s`
echo "script time: $(($end-$start))"