#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80

CONFIG_FILE="coral-final-spotless-ht.yml"

mkdir -p logs/CORAL

for i in 52 235426 157217 345 28323; do
    echo random seed: $i
    python -u coral.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs" --model_dir="model_FINAL" --seed_override=$i 2>> logs/CORAL/training_FINAL.err 1>> logs/CORAL/training_FINAL.out
    echo "Evaluating"
    ./eval_config.py -n CORAL -f "${CONFIG_FILE}" -cdir "configs" --njobs 16 -t --model_dir="model_FINAL" --seed_override=$i --results_dir="results_final" >> logs/CORAL/eval_FINAL.out
done


end=`date +%s`
echo "script time: $(($end-$start))"