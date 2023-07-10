#!/usr/bin/env bash

start=`date +%s`

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --nmarkers 80

CONFIG_FILE="adda-final-spotless-ht.yml"

mkdir -p logs/ADDA

echo "ADDA config file: ${CONFIG_FILE}"

for i in 2353 24385 284 86322 98237; do
    echo random seed: $i
    python -u adda.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs" --model_dir="model_FINAL" --seed_override=$i 2>> logs/ADDA/training_FINAL.err 1>> logs/ADDA/training_FINAL.out
    echo "Evaluating"
    ./eval_config.py -n ADDA -f "${CONFIG_FILE}" -cdir "configs" --njobs 16 --early_stopping -t --model_dir="model_FINAL" --seed_override=$i --results_dir="results_final" >> logs/ADDA/eval_FINAL.out
done

end=`date +%s`
echo "script time: $(($end-$start))"