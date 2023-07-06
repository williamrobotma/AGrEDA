#!/usr/bin/env bash

start=`date +%s`

for n in {1..50}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_pdac/DANN/a_list.txt)
    echo "DANN config file no. ${n}: ${CONFIG_FILE}"
    python -u dann.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_pdac" -c 3 2>> logs/DANN/generated_pdac/training.err 1>> logs/DANN/generated_pdac/training.out
    echo "Evaluating"
    ./eval_config.py -n DANN -f "${CONFIG_FILE}" -cdir "configs/generated_pdac" --early_stopping --njobs 32 -c 3

done

end=`date +%s`
echo "script time: $(($end-$start))" 