#!/usr/bin/env bash

start=`date +%s`

for n in {1..100}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_spotless/DANN/a_list.txt)
    echo "DANN config file no. ${n}: ${CONFIG_FILE}"
    # python -u dann.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" -c 2 2>> logs/DANN/generated_spotless/training.err 1>> logs/DANN/generated_spotless/training.out
    echo "Evaluating"
    ./eval_config.py -n DANN -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --early_stopping --njobs 32 -c 2

done

end=`date +%s`
echo "script time: $(($end-$start))" 