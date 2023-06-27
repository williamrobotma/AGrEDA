#!/usr/bin/env bash

start=`date +%s`

for n in {1..100}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_spotless/CORAL/a_list.txt)
    echo "CORAL config file no. ${n}: ${CONFIG_FILE}"
    # python -u coral.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" -c 1 2>> logs/CORAL/generated_spotless/training.err 1>> logs/CORAL/generated_spotless/training.out
    echo "Evaluating"
    ./eval_config.py -n CORAL -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --njobs 32 -c 1

done

end=`date +%s`
echo "script time: $(($end-$start))" 