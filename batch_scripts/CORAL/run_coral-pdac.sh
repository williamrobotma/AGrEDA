#!/usr/bin/env bash

start=`date +%s`

for n in {1..50}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_pdac/CORAL/a_list.txt)
    echo "CORAL config file no. ${n}: ${CONFIG_FILE}"
    python -u coral.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_pdac" -c 2 2>> logs/CORAL/generated_pdac/training.err 1>> logs/CORAL/generated_pdac/training.out
    echo "Evaluating"
    ./eval_config.py -n CORAL -f "${CONFIG_FILE}" -cdir "configs/generated_pdac" --njobs 32 -c 2

done

end=`date +%s`
echo "script time: $(($end-$start))" 