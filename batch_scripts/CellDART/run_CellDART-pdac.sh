#!/usr/bin/env bash

start=`date +%s`

for n in {1..50}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_pdac/CellDART/a_list.txt)
    echo "CellDART config file no. ${n}: ${CONFIG_FILE}"
    python -u reproduce_celldart.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_pdac" -c 1 2>> logs/CellDART/generated_pdac/training.err 1>> logs/CellDART/generated_pdac/training.out
    echo "Evaluating"
    ./eval_config.py -n CellDART -f "${CONFIG_FILE}" -cdir "configs/generated_pdac" --early_stopping --njobs 32 -c 1

done

end=`date +%s`
echo "script time: $(($end-$start))" 