#!/usr/bin/env bash

start=`date +%s`

for n in {1..100}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_dlpfc/CellDART/a_list.txt)
    echo "CellDART config file no. ${n}: ${CONFIG_FILE}"
    python -u reproduce_celldart.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_dlpfc" -c 1 2>> logs/CellDART/generated_dlpfc/training.err 1>> logs/CellDART/generated_dlpfc/training.out
    echo "Evaluating"
    ./eval_config.py -n CellDART -f "${CONFIG_FILE}" -cdir "configs/generated_dlpfc" --early_stopping --njobs 32 -c 1

done

end=`date +%s`
echo "script time: $(($end-$start))" 