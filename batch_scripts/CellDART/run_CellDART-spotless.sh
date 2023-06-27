#!/usr/bin/env bash

start=`date +%s`

for n in {44..57}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_spotless/CellDART/a_list.txt)
    echo "CellDART config file no. ${n}: ${CONFIG_FILE}"
    python -u reproduce_celldart.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" -c 0 2>> logs/CellDART/generated_spotless/training.err 1>> logs/CellDART/generated_spotless/training.out
    echo "Evaluating"
    ./eval_config.py -n CellDART -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --early_stopping --njobs 32 -c 0

done

end=`date +%s`
echo "script time: $(($end-$start))" 