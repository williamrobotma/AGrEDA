#!/usr/bin/env bash

start=`date +%s`

for n in {1..50}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_pdac/ADDA/a_list.txt)
    echo "ADDA config file no. ${n}: ${CONFIG_FILE}"
    python -u adda.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_pdac" -c 0 2>> logs/ADDA/generated_pdac/training.err 1>> logs/ADDA/generated_pdac/training.out
    echo "Evaluating"
    ./eval_config.py -n ADDA -f "${CONFIG_FILE}" -cdir "configs/generated_pdac" --early_stopping --njobs 32 -c 0

done

end=`date +%s`
echo "script time: $(($end-$start))" 