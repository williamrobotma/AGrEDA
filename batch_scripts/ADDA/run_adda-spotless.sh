#!/usr/bin/env bash

start=`date +%s`

for n in {1..100}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_spotless/ADDA/a_list.txt)
    echo "ADDA config file no. ${n}: ${CONFIG_FILE}"
    # python -u adda.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" -c 3 2>> logs/ADDA/generated_spotless/training.err 1>> logs/ADDA/generated_spotless/training.out
    echo "Evaluating"
    ./eval_config.py -n ADDA -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --early_stopping --njobs 32 -c 3

done

end=`date +%s`
echo "script time: $(($end-$start))" 