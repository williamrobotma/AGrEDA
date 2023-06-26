#!/usr/bin/env bash

# for n in 5 10 20 40 80
# do
#     for m in 5 8 10 15 20
#     do
#         echo "Preprocessing nmarkers: ${n} nmix: ${m}"
#         ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --nmarkers $n
#     done
# done

start=`date +%s`

for n in {1..100}; do

    CONFIG_FILE=$(sed -n "${n}p" configs/generated_spotless/ADDA/a_list.txt)
    echo "ADDA config file no. ${n}: ${CONFIG_FILE}"
    python -u adda.py -f "${CONFIG_FILE}" -l "log.txt" -cdir "configs/generated_spotless" 2>> logs/ADDA/generated_spotless/training.err 1>> logs/ADDA/generated_spotless/training.out
    echo "Evaluating"
    ./eval_config.py -n ADDA -f "${CONFIG_FILE}" -cdir "configs/generated_spotless" --early_stopping --njobs -1

done

end=`date +%s`
echo "script time: $(($end-$start))" 