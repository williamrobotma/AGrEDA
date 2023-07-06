#!/usr/bin/env bash

set -x
for n in 20 40 80
do
    for m in 30 50 70
    do
        # ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --nmarkers $n

        # ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --nmarkers $n

        # ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --nmarkers $n
        # ./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix $m --samp_split --nmarkers $n

        ./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix $m --one_model --nmarkers $n --njobs 32
        ./prep_data.py -s minmax --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix $m --one_model --nmarkers $n --njobs 32

    done
done