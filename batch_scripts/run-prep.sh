#!/usr/bin/env bash

set -x
for n in 5 10 20 40 80
do
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50 --nmarkers $n
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50 --nmarkers $n

./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50 --stsplit  --nmarkers $n
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50 --stsplit --nmarkers $n

./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id GSE111672 --nmix 50 --stsplit --one_model --nmarkers $n
./prep_data.py -s standard --dset pdac --st_id GSE111672 --sc_id CA001063 --nmix 50 --stsplit --one_model --nmarkers $n

for m in 5 8 10 15 20
do
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --nmarkers $n
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --nmarkers $n
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --nmarkers $n
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --one_model --nmarkers $n

    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --nmarkers $n
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --samp_split --nmarkers $n
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --nmarkers $n
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix $m --stsplit --one_model --nmarkers $n
done

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --nmix 10 --nmarkers $n
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --nmix 10 --samp_split --nmarkers $n
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --nmix 10 --stsplit --nmarkers $n
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id spotless_mouse_cortex --nmix 10 --stsplit --one_model --nmarkers $n

./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --nmarkers $n
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --nmarkers $n
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --stsplit --nmarkers $n
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --stsplit --one_model --nmarkers $n

done