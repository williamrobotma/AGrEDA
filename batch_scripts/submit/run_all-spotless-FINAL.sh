#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA
mkdir -p logs/CellDART
mkdir -p logs/CORAL
mkdir -p logs/DANN

./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --val_samp --nmarkers 80
./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 80
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 20
./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --val_samp --nmarkers 20
./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 40

ps_seeds=(3679 343 25 234 98098)

for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    ./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 5 --samp_split --val_samp --nmarkers 80 --ps_seed=$ps_seed
    ./prep_data.py -s minmax --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 80 --ps_seed=$ps_seed
    ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 20 --ps_seed=$ps_seed
    ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 40 --ps_seed=$ps_seed
    ./prep_data.py -s standard --dset mouse_cortex --st_id spotless_mouse_cortex --sc_id GSE115746 --nmix 10 --samp_split --val_samp --nmarkers 40 --ps_seed=$ps_seed
done

bash ./batch_scripts/ADDA/run_adda-spotless-FINAL.sh 1> logs/ADDA/run-spotless-FINAL.out 2> logs/ADDA/run-spotless-FINAL.err
bash ./batch_scripts/CellDART/run_CellDART-spotless-FINAL.sh 1> logs/CellDART/run-spotless-FINAL.out 2> logs/CellDART/run-spotless-FINAL.err
bash ./batch_scripts/CORAL/run_coral-spotless-FINAL.sh 1> logs/CORAL/run-spotless-FINAL.out 2> logs/CORAL/run-spotless-FINAL.err
bash ./batch_scripts/DANN/run_dann-spotless-FINAL.sh 1> logs/DANN/run-spotless-FINAL.out 2> logs/DANN/run-spotless-FINAL.err
bash ./batch_scripts/DANN/run_dann-spotless-FINAL-STANDARD.sh 1> logs/DANN/run-spotless-FINAL-STANDARD.out 2> logs/DANN/run-spotless-FINAL-STANDARD.err
