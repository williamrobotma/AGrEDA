#!/usr/bin/env bash

set -x

mkdir -p logs/ADDA
mkdir -p logs/CellDART
mkdir -p logs/CORAL
mkdir -p logs/DANN


./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 3 --samp_split --val_samp --nmarkers 40 # celldart, adda
./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 10 --samp_split --val_samp --nmarkers 80 # coral
./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 3 --samp_split --val_samp --nmarkers 80 # dann
./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 5 --samp_split --val_samp --nmarkers 40

ps_seeds=(3679 343 25 234 98098)

for i in "${!ps_seeds[@]}"; do
    ps_seed=${ps_seeds[$i]}
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 3 --samp_split --val_samp --nmarkers 40 --ps_seed=$ps_seed
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 10 --samp_split --val_samp --nmarkers 80 --ps_seed=$ps_seed
    ./prep_data.py -s minmax --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 3 --samp_split --val_samp --nmarkers 80 --ps_seed=$ps_seed
    ./prep_data.py -s standard --dset dlpfc --st_id spatialLIBD --sc_id GSE144136 --nmix 5 --samp_split --val_samp --nmarkers 40 --ps_seed=$ps_seed
done

nohup ./batch_scripts/ADDA/run_adda-dlpfc-FINAL.sh 1> logs/ADDA/run-dlpfc-FINAL.out 2> logs/ADDA/run-dlpfc-FINAL.err &
nohup ./batch_scripts/CellDART/run_CellDART-dlpfc-FINAL.sh 1> logs/CellDART/run-dlpfc-FINAL.out 2> logs/CellDART/run-dlpfc-FINAL.err &
nohup ./batch_scripts/CORAL/run_coral-dlpfc-FINAL.sh 1> logs/CORAL/run-dlpfc-FINAL.out 2> logs/CORAL/run-dlpfc-FINAL.err &
nohup ./batch_scripts/DANN/run_dann-dlpfc-FINAL.sh 1> logs/DANN/run-dlpfc-FINAL.out 2> logs/DANN/run-dlpfc-FINAL.err &
nohup ./batch_scripts/DANN/run_dann-dlpfc-FINAL-STANDARD.sh 1> logs/DANN/run-dlpfc-FINAL-STANDARD.out 2> logs/DANN/run-dlpfc-FINAL-STANDARD.err &
