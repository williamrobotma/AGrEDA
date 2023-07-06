#!/usr/bin/env bash

./hyperval_generated.py -n ADDA -cdir configs/generated_pdac --njobs 32 -e > logs/ADDA/generated_pdac/final_hyperval.txt
./hyperval_generated.py -n CellDART -cdir configs/generated_pdac --njobs 32 -e > logs/CellDART/generated_pdac/final_hyperval.txt
./hyperval_generated.py -n DANN -cdir configs/generated_pdac --njobs 32 -e > logs/DANN/generated_pdac/final_hyperval.txt
./hyperval_generated.py -n CORAL -cdir configs/generated_pdac --njobs 32 > logs/CORAL/generated_pdac/final_hyperval.txt
