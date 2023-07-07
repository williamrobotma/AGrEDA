#!/bin/bash

for (( SLURM_ARRAY_TASK_ID=1; SLURM_ARRAY_TASK_ID<=991; SLURM_ARRAY_TASK_ID+=10 )); do
    sbatch --output="./logs/CellDART/generated_spotless/gen_v1-${SLURM_ARRAY_TASK_ID}-eval.out" --export=SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID ./batch_scripts/CellDART/sbatch-celldart-ht-spotless-eval.sh
    sleep 2
done

for (( SLURM_ARRAY_TASK_ID=1; SLURM_ARRAY_TASK_ID<=991; SLURM_ARRAY_TASK_ID+=10 )); do
    sbatch --output="./logs/ADDA/generated_spotless/gen_v1-${SLURM_ARRAY_TASK_ID}-eval.out" --export=SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID ./batch_scripts/ADDA/sbatch-adda-ht-spotless-eval.sh
    sleep 2
done

for (( SLURM_ARRAY_TASK_ID=1; SLURM_ARRAY_TASK_ID<=991; SLURM_ARRAY_TASK_ID+=100 )); do
    sbatch --output="./logs/DANN/generated_spotless/gen_v1-${SLURM_ARRAY_TASK_ID}-eval.out" --export=SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID ./batch_scripts/DANN/sbatch-dann-ht-spotless-eval.sh
    sleep 2
done