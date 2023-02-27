#!/usr/bin/env bash
salloc --time=2:00:00 --ntasks=1 --gpus=1 --cpus-per-task=8 --mem-per-cpu=2G srun $VIRTUAL_ENV/bin/jupyterlab.sh
