#!/usr/bin/env bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_adda.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gen_configs_celldart.ipynb