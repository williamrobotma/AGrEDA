#!/usr/bin/env bash

set -x

./scripts/data/download_pdac.py
./scripts/data/preprocess_pdac.py
./scripts/data/preprocessing_libd.py
./scripts/data/preprocessing_dlpfc.py
