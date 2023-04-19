#!/usr/bin/env bash

set -x

./scripts/data/download_pdac.py
./scripts/data/preprocess_pdac_GSE111672.py
./scripts/data/preprocess_pdac_zenodo6024273.py
./scripts/data/preprocessing_libd.py
./scripts/data/preprocessing_dlpfc.py
