#!/usr/bin/env bash

set -x

python -m scripts.data.download_pdac
python -m scripts.data.preprocess_pdac_GSE111672
python -m scripts.data.preprocess_pdac_zenodo6024273
python -m scripts.data.preprocessing_libd
python -m scripts.data.preprocessing_dlpfc
python -m scripts.data.preprocessing_spotless
python -m scripts.data.preprocessing_mouse_GSE115746
