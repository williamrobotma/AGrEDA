#!/usr/bin/env bash

set -x

python -m src.da_utils.scripts.data.download_pdac
python -m src.da_utils.scripts.data.preprocess_pdac_GSE111672
python -m src.da_utils.scripts.data.preprocess_pdac_zenodo6024273
python -m src.da_utils.scripts.data.preprocessing_libd
python -m src.da_utils.scripts.data.preprocessing_dlpfc
python -m src.da_utils.scripts.data.preprocessing_spotless
python -m src.da_utils.scripts.data.preprocessing_mouse_GSE115746
