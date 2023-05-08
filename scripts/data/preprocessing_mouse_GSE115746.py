#!/usr/bin/env python3
# %%
import glob
import os

import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from src.da_utils.data_processing import qc_sc


# %%
DSET_DIR = "data/mouse_cortex"

SC_ID = "GSE115746"

DATA_DIR = os.path.join(DSET_DIR, SC_ID)
RAW_PATH = os.path.join(DATA_DIR, "GSE115746_cells_exon_counts.csv")
RAW_PATH_META = os.path.join(DATA_DIR, "GSE115746_complete_metadata_28706-cells.csv")

sc_dir = os.path.join(DSET_DIR, "sc_adata")


# %%
os.system(
    f"wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746%5Fcells%5Fexon%5Fcounts%2Ecsv%2Egz -P '{DATA_DIR}'"
)

os.system(
    f"wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746%5Fcomplete%5Fmetadata%5F28706%2Dcells%2Ecsv%2Egz -P '{DATA_DIR}'"
)

os.system(f"gunzip -r '{DATA_DIR}' -f")
for f in glob.glob(os.path.join(DATA_DIR, "*.gz")):
    os.remove(f)

# %%
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# %%


adata_cortex = sc.read_csv(RAW_PATH).T
adata_cortex_meta = pd.read_csv(RAW_PATH_META, index_col=0)
adata_cortex_meta_ = adata_cortex_meta.loc[adata_cortex.obs.index,]

adata_cortex.obs = adata_cortex_meta_
adata_cortex.var_names_make_unique()

qc_sc(adata_cortex)


# %%
adata_cortex.X = csr_matrix(adata_cortex.X)

if not os.path.exists(sc_dir):
    os.makedirs(sc_dir)
adata_cortex.write(os.path.join(sc_dir, f"{SC_ID}.h5ad"))
