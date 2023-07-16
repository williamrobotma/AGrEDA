#!/usr/bin/env python3

# %%
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    from da_utils.data_processing import qc_sc

# %%
dset_dir = "data/dlpfc"
spatialLIBD_dir = os.path.join(dset_dir, "spatialLIBD_data")
SC_ID = "GSE144136"
ST_ID = "spatialLIBD"
st_dir = os.path.join(dset_dir, "st_adata")
sc_dir = os.path.join(dset_dir, "sc_adata")


# %%
sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3

# %%
temp = pd.read_pickle(os.path.join(spatialLIBD_dir, "temp.pkl"))
dlpfc = pd.read_pickle(os.path.join(spatialLIBD_dir, "dlpfc.pkl"))

# %%
sample_ids = temp.index.get_level_values("sample_id").unique()

# %%
print(temp.loc[sample_ids[0]])

# %%

# adata_spatial_anterior = sc.datasets.visium_sge(
#     sample_id="V1_Mouse_Brain_Sagittal_Anterior"
# )
# adata_spatial_posterior = sc.datasets.visium_sge(
#     sample_id="V1_Mouse_Brain_Sagittal_Posterior"
# )
# adata_dir = os.path.join(spatialLIBD_dir, 'adata')
if not os.path.exists(st_dir):
    os.makedirs(st_dir)

for sample_id in sample_ids:
    print(f"Creating adata for sample_id {sample_id}")

    adata_spatialLIBD = sc.AnnData(temp.loc[sample_id], dtype=np.float32)
    # sc.pp.normalize_total(adata_spatialLIBD, inplace=True)
    adata_spatialLIBD.obs = pd.concat(
        [adata_spatialLIBD.obs, dlpfc.loc[sample_id]], axis=1, join="inner"
    )
    adata_spatialLIBD.X = csr_matrix(adata_spatialLIBD.X)
    adata_spatialLIBD.write(os.path.join(st_dir, f"{ST_ID}-{sample_id}.h5ad"))

# %%
# adata_spatialLIBD = sc.read_h5ad(os.path.join(spatialLIBD_dir, 'adata_spatialLIBD.h5ad'))

# %% [markdown]
# # Single cell Data: GSE115746
#

# %%
raw_sc_dir = os.path.join(dset_dir, "sc_dlpfc")
try:
    adata_sc_dlpfc = sc.read_mtx(
        os.path.join(raw_sc_dir, "GSE144136_GeneBarcodeMatrix_Annotated.mtx")
    ).T
    adata_sc_dlpfc_var_names = pd.read_csv(
        os.path.join(raw_sc_dir, "GSE144136_GeneNames.csv"), index_col=0
    ).x
    adata_sc_dlpfc_obs_names = pd.read_csv(
        os.path.join(raw_sc_dir, "GSE144136_CellNames.csv"), index_col=0
    ).x
except FileNotFoundError as e:
    raise FileNotFoundError(f"Download from GSE144136 into {raw_sc_dir}") from e

# %%
assert (adata_sc_dlpfc_var_names.index - 1 == adata_sc_dlpfc.var.index.astype(int)).all()
assert (adata_sc_dlpfc_obs_names.index - 1 == adata_sc_dlpfc.obs.index.astype(int)).all()

# %%
adata_sc_dlpfc.var_names = adata_sc_dlpfc_var_names
adata_sc_dlpfc.obs_names = adata_sc_dlpfc_obs_names

# %%
print(
    adata_sc_dlpfc.obs_names.to_series()
    .str.split(".", 1, expand=True)[1]
    .str.split("_", 1, expand=True)[0]
    .nunique()
)

# %%
adata_sc_dlpfc = adata_sc_dlpfc[adata_sc_dlpfc.obs.index.str.contains("Control")]

# %%
adata_sc_dlpfc.obs["patient"] = (
    adata_sc_dlpfc.obs_names.to_series()
    .str.split(".", 1, expand=True)[1]
    .str.split("_", 1, expand=True)[0]
    .tolist()
)

sc.pp.filter_genes(adata_sc_dlpfc, min_cells=3)
sc.pp.filter_cells(adata_sc_dlpfc, min_genes=200)

qc_sc(adata_sc_dlpfc)

assert adata_sc_dlpfc.obs["patient"].nunique() == 17

# %%
adata_sc_dlpfc.obs["cell_type"] = adata_sc_dlpfc.obs.index.str.split(
    ".", expand=True
).get_level_values(0)

# %%
print(adata_sc_dlpfc.obs["cell_type"].unique())

# %%
adata_sc_dlpfc.X = csr_matrix(adata_sc_dlpfc.X)

# %%
# sc.pp.normalize_total(adata_sc_dlpfc, inplace=True)
adata_sc_dlpfc.var_names_make_unique()
if not os.path.exists(sc_dir):
    os.makedirs(sc_dir)
adata_sc_dlpfc.write(os.path.join(sc_dir, f"{SC_ID}.h5ad"))

# %%
