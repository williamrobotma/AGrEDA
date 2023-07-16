#!/usr/bin/env python3

# %%
import os
import pickle

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    from da_utils.data_processing import qc_sc

# %% [markdown]
# # Load and clean data

# %%
# pdac_a_indrop = pd.read_csv('data/pdac/GSE111672/suppl/GSE111672_PDAC-A-indrop-filtered-expMat.txt', sep='\t', index_col=0)
# pdac_b_indrop = pd.read_csv('data/pdac/GSE111672/suppl/GSE111672_PDAC-B-indrop-filtered-expMat.txt', sep='\t', index_col=0)
PDAC_DIR = "./data/pdac"
SERIES_ACCESSION = "GSE111672"
DATA_DIR = os.path.join(PDAC_DIR, SERIES_ACCESSION)
SC_DIR = os.path.join(DATA_DIR, "suppl")

pdac_a_indrop = sc.read_csv(
    os.path.join(SC_DIR, "GSE111672_PDAC-A-indrop-filtered-expMat.txt"), delimiter="\t"
).T
pdac_a_indrop.X = csr_matrix(pdac_a_indrop.X)
pdac_a_indrop.obs["cell_type"] = pdac_a_indrop.obs_names.to_series().astype("category")
pdac_a_indrop.obs_names_make_unique(join=".")
pdac_a_indrop.var_names_make_unique(join=".")
pdac_a_indrop = pdac_a_indrop[:, ~pdac_a_indrop.var_names.str.contains("-Mar")]


pdac_b_indrop = sc.read_csv(
    os.path.join(SC_DIR, "GSE111672_PDAC-B-indrop-filtered-expMat.txt"), delimiter="\t"
).T
pdac_b_indrop.X = csr_matrix(pdac_b_indrop.X)
pdac_b_indrop.obs["cell_type"] = pdac_b_indrop.obs_names.to_series().astype("category")
pdac_b_indrop.obs_names_make_unique(join=".")
pdac_b_indrop.var_names_make_unique(join=".")
pdac_b_indrop = pdac_b_indrop[:, ~pdac_b_indrop.var_names.str.contains("-Mar")]


# %%
# pdac_a_indrop.obs["cell_type"].map(lambda x: x.rstrip(" AB")).value_counts()

# %%
# pdac_b_indrop.obs["cell_type"].map(lambda x: x.rstrip(" AB")).value_counts()

# %%
ST_DIR = os.path.join(DATA_DIR, "st")

# pdac_a_st = pd.read_csv(os.path.join(ST_DIR, "GSM3036911/GSM3036911_PDAC-A-ST1-filtered.txt"), sep='\t', index_col=0)
# pdac_b_st = pd.read_csv(os.path.join(ST_DIR, "GSM3405534_PDAC-B-ST1-filtered.txt"), sep='\t', index_col=0)
pdac_a_st = sc.read_csv(
    os.path.join(ST_DIR, "GSM3036911/GSM3036911_PDAC-A-ST1-filtered.txt"), delimiter="\t"
).T
pdac_b_st = sc.read_csv(
    os.path.join(ST_DIR, "GSM3405534/GSM3405534_PDAC-B-ST1-filtered.txt"), delimiter="\t"
).T
pdac_a_st.X = csr_matrix(pdac_a_st.X)
pdac_b_st.X = csr_matrix(pdac_b_st.X)
pdac_a_st = pdac_a_st[:, ~pdac_a_st.var_names.str.endswith("-Mar")]
pdac_b_st = pdac_b_st[:, ~pdac_b_st.var_names.str.endswith("-Mar")]

# %%


# %%
# normalize_cpm = lambda x: np.log1p(x * 1e6 / x.sum())
# pdac_a_st = pdac_a_st.apply(normalize_cpm, axis=0)
# pdac_b_st = pdac_b_st.apply(normalize_cpm, axis=0)


# %% [markdown]
# ## Get coordinates for ST

# %%
pdac_a_st.obs = pdac_a_st.obs_names.to_series().str.split("x", expand=True).astype(int)
pdac_a_st.obs.columns = ["X", "Y"]

pdac_b_st.obs = pdac_b_st.obs_names.to_series().str.split("x", expand=True).astype(int)
pdac_b_st.obs.columns = ["X", "Y"]


# %%
# pdac_b_st.obs = pdac_b_st.columns.to_series().str.split("x", expand=True).astype(int)
pdac_a_st.obs["cell_type"] = pd.array([None] * pdac_a_st.shape[0], dtype="Int64")
pdac_b_st.obs["cell_type"] = pd.array([None] * pdac_b_st.shape[0], dtype="Int64")
# pdac_a_st.obs["cell_type"].dtype

# %% [markdown]
# # Annotate ST

# %%
cluster_assignments = [
    "Cancer region",
    "Pancreatic tissue",
    "Interstitium",
    "Duct epithelium",
    "Stroma",
]
cluster_to_ordinal = {c: i for i, c in enumerate(cluster_assignments)}
cluster_to_rgb = {
    "Cancer region": "#686868",
    "Pancreatic tissue": "#7670b0",
    "Interstitium": "#8dccc2",
    "Duct epithelium": "#e42a89",
    "Stroma": "#169c78",
}
print(pd.Series(cluster_to_ordinal))

# %%
pdac_a_st.obs.loc["10x10", "cell_type"] = 3
pdac_a_st.obs.loc["10x13", "cell_type"] = 3
pdac_a_st.obs.loc["10x14", "cell_type"] = 3
pdac_a_st.obs.loc["10x15", "cell_type"] = 3
pdac_a_st.obs.loc["10x16", "cell_type"] = 3
pdac_a_st.obs.loc["10x17", "cell_type"] = 3
pdac_a_st.obs.loc["10x19", "cell_type"] = 3
pdac_a_st.obs.loc["10x20", "cell_type"] = 3
pdac_a_st.obs.loc["10x24", "cell_type"] = 3
pdac_a_st.obs.loc["10x25", "cell_type"] = 3
pdac_a_st.obs.loc["10x26", "cell_type"] = 3
pdac_a_st.obs.loc["10x27", "cell_type"] = 3
pdac_a_st.obs.loc["10x28", "cell_type"] = 3
pdac_a_st.obs.loc["10x29", "cell_type"] = 4
pdac_a_st.obs.loc["10x30", "cell_type"] = 3
pdac_a_st.obs.loc["10x31", "cell_type"] = 4
pdac_a_st.obs.loc["10x32", "cell_type"] = 4
pdac_a_st.obs.loc["10x33", "cell_type"] = 3


# %%

pdac_a_st.obs.loc["11x11", "cell_type"] = 3
pdac_a_st.obs.loc["11x13", "cell_type"] = 3
pdac_a_st.obs.loc["11x14", "cell_type"] = 3
pdac_a_st.obs.loc["11x15", "cell_type"] = 3
pdac_a_st.obs.loc["11x16", "cell_type"] = 3
pdac_a_st.obs.loc["11x17", "cell_type"] = 3
pdac_a_st.obs.loc["11x19", "cell_type"] = 3
pdac_a_st.obs.loc["11x21", "cell_type"] = 3
pdac_a_st.obs.loc["11x22", "cell_type"] = 3
pdac_a_st.obs.loc["11x23", "cell_type"] = 3
pdac_a_st.obs.loc["11x24", "cell_type"] = 3
pdac_a_st.obs.loc["11x25", "cell_type"] = 3
pdac_a_st.obs.loc["11x26", "cell_type"] = 4
pdac_a_st.obs.loc["11x27", "cell_type"] = 3
pdac_a_st.obs.loc["11x28", "cell_type"] = 3
pdac_a_st.obs.loc["11x29", "cell_type"] = 3
pdac_a_st.obs.loc["11x30", "cell_type"] = 3
pdac_a_st.obs.loc["11x31", "cell_type"] = 3
pdac_a_st.obs.loc["11x32", "cell_type"] = 4
pdac_a_st.obs.loc["11x33", "cell_type"] = 0
pdac_a_st.obs.loc["11x34", "cell_type"] = 0


# %%
col_no = 12
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 4


# %%
col_no = 13
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 14
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1
# pdac_a_st.obs.iloc[0:20]


# %%
col_no = 15
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 16
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 17
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 18
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 19
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 20
pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 21
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 22
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 23
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 24
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 25
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 26
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 1
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 27
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 28
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 29
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 30
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 4
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 1
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 1


# %%
col_no = 9
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3


# %%
col_no = 8
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3


# %%
col_no = 7
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3


# %%
col_no = 6
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x9", "cell_type"] = 3


# %%
col_no = 5
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x9", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x8", "cell_type"] = 4


# %%
col_no = 4
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 4
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x9", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x8", "cell_type"] = 4


# %%
col_no = 3
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

# pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x9", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x8", "cell_type"] = 4


# %%
col_no = 2
# pdac_a_st.obs.loc[f"{col_no}x34", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x33", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x32", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x31", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x30", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x29", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x28", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x27", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x26", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x25", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x24", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x23", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x22", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_a_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
pdac_a_st.obs.loc[f"{col_no}x18", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x17", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x16", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x15", "cell_type"] = 3
pdac_a_st.obs.loc[f"{col_no}x14", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x13", "cell_type"] = 3

# pdac_a_st.obs.loc[f"{col_no}x11", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x10", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x9", "cell_type"] = 3
# pdac_a_st.obs.loc[f"{col_no}x8", "cell_type"] = 4


# %%

fig, axs = plt.subplots(1, 1, figsize=(20, 20))
for col_name, coords in pdac_a_st.obs.iterrows():
    if not pd.isna(coords["cell_type"]):
        spot_colour = cluster_to_rgb[cluster_assignments[coords["cell_type"]]]
        facecolor = spot_colour
    else:
        spot_colour = "k"
        facecolor = "none"
    axs.scatter(
        coords["X"], coords["Y"], marker="s", s=1000, facecolors=facecolor, edgecolors=spot_colour
    )
    axs.text(coords["X"], coords["Y"], col_name, ha="center", va="center")
    axs.set_aspect("equal")

print(pd.Series(cluster_to_ordinal))

# %%
col_no = 15
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3


# %%
col_no = 16

pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0

pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3


# %%
col_no = 17

pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3


# %%
col_no = 18

pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3

# %%
col_no = 19
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3

# %%
col_no = 20
pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
# pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 21
pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 22
pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 23
pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3


# %%
col_no = 24
pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 25
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 26
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 27
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 28
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 29
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 30
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
# pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 31
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
# pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
col_no = 32
# pdac_b_st.obs.loc[f"{col_no}x21", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x20", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x19", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x18", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x17", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x16", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x15", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x14", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x13", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x12", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x11", "cell_type"] = 0
# pdac_b_st.obs.loc[f"{col_no}x10", "cell_type"] = 0
pdac_b_st.obs.loc[f"{col_no}x9", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x8", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x7", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x6", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x5", "cell_type"] = 2
pdac_b_st.obs.loc[f"{col_no}x4", "cell_type"] = 2
# pdac_b_st.obs.loc[f"{col_no}x3", "cell_type"] = 3
# pdac_b_st.obs.loc[f"{col_no}x2", "cell_type"] = 3

# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
for col_name, coords in pdac_b_st.obs.iterrows():
    if not pd.isna(coords["cell_type"]):
        spot_colour = cluster_to_rgb[cluster_assignments[coords["cell_type"]]]
        facecolor = spot_colour
    else:
        spot_colour = "k"
        facecolor = "none"
    axs.scatter(
        coords["X"], coords["Y"], marker="s", s=600, facecolors=facecolor, edgecolors=spot_colour
    )
    axs.text(coords["X"], coords["Y"], col_name, ha="center", va="center", fontsize=7)
axs.set_aspect("equal")
pd.Series(cluster_to_ordinal)

# %% [markdown]
# # Save

# %%


SC_OUT_DIR = os.path.join(PDAC_DIR, "sc_adata")
ST_OUT_DIR = os.path.join(PDAC_DIR, "st_adata")

if not os.path.isdir(SC_OUT_DIR):
    os.makedirs(SC_OUT_DIR)
if not os.path.isdir(ST_OUT_DIR):
    os.makedirs(ST_OUT_DIR)

sample_ids = ["pdac_a", "pdac_b"]
indrop_out = ad.concat([pdac_a_indrop, pdac_b_indrop], label="sample_id", keys=sample_ids)
indrop_out.obs_names_make_unique()
# indrop_out.obs["cell_type"] = indrop_out.obs["cell_type"].map(
#     lambda x: x.split(" - ")[0].rstrip(" AB")
# )
indrop_out.obs["cell_type"] = indrop_out.obs["cell_type"].str.rstrip(" AB")


qc_sc(indrop_out)

cell_type_value_counts = indrop_out.obs["cell_type"].value_counts()
cell_types_to_keep = set(cell_type_value_counts[cell_type_value_counts >= 10].index.tolist())

indrop_out = indrop_out[indrop_out.obs["cell_type"].isin(cell_types_to_keep)]

print(indrop_out.obs["cell_type"].value_counts())

indrop_out.write(os.path.join(SC_OUT_DIR, f"{SERIES_ACCESSION}.h5ad"))
# pdac_b_indrop.write(os.path.join(SC_OUT_DIR, "pdac_b_indrop.h5ad"))
ordinal_to_cluster = {v: k for k, v in cluster_to_ordinal.items()}


for sample_id, st_adata in zip(sample_ids, [pdac_a_st, pdac_b_st]):
    # st_samp_dir = os.path.join(ST_OUT_DIR, sample_id)
    # if not os.path.isdir(st_samp_dir):
    #     os.makedirs(st_samp_dir)
    st_adata.obs["cell_type"] = st_adata.obs["cell_type"].map(ordinal_to_cluster)

    st_adata.write(os.path.join(ST_OUT_DIR, f"{SERIES_ACCESSION}-{sample_id}.h5ad"))


# pdac_a_st.write(os.path.join(ST_OUT_DIR, "pdac_a_st.h5ad"))
# pdac_b_st.write(os.path.join(ST_OUT_DIR, "pdac_b_st.h5ad"))

# pdac_a_indrop.write(os.path.join(SC_OUT_DIR, "pdac_a_indrop.h5ad"), compression="gzip")
# pdac_b_indrop.write(os.path.join(SC_OUT_DIR, "pdac_b_indrop.h5ad"), compression="gzip")
# pdac_a_st.write(os.path.join(ST_OUT_DIR, "pdac_a_st.h5ad"), compression="gzip")
# pdac_b_st.write(os.path.join(ST_OUT_DIR, "pdac_b_st.h5ad"), compression="gzip")

with open(os.path.join(ST_OUT_DIR, f"{SERIES_ACCESSION}-cluster_to_ordinal.pkl"), "wb") as f:
    pickle.dump(cluster_to_ordinal, f)
with open(os.path.join(ST_OUT_DIR, f"{SERIES_ACCESSION}-cluster_to_rgb.pkl"), "wb") as f:
    pickle.dump(cluster_to_rgb, f)

# %%
val_counts = indrop_out.obs["cell_type"].value_counts().reset_index()
val_counts["text"] = val_counts.apply(lambda x: f"{x['index']} ({x['cell_type']})", axis=1)

# %%
fig = plt.figure(figsize=(8, 8))

plt.pie(
    val_counts["cell_type"],
    labels=val_counts["text"],
    autopct="%1.1f%%",
    pctdistance=0.8,
    labeldistance=1.1,
)

# %%
plt.close()
