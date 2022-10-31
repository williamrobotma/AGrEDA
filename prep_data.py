#!/usr/bin/env python3
"""Preps the data into sets."""
# %%
import glob
import pickle
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import model_selection
import matplotlib.pyplot as plt

from src.utils import data_processing

# %%
SPATIALLIBD_DIR = "./data/spatialLIBD"
SC_DLPFC_PATH = "./data/sc_dlpfc/adata_sc_dlpfc.h5ad"
GENELISTS_PATH = "data/sc_dlpfc/df_genelists.pkl"
SAVE_DIR = "data/preprocessed"

# %%
NUM_MARKERS = 20
N_MIX = 8
N_SPOTS = 20000

# %%[markdown]
#  # Prepare Data
#  ## Data load
#  ### Load SpatialLIBD Data


# %%
adata_dir = os.path.join(SPATIALLIBD_DIR, "adata")

adata_spatialLIBD_d = {}

for name in glob.glob(os.path.join(adata_dir, "adata_spatialLIBD-*.h5ad")):
    sample_id = name.partition("-")[2].rpartition(".")[0]
    adata_spatialLIBD_d[sample_id] = sc.read_h5ad(name)

adata_spatialLIBD = ad.concat(
    adata_spatialLIBD_d.values(), label="sample_id", keys=adata_spatialLIBD_d.keys()
)
adata_spatialLIBD.obs_names_make_unique()
sc.pp.normalize_total(adata_spatialLIBD, inplace=True, target_sum=1e4)
st_sample_id_l = adata_spatialLIBD.obs["sample_id"].unique()

adata_spatialLIBD.var_names_make_unique()

# %%[markdown]
#  ### Load Single Cell Data

# %%
adata_sc_dlpfc = sc.read_h5ad(SC_DLPFC_PATH)
sc.pp.normalize_total(adata_sc_dlpfc, inplace=True, target_sum=1e4)
adata_sc_dlpfc.var_names_make_unique()
# %%
(adata_sc_dlpfc, adata_spatialLIBD), df_genelists, (fig, ax) = data_processing.select_marker_genes(
    adata_sc_dlpfc,
    adata_spatialLIBD,
    NUM_MARKERS,
    genelists_path=GENELISTS_PATH,
)

fig.savefig('results/venn.png')

# %%[markdown]
#  ## Format Data

# %%[markdown]
# ### Array of single cell & spatial data
# - Single cell data with labels
# - Spatial data without labels

# %%[markdown]
# ### Generate Pseudospots

# %%
df_sc = adata_sc_dlpfc.to_df()
df_sc.index = pd.MultiIndex.from_frame(adata_sc_dlpfc.obs.reset_index())

sc_sub_dict = dict(zip(range(df_genelists.shape[1]), df_genelists.columns.tolist()))
sc_sub_dict2 = dict((y, x) for x, y in sc_sub_dict.items())

lab_sc_sub = df_sc.index.get_level_values("cell_subclass")
lab_sc_num = [sc_sub_dict2[ii] for ii in lab_sc_sub]
lab_sc_num = np.asarray(lab_sc_num, dtype="int")

(
    mat_sc_train,
    mat_sc_eval,
    lab_sc_num_train,
    lab_sc_num_eval,
) = model_selection.train_test_split(
    df_sc.to_numpy(),
    lab_sc_num,
    test_size=0.2,
    random_state=225,
    stratify=lab_sc_num,
)

(
    mat_sc_val,
    mat_sc_test,
    lab_sc_num_val,
    lab_sc_num_test,
) = model_selection.train_test_split(
    mat_sc_eval,
    lab_sc_num_eval,
    test_size=0.5,
    random_state=263,
    stratify=lab_sc_num_eval,
)

sc_mix_train, lab_mix_train = data_processing.random_mix(
    mat_sc_train, lab_sc_num_train, nmix=N_MIX, n_samples=N_SPOTS
)
sc_mix_val, lab_mix_val = data_processing.random_mix(
    mat_sc_val, lab_sc_num_val, nmix=N_MIX, n_samples=N_SPOTS // 8
)
sc_mix_test, lab_mix_test = data_processing.random_mix(
    mat_sc_test, lab_sc_num_test, nmix=N_MIX, n_samples=N_SPOTS // 8
)


sc_mix_train_s = data_processing.log_minmaxscale(sc_mix_train)
sc_mix_val_s = data_processing.log_minmaxscale(sc_mix_val)
sc_mix_test_s = data_processing.log_minmaxscale(sc_mix_test)

# %%[markdown]
# ### Format Spatial Data

# %%
mat_sp_test_d = {}
mat_sp_test_s_d = {}
for sample_id in st_sample_id_l:
    mat_sp_test_d[sample_id] = adata_spatialLIBD[
        adata_spatialLIBD.obs.sample_id == sample_id
    ].X.todense()

    mat_sp_test_s_d[sample_id] = data_processing.log_minmaxscale(
        mat_sp_test_d[sample_id]
    )

# if TRAIN_USING_ALL_ST_SAMPLES:
mat_sp_train = adata_spatialLIBD.X.todense()
mat_sp_train_s = data_processing.log_minmaxscale(mat_sp_train)

# %%[markdown]
# # Export

# %%
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with h5py.File(os.path.join(SAVE_DIR, "mat_sp_test_s_d.hdf5"), "w") as f:
    for dset_name in mat_sp_test_s_d:
        dset = f.create_dataset(dset_name, data=mat_sp_test_s_d[dset_name])

with h5py.File(os.path.join(SAVE_DIR, "mat_sp_train_s.hdf5"), "w") as f:
    dset = f.create_dataset("all", data=mat_sp_train_s)

with h5py.File(os.path.join(SAVE_DIR, "sc.hdf5"), "w") as f:
    grp_x = f.create_group("X")
    dset = grp_x.create_dataset("train", data=sc_mix_train_s)
    dset = grp_x.create_dataset("val", data=sc_mix_val_s)
    dset = grp_x.create_dataset("test", data=sc_mix_test_s)

    grp_y = f.create_group("y")
    dset = grp_y.create_dataset("train", data=lab_mix_train)
    dset = grp_y.create_dataset("val", data=lab_mix_val)
    dset = grp_y.create_dataset("test", data=lab_mix_test)


adata_sc_dlpfc.write(os.path.join(SAVE_DIR, "adata_sc_dlpfc.h5ad"))
adata_spatialLIBD.write(os.path.join(SAVE_DIR, "adata_spatialLIBD.h5ad"))

# %%
with open(os.path.join(SAVE_DIR, "sc_sub_dict.pkl"), "wb") as f:
    pickle.dump(sc_sub_dict, f)

with open(os.path.join(SAVE_DIR, "sc_sub_dict2.pkl"), "wb") as f:
    pickle.dump(sc_sub_dict2, f)

with open(os.path.join(SAVE_DIR, "st_sample_id_l.pkl"), "wb") as f:
    pickle.dump(st_sample_id_l.tolist(), f)
