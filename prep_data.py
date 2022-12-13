#!/usr/bin/env python3
"""Preps the data into sets."""
# %%
import glob
import pickle
import os
import argparse
from collections import OrderedDict

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import model_selection, preprocessing

from src.utils import data_processing

scaler_opts = ["minmax", "standard", "celldart"]

parser = argparse.ArgumentParser(description="Preps the data into sets.")
parser.add_argument(
    "--scaler",
    "-s",
    type=str,
    default="celldart",
    choices=scaler_opts,
    help="Scaler to use.",
)
parser.add_argument("--stsplit", action="store_true", help="Whether to split ST data.")
parser.add_argument(
    "--allgenes", "-a", action="store_true", help="Turn off marker selection."
)
args = parser.parse_args()

# %%
SPATIALLIBD_DIR = "./data/spatialLIBD"
SC_DLPFC_PATH = "./data/sc_dlpfc/adata_sc_dlpfc.h5ad"
GENELISTS_PATH = "data/sc_dlpfc/df_genelists.pkl"

SAVE_DIR = f"data/preprocessed_{'all' if args.allgenes else 'markers'}_{args.scaler}"

# %%
NUM_MARKERS = 20
N_MIX = 8
N_SPOTS = 20000


def main():
    if args.scaler == "minmax":
        Scaler = preprocessing.MinMaxScaler
    elif args.scaler == "standard":
        Scaler = preprocessing.StandardScaler
    elif args.scaler == "celldart":
        pass
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # %%[markdown]
    #  # Prepare Data
    #  ## Data load
    #  ### Load SpatialLIBD Data

    # %%
    print("Loading SpatialLIBD Data")

    adata_dir = os.path.join(SPATIALLIBD_DIR, "adata")

    adata_spatialLIBD_d = OrderedDict()

    for name in sorted(glob.glob(os.path.join(adata_dir, "adata_spatialLIBD-*.h5ad"))):
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
    print("Loading Single Cell Data")

    adata_sc_dlpfc = sc.read_h5ad(SC_DLPFC_PATH)
    sc.pp.normalize_total(adata_sc_dlpfc, inplace=True, target_sum=1e4)
    adata_sc_dlpfc.var_names_make_unique()

    # %%[markdown]
    #  ## Format Data

    # %%[markdown]
    # ### Array of single cell & spatial data
    # - Single cell data with labels
    # - Spatial data without labels

    # %%[markdown]
    # ### Generate Pseudospots

    # %%
    print("Splitting single cell data")
    df_sc = adata_sc_dlpfc.to_df()
    df_sc.index = pd.MultiIndex.from_frame(adata_sc_dlpfc.obs.reset_index())

    lab_sc_sub = df_sc.index.get_level_values("cell_subclass")

    (
        adata_sc_dlpfc_train,
        adata_sc_dlpfc_eval,
        lab_sc_sub_train,
        lab_sc_sub_eval,
    ) = model_selection.train_test_split(
        adata_sc_dlpfc,
        lab_sc_sub,
        test_size=0.2,
        random_state=225,
        stratify=lab_sc_sub,
    )

    (
        adata_sc_dlpfc_val,
        adata_sc_dlpfc_test,
        lab_sc_sub_val,
        lab_sc_sub_test,
    ) = model_selection.train_test_split(
        adata_sc_dlpfc_eval,
        lab_sc_sub_eval,
        test_size=0.5,
        random_state=263,
        stratify=lab_sc_sub_eval,
    )

    # %%
    print("Selecting genes")
    (
        (adata_sc_dlpfc_train, adata_spatialLIBD),
        df_genelists,
        (fig, ax),
    ) = data_processing.select_marker_genes(
        adata_sc_dlpfc_train,
        adata_spatialLIBD,
        n_markers=None if args.allgenes else NUM_MARKERS,
        genelists_path=GENELISTS_PATH,
    )
    fig.savefig(os.path.join(SAVE_DIR, "venn.png"))
    # %%

    sc_sub_dict = dict(zip(range(df_genelists.shape[1]), df_genelists.columns.tolist()))
    sc_sub_dict2 = dict((y, x) for x, y in sc_sub_dict.items())

    lab_sc_num_test = [sc_sub_dict2[ii] for ii in lab_sc_sub_test]
    lab_sc_num_test = np.asarray(lab_sc_num_test, dtype="int")

    lab_sc_num_val = [sc_sub_dict2[ii] for ii in lab_sc_sub_val]
    lab_sc_num_val = np.asarray(lab_sc_num_val, dtype="int")

    lab_sc_num_train = [sc_sub_dict2[ii] for ii in lab_sc_sub_train]
    lab_sc_num_train = np.asarray(lab_sc_num_train, dtype="int")

    adata_sc_dlpfc = adata_sc_dlpfc[:, adata_sc_dlpfc_train.var.index]
    adata_sc_dlpfc_val = adata_sc_dlpfc_val[:, adata_sc_dlpfc_train.var.index]
    adata_sc_dlpfc_test = adata_sc_dlpfc_test[:, adata_sc_dlpfc_train.var.index]

    # %%
    print("Generating Pseudospots")

    sc_mix_train, lab_mix_train = data_processing.random_mix(
        adata_sc_dlpfc_train.to_df().to_numpy(),
        lab_sc_num_train,
        nmix=N_MIX,
        n_samples=N_SPOTS,
        seed=251,
    )
    sc_mix_val, lab_mix_val = data_processing.random_mix(
        adata_sc_dlpfc_val.to_df().to_numpy(),
        lab_sc_num_val,
        nmix=N_MIX,
        n_samples=N_SPOTS // 8,
        seed=55,
    )
    sc_mix_test, lab_mix_test = data_processing.random_mix(
        adata_sc_dlpfc_test.to_df().to_numpy(),
        lab_sc_num_test,
        nmix=N_MIX,
        n_samples=N_SPOTS // 8,
        seed=119,
    )

    print("Log scaling pseudospots")
    if args.scaler == "celldart":
        sc_mix_train_s = data_processing.log_minmaxscale(sc_mix_train)
        sc_mix_val_s = data_processing.log_minmaxscale(sc_mix_val)
        sc_mix_test_s = data_processing.log_minmaxscale(sc_mix_test)
    else:
        sc_scaler = Scaler()
        sc_mix_train_s = sc_scaler.fit_transform(np.log1p(sc_mix_train))
        sc_mix_val_s = sc_scaler.transform(np.log1p(sc_mix_val))
        sc_mix_test_s = sc_scaler.transform(np.log1p(sc_mix_test))

    # %%[markdown]
    # ### Format Spatial Data

    # %%
    print("Log scaling spatial data")
    mat_sp_train_d = {}
    mat_sp_train_s_d = {}
    if args.stsplit:
        mat_sp_test_d = {}
        mat_sp_test_s_d = {}
        mat_sp_val_d = {}
        mat_sp_val_s_d = {}
    for sample_id in st_sample_id_l:
        X_st_train = adata_spatialLIBD[
            adata_spatialLIBD.obs.sample_id == sample_id
        ].X.toarray()

        if args.stsplit:
            X_st_train, X_st_val = model_selection.train_test_split(
                X_st_train,
                test_size=0.2,
                random_state=163,
            )

            X_st_val, X_st_test = model_selection.train_test_split(
                X_st_val,
                test_size=0.5,
                random_state=195,
            )
            mat_sp_val_d[sample_id] = X_st_val
            mat_sp_test_d[sample_id] = X_st_test
        mat_sp_train_d[sample_id] = X_st_train

        if args.scaler == "celldart":
            mat_sp_train_s_d[sample_id] = data_processing.log_minmaxscale(
                mat_sp_train_d[sample_id]
            )
            if args.stsplit:
                mat_sp_val_s_d[sample_id] = data_processing.log_minmaxscale(
                    mat_sp_val_d[sample_id]
                )
                mat_sp_test_s_d[sample_id] = data_processing.log_minmaxscale(
                    mat_sp_test_d[sample_id]
                )
        else:
            sp_scaler = Scaler()
            mat_sp_train_s_d[sample_id] = sp_scaler.fit_transform(
                np.log1p(mat_sp_train_d[sample_id])
            )
            if args.stsplit:
                mat_sp_val_s_d[sample_id] = sp_scaler.transform(
                    np.log1p(mat_sp_val_d[sample_id])
                )
                mat_sp_test_s_d[sample_id] = sp_scaler.transform(
                    np.log1p(mat_sp_test_d[sample_id])
                )

    # if TRAIN_USING_ALL_ST_SAMPLES:
    mat_sp_train = adata_spatialLIBD.X.toarray()
    if args.stsplit:
        mat_sp_train, mat_sp_val = model_selection.train_test_split(
            mat_sp_train,
            test_size=0.2,
            random_state=629,
        )
        mat_sp_val, mat_sp_test = model_selection.train_test_split(
            mat_sp_val,
            test_size=0.5,
            random_state=18,
        )
    if args.scaler == "celldart":
        mat_sp_train_s = data_processing.log_minmaxscale(mat_sp_train)
        if args.stsplit:
            mat_sp_val_s = data_processing.log_minmaxscale(mat_sp_val)
            mat_sp_test_s = data_processing.log_minmaxscale(mat_sp_test)
    else:
        sp_all_scaler = Scaler()
        mat_sp_train_s = sp_all_scaler.fit_transform(np.log1p(mat_sp_train))
        if args.stsplit:
            mat_sp_val_s = sp_all_scaler.transform(np.log1p(mat_sp_val))
            mat_sp_test_s = sp_all_scaler.transform(np.log1p(mat_sp_test))

    # %%[markdown]
    # # Export

    # %%
    print("Exporting")
    if args.stsplit:
        with h5py.File(os.path.join(SAVE_DIR, "mat_sp_split_s_d.hdf5"), "w") as f:
            for grp_name in mat_sp_train_s_d:
                grp_samp = f.create_group(grp_name)
                dset = grp_samp.create_dataset("train", data=mat_sp_train_s_d[grp_name])
                dset = grp_samp.create_dataset("val", data=mat_sp_val_s_d[grp_name])
                dset = grp_samp.create_dataset("test", data=mat_sp_test_s_d[grp_name])

        with h5py.File(os.path.join(SAVE_DIR, "mat_sp_split_s.hdf5"), "w") as f:
            grp_samp = f.create_group("all")
            dset = grp_samp.create_dataset("train", data=mat_sp_train_s)
            dset = grp_samp.create_dataset("val", data=mat_sp_val_s)
            dset = grp_samp.create_dataset("test", data=mat_sp_test_s)
    else:
        with h5py.File(os.path.join(SAVE_DIR, "mat_sp_train_s_d.hdf5"), "w") as f:
            for dset_name in mat_sp_train_s_d:
                dset = f.create_dataset(dset_name, data=mat_sp_train_s_d[dset_name])

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

    print("Done")


if __name__ == "__main__":
    main()
