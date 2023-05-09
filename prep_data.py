#!/usr/bin/env python3
"""Preps the data into sets."""

import argparse
import glob
import logging
import os
import pickle
import warnings
from collections import OrderedDict
from itertools import accumulate

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
from sklearn import model_selection, preprocessing

from src.da_utils import data_loading, data_processing, misc

SPLIT_RATIOS = (0.8, 0.1, 0.1)
DATA_DIR = "./data"
# SPATIALLIBD_BASEPATH = "spatialLIBD"
# SC_DLPFC_PATH = os.path.join(DATA_DIR, "sc_dlpfc", "adata_sc_dlpfc.h5ad")


SCALER_OPTS = ("minmax", "standard", "celldart")

logger = logging.getLogger(__name__)


# %%
def main(args):
    dset_dir = data_loading.get_dset_dir(DATA_DIR, args.dset)
    selected_dir = data_loading.get_selected_dir(
        dset_dir,
        sc_id=args.sc_id,
        st_id=args.st_id,
        n_markers=args.nmarkers,
        all_genes=args.allgenes,
    )

    print("Selecting subset genes and splitting single-cell data")
    print("-" * 80)
    select_genes_and_split(
        dset_dir,
        sc_id=args.sc_id,
        st_id=args.st_id,
        n_markers=args.nmarkers,
        all_genes=args.allgenes,
        rng=462,
    )

    print("Generating Pseudospots")
    print("-" * 80)
    sc_mix_d, lab_mix_d = gen_pseudo_spots(
        selected_dir, n_mix=args.nmix, n_spots=args.nspots, rng=623, n_jobs=args.njobs
    )

    print("Log scaling pseudospots")
    print("-" * 80)
    log_scale_pseudospots(
        selected_dir,
        args.scaler,
        n_mix=args.nmix,
        n_spots=args.nspots,
        sc_mix_d=sc_mix_d,
        lab_mix_d=lab_mix_d,
    )

    print("Log scaling and maybe splitting spatial data")
    print("-" * 80)
    split_st(
        selected_dir,
        stsplit=args.stsplit,
        samp_split=args.samp_split,
        rng=16,
    )
    log_scale_st(
        selected_dir,
        scaler_name=args.scaler,
        stsplit=args.stsplit,
        samp_split=args.samp_split,
    )

    print("Log scaling all spatial data...")
    print("-" * 80)
    # if TRAIN_USING_ALL_ST_SAMPLES:
    log_scale_all_st(selected_dir, args.scaler)


def scale(scaler, *unscaled):
    if scaler == "celldart":
        yield from (data_processing.log_minmaxscale(x) for x in unscaled)

    else:
        sp_scaler = scaler().fit(np.log1p(unscaled[0]))
        yield from (sp_scaler.transform(np.log1p(x)) for x in unscaled)


def get_scaler(scaler_name):
    if scaler_name == "minmax":
        return preprocessing.MinMaxScaler
    if scaler_name == "standard":
        return preprocessing.StandardScaler
    if scaler_name == "celldart":
        warnings.warn(
            "celldart scaler is provided for legacy purposes only. " "Use minmax instead."
        )
        return scaler_name

    raise ValueError(f"Scaler '{scaler_name}' not recognized.")


def check_selected_split_exists(selected_dir):
    """Check if selected and split data exists and all files are present

    Args:
        selected_dir (str): Path to selected data directory.

    Returns:
        bool: Whether selected and split data exists.

    """

    # label numbers
    if not os.path.isfile(os.path.join(selected_dir, f"lab_sc_num.hdf5")):
        return False

    # df genelists
    if not os.path.isfile(os.path.join(selected_dir, f"df_genelists.pkl")):
        return False

    if not os.path.isfile(os.path.join(selected_dir, f"venn.png")):
        return False

    # sc adatas
    if not os.path.isfile(os.path.join(selected_dir, f"sc.h5ad")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, f"sc_train.h5ad")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, f"sc_val.h5ad")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, f"sc_test.h5ad")):
        return False

    # st adata
    if not os.path.isfile(os.path.join(selected_dir, f"st.h5ad")):
        return False

    # dicts and helpers
    if not os.path.isfile(os.path.join(selected_dir, f"sc_sub_dict.pkl")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, f"sc_sub_dict2.pkl")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, f"st_sample_id_l.pkl")):
        return False

    # All files present
    return True


def select_genes_and_split(
    dset_dir,
    sc_id=data_loading.DEF_SC_ID,
    st_id=data_loading.DEF_ST_ID,
    n_markers=data_loading.DEFAULT_N_MARKERS,
    all_genes=False,
    rng=None,
):
    """Select genes and split sc data into train, val, and test sets

    Args:
        n_markers (int): Number of top sc train genes to select. Ignored if
            `allgenes`. Default: 20.
        allgenes (bool): Whether to use all intersecting genes. Default: False.
        spatiallibd_dir (str): Path to spatialLIBD directory. Default:
            SPATIALLIBD_DIR.
        sc_dlpfc_path (str): Path to sc DLPFC data. Default: SC_DLPFC_PATH.
        rng: Random number generator or seed for numpy's rng. Default: None.

    """
    rng_integers = misc.check_integer_rng(rng)

    selected_dir = data_loading.get_selected_dir(
        dset_dir, sc_id=sc_id, st_id=st_id, n_markers=n_markers, all_genes=all_genes
    )

    if check_selected_split_exists(selected_dir):
        print("Selected and split data already exists. Skipping.")
        return

    if not os.path.isdir(selected_dir):
        os.makedirs(selected_dir)

    print("Loading ST Data")

    st_dir = os.path.join(dset_dir, "st_adata")

    adata_st_d = OrderedDict()

    for name in sorted(glob.glob(os.path.join(st_dir, f"{st_id}-*.h5ad"))):
        sample_id = name.partition("-")[2].rpartition(".")[0]
        adata_st_d[sample_id] = sc.read_h5ad(name)

    adata_st = ad.concat(adata_st_d.values(), label="sample_id", keys=adata_st_d.keys())
    adata_st.obs_names_make_unique()
    sc.pp.normalize_total(adata_st, inplace=True, target_sum=1e4)
    adata_st.var_names_make_unique()

    st_sample_id_l = adata_st.obs["sample_id"].unique()

    print("Loading Single Cell Data")
    sc_path = os.path.join(dset_dir, "sc_adata", f"{sc_id}.h5ad")
    adata_sc = sc.read_h5ad(sc_path)
    sc.pp.normalize_total(adata_sc, inplace=True, target_sum=1e4)
    adata_sc.var_names_make_unique()

    print("Splitting single cell data")
    # df_sc = adata_sc_dlpfc.to_df()
    # df_sc.index = pd.MultiIndex.from_frame(adata_sc_dlpfc.obs.reset_index())

    # lab_sc_sub = df_sc.index.get_level_values("cell_subclass")
    lab_sc_sub = adata_sc.obs["cell_subclass"]
    logger.debug(f"lab_sc_sub counts: {lab_sc_sub.value_counts()}")
    (
        adata_sc_train,
        adata_sc_eval,
        lab_sc_sub_train,
        lab_sc_sub_eval,
    ) = model_selection.train_test_split(
        adata_sc,
        lab_sc_sub,
        test_size=0.2,
        random_state=rng_integers(2**32),
        stratify=lab_sc_sub,
    )
    logger.debug(f"lab_sc_sub_eval counts: {lab_sc_sub_eval.value_counts()}")
    (
        adata_sc_val,
        adata_sc_test,
        lab_sc_sub_val,
        lab_sc_sub_test,
    ) = model_selection.train_test_split(
        adata_sc_eval,
        lab_sc_sub_eval,
        test_size=0.5,
        random_state=rng_integers(2**32),
        stratify=lab_sc_sub_eval,
    )

    print("Selecting genes")
    (
        (adata_sc_train, adata_st),
        df_genelists,
        (fig, ax),
    ) = data_processing.select_marker_genes(
        adata_sc_train,
        adata_st,
        n_markers=None if all_genes else n_markers,
        genelists_path=os.path.join(selected_dir, "df_genelists.pkl"),
    )

    fig.savefig(os.path.join(selected_dir, "venn.png"))

    sc_sub_dict = dict(zip(range(df_genelists.shape[1]), df_genelists.columns.tolist()))
    sc_sub_dict2 = dict((y, x) for x, y in sc_sub_dict.items())

    lab_sc_num_test = [sc_sub_dict2[ii] for ii in lab_sc_sub_test]
    lab_sc_num_test = np.asarray(lab_sc_num_test, dtype="int")

    lab_sc_num_val = [sc_sub_dict2[ii] for ii in lab_sc_sub_val]
    lab_sc_num_val = np.asarray(lab_sc_num_val, dtype="int")

    lab_sc_num_train = [sc_sub_dict2[ii] for ii in lab_sc_sub_train]
    lab_sc_num_train = np.asarray(lab_sc_num_train, dtype="int")

    adata_sc = adata_sc[:, adata_sc_train.var.index]
    adata_sc_val = adata_sc_val[:, adata_sc_train.var.index]
    adata_sc_test = adata_sc_test[:, adata_sc_train.var.index]

    print("Saving sc labels")
    with h5py.File(os.path.join(selected_dir, f"lab_sc_num.hdf5"), "w") as f:
        f.create_dataset("train", data=lab_sc_num_train)
        f.create_dataset("val", data=lab_sc_num_val)
        f.create_dataset("test", data=lab_sc_num_test)

    print("Saving sc adatas")
    adata_sc.write(os.path.join(selected_dir, "sc.h5ad"))
    adata_sc_train.write(os.path.join(selected_dir, "sc_train.h5ad"))
    adata_sc_val.write(os.path.join(selected_dir, "sc_val.h5ad"))
    adata_sc_test.write(os.path.join(selected_dir, "sc_test.h5ad"))

    print("Saving st adata")

    adata_st.write(os.path.join(selected_dir, "st.h5ad"))

    print("Saving dicts and helpers")
    with open(os.path.join(selected_dir, "sc_sub_dict.pkl"), "wb") as f:
        pickle.dump(sc_sub_dict, f)

    with open(os.path.join(selected_dir, "sc_sub_dict2.pkl"), "wb") as f:
        pickle.dump(sc_sub_dict2, f)

    with open(os.path.join(selected_dir, "st_sample_id_l.pkl"), "wb") as f:
        pickle.dump(st_sample_id_l.tolist(), f)


def gen_pseudo_spots(
    selected_dir,
    n_mix=data_loading.DEFAULT_N_MIX,
    n_spots=data_loading.DEFAULT_N_SPOTS,
    rng=None,
    n_jobs=1,
):
    """Generate pseudo spots for the spatial data.

    Args:
        selected_dir (str): Directory containing the unmixed data.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        n_spots (int): Number of spots to generate. for training set. Default:
            20000.
        rng: Random number generator or seed for numpy's rng. Default: None.

    Returns:
        Tuple of dictionaries containing the pseudo spots and their labels.

    """
    rng_integers = misc.check_integer_rng(rng)

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    try:
        sc_mix_d, lab_mix_d = data_loading.load_pseudospots(unscaled_data_dir, n_mix, n_spots)
        print("Unscaled pseudospots already exist. " "Skipping generation and loading from disk.")
        return sc_mix_d, lab_mix_d

    except FileNotFoundError:
        pass

    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    adata_sc_d = {}
    for split in data_loading.SPLITS:
        adata_sc_d[split] = sc.read_h5ad(os.path.join(selected_dir, f"sc_{split}.h5ad"))

    lab_sc_num_d = {}
    with h5py.File(os.path.join(selected_dir, f"lab_sc_num.hdf5"), "r") as f:
        for split in data_loading.SPLITS:
            lab_sc_num_d[split] = f[split][()]

    lab_mix_d = {}
    sc_mix_d = {}

    total_spots = n_spots / SPLIT_RATIOS[data_loading.SPLITS.index("train")]
    for split, ratio in zip(data_loading.SPLITS, SPLIT_RATIOS):
        sc_mix_d[split], lab_mix_d[split] = data_processing.random_mix(
            adata_sc_d[split].X.toarray(),
            lab_sc_num_d[split],
            nmix=n_mix,
            n_samples=round(total_spots * ratio),
            seed=rng_integers(2**32),
            n_jobs=n_jobs,
        )

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    print("Saving unscaled pseudospots")
    data_loading.save_pseudospots(lab_mix_d, sc_mix_d, unscaled_data_dir, n_mix, n_spots)

    return sc_mix_d, lab_mix_d


def log_scale_pseudospots(
    selected_dir,
    scaler_name,
    n_mix=data_loading.DEFAULT_N_MIX,
    n_spots=data_loading.DEFAULT_N_SPOTS,
    sc_mix_d=None,
    lab_mix_d=None,
):
    """Log scales the pseudospots.

    Args:
        selected_dir (str): Directory of selected data. If `sc_mix_d` or
            `lab_mix_d` are None, will also load unscaled pseudospots from this
            directory.
        scaler_name (str): Name of the scaler to use.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        n_spots (int): Number of spots to generate. for training set. Default:
            20000.
        sc_mix_d (dict): Dictionary of sc mixtures. Default: None.
        lab_mix_d (dict): Dictionary of labels for sc mixtures. Default: None.


    """

    scaler = get_scaler(scaler_name)

    if sc_mix_d is None or lab_mix_d is None:
        unscaled_data_dir = os.path.join(selected_dir, "unscaled")
        sc_mix_d, lab_mix_d = data_loading.load_pseudospots(
            unscaled_data_dir, n_mix=n_mix, n_spots=n_spots
        )

    scaled = scale(scaler, *(sc_mix_d[split] for split in data_loading.SPLITS))
    sc_mix_s_d = {split: next(scaled) for split in data_loading.SPLITS}

    print("Saving pseudospots")
    preprocessed_data_dir = os.path.join(selected_dir, scaler_name)
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    data_loading.save_pseudospots(lab_mix_d, sc_mix_s_d, preprocessed_data_dir, n_mix, n_spots)


def split_st(selected_dir, stsplit=False, samp_split=False, rng=None):
    """Split and save spatial data into train, val, and test sets if applicable.

    Args:
        selected_dir (str): Directory containing the gene selected data.
        stsplit (bool): Whether to use a train/val/test split for spatial data.
            Default: False.
        rng: Random number generator or seed for numpy's rng. Default: None.

    """
    if stsplit and samp_split:
        raise ValueError("Cannot use both stsplit and samp_split.")

    rng_integers = misc.check_integer_rng(rng)

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    if samp_split:
        fname = "mat_sp_samp_split_d.hdf5"
    elif stsplit:
        fname = "mat_sp_split_d.hdf5"
    else:
        fname = "mat_sp_train_d.hdf5"

    out_path = os.path.join(unscaled_data_dir, fname)
    if os.path.isfile(out_path):
        print("Unscaled spatial data already exists at:")
        print(out_path)
        print("Skipping unscaled data generation.")
        return

    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    adata_st = sc.read_h5ad(os.path.join(selected_dir, "st.h5ad"))
    st_sample_id_l = data_loading.load_st_sample_names(selected_dir)

    if samp_split:
        holdout_idxs = [rng_integers(len(st_sample_id_l))]
        # Ensure that the holdout samples are different
        while holdout_idxs[0] == (i_2 := rng_integers(len(st_sample_id_l))):
            pass
        holdout_idxs.append(i_2)

        holdout_sids = [st_sample_id_l[i] for i in holdout_idxs]
        st_sample_id_l = [sid for sid in st_sample_id_l if sid not in holdout_sids]

        with h5py.File(out_path, "w") as f:
            grp_samp = f.create_group("train")
            for sample_id in st_sample_id_l:
                x_st_train = adata_st[adata_st.obs.sample_id == sample_id].X.toarray()
                grp_samp.create_dataset(sample_id, data=x_st_train)

            for sample_id, split in zip(holdout_sids, ["val", "test"]):
                grp_samp = f.create_group(split)

                x_st_test = adata_st[adata_st.obs.sample_id == sample_id].X.toarray()
                grp_samp.create_dataset(sample_id, data=x_st_test)

        return

    with h5py.File(out_path, "w") as f:
        for sample_id in st_sample_id_l:
            x_st_train = adata_st[adata_st.obs.sample_id == sample_id].X.toarray()
            grp_samp = f.create_group(sample_id)
            if stsplit:
                x_st_train, x_st_val = model_selection.train_test_split(
                    x_st_train, test_size=0.2, random_state=rng_integers(2**32)
                )

                x_st_val, x_st_test = model_selection.train_test_split(
                    x_st_val,
                    test_size=0.5,
                    random_state=rng_integers(2**32),
                )
                grp_samp.create_dataset("train", data=x_st_train)
                grp_samp.create_dataset("val", data=x_st_val)
                grp_samp.create_dataset("test", data=x_st_test)
            else:
                grp_samp.create_dataset("train", data=x_st_train)


def log_scale_st(selected_dir, scaler_name, stsplit=False, samp_split=False):
    """Log scale spatial data and save to file.

    Args:
        selected_dir (str): Directory containing selected data.
        scaler_name (str): Name of scaler to use.
        stsplit (bool): Whether to split the spatial data into train, val, and
            test. Defaults to False.

    """
    if stsplit and samp_split:
        raise ValueError("Cannot use both stsplit and samp_split.")

    scaler = get_scaler(scaler_name)

    # st_sample_id_l = data_loading.load_st_sample_names(selected_dir)

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    preprocessed_data_dir = os.path.join(selected_dir, scaler_name)
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    if samp_split:
        st_fname = "mat_sp_samp_split_d.hdf5"
    elif stsplit:
        st_fname = "mat_sp_split_d.hdf5"
    else:
        st_fname = "mat_sp_train_d.hdf5"

    in_path = os.path.join(unscaled_data_dir, st_fname)
    out_path = os.path.join(preprocessed_data_dir, st_fname)
    with h5py.File(out_path, "w") as fout, h5py.File(in_path, "r") as fin:
        if samp_split:
            # x_all = {}
            # sids_lens_all = {}
            # for split in data_loading.SPLITS:
            #     sids_lens_l = []
            #     x_l = []
            #     for sample_id in fin[split]:
            #         x = fin[split][sample_id][()]
            #         sids_lens_l.append((sample_id, x.shape[0]))
            #         x_l.append(x)
            #     x_all[split] = np.concatenate(x_l, axis=0)
            #     sids_lens_all[split] = sids_lens_l

            # scaled = scale(scaler, *(x_all[split] for split in data_loading.SPLITS))
            # for split, x_out in zip(data_loading.SPLITS, scaled):
            #     fout.create_group(split)

            #     _, lens = zip(*sids_lens_all[split])
            #     for (sid, l), i_n in zip(sids_lens_all[split], accumulate(lens)):
            #         fout[split].create_dataset(sid, data=x_out[i_n - l : i_n])

            # return
            for split in data_loading.SPLITS:
                grp = fin[split]
                grp_samp = fout.create_group(split)
                for sample_id in grp:
                    x = grp[sample_id][()]
                    grp_samp.create_dataset(sample_id, data=next(scale(scaler, x)))

            return

        for sample_id in fin:
            grp = fin[sample_id]
            grp_samp = fout.create_group(sample_id)

            if stsplit:
                scaled = scale(scaler, *(grp[split][()] for split in data_loading.SPLITS))
                for split in data_loading.SPLITS:
                    grp_samp.create_dataset(split, data=next(scaled))
            else:
                grp_samp.create_dataset("train", data=next(scale(scaler, grp["train"][()])))


def log_scale_all_st(selected_dir, scaler_name):
    """Log scales all spatial data.

    Args:
        selected_dir: Directory containing selected data.
        scaler_name: Name of scaler to use.

    """
    scaler = get_scaler(scaler_name)

    mat_sp_train_all = sc.read_h5ad(os.path.join(selected_dir, "st.h5ad")).X.toarray()

    mat_sp_train_all = next(scale(scaler, mat_sp_train_all))

    preprocessed_data_dir = os.path.join(selected_dir, scaler_name)
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    print("Saving all spatial data...")
    with h5py.File(os.path.join(preprocessed_data_dir, "mat_sp_train_s.hdf5"), "w") as f:
        f.create_dataset("all", data=mat_sp_train_all)


if __name__ == "__main__":
    # logging.basicConfig(
    #     level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
    # )

    parser = argparse.ArgumentParser(description="Preps the data into sets.")

    parser.add_argument(
        "--scaler",
        "-s",
        type=str,
        default="minmax",
        choices=SCALER_OPTS,
        help="Scaler to use.",
    )
    parser.add_argument("--stsplit", action="store_true", help="Split ST data by spot.")
    parser.add_argument("--samp_split", action="store_true", help="Split ST data by sample.")
    parser.add_argument("--allgenes", "-a", action="store_true", help="Turn off marker selection.")
    parser.add_argument(
        "--nmarkers",
        type=int,
        default=data_loading.DEFAULT_N_MARKERS,
        help=(
            "Number of top markers in sc training data to used. "
            "Ignored if --allgenes flag is used."
        ),
    )
    parser.add_argument(
        "--nmix",
        type=int,
        default=data_loading.DEFAULT_N_MIX,
        help="number of sc samples to use to generate pseudospots.",
    )
    parser.add_argument(
        "--nspots",
        type=int,
        default=data_loading.DEFAULT_N_SPOTS,
        help="Number of training pseudospots to use.",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=-1,
        help="Number of jobs to use for parallel processing.",
    )
    parser.add_argument(
        "--dset",
        "-d",
        type=str,
        default="dlpfc",
        help="dataset type to use. Default: dlpfc.",
    )
    parser.add_argument(
        "--st_id",
        type=str,
        default="spatialLIBD",
        help="st set to use. Default: spatialLIBD.",
    )
    parser.add_argument(
        "--sc_id",
        type=str,
        default="GSE144136",
        help="sc set to use. Default: GSE144136.",
    )

    args = parser.parse_args()

    main(args)
