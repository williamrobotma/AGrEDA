#!/usr/bin/env python3
"""Preps the data into sets."""

import glob
import pickle
import os
import argparse
from collections import OrderedDict

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
from sklearn import model_selection, preprocessing

from src.utils import data_loading, data_processing, misc

SPLIT_RATIOS = (0.8, 0.1, 0.1)
DATA_DIR = "./data"
SPATIALLIBD_DIR = os.path.join(DATA_DIR, "spatialLIBD")
SC_DLPFC_PATH = os.path.join(DATA_DIR, "sc_dlpfc", "adata_sc_dlpfc.h5ad")

SCALER_OPTS = ("minmax", "standard", "celldart")


parser = argparse.ArgumentParser(description="Preps the data into sets.")

parser.add_argument(
    "--scaler",
    "-s",
    type=str,
    default="celldart",
    choices=SCALER_OPTS,
    help="Scaler to use.",
)
parser.add_argument("--stsplit", action="store_true", help="Whether to split ST data.")
parser.add_argument(
    "--allgenes", "-a", action="store_true", help="Turn off marker selection."
)
parser.add_argument(
    "--nmarkers",
    type=int,
    default=data_loading.DEFAULT_N_MARKERS,
    help="Number of top markers in sc training data to used. Ignored if --allgenes flag is used.",
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

args = parser.parse_args()


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
        return scaler_name

    raise ValueError(f"Scaler '{scaler_name}' not recognized.")


def check_selected_split_exists(selected_dir):
    """Check if selected and split data exists and all files are present

    Args:
        selected_dir (str): Path to selected data directory.

    Returns:
        bool: Whether selected and split data exists.

    """
    # df genelists
    if not os.path.isfile(os.path.join(selected_dir, "df_genelists.pkl")):
        return False

    if not os.path.isfile(os.path.join(selected_dir, "venn.png")):
        return False

    # label numbers
    if not os.path.isfile(os.path.join(selected_dir, f"lab_sc_num.hdf5")):
        return False

    # sc adatas
    if not os.path.isfile(os.path.join(selected_dir, "adata_sc_dlpfc.h5ad")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, "adata_sc_dlpfc_train.h5ad")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, "adata_sc_dlpfc_val.h5ad")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, "adata_sc_dlpfc_test.h5ad")):
        return False

    # st adata
    if not os.path.isfile(os.path.join(selected_dir, "adata_spatialLIBD.h5ad")):
        return False

    # dicts and helpers
    if not os.path.isfile(os.path.join(selected_dir, "sc_sub_dict.pkl")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, "sc_sub_dict2.pkl")):
        return False
    if not os.path.isfile(os.path.join(selected_dir, "st_sample_id_l.pkl")):
        return False

    # All files present
    return True


def select_genes_and_split(
    n_markers=data_loading.DEFAULT_N_MARKERS,
    allgenes=False,
    spatiallibd_dir=SPATIALLIBD_DIR,
    sc_dlpfc_path=SC_DLPFC_PATH,
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

    selected_dir = data_loading.get_selected_dir(DATA_DIR, n_markers, allgenes)

    if check_selected_split_exists(selected_dir):
        print("Selected and split data already exists. Skipping.")
        return

    if not os.path.isdir(selected_dir):
        os.makedirs(selected_dir)

    print("Loading SpatialLIBD Data")

    adata_dir = os.path.join(spatiallibd_dir, "adata")

    adata_spatiallibd_d = OrderedDict()

    for name in sorted(glob.glob(os.path.join(adata_dir, "adata_spatialLIBD-*.h5ad"))):
        sample_id = name.partition("-")[2].rpartition(".")[0]
        adata_spatiallibd_d[sample_id] = sc.read_h5ad(name)

    adata_spatiallibd = ad.concat(
        adata_spatiallibd_d.values(), label="sample_id", keys=adata_spatiallibd_d.keys()
    )
    adata_spatiallibd.obs_names_make_unique()
    sc.pp.normalize_total(adata_spatiallibd, inplace=True, target_sum=1e4)
    adata_spatiallibd.var_names_make_unique()

    st_sample_id_l = adata_spatiallibd.obs["sample_id"].unique()

    print("Loading Single Cell Data")

    adata_sc_dlpfc = sc.read_h5ad(sc_dlpfc_path)
    sc.pp.normalize_total(adata_sc_dlpfc, inplace=True, target_sum=1e4)
    adata_sc_dlpfc.var_names_make_unique()

    print("Splitting single cell data")
    # df_sc = adata_sc_dlpfc.to_df()
    # df_sc.index = pd.MultiIndex.from_frame(adata_sc_dlpfc.obs.reset_index())

    # lab_sc_sub = df_sc.index.get_level_values("cell_subclass")
    lab_sc_sub = adata_sc_dlpfc.obs["cell_subclass"]

    (
        adata_sc_dlpfc_train,
        adata_sc_dlpfc_eval,
        lab_sc_sub_train,
        lab_sc_sub_eval,
    ) = model_selection.train_test_split(
        adata_sc_dlpfc,
        lab_sc_sub,
        test_size=0.2,
        random_state=rng_integers(2**32),
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
        random_state=rng_integers(2**32),
        stratify=lab_sc_sub_eval,
    )

    print("Selecting genes")
    (
        (adata_sc_dlpfc_train, adata_spatiallibd),
        df_genelists,
        (fig, ax),
    ) = data_processing.select_marker_genes(
        adata_sc_dlpfc_train,
        adata_spatiallibd,
        n_markers=None if allgenes else n_markers,
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

    adata_sc_dlpfc = adata_sc_dlpfc[:, adata_sc_dlpfc_train.var.index]
    adata_sc_dlpfc_val = adata_sc_dlpfc_val[:, adata_sc_dlpfc_train.var.index]
    adata_sc_dlpfc_test = adata_sc_dlpfc_test[:, adata_sc_dlpfc_train.var.index]

    print("Saving sc labels")
    with h5py.File(os.path.join(selected_dir, f"lab_sc_num.hdf5"), "w") as f:
        f.create_dataset("train", data=lab_sc_num_train)
        f.create_dataset("val", data=lab_sc_num_val)
        f.create_dataset("test", data=lab_sc_num_test)

    print("Saving sc adatas")
    adata_sc_dlpfc.write(os.path.join(selected_dir, "adata_sc_dlpfc.h5ad"))
    adata_sc_dlpfc_train.write(os.path.join(selected_dir, "adata_sc_dlpfc_train.h5ad"))
    adata_sc_dlpfc_val.write(os.path.join(selected_dir, "adata_sc_dlpfc_val.h5ad"))
    adata_sc_dlpfc_test.write(os.path.join(selected_dir, "adata_sc_dlpfc_test.h5ad"))

    print("Saving st adata")

    adata_spatiallibd.write(os.path.join(selected_dir, "adata_spatialLIBD.h5ad"))

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
        sc_mix_d, lab_mix_d = data_loading.load_pseudospots(
            unscaled_data_dir, n_mix, n_spots
        )
        print(
            "Unscaled pseudospots already exist. "
            "Skipping generation and loading from disk."
        )
        return sc_mix_d, lab_mix_d

    except FileNotFoundError:
        pass

    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    adata_sc_dlpfc_d = {}
    for split in data_loading.SPLITS:
        adata_sc_dlpfc_d[split] = sc.read_h5ad(
            os.path.join(selected_dir, f"adata_sc_dlpfc_{split}.h5ad")
        )

    lab_sc_num_d = {}
    with h5py.File(os.path.join(selected_dir, f"lab_sc_num.hdf5"), "r") as f:
        for split in data_loading.SPLITS:
            lab_sc_num_d[split] = f[split][()]

    lab_mix_d = {}
    sc_mix_d = {}

    total_spots = n_spots / SPLIT_RATIOS[data_loading.SPLITS.index("train")]
    for split, ratio in zip(data_loading.SPLITS, SPLIT_RATIOS):
        sc_mix_d[split], lab_mix_d[split] = data_processing.random_mix(
            adata_sc_dlpfc_d[split].X.toarray(),
            lab_sc_num_d[split],
            nmix=n_mix,
            n_samples=round(total_spots * ratio),
            seed=rng_integers(2**32),
            n_jobs=args.njobs,
        )

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    print("Saving unscaled pseudospots")
    data_loading.save_pseudospots(
        lab_mix_d, sc_mix_d, unscaled_data_dir, n_mix, n_spots
    )

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

    data_loading.save_pseudospots(
        lab_mix_d, sc_mix_s_d, preprocessed_data_dir, n_mix, n_spots
    )


def split_st(selected_dir, stsplit=False, rng=None):
    """Split and save spatial data into train, val, and test sets if applicable.

    Args:
        selected_dir (str): Directory containing the gene selected data.
        stsplit (bool): Whether to use a train/val/test split for spatial data.
            Default: False.
        rng: Random number generator or seed for numpy's rng. Default: None.

    """

    rng_integers = misc.check_integer_rng(rng)

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    out_path = os.path.join(
        unscaled_data_dir, f"mat_sp_{'split' if stsplit else 'train'}_d.hdf5"
    )
    if os.path.isfile(out_path):
        print("Unscaled spatial data already exists at:")
        print(out_path)
        print("Skipping unscaled data generation.")
        return

    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    adata_spatiallibd = sc.read_h5ad(
        os.path.join(selected_dir, "adata_spatialLIBD.h5ad")
    )
    st_sample_id_l = data_loading.load_st_sample_names(selected_dir)

    with h5py.File(out_path, "w") as f:
        for sample_id in st_sample_id_l:
            x_st_train = adata_spatiallibd[
                adata_spatiallibd.obs.sample_id == sample_id
            ].X.toarray()
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


def log_scale_st(selected_dir, scaler_name, stsplit=False):
    """Log scale spatial data and save to file.

    Args:
        selected_dir (str): Directory containing selected data.
        scaler_name (str): Name of scaler to use.
        stsplit (bool): Whether to split the spatial data into train, val, and
            test. Defaults to False.

    """
    scaler = get_scaler(scaler_name)

    # st_sample_id_l = data_loading.load_st_sample_names(selected_dir)

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    preprocessed_data_dir = os.path.join(selected_dir, scaler_name)
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    st_fname = f"mat_sp_{'split' if stsplit else 'train'}_d.hdf5"
    in_path = os.path.join(unscaled_data_dir, st_fname)
    out_path = os.path.join(preprocessed_data_dir, st_fname)
    with h5py.File(out_path, "w") as fout, h5py.File(in_path, "r") as fin:

        for sample_id in fin:

            grp = fin[sample_id]
            grp_samp = fout.create_group(sample_id)

            if stsplit:
                scaled = scale(
                    scaler, *(grp[split][()] for split in data_loading.SPLITS)
                )
                for split in data_loading.SPLITS:
                    grp_samp.create_dataset(split, data=next(scaled))
            else:
                grp_samp.create_dataset(
                    "train", data=next(scale(scaler, grp["train"][()]))
                )


def log_scale_all_st(selected_dir, scaler_name):
    """Log scales all spatial data.

    Args:
        selected_dir: Directory containing selected data.
        scaler_name: Name of scaler to use.

    """
    scaler = get_scaler(scaler_name)

    mat_sp_train_all = sc.read_h5ad(
        os.path.join(selected_dir, "adata_spatialLIBD.h5ad")
    ).X.toarray()

    mat_sp_train_all = next(scale(scaler, mat_sp_train_all))

    preprocessed_data_dir = os.path.join(selected_dir, scaler_name)
    if not os.path.isdir(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    print("Saving all spatial data...")
    with h5py.File(
        os.path.join(preprocessed_data_dir, "mat_sp_train_s.hdf5"), "w"
    ) as f:
        f.create_dataset("all", data=mat_sp_train_all)


# %%
def main():
    selected_dir = data_loading.get_selected_dir(DATA_DIR, args.nmarkers, args.allgenes)

    print("Selecting subset genes and splitting single-cell data")
    print("-" * 80)
    select_genes_and_split(
        n_markers=args.nmarkers,
        allgenes=args.allgenes,
        spatiallibd_dir=SPATIALLIBD_DIR,
        sc_dlpfc_path=SC_DLPFC_PATH,
        rng=462,
    )

    print("Generating Pseudospots")
    print("-" * 80)
    sc_mix_d, lab_mix_d = gen_pseudo_spots(
        selected_dir, n_mix=args.nmix, n_spots=args.nspots, rng=623
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
    split_st(selected_dir, stsplit=args.stsplit, rng=16)
    log_scale_st(selected_dir, scaler_name=args.scaler, stsplit=args.stsplit)

    print("Log scaling all spatial data...")
    print("-" * 80)
    # if TRAIN_USING_ALL_ST_SAMPLES:
    log_scale_all_st(selected_dir, args.scaler)


if __name__ == "__main__":
    main()
