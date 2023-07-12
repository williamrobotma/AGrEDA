#!/usr/bin/env python3
"""Preps the data into sets."""

import argparse
import glob
import logging
import math
import os
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import model_selection, preprocessing

from src.da_utils import data_loading, data_processing, misc

logger = logging.getLogger(__name__)

SPLIT_RATIOS = (0.8, 0.1, 0.1)
DATA_DIR = "./data"

SCALER_OPTS = ("minmax", "standard", "celldart")


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
        selected_dir, n_mix=args.nmix, rng=args.ps_seed, n_jobs=args.njobs
    )

    print("Log scaling pseudospots")
    print("-" * 80)
    log_scale_pseudospots(
        selected_dir,
        args.scaler,
        n_mix=args.nmix,
        # n_spots=args.nspots,
        sc_mix_d=sc_mix_d,
        lab_mix_d=lab_mix_d,
        seed_int=-1 if args.ps_seed == 623 else args.ps_seed,
    )

    print("Log scaling and maybe splitting spatial data")
    print("-" * 80)
    split_st(
        selected_dir,
        stsplit=args.stsplit,
        samp_split=args.samp_split,
        one_model=args.one_model,
        rng=16,
    )
    log_scale_st(
        selected_dir,
        scaler_name=args.scaler,
        stsplit=args.stsplit,
        samp_split=args.samp_split,
        one_model=args.one_model,
    )


def scale(scaler, *unscaled):
    if scaler == "celldart":
        yield from (data_processing.log_minmaxscale(x) for x in unscaled)

    else:
        sp_scaler = scaler().fit(np.log1p(unscaled[0]))
        yield from (sp_scaler.transform(np.log1p(x)) for x in unscaled)


def get_scaler(scaler_name):
    """Get the scaler class from its name."""
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
    # drop cells with no labels
    adata_sc = adata_sc[~adata_sc.obs["cell_type"].isna().to_numpy(), :]

    adata_sc_train, adata_sc_eval = model_selection.train_test_split(
        adata_sc,
        test_size=0.2,
        random_state=rng_integers(2**32),
        stratify=adata_sc.obs["cell_type"],
    )
    adata_sc_val, adata_sc_test = model_selection.train_test_split(
        adata_sc_eval,
        test_size=0.5,
        random_state=rng_integers(2**32),
        stratify=adata_sc_eval.obs["cell_type"],
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

    # propogate gene subset to all sc adatas
    adata_sc = adata_sc[:, adata_sc_train.var.index]
    adata_sc_val = adata_sc_val[:, adata_sc_train.var.index]
    adata_sc_test = adata_sc_test[:, adata_sc_train.var.index]

    # gen label numbers
    _create_lab_sc_sub_col(adata_sc, sc_sub_dict2)
    _create_lab_sc_sub_col(adata_sc_test, sc_sub_dict2)
    _create_lab_sc_sub_col(adata_sc_val, sc_sub_dict2)
    _create_lab_sc_sub_col(adata_sc_train, sc_sub_dict2)

    print("Saving sc adatas")
    adata_sc.write(Path(os.path.join(selected_dir, "sc.h5ad")))
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


def _create_lab_sc_sub_col(adata_sc, sc_sub_dict2):
    adata_sc.obs["lab_sc_num"] = adata_sc.obs["cell_type"].map(sc_sub_dict2).astype(int)


def gen_pseudo_spots(
    selected_dir,
    n_mix=data_loading.DEFAULT_N_MIX,
    # n_spots=data_loading.DEFAULT_N_SPOTS,
    rng=None,
    n_jobs=1,
):
    """Generate pseudo spots for the spatial data.

    Args:
        selected_dir (str): Directory containing the unmixed data.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        rng: Random number generator or seed for numpy's rng. Default: None.

    Returns:
        Tuple of dictionaries containing the pseudo spots and their labels.

    """

    if isinstance(rng, int):
        if rng == 623:
            seed_int = -1
        else:
            seed_int = rng
    else:
        seed_int = -1

    rng_integers = misc.check_integer_rng(rng)

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    try:
        sc_mix_d, lab_mix_d = data_loading.load_pseudospots(
            unscaled_data_dir, n_mix, seed_int=seed_int
        )
        # lab_mix_d["train"][:n_spots]
        print("Unscaled pseudospots already exist. " "Skipping generation and loading from disk.")
        return sc_mix_d, lab_mix_d

    except FileNotFoundError:
        pass
    # except IndexError:
    #     print("Unscaled pseudospots already exist but are incomplete. " "Regenerating.")

    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    adata_sc_d = {}
    for split in data_loading.SPLITS:
        adata_sc_d[split] = sc.read_h5ad(os.path.join(selected_dir, f"sc_{split}.h5ad"))

    lab_mix_d = {}
    sc_mix_d = {}

    total_spots = data_loading.DEFAULT_N_SPOTS / SPLIT_RATIOS[data_loading.SPLITS.index("train")]
    for split, ratio in zip(data_loading.SPLITS, SPLIT_RATIOS):
        sc_mix_d[split], lab_mix_d[split] = data_processing.random_mix(
            adata_sc_d[split].X,
            adata_sc_d[split].obs["lab_sc_num"].to_numpy(),
            nmix=n_mix,
            n_samples=round(total_spots * ratio),
            seed=rng_integers(2**32),
            n_jobs=n_jobs,
        )

    unscaled_data_dir = os.path.join(selected_dir, "unscaled")
    if not os.path.exists(unscaled_data_dir):
        os.makedirs(unscaled_data_dir)

    print("Saving unscaled pseudospots")
    data_loading.save_pseudospots(
        lab_mix_d,
        sc_mix_d,
        unscaled_data_dir,
        n_mix,
        seed_int,
    )

    return sc_mix_d, lab_mix_d


def log_scale_pseudospots(
    selected_dir,
    scaler_name,
    n_mix=data_loading.DEFAULT_N_MIX,
    # n_spots=data_loading.DEFAULT_N_SPOTS,
    sc_mix_d=None,
    lab_mix_d=None,
    seed_int=-1,
):
    """Log scales the pseudospots.

    Args:
        selected_dir (str): Directory of selected data. If `sc_mix_d` or
            `lab_mix_d` are None, will also load unscaled pseudospots from this
            directory.
        scaler_name (str): Name of the scaler to use.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        sc_mix_d (dict): Dictionary of sc mixtures. Default: None.
        lab_mix_d (dict): Dictionary of labels for sc mixtures. Default: None.


    """

    scaler = get_scaler(scaler_name)

    if sc_mix_d is None or lab_mix_d is None:
        unscaled_data_dir = os.path.join(selected_dir, "unscaled")
        sc_mix_d, lab_mix_d = data_loading.load_pseudospots(
            unscaled_data_dir, n_mix=n_mix, seed_int=seed_int
        )

    scaled = scale(scaler, *(sc_mix_d[split] for split in data_loading.SPLITS))
    sc_mix_s_d = {split: next(scaled) for split in data_loading.SPLITS}

    print("Saving pseudospots")
    preprocessed_data_dir = os.path.join(selected_dir, scaler_name)
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    data_loading.save_pseudospots(
        lab_mix_d,
        sc_mix_s_d,
        preprocessed_data_dir,
        n_mix,
        seed_int,
    )


def split_st(selected_dir, stsplit=False, samp_split=False, one_model=False, rng=None):
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
        fname = "mat_sp_samp_split_d.h5ad"
    elif stsplit:
        fname = "mat_sp_split_d.h5ad"
    else:
        fname = "mat_sp_train_d.h5ad"

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
        # for spotless, gold standard 1 fovs 5 and 6 is not representative
        # for gs2, 1-4 do not have enough cell types

        if "spotless_mouse_cortex" in selected_dir:
            exclude_sids = {"Eng2019_cortex_svz_fov5", "Eng2019_cortex_svz_fov6"}
        elif "spotless_mouse_olfactory" in selected_dir:
            exclude_sids = {
                "Eng2019_ob_fov1",
                "Eng2019_ob_fov2",
                "Eng2019_ob_fov3",
                "Eng2019_ob_fov4",
            }
        elif "spatialLIBD" in selected_dir:
            # these samples do not contain L1
            exclude_sids = {"151669", "151670", "151671", "151672"}
        else:
            exclude_sids = set()

        holdout_idxs = [rng_integers(len(st_sample_id_l) - len(exclude_sids))]
        # Ensure that the holdout samples are different
        # while holdout_idxs[0] == (i_2 := rng_integers(len(st_sample_id_l))):
        #     pass
        # holdout_idxs.append(i_2)

        holdout_candidate_sids = [sid for sid in st_sample_id_l if sid not in exclude_sids]
        holdout_sids = [holdout_candidate_sids[i] for i in holdout_idxs]
        st_sample_id_l = [sid for sid in st_sample_id_l if sid not in holdout_sids]

        # put each sample into own adata in dict
        adata_train_sample_d = {}
        for sid in st_sample_id_l:
            adata_train_sample_d[sid] = adata_st[adata_st.obs.sample_id == sid]
            adata_train_sample_d[sid].obs.drop(columns="sample_id", inplace=True)
            adata_train_sample_d[sid].obs.insert(loc=0, column="split", value="train")
        for sid, split in zip(holdout_sids, ["test"]):
            adata_train_sample_d[sid] = adata_st[adata_st.obs.sample_id == sid]
            adata_train_sample_d[sid].obs.drop(columns="sample_id", inplace=True)
            adata_train_sample_d[sid].obs.insert(loc=0, column="split", value=split)

        # concatenate all samples together
        adata_st = ad.concat(adata_train_sample_d, label="sample_id")
        adata_st.obs.insert(1, "sample_id", adata_st.obs.pop("sample_id"))

        # save to file
        adata_st.write_h5ad(Path(out_path))

        return
    # not samp_split
    if stsplit:
        holdout_frac = 0.1

        # ensure that holdout proportion is at least 1 cell (1 test)
        true_holdout_frac = math.ceil(len(adata_st) * holdout_frac) / len(adata_st)
        min_holdout_size = adata_st.obs["sample_id"].value_counts().min()
        if min_holdout_size * true_holdout_frac < 1:
            holdout_frac = 1 / min_holdout_size

            warnings.warn(
                "Holdout proportion too small. Increasing to 1 cells per sample.\n"
                "Using train/test split of "
                f"{1 - holdout_frac}/{holdout_frac}.",
                UserWarning,
            )

        adata_st_train, adata_st_test = model_selection.train_test_split(
            adata_st,
            test_size=holdout_frac,
            random_state=rng_integers(2**32),
            # stratify=data_processing.safe_stratify(adata_st.obs["sample_id"]),
            stratify=adata_st.obs["sample_id"],
        )
        # adata_st_val, adata_st_test = model_selection.train_test_split(
        #     adata_st_val,
        #     test_size=0.5,
        #     random_state=rng_integers(2**32),
        #     # stratify=data_processing.safe_stratify(adata_st_val.obs["sample_id"]),
        #     stratify=adata_st_val.obs["sample_id"],
        # )
        adata_st = ad.concat(
            [adata_st_train, adata_st_test],
            label="split",
            keys=["train", "test"],
        )
        adata_st.obs.insert(0, "split", adata_st.obs.pop("split"))

        # sort by sample_id then split
        samp_splits_l = []
        for sid in st_sample_id_l:
            # get split column for this sample
            samp_to_split = adata_st.obs["split"][adata_st.obs["sample_id"] == sid]
            for split in ["train", "test"]:
                # get split for sample and accumulate
                samp_splits_l.append(samp_to_split[samp_to_split == split])

        # concat list of series into one series and use to index adata
        adata_st = adata_st[pd.concat(samp_splits_l, axis="index").index.to_numpy()]

        adata_st.obs.insert(0, "sample_id", adata_st.obs.pop("sample_id"))
        adata_st.write_h5ad(Path(out_path))

        return

    adata_st.obs.insert(0, "split", "train")
    adata_st = adata_st[adata_st.obs.sort_values("sample_id").index.to_numpy()]

    if one_model:
        adata_st.obs.insert(1, "sample_id", adata_st.obs.pop("sample_id"))
    else:
        adata_st.obs.insert(0, "sample_id", adata_st.obs.pop("sample_id"))

    adata_st.write_h5ad(Path(out_path))


def log_scale_st(selected_dir, scaler_name, stsplit=False, samp_split=False, one_model=False):
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
        st_fname = "mat_sp_samp_split_d"
    elif stsplit:
        st_fname = "mat_sp_split_d"
    else:
        st_fname = "mat_sp_train_d"

    in_path = os.path.join(unscaled_data_dir, f"{st_fname}.h5ad")

    if one_model:
        st_fname = f"{st_fname}_one_model.h5ad"
    else:
        st_fname = f"{st_fname}.h5ad"

    out_path = os.path.join(preprocessed_data_dir, st_fname)

    adata_st = sc.read_h5ad(in_path)
    adata_st.X = adata_st.X.toarray()  # type: ignore

    if samp_split or (stsplit and one_model):
        adata_splits = (adata_st[adata_st.obs["split"] == split].X for split in ("train", "test"))
        scaled = scale(scaler, *(adata_split for adata_split in adata_splits))
        for split, scaled_split in zip(("train", "test"), scaled):
            adata_st[adata_st.obs["split"] == split].X = scaled_split

    elif one_model:
        adata_st.X = next(scale(scaler, adata_st.X))
    else:  # just stsplit or none
        for sample_id in adata_st.obs["sample_id"].unique():
            adata_samp = adata_st[adata_st.obs["sample_id"] == sample_id]
            if stsplit:
                adata_splits = (
                    adata_samp[adata_samp.obs["split"] == split].X for split in ("train", "test")
                )
            else:
                # easy len 1 generator
                adata_splits = (adata_samp.X for _ in range(1))

            scaled = scale(scaler, *adata_splits)

            # if not stplit scaled is len 1 and zip will quit after "train"
            for split, scaled_split in zip(("train", "test"), scaled):
                adata_st[
                    (adata_st.obs["split"] == split) & (adata_st.obs["sample_id"] == sample_id)
                ].X = scaled_split

    adata_st.write_h5ad(Path(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preps the data into sets.")

    parser.add_argument(
        "--scaler", "-s", type=str, default="minmax", choices=SCALER_OPTS, help="Scaler to use."
    )
    parser.add_argument("--stsplit", action="store_true", help="Split ST data by spot.")
    parser.add_argument(
        "--samp_split", action="store_true", help="Split ST data by sample. Will use single model."
    )
    parser.add_argument(
        "--one_model",
        action="store_true",
        help="Use single model for all samples. Ignored if --samp_split is used.",
    )
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
    # parser.add_argument(
    #     "--nspots",
    #     type=int,
    #     default=data_loading.DEFAULT_N_SPOTS,
    #     help="Number of training pseudospots to use.",
    # )
    parser.add_argument(
        "--njobs", type=int, default=-1, help="Number of jobs to use for parallel processing."
    )
    parser.add_argument(
        "--dset", "-d", type=str, default="dlpfc", help="dataset type to use. Default: dlpfc."
    )
    parser.add_argument(
        "--st_id", type=str, default="spatialLIBD", help="st set to use. Default: spatialLIBD."
    )
    parser.add_argument(
        "--sc_id", type=str, default="GSE144136", help="sc set to use. Default: GSE144136."
    )
    parser.add_argument(
        "--ps_seed", type=int, default=623, help="Seed to use for pseudospot generation."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    )

    main(args)

# %%
