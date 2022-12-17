"""Data loading functions."""
import os
import pickle
import h5py

DEFAULT_N_SPOTS = 20000
DEFAULT_N_MARKERS = 20
DEFAULT_N_MIX = 8
SPLITS = ("train", "val", "test")


def get_selected_dir(data_dir, n_markers=DEFAULT_N_MARKERS, all_genes=False):
    if all_genes:
        return os.path.join(data_dir, "preprocessed", "all")
    else:
        return os.path.join(data_dir, "preprocessed", f"{n_markers}markers")


def load_spatial(selected_dir, scaler_name, **kwargs):
    """Loads preprocessed spatial data.

    Args:
        selected_dir (str): Directory of selected data.
        scaler_name (str): Name of the scaler to use.
        train_using_all_st_samples (bool): Whether to use all spatial samples
            for training, or separate by sample.
        st_split (bool): Whether to use a train/val/test split for spatial data.

    Returns:
        Tuple::

            (`mat_sp_d`, `mat_sp_train`, `st_sample_id_l`)


        `mat_sp_d` is a dict of spatial data by sample and split. If not
        `st_split`, then 'val' will not be contained and 'test' will point to
        'train'.

        `mat_sp_train` is a numpy array of all spatial data together for
        training; None if not `train_using_all_st_samples`.

        `st_sample_id_l` is a list of sample ids for spatial data.

    """
    processed_data_dir = os.path.join(selected_dir, scaler_name)
    mat_sp_d, mat_sp_train = load_st_spots(processed_data_dir, **kwargs)

    st_sample_id_l = load_st_sample_names(selected_dir)
    return mat_sp_d, mat_sp_train, st_sample_id_l


def load_st_spots(data_dir, train_using_all_st_samples=False, st_split=False):
    """Loads spatial spots.

    Args:
        data_dir (str): Directory to load data from.
        train_using_all_st_samples (bool): Whether to use all spatial samples
            for training, or separate by sample.
        st_split (bool): Whether to use a train/val/test split for spatial data.

    Returns:
        Tuple::

            (`mat_sp_d`, `mat_sp_train`)


        `mat_sp_d` is a dict of spatial data by sample and split. If not
        `st_split`, then 'val' will not be contained and 'test' will point to
        'train'.

        `mat_sp_train` is a numpy array of all spatial data together for
        training; None if not `train_using_all_st_samples`.


    """
    fname = f"mat_sp_{'split' if st_split else 'train'}_d.hdf5"
    in_path = os.path.join(data_dir, fname)

    mat_sp_d = {}
    with h5py.File(in_path, "r") as f:
        for sample_id in f:
            mat_sp_d[sample_id] = {}
            mat_sp_d[sample_id]["train"] = f[f"{sample_id}/train"][()]
            if st_split:
                mat_sp_d[sample_id]["val"] = f[f"{sample_id}/val"][()]
                mat_sp_d[sample_id]["test"] = f[f"{sample_id}/test"][()]
            else:
                mat_sp_d[sample_id]["test"] = mat_sp_d[sample_id]["train"]

    if train_using_all_st_samples:
        with h5py.File(os.path.join(data_dir, "mat_sp_train_s.hdf5"), "r") as f:
            mat_sp_train = f["all"][()]

    else:
        mat_sp_train = None
    return mat_sp_d, mat_sp_train


def save_st_spots(mat_sp_d, data_dir, stsplit=False):
    """Saves spatial data to hdf5 files.

    Args:
        mat_sp_train_s_d (dict): Dict of spatial data by sample for training.
        mat_sp_test_s_d (dict): Dict of spatial data by sample for testing.
        mat_sp_val_s_d (dict): Dict of spatial data by sample for validation.
        data_dir (str): Directory to save data to.
        stsplit (bool): Whether to use a train/val/test split for spatial data.
            Default: False.

    """
    fname = f"mat_sp_{'split' if stsplit else 'train'}_d.hdf5"
    out_path = os.path.join(data_dir, fname)

    with h5py.File(out_path, "w") as f:
        for sample_id in mat_sp_d:
            grp_samp = f.create_group(sample_id)
            if stsplit:
                for split in mat_sp_d[sample_id]:
                    grp_samp.create_dataset(split, data=mat_sp_d[sample_id][split])
            else:
                grp_samp.create_dataset("train", data=mat_sp_d[sample_id]["train"])


def load_st_sample_names(selected_dir):
    with open(os.path.join(selected_dir, "st_sample_id_l.pkl"), "rb") as f:
        st_sample_id_l = pickle.load(f)
    return st_sample_id_l


def load_sc(selected_dir, scaler_name, **kwargs):
    """Loads preprocessed sc data.

    Args:
        selected_dir (str): Directory of selected data.
        scaler_name (str): Name of the scaler to use.
        n_mix (int): Number of sc samples in each spot. Default: 8.
        n_spots (int): Number of spots to generate. for training set. Default:
            20000.

    Returns:
        Tuple::

            (
                sc_mix_d,
                lab_mix_d,
                sc_sub_dict,
                sc_sub_dict2,
            )

         - `sc_mix_d` is a dict of sc data by split.
         - `lab_mix_d` is a dict of sc labels by split.
         - `sc_sub_dict` is a dict of 'label_id' to 'label_name' for sc data.
         - `sc_sub_dict2` is a dict of 'label_name' to 'label_id' for sc data.

    """
    preprocessed_dir = os.path.join(selected_dir, scaler_name)
    sc_mix_d, lab_mix_d = load_pseudospots(preprocessed_dir, **kwargs)

    # Load helper dicts / lists
    sc_sub_dict, sc_sub_dict2 = load_sc_dicts(selected_dir)
    return sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2


def load_sc_dicts(selected_dir):
    with open(os.path.join(selected_dir, "sc_sub_dict.pkl"), "rb") as f:
        sc_sub_dict = pickle.load(f)

    with open(os.path.join(selected_dir, "sc_sub_dict2.pkl"), "rb") as f:
        sc_sub_dict2 = pickle.load(f)
    return sc_sub_dict, sc_sub_dict2


def ps_fname(n_mix, n_spots):
    return f"sc_{n_mix}mix_{n_spots}spots.hdf5"


def load_pseudospots(data_dir, n_mix=DEFAULT_N_MIX, n_spots=DEFAULT_N_SPOTS):
    sc_mix_d = {}
    lab_mix_d = {}
    with h5py.File(os.path.join(data_dir, ps_fname(n_mix, n_spots)), "r") as f:
        for split in SPLITS:
            sc_mix_d[split] = f[f"X/{split}"][()]
            lab_mix_d[split] = f[f"y/{split}"][()]
    return sc_mix_d, lab_mix_d


def save_pseudospots(lab_mix_d, sc_mix_s_d, data_dir, n_mix, n_spots):
    with h5py.File(os.path.join(data_dir, ps_fname(n_mix, n_spots)), "w") as f:
        grp_x = f.create_group("X")
        grp_y = f.create_group("y")
        for split in SPLITS:
            dset = grp_x.create_dataset(split, data=sc_mix_s_d[split])
            dset = grp_y.create_dataset(split, data=lab_mix_d[split])
