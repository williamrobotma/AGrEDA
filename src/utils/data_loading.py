"""Data loading functions."""
import os
import pickle
import h5py


def load_spatial(train_using_all_st_samples, processed_data_dir, st_split):
    """Loads preprocessed spatial data.

    Args:
        train_using_all_st_samples (bool): Whether to use all spatial samples
            for training, or separate by sample.
        processed_data_dir (str): Path to directory containing preprocessed
            data.
        st_split (bool): Whether to use a train/val/test split for spatial data.

    Returns:
        Tuple::

            (`mat_sp_d`, `mat_sp_train_s`, `st_sample_id_l`)


        `mat_sp_d` is a dict of spatial data by split and sample. If not
        `st_split`, then 'val' will not be contained and 'test' will point to
        'train'.

        `mat_sp_train_s` is a numpy array of all spatial data together for
        training; None if not `train_using_all_st_samples`.

        `st_sample_id_l` is a list of sample ids for spatial data.

    """
    mat_sp_d = {}
    mat_sp_d["train"] = {}
    if st_split:
        mat_sp_d["val"] = {}
        mat_sp_d["test"] = {}
        with h5py.File(
            os.path.join(processed_data_dir, "mat_sp_split_s_d.hdf5"), "r"
        ) as f:
            for sample_id in f:
                mat_sp_d["train"][sample_id] = f[f"{sample_id}/train"][()]
                mat_sp_d["val"] = f[f"{sample_id}/val"][()]
                mat_sp_d["test"] = f[f"{sample_id}/test"][()]
    else:
        with h5py.File(
            os.path.join(processed_data_dir, "mat_sp_train_s_d.hdf5"), "r"
        ) as f:
            for sample_id in f:
                mat_sp_d["train"][sample_id] = f[sample_id][()]

        mat_sp_d["test"] = mat_sp_d["train"]

    if train_using_all_st_samples:
        with h5py.File(
            os.path.join(processed_data_dir, "mat_sp_train_s.hdf5"), "r"
        ) as f:
            mat_sp_train_s = f["all"][()]

    else:
        mat_sp_train_s = None

    with open(os.path.join(processed_data_dir, "st_sample_id_l.pkl"), "rb") as f:
        st_sample_id_l = pickle.load(f)
    return mat_sp_d, mat_sp_train_s, st_sample_id_l


def load_sc(processed_data_dir):
    """Loads preprocessed sc data.

    Args:
        processed_data_dir (str): Path to directory containing preprocessed
            data.

    Returns:
        Tuple::

            (
                `sc_mix_d`,
                `lab_mix_d`,
                `sc_sub_dict`,
                `sc_sub_dict2`,
            )

        `sc_mix_d` is a dict of sc data by split.
        `lab_mix_d` is a dict of sc labels by split.
        `sc_sub_dict` is a dict of 'label_id' to 'label_name' for sc data.
        `sc_sub_dict2` is a dict of 'label_name' to 'label_id' for sc data.

    """
    sc_mix_d = {}
    lab_mix_d = {}
    with h5py.File(os.path.join(processed_data_dir, "sc.hdf5"), "r") as f:
        sc_mix_d["train"] = f["X/train"][()]
        sc_mix_d["val"] = f["X/val"][()]
        sc_mix_d["test"] = f["X/test"][()]

        lab_mix_d["train"] = f["y/train"][()]
        lab_mix_d["val"] = f["y/val"][()]
        lab_mix_d["test"] = f["y/test"][()]

    # Load helper dicts / lists
    with open(os.path.join(processed_data_dir, "sc_sub_dict.pkl"), "rb") as f:
        sc_sub_dict = pickle.load(f)

    with open(os.path.join(processed_data_dir, "sc_sub_dict2.pkl"), "rb") as f:
        sc_sub_dict2 = pickle.load(f)

    return sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2
