#!/usr/bin/env python3

# %%
import glob
import os

import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix


from src.da_utils.data_processing import qc_sc

# %%

SPOTLESS_DIR = "data/spotless/standards"

CORTEX_DIR = "data/mouse_cortex"
OLFACTORY_DIR = "data/mouse_olfactory"
# SC_ID_OLFACTORY = "spotless_mouse_olfactory"
# SC_ID_CORTEX = "spotless_mouse_cortex"
# SC_ID_VISUAL = "spotless_mouse_visual"


# %%

standard_to_id = {
    "gold_standard_1": "spotless_mouse_cortex",
    "gold_standard_2": "spotless_mouse_olfactory",
    "gold_standard_3": "spotless_mouse_visual",
    "gold_standard_3_12celltypes": "spotless_mouse_visual",
}

id_to_dir = {
    "spotless_mouse_cortex": CORTEX_DIR,
    "spotless_mouse_olfactory": OLFACTORY_DIR,
    "spotless_mouse_visual": CORTEX_DIR,
}


# %%
def cat_to_obsm(cat, adata, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    selected_cols = adata.obs.columns.to_series().map(lambda x: x.split(".")[0] == cat)
    adata.obsm[cat] = (
        adata.obs.loc[:, selected_cols]
        .rename(columns=lambda x: x[len(cat + ".") :])
        .drop(columns=drop_cols)
    )
    keep_cols = ~selected_cols
    # print(keep_cols)

    for drop_col in drop_cols:
        keep_cols[cat + "." + drop_col] = True
    adata.obs = adata.obs.loc[:, keep_cols]


# %%
for dir in glob.glob(os.path.join(SPOTLESS_DIR, "*/")):
    name = os.path.basename(os.path.normpath(dir))
    if name != "reference":
        id = standard_to_id[name]
        dset_dir = id_to_dir[id]

        print(f"Processing {name} to {id} in {dset_dir}")
        st_dir = os.path.join(dset_dir, "st_adata")
        if not os.path.exists(st_dir):
            os.makedirs(st_dir)

        fpaths = sorted(glob.glob(os.path.join(dir, "*.h5ad")))
        sample_ids = [os.path.basename(f).split(".")[0] for f in fpaths]

        fovs = [sc.read_h5ad(name) for name in fpaths]
        obs_cols = sorted(list(set.union(*[set(fov.obs.columns) for fov in fovs])))
        for fov, sample_id in zip(fovs, sample_ids):
            fov.obs = fov.obs.reindex(columns=obs_cols)
            fov.obs = fov.obs.fillna(0)
            fov.obs = fov.obs.rename(columns={"coordinates.x": "X", "coordinates.y": "Y"})
            fov.X = csr_matrix(fov.X.astype("float32"))
            fov.raw = fov

            fov.obs = fov.obs.loc[:, ~fov.obs.columns.str.contains("spot_no")]
            cat_to_obsm("relative_spot_composition", fov)
            cat_to_obsm("spot_composition", fov)

            fov.write(os.path.join(st_dir, f"{id}-{sample_id}.h5ad"))


# %%
for ref_path in glob.glob(os.path.join(SPOTLESS_DIR, "reference", "*.h5ad")):
    if "19celltypes" in ref_path:
        continue

    name = os.path.basename(ref_path).split(".")[0]
    id = standard_to_id[name]

    dset_dir = id_to_dir[id]
    print(f"Processing {name} to {id} in {dset_dir}")

    sc_dir = os.path.join(dset_dir, "sc_adata")
    if not os.path.exists(sc_dir):
        os.makedirs(sc_dir)

    adata_sc = sc.read_h5ad(ref_path)

    qc_sc(adata_sc)

    adata_sc.obs = adata_sc.obs.rename(columns={"celltype": "cell_subclass"})

    adata_sc.X = csr_matrix(adata_sc.X.astype("float32"))
    adata_sc.raw = adata_sc

    adata_sc.write(os.path.join(sc_dir, f"{id}.h5ad"))

# %%
