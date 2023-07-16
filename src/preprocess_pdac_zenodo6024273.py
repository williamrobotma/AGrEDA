#!/usr/bin/env python3

import os

import scanpy as sc

if __name__ == "__main__":
    from da_utils.data_processing import qc_sc

PDAC_DIR = "./data/pdac"
SC_OUT_DIR = os.path.join(PDAC_DIR, "sc_adata")
PROJECT_ID = "zenodo6024273"
SRC_FILE = os.path.join(PDAC_DIR, PROJECT_ID, "pk_all.h5ad")


def process_dset(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)

    # annotate the group of mitochondrial genes as 'mt'
    qc_sc(adata)

    adata.obs.rename(
        columns={
            "Cell_type": "cell_type",
            "Patient2": "subject",
        },
        inplace=True,
    )

    adata.obs["sample_id"] = adata.obs["subject"]
    adata.raw = adata


print(f"reading data from {SRC_FILE} ...")
master_ad = sc.read_h5ad(SRC_FILE)
master_ad.X = master_ad.X.astype("float32")

if not os.path.exists(SC_OUT_DIR):
    os.makedirs(SC_OUT_DIR)

# each project individually
for proj_id in master_ad.obs["Project"].unique():
    if proj_id == "GSE111672":
        continue
    print(f"processing project {proj_id} ...")
    subset = master_ad[(master_ad.obs[["Project", "Type"]] == (proj_id, "Tumor")).all(axis=1)]
    process_dset(subset)
    subset.write_h5ad(os.path.join(SC_OUT_DIR, f"{proj_id}.h5ad"))

# all projects together
print(f"processing integrated set ...")
subset = master_ad[master_ad.obs["Type"] == "Tumor"]
process_dset(subset)
subset.write_h5ad(os.path.join(SC_OUT_DIR, f"{PROJECT_ID}.h5ad"))
