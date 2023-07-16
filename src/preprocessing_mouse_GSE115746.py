#!/usr/bin/env python3
# %%
import glob
import itertools
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

cell_subclass_to_spot_composition = {
    "Astro": {"Astrocytes deep", "Astrocytes superficial"},
    "CR": set(),
    "Batch Grouping": {"Excitatory layer 5/6"},  # all cell clusters are L5
    "L2/3 IT": {"Excitatory layer II", "Excitatory layer 3"},
    "L4": {"Excitatory layer 4"},
    "L5 PT": {"Excitatory layer 5/6"},
    "L5 IT": {"Excitatory layer 5/6"},
    "L6 CT": {"Excitatory layer 5/6"},
    "L6 IT": {"Excitatory layer 5/6"},
    "L6b": {"Excitatory layer 5/6"},
    "NP": {"Excitatory layer 5/6"},  # all NP are L5 or L6
    "Endo": {"Endothelial", "Choroid plexus"},
    "High Intronic": {"Excitatory layer 5/6"},  # all High Intronic are VISp L5 Endou
    ## Doublets; these are cell clusters
    "Peri": {"Endothelial", "Choroid plexus"},  # pericyte
    "SMC": set(),  # smooth muscle cell
    "VLMC": set(),  # vascular leptomeningeal cell
    "Macrophage": {"Microglia"},
    "Lamp5": {
        "Interneurons",
        "Interneurons deep",
    },  # "We define six subclasses of GABAergic cells: Sst, Pvalb, Vip, Lamp5, Sncg and Serpinf1, and two distinct types: Sst–Chodl and Meis2–Adamts19 (Fig. 1c). We represent the taxonomy by constellation diagrams, dendrograms, layer-of-isolation, and the expression of select marker genes (Fig. 5a–f). The major division among GABAergic types largely corresponds to their developmental origin in the medial ganglionic eminence (Pvalb and Sst subclasses) or caudal ganglionic eminence (Lamp5, Sncg, Serpinf1 and Vip subclasses)."
    "Meis2": {"Interneurons", "Interneurons deep"},
    "Pvalb": {"Interneurons", "Interneurons deep"},
    "Serpinf1": {"Interneurons", "Interneurons deep"},
    "Sncg": {"Interneurons", "Interneurons deep"},
    "Sst": {"Interneurons", "Interneurons deep"},
    "Vip": {"Interneurons", "Interneurons deep"},
    "Oligo": {"Oligodendrocytes", "OPC"},
    "keep_the_rest": {"Ependymal", "NSC", "Neural progenitors", "Neuroblasts"},
}

cell_cluster_cell_type_to_spot_composition = {
    "Doublet VISp L5 NP and L6 CT": {"Excitatory layer 5/6"},
    "Doublet Endo": {
        "Endothelial",
        "Choroid plexus",
    },  # no choroid plexus in other dataset
    "Doublet Astro Aqp4 Ex": {"Astrocytes deep", "Astrocytes superficial"},
}

cell_cluster_to_cell_type = {
    "Doublet VISp L5 NP and L6 CT": "Doublet VISp L5 NP and L6 CT",
    "Doublet Endo and Peri_1": "Doublet Endo",
    "Doublet Astro Aqp4 Ex": "Doublet Astro Aqp4 Ex",
    "Doublet SMC and Glutamatergic": "Doublet Endo",
    "Doublet Endo Peri SMC": "Doublet Endo",
}


def gen_cell_type(row):
    if row["cell_subclass"] in cell_subclass_to_spot_composition:
        return row["cell_subclass"]
    if row["cell_cluster"] in cell_cluster_to_cell_type:
        return cell_cluster_to_cell_type[row["cell_cluster"]]

    return np.nan


# %%
DSET_DIR = "data/mouse_cortex"

SC_ID = "GSE115746"

DATA_DIR = os.path.join(DSET_DIR, SC_ID)
RAW_PATH = os.path.join(DATA_DIR, "GSE115746_cells_exon_counts.csv")
RAW_PATH_META = os.path.join(DATA_DIR, "GSE115746_complete_metadata_28706-cells.csv")

sc_dir = os.path.join(DSET_DIR, "sc_adata")


def main():
    # %%
    if not os.path.exists(os.path.join(DATA_DIR, "GSE115746_cells_exon_counts.csv")):
        os.system(
            f"wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746%5Fcells%5Fexon%5Fcounts%2Ecsv%2Egz -P '{DATA_DIR}'"
        )
    if not os.path.exists(os.path.join(DATA_DIR, "GSE115746_complete_metadata_28706-cells.csv")):
        os.system(
            f"wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115746/suppl/GSE115746%5Fcomplete%5Fmetadata%5F28706%2Dcells%2Ecsv%2Egz -P '{DATA_DIR}'"
        )

    os.system(f"gunzip -r '{DATA_DIR}' -f")
    for f in glob.glob(os.path.join(DATA_DIR, "*.gz")):
        os.remove(f)

    # %%
    sc.logging.print_versions()
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    sc.settings.verbosity = 3

    # %%

    adata_cortex = sc.read_csv(RAW_PATH).T
    adata_cortex_meta = pd.read_csv(RAW_PATH_META, index_col=0)
    adata_cortex_meta_ = adata_cortex_meta.loc[adata_cortex.obs.index,]

    adata_cortex.obs = adata_cortex_meta_
    adata_cortex.var_names_make_unique()

    qc_sc(adata_cortex)

    print("converting cell type")
    adata_cortex.obs["cell_type"] = adata_cortex.obs[["cell_subclass", "cell_cluster"]].apply(
        gen_cell_type, axis=1
    )
    # %%
    adata_cortex.X = csr_matrix(adata_cortex.X)

    if not os.path.exists(sc_dir):
        os.makedirs(sc_dir)
    adata_cortex.write(os.path.join(sc_dir, f"{SC_ID}.h5ad"))


if __name__ == "__main__":
    from da_utils.data_processing import qc_sc

    main()
