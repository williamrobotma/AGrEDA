# %%
from collections import defaultdict
import gc
import os

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_array, csr_matrix

import anndata as ad
import numpy as np


# %%
SPATIALLIBD_DIR = "data/dlpfc/spatialLIBD_data"


# %%
try:
    spots = pd.read_pickle(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_counts.pkl"))
    st = pd.read_pickle(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_st.pkl"))
    gene_meta = pd.read_pickle(os.path.join(SPATIALLIBD_DIR, "gene_meta.pkl"))
    cell_type = pd.read_pickle(os.path.join(SPATIALLIBD_DIR, "RowDataTable1.pkl"))
    csr = pd.read_pickle(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_csr_counts_sample_id.pkl"))
except FileNotFoundError as e:
    try:
        spots = pd.read_csv(
            os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_counts.csv"),
            header=0,
            index_col=0,
            sep=",",
        )
        st = pd.read_csv(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_st.csv"), index_col=0)
        gene_meta = pd.read_csv(os.path.join(SPATIALLIBD_DIR, "gene_meta.csv"))
        cell_type = pd.read_csv(os.path.join(SPATIALLIBD_DIR, "RowDataTable1.csv"))
        csr = pd.read_csv(
            os.path.join(SPATIALLIBD_DIR, "spatialLIBD_csr_counts_sample_id.csv"),
            index_col=0,
            dtype=defaultdict(lambda: "string", count=np.float32),
        )

        spots.to_pickle(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_counts.pkl"))
        st.to_pickle(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_st.pkl"))
        gene_meta.to_pickle(os.path.join(SPATIALLIBD_DIR, "gene_meta.pkl"))
        cell_type.to_pickle(os.path.join(SPATIALLIBD_DIR, "RowDataTable1.pkl"))
        csr.to_pickle(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_csr_counts_sample_id.pkl"))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Download from spatialLIBD and convert to .csv into {SPATIALLIBD_DIR}"
        ) from e


# %%
print(spots)
# display(spots)
print(st)
# display(st)
print(gene_meta)
# display(gene_meta)
print(cell_type)
# display(cell_type)
print(csr)
# display(csr)


# %%
print(spots.columns)


# %%
# rename st column names
st.columns = ["X", "Y"]
st.index.name = "spot"
st = st.reset_index()
print(st.head())


# %%
# spot = spots[
#     [
#         "sample_id",
#         "key",
#         "subject",
#         "replicate",
#         "Cluster",
#         "sum_umi",
#         "sum_gene",
#         "cell_count",
#         "in_tissue",
#         "spatialLIBD",
#         "array_col",
#         "array_row",
#     ]
# ]
spot = spots
print(spot.shape, st.shape)


# %%
# merge spot and st info -- merging based on index... no other specifying info in st:S, seems okay?
spot_meta = st.join(spot.reset_index())
spot_meta["sample_id"] = spot_meta["sample_id"].astype("string").astype("object")
print(spot_meta)


# %%


# %%
assert (spot_meta.spot.isin(spot_meta["index"])).all()


# %%
def plot_cell_layers(df):
    layer_idx = df["spatialLIBD"].unique()

    fig, ax = plt.subplots(nrows=1, ncols=12, figsize=(50, 6))
    samples = df["sample_id"].unique()

    for idx, sample in enumerate(samples):
        cells_of_samples = df[df["sample_id"] == sample]
        for index in layer_idx:
            cells_of_layer = cells_of_samples[cells_of_samples["spatialLIBD"] == index]
            ax[idx].scatter(-cells_of_layer["Y"], cells_of_layer["X"], label=index)
        ax[idx].set_title(sample)
    plt.legend()
    plt.show()


# %%
plot_cell_layers(spot_meta)


# %%
print(cell_type)


# %%
cell_type = cell_type.set_index("ID")
cell_type.index.name = "gene_id"


# %%
# cell_type_idx_df = cell_type[["gene_biotype", "ID"]]


# %%
# cell_type.drop(columns=["Unnamed: 0", "gene_biotype", "ID"], inplace=True)


# %%
# ID_to_symbol = cell_type_idx_df.ID.reset_index().set_index("ID")["Symbol"]
# ID_to_symbol_d = ID_to_symbol.to_dict()


# %%
# cell_type_idx_df = cell_type_idx_df.reset_index().set_index("ID")


# %%
gene_meta = gene_meta.drop(columns=["Unnamed: 0"]).set_index("gene_id")


# %%

# %%
del spots
del spot
# del gene_meta
del st
# del cell_type
# del cell_type_idx_df

gc.collect()


# %%
print("LONG TO WIDE")


# %%
idx = pd.Series(zip(csr["sample_id"], csr["spot"]))

idx_c = CategoricalDtype(sorted(idx.unique()), ordered=True)
gene_c = CategoricalDtype(sorted(csr["gene"].unique()), ordered=True)

row = idx.astype(idx_c).cat.codes
col = csr["gene"].astype(gene_c).cat.codes
# sparse_matrix =
# sparse_matrix #, sparse_matrix.toarray()

# %%
# wide = csr.pivot_table(
#     index=["sample_id", "spot"],
#     columns="gene",
#     values="count",
#     fill_value=0.0,
#     sort=False,
# )  # .astype(pd.SparseDtype(np.float32, 0.0))
# wide = wide.fillna(0)
# wide = wide.astype(pd.SparseDtype("float", 0.0))

# %%
counts_df = pd.DataFrame.sparse.from_spmatrix(
    csr_array(
        (csr["count"], (row, col)),
        shape=(idx_c.categories.size, gene_c.categories.size),
    ),
    index=pd.MultiIndex.from_tuples(idx_c.categories, names=["sample_id", "spot"]),
    columns=gene_c.categories,
)  # .astype(pd.SparseDtype(np.float32, 0.0))

# %%
# counts_df = wide
print(counts_df.iloc[:5, :5])


# %%
# # working with sampleID 151673 only, for now
# dlpfc = spot_meta[spot_meta['sample_id'] == 151673]
spot_meta = spot_meta.set_index(["sample_id", "spot"])
print(spot_meta.index)


# %%
# gene_meta = gene_meta.reindex(counts_df.columns)
counts_df = (
    counts_df.reindex(spot_meta.index, axis=0, fill_value=np.float32(0.0))
    .reindex(gene_meta.index, axis=1, fill_value=np.float32(0.0))
    .astype(pd.SparseDtype(np.float32, 0.0))
)
# counts_df = counts_df.loc[spot_meta.index, gene_meta.index]
print(counts_df)

# %%
temp = counts_df.loc[counts_df.any(axis=1), counts_df.any(axis=0)]  # .dropna(axis=1).dropna(axis=0)


# %%
counts_df.to_pickle(os.path.join(SPATIALLIBD_DIR, "counts_df.pkl"))


# %%
spot_meta.to_pickle(os.path.join(SPATIALLIBD_DIR, "dlpfc.pkl"))


# %%
gene_meta.to_pickle(os.path.join(SPATIALLIBD_DIR, "gene_meta_processed.pkl"))
temp.to_pickle(os.path.join(SPATIALLIBD_DIR, "temp.pkl"))


# %%
counts_df.iloc[:, 0]

# %%
adata = ad.AnnData(
    X=csr_matrix(counts_df.sparse.to_coo()),
    obs=spot_meta.reset_index().set_index("index"),
    var=gene_meta,
    varm=cell_type.iloc[:, 3:].reindex(gene_meta.index),
    dtype=np.float32,
)
print(adata)


# %%
adata.varm


# %%
adata.varm["propNucleiExprs"]


# %%
adata.X


# %%
for k, v in adata.varm.items():
    adata.varm[k] = v.to_numpy(dtype=np.float32)


# %%
adata.write_h5ad(os.path.join(SPATIALLIBD_DIR, "spatialLIBD.h5ad"))


# %%
# # %%
# adata2 = ad.read_h5ad(os.path.join(SPATIALLIBD_DIR, "spatialLIBD.h5ad"))


# %%
# # %%
# adata2.X.toarray().sum()


# %%
