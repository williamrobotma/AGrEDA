#!/usr/bin/env python3

# %%
import gc
import os

import matplotlib.pyplot as plt
import pandas as pd

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
        st = pd.read_csv(os.path.join(SPATIALLIBD_DIR, "spatialLIBD_spot_st.csv"))
        gene_meta = pd.read_csv(os.path.join(SPATIALLIBD_DIR, "gene_meta.csv"))
        cell_type = pd.read_csv(os.path.join(SPATIALLIBD_DIR, "RowDataTable1.csv"))
        csr = pd.read_csv(
            os.path.join(SPATIALLIBD_DIR, "spatialLIBD_csr_counts_sample_id.csv"), index_col=0
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
print("spots")
# display(spots)
print("st")
# display(st)
print("gene_meta")
# display(gene_meta)
print("cell_type")
# display(cell_type)
print("csr")
# display(csr)


# %%
print(spots.columns)


# %%
# rename st column names
st.columns = ["spot", "X", "Y"]
print(st.head())


# %%
spot = spots[
    [
        "sample_id",
        "key",
        "subject",
        "replicate",
        "Cluster",
        "sum_umi",
        "sum_gene",
        "cell_count",
        "in_tissue",
        "spatialLIBD",
        "array_col",
        "array_row",
    ]
]
print(spot)


# %%
# merge spot and st info -- merging based on index... no other specifying info in st:S, seems okay?
spot_meta = st.join(spot.reset_index())
print(spot_meta)


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
print(plot_cell_layers(spot_meta))


# %%
print(cell_type)


# %%
cell_type = cell_type.set_index("Symbol")


# %%
cell_type_idx_df = cell_type.iloc[:, :3]


# %%
cell_type = cell_type.drop(["Unnamed: 0", "gene_biotype", "ID"], axis=1)


# %%
ID_to_symbol_d = cell_type_idx_df.ID.reset_index().set_index("ID")["Symbol"].to_dict()


# %%
del spots
del spot
del gene_meta
del st
del cell_type
del cell_type_idx_df

gc.collect()

# %%
wide = (
    csr.pivot_table(index=["sample_id", "spot"], columns="gene", values="count")
    .fillna(0)
    .astype(pd.SparseDtype("float", 0.0))
)
# wide = wide.fillna(0)
# wide = wide.astype(pd.SparseDtype("float", 0.0))


# %%
counts_df = wide
print(counts_df)


# %%
counts_df.columns = counts_df.columns.map(ID_to_symbol_d, na_action=None)
print(counts_df)


# %%
# # working with sampleID 151673 only, for now
# dlpfc = spot_meta[spot_meta['sample_id'] == 151673]
dlpfc = spot_meta


# %%
dlpfc = dlpfc.set_index(["sample_id", "spot"])


# %%
print(dlpfc)


# %%
temp = pd.concat([dlpfc, counts_df], join="inner", axis=1)


# %%
temp = temp.drop(
    columns=[
        "X",
        "Y",
        "index",
        "key",
        "subject",
        "replicate",
        "Cluster",
        "sum_umi",
        "sum_gene",
        "cell_count",
        "in_tissue",
        "spatialLIBD",
        "array_col",
        "array_row",
    ]
)
print(temp)


# %%
# same_genes = cell_type[cell_type.index.isin(temp.columns)]
# print(same_genes)


# %%
counts_df.to_pickle(os.path.join(SPATIALLIBD_DIR, "counts_df.pkl"))


# %%
print(dlpfc)


# %%
dlpfc.to_pickle(os.path.join(SPATIALLIBD_DIR, "dlpfc.pkl"))


# %%
temp.to_pickle(os.path.join(SPATIALLIBD_DIR, "temp.pkl"))
