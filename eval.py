#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import os
import datetime
from collections import defaultdict
import pickle
import warnings


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import scanpy as sc

from sklearn.metrics import RocCurveDisplay

from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import metrics
import umap

from imblearn.ensemble import BalancedRandomForestClassifier


import torch
import harmonypy as hm

from src.da_models.adda import ADDAST
from src.da_models.datasets import SpotDataset
from src.utils.evaluation import JSD

# datetime object containing current date and time
script_start_time = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    warnings.warn(
        'Using CPU', 
        stacklevel=2)

# NUM_MARKERS = 20
# N_MIX = 8
# N_SPOTS = 20000
TRAIN_USING_ALL_ST_SAMPLES = False

SAMPLE_ID_N = "151673"

BATCH_SIZE = 1024
NUM_WORKERS = 4
INITIAL_TRAIN_EPOCHS = 100


MIN_EPOCHS = 0.4 * INITIAL_TRAIN_EPOCHS
EARLY_STOP_CRIT = INITIAL_TRAIN_EPOCHS

PROCESSED_DATA_DIR = "data/preprocessed"

MODEL_NAME = "ADDA"

MILISI = True

# %%
results_folder = os.path.join("results", MODEL_NAME, script_start_time)
model_folder = os.path.join("model", MODEL_NAME, script_start_time)

model_folder = os.path.join("model", MODEL_NAME, "TESTING")
results_folder = os.path.join("results", MODEL_NAME, "TESTING")


if not os.path.isdir(results_folder):
    os.makedirs(results_folder)


# %%
# sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


# %% [markdown]
#   # Data load

# %%
print("Loading Data")
# Load spatial data
mat_sp_test_s_d = {}
with h5py.File(os.path.join(PROCESSED_DATA_DIR, "mat_sp_test_s_d.hdf5"), "r") as f:
    for sample_id in f:
        mat_sp_test_s_d[sample_id] = f[sample_id][()]

if TRAIN_USING_ALL_ST_SAMPLES:
    with h5py.File(os.path.join(PROCESSED_DATA_DIR, "mat_sp_train_s.hdf5"), "r") as f:
        mat_sp_train_s = f["all"][()]
else:
    mat_sp_train_s_d = mat_sp_test_s_d

# Load sc data
with h5py.File(os.path.join(PROCESSED_DATA_DIR, "sc.hdf5"), "r") as f:
    sc_mix_train_s = f["X/train"][()]
    sc_mix_val_s = f["X/val"][()]
    sc_mix_test_s = f["X/test"][()]

    lab_mix_train = f["y/train"][()]
    lab_mix_val = f["y/val"][()]
    lab_mix_test = f["y/test"][()]

# Load helper dicts / lists
with open(os.path.join(PROCESSED_DATA_DIR, "sc_sub_dict.pkl"), "rb") as f:
    sc_sub_dict = pickle.load(f)

with open(os.path.join(PROCESSED_DATA_DIR, "sc_sub_dict2.pkl"), "rb") as f:
    sc_sub_dict2 = pickle.load(f)

with open(os.path.join(PROCESSED_DATA_DIR, "st_sample_id_l.pkl"), "rb") as f:
    st_sample_id_l = pickle.load(f)


# %% [markdown]
#   # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#   ## Prepare dataloaders

# %%
print("Setting up dataloaders:")
### source dataloaders
source_train_set = SpotDataset(sc_mix_train_s, lab_mix_train)
source_val_set = SpotDataset(sc_mix_val_s, lab_mix_val)
source_test_set = SpotDataset(sc_mix_test_s, lab_mix_test)

dataloader_source_train = torch.utils.data.DataLoader(
    source_train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
dataloader_source_val = torch.utils.data.DataLoader(
    source_val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
dataloader_source_test = torch.utils.data.DataLoader(
    source_test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

### target dataloaders
target_test_set_d = {}
for sample_id in st_sample_id_l:
    target_test_set_d[sample_id] = SpotDataset(mat_sp_test_s_d[sample_id])

dataloader_target_test_d = {}
for sample_id in st_sample_id_l:
    dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
        target_test_set_d[sample_id],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

if TRAIN_USING_ALL_ST_SAMPLES:
    target_train_set = SpotDataset(mat_sp_train_s)
    dataloader_target_train = torch.utils.data.DataLoader(
        target_train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
else:
    target_train_set_d = {}
    dataloader_target_train_d = {}
    for sample_id in st_sample_id_l:
        target_train_set_d[sample_id] = SpotDataset(mat_sp_test_s_d[sample_id])
        dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
            target_train_set_d[sample_id],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )


# %% [markdown]
#   ## Define Model

# %%
pretrain_folder = os.path.join(model_folder, "pretrain")
advtrain_folder = os.path.join(model_folder, "advtrain")


# %%
st_sample_id_l = [SAMPLE_ID_N]


# %% [markdown]
#  ## Load Models

# %%
# checkpoints_da_d = {}
print("Getting predictions: ")
pred_sp_d, pred_sp_noda_d = {}, {}
if TRAIN_USING_ALL_ST_SAMPLES:
    check_point_da = torch.load(os.path.join(advtrain_folder, f"final_model.pth"))
    model = check_point_da["model"]
    model.to(device)

    model.eval()
    model.target_inference()
    with torch.no_grad():
        for sample_id in st_sample_id_l:
            pred_sp_d[sample_id] = (
                torch.exp(model(torch.Tensor(mat_sp_test_s_d[sample_id]).to(device)))
                .detach()
                .cpu()
                .numpy()
            )

else:
    for sample_id in st_sample_id_l:
        check_point_da = torch.load(
            os.path.join(advtrain_folder, sample_id, f"final_model.pth")
        )
        model = check_point_da["model"]
        model.to(device)

        model.eval()
        model.target_inference()

        with torch.no_grad():
            pred_sp_d[sample_id] = (
                torch.exp(model(torch.Tensor(mat_sp_test_s_d[sample_id]).to(device)))
                .detach()
                .cpu()
                .numpy()
            )

checkpoint_noda = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
model_noda = checkpoint_noda["model"]
model_noda.to(device)

model_noda.eval()
model_noda.set_encoder("source")

with torch.no_grad():
    for sample_id in st_sample_id_l:
        pred_sp_noda_d[sample_id] = (
            torch.exp(model_noda(torch.Tensor(mat_sp_test_s_d[sample_id]).to(device)))
            .detach()
            .cpu()
            .numpy()
        )


# %% [markdown]
#  ## Evaluation of latent space

# %%
rf50_d = {"da": {}, "noda": {}}
splits = ["train", "val", "test"]
for split in splits:
    for k in rf50_d:
        rf50_d[k][split] = {}

if MILISI:
    miLISI_d = {"da": {}, "noda": {}}
    splits = ["train", "val", "test"]
    for split in splits:
        for k in miLISI_d:
            miLISI_d[k][split] = {}

Xs = [sc_mix_train_s, sc_mix_val_s, sc_mix_test_s]
random_states = np.asarray([225, 53, 92])

checkpoint_noda = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
model_noda = checkpoint_noda["model"]
model_noda.to(device)

model_noda.eval()
model_noda.set_encoder("source")

for sample_id in st_sample_id_l:
    print(f"Calculating domain shift for {sample_id}:", end=" ")
    random_states = random_states + 1

    check_point_da = torch.load(
        os.path.join(advtrain_folder, sample_id, f"final_model.pth")
    )
    model = check_point_da["model"]
    model.to(device)

    model.eval()
    model.target_inference()

    for i, (split, X, rs) in enumerate(zip(splits, Xs, random_states)):
        print(split, end=" ")
        figs = []
        with torch.no_grad():
            X = torch.Tensor(X).to(device)
            X_target = torch.Tensor(mat_sp_test_s_d[sample_id]).to(device)

            source_emb = model.source_encoder(X)
            target_emb = model.target_encoder(X_target)

            source_emb_noda = model_noda.source_encoder(X)
            target_emb_noda = model_noda.source_encoder(X_target)

            y_dis = torch.cat(
                [
                    torch.zeros(source_emb.shape[0], device=device, dtype=torch.long),
                    torch.ones(target_emb.shape[0], device=device, dtype=torch.long),
                ]
            )

            emb = torch.cat([source_emb, target_emb])
            emb_noda = torch.cat([source_emb_noda, target_emb_noda])

            emb = emb.detach().cpu().numpy()
            emb_noda = emb_noda.detach().cpu().numpy()
            y_dis = y_dis.detach().cpu().numpy()

        pca_da_df = pd.DataFrame(
            PCA(n_components=2).fit_transform(emb), columns=["PC1", "PC2"]
        )
        pca_noda_df = pd.DataFrame(
            PCA(n_components=2).fit_transform(emb_noda), columns=["PC1", "PC2"]
        )
        pca_da_df["domain"] = pca_noda_df["domain"] = [
            "source" if x == 0 else "target" for x in y_dis
        ]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
        sns.scatterplot(
            data=pca_da_df, x="PC1", y="PC2", hue="domain", ax=axs[0][0], marker="."
        )
        sns.scatterplot(
            data=pca_noda_df, x="PC1", y="PC2", hue="domain", ax=axs[0][1], marker="."
        )

        for ax in axs.flat:
            ax.set_aspect("equal", "box")
        fig.suptitle(f"{sample_id} {split}")
        fig.savefig(
            os.path.join(results_folder, f"PCA_{sample_id}_{split}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        if MILISI:
            meta_df = pd.DataFrame(y_dis, columns=["Domain"])
            miLISI_d["da"][split][sample_id] = np.median(
                hm.compute_lisi(emb, meta_df, ["Domain"])
            )
            miLISI_d["noda"][split][sample_id] = np.median(
                hm.compute_lisi(emb_noda, meta_df, ["Domain"])
            )

        (
            emb_train,
            emb_test,
            emb_noda_train,
            emb_noda_test,
            y_dis_train,
            y_dis_test,
        ) = model_selection.train_test_split(
            emb,
            emb_noda,
            y_dis,
            test_size=0.2,
            random_state=rs,
            stratify=y_dis,
        )

        pca = PCA(n_components=50)
        pca.fit(emb_train)

        emb_train_50 = pca.transform(emb_train)
        emb_test_50 = pca.transform(emb_test)

        clf = BalancedRandomForestClassifier(random_state=145, n_jobs=-1)
        clf.fit(emb_train_50, y_dis_train)
        y_pred_test = clf.predict(emb_test_50)

        # bal_accu_train = metrics.balanced_accuracy_score(y_dis_train, y_pred_train)
        bal_accu_test = metrics.balanced_accuracy_score(y_dis_test, y_pred_test)

        rf50_d["da"][split][sample_id] = bal_accu_test

        pca = PCA(n_components=50)
        pca.fit(emb_noda_train)

        emb_noda_train_50 = pca.transform(emb_noda_train)
        emb_noda_test_50 = pca.transform(emb_noda_test)

        clf = BalancedRandomForestClassifier(random_state=145, n_jobs=-1)
        clf.fit(emb_noda_train_50, y_dis_train)
        y_pred_noda_test = clf.predict(emb_noda_test_50)

        # bal_accu_train = metrics.balanced_accuracy_score(y_dis_train, y_pred_train)
        bal_accu_noda_test = metrics.balanced_accuracy_score(
            y_dis_test, y_pred_noda_test
        )

        rf50_d["noda"][split][sample_id] = bal_accu_noda_test
    # newline at end of split
    print("")


# %% [markdown]
#   # 4. Predict cell fraction of spots and visualization

# %%
adata_spatialLIBD = sc.read_h5ad(
    os.path.join(PROCESSED_DATA_DIR, "adata_spatialLIBD.h5ad")
)

adata_spatialLIBD_d = {}
for sample_id in st_sample_id_l:
    adata_spatialLIBD_d[sample_id] = adata_spatialLIBD[
        adata_spatialLIBD.obs.sample_id == sample_id
    ]
    adata_spatialLIBD_d[sample_id].obsm["spatial"] = (
        adata_spatialLIBD_d[sample_id].obs[["X", "Y"]].values
    )


# %%
num_name_exN_l = []
for k, v in sc_sub_dict.items():
    if "Ex" in v:
        num_name_exN_l.append((k, v, int(v.split("_")[1])))
num_name_exN_l.sort(key=lambda a: a[2])
num_name_exN_l


# %%
Ex_to_L_d = {
    1: {5, 6},
    2: {5},
    3: {4, 5},
    4: {6},
    5: {5},
    6: {4, 5, 6},
    7: {4, 5, 6},
    8: {5, 6},
    9: {5, 6},
    10: {2, 3, 4},
}


# %%
numlist = [t[0] for t in num_name_exN_l]
Ex_l = [t[2] for t in num_name_exN_l]
num_to_ex_d = dict(zip(numlist, Ex_l))


# %%
def plot_cellfraction(visnum, adata, pred_sp, ax=None):
    """Plot predicted cell fraction for a given visnum"""
    adata.obs["Pred_label"] = pred_sp[:, visnum]
    # vmin = 0
    # vmax = np.amax(pred_sp)

    sc.pl.spatial(
        adata,
        img_key="hires",
        color="Pred_label",
        palette="Set1",
        size=1.5,
        legend_loc=None,
        title=f"{sc_sub_dict[visnum]}",
        spot_size=100,
        show=False,
        # vmin=vmin,
        # vmax=vmax,
        ax=ax,
    )


# %%
def plot_roc(visnum, adata, pred_sp, name, ax=None):
    """Plot ROC for a given visnum"""

    def layer_to_layer_number(x):
        for char in x:
            if char.isdigit():
                if int(char) in Ex_to_L_d[num_to_ex_d[visnum]]:
                    return 1
        return 0

    y_pred = pred_sp[:, visnum]
    y_true = adata.obs["spatialLIBD"].map(layer_to_layer_number).fillna(0)
    # print(y_true)
    # print(y_true.isna().sum())
    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)

    return metrics.roc_auc_score(y_true, y_pred)


# %%
fig, ax = plt.subplots(
    nrows=1,
    ncols=len(st_sample_id_l),
    figsize=(3 * len(st_sample_id_l), 3),
    squeeze=False,
    constrained_layout=True,
    dpi=50,
)

cmap = mpl.cm.get_cmap("Accent_r")

color_range = list(
    np.linspace(
        0.125, 1, len(adata_spatialLIBD.obs.spatialLIBD.cat.categories), endpoint=True
    )
)
colors = [cmap(x) for x in color_range]

color_dict = defaultdict(lambda: "lightgrey")
for cat, color in zip(adata_spatialLIBD.obs.spatialLIBD.cat.categories, colors):
    color_dict[cat] = color

color_dict["NA"] = "lightgrey"

legend_elements = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=color,
        markerfacecolor=color_dict[color],
        markersize=10,
    )
    for color in color_dict
]
fig.legend(bbox_to_anchor=(0, 0.5), handles=legend_elements, loc="center right")

for i, sample_id in enumerate(st_sample_id_l):
    sc.pl.spatial(
        adata_spatialLIBD_d[sample_id],
        img_key=None,
        color="spatialLIBD",
        palette=color_dict,
        size=1,
        title=sample_id,
        legend_loc=4,
        na_color="lightgrey",
        spot_size=100,
        show=False,
        ax=ax[0][i],
    )

    ax[0][i].axis("equal")
    ax[0][i].set_xlabel("")
    ax[0][i].set_ylabel("")


# fig.legend(loc=7)
fig.savefig(os.path.join(results_folder, "layers.png"), bbox_inches="tight", dpi=300)
plt.close()


# %%
realspots_d = {"da": {}, "noda": {}}

for sample_id in st_sample_id_l:
    fig, ax = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, dpi=10)

    for i, num in enumerate(numlist):
        plot_cellfraction(
            num, adata_spatialLIBD_d[sample_id], pred_sp_d[sample_id], ax.flat[i]
        )
        ax.flat[i].axis("equal")
        ax.flat[i].set_xlabel("")
        ax.flat[i].set_ylabel("")
    fig.suptitle(sample_id)

    fig.savefig(
        os.path.join(results_folder, f"{sample_id}_cellfraction.png"),
        bbox_inches="tight",
        dpi=300,
    )
    # fig.show()
    plt.close()

    fig, ax = plt.subplots(
        2, 5, figsize=(20, 8), constrained_layout=True, sharex=True, sharey=True, dpi=10
    )

    da_aucs = []
    noda_aucs = []
    for i, num in enumerate(numlist):
        da_aucs.append(
            plot_roc(
                num,
                adata_spatialLIBD_d[sample_id],
                pred_sp_d[sample_id],
                MODEL_NAME,
                ax.flat[i],
            )
        )
        noda_aucs.append(
            plot_roc(
                num,
                adata_spatialLIBD_d[sample_id],
                pred_sp_noda_d[sample_id],
                f"{MODEL_NAME}_wo_da",
                ax.flat[i],
            )
        )

        ax.flat[i].plot(
            [0, 1], [0, 1], transform=ax.flat[i].transAxes, ls="--", color="k"
        )
        ax.flat[i].set_aspect("equal")
        ax.flat[i].set_xlim([0, 1])
        ax.flat[i].set_ylim([0, 1])

        ax.flat[i].set_title(f"{sc_sub_dict[num]}")

        if i >= len(numlist) - 5:
            ax.flat[i].set_xlabel("FPR")
        else:
            ax.flat[i].set_xlabel("")
        if i % 5 == 0:
            ax.flat[i].set_ylabel("TPR")
        else:
            ax.flat[i].set_ylabel("")

    realspots_d["da"][sample_id] = np.mean(da_aucs)
    realspots_d["noda"][sample_id] = np.mean(noda_aucs)

    fig.suptitle(sample_id)
    fig.savefig(
        os.path.join(results_folder, f"{sample_id}_roc.png"),
        bbox_inches="tight",
        dpi=300,
    )
    # fig.show()
    plt.close()


# %%
metric_ctp = JSD()


# %%
jsd_d = {"da": {}, "noda": {}}
for k in jsd_d:
    jsd_d[k] = {"train": {}, "val": {}, "test": {}}

checkpoint_noda = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
model_noda = checkpoint_noda["model"]
model_noda.to(device)

model_noda.eval()
model_noda.set_encoder("source")

for sample_id in st_sample_id_l:
    if TRAIN_USING_ALL_ST_SAMPLES:
        check_point_da = torch.load(os.path.join(advtrain_folder, f"final_model.pth"))

    else:
        check_point_da = torch.load(
            os.path.join(advtrain_folder, sample_id, f"final_model.pth")
        )

    model = check_point_da["model"]
    model.to(device)

    model.eval()
    model.set_encoder("source")

    with torch.no_grad():
        pred_mix_train = model(torch.Tensor(sc_mix_train_s).to(device))

        pred_mix_val = model(torch.Tensor(sc_mix_val_s).to(device))

        pred_mix_test = model(torch.Tensor(sc_mix_test_s).to(device))

        target_names = [sc_sub_dict[i] for i in range(len(sc_sub_dict))]

        jsd_train = metric_ctp(
            torch.Tensor(lab_mix_train).to(device), torch.exp(pred_mix_train)
        )
        jsd_val = metric_ctp(
            torch.Tensor(lab_mix_val).to(device), torch.exp(pred_mix_val)
        )
        jsd_test = metric_ctp(
            torch.Tensor(lab_mix_test).to(device), torch.exp(pred_mix_test)
        )

        jsd_d["da"]["train"][sample_id] = jsd_train.detach().cpu().numpy()
        jsd_d["da"]["val"][sample_id] = jsd_val.detach().cpu().numpy()
        jsd_d["da"]["test"][sample_id] = jsd_test.detach().cpu().numpy()

    with torch.no_grad():
        pred_mix_train = model_noda(torch.Tensor(sc_mix_train_s).to(device))

        pred_mix_val = model_noda(torch.Tensor(sc_mix_val_s).to(device))

        pred_mix_test = model_noda(torch.Tensor(sc_mix_test_s).to(device))

        target_names = [sc_sub_dict[i] for i in range(len(sc_sub_dict))]

        jsd_train = metric_ctp(
            torch.Tensor(lab_mix_train).to(device), torch.exp(pred_mix_train)
        )
        jsd_val = metric_ctp(
            torch.Tensor(lab_mix_val).to(device), torch.exp(pred_mix_val)
        )
        jsd_test = metric_ctp(
            torch.Tensor(lab_mix_test).to(device), torch.exp(pred_mix_test)
        )

        jsd_d["noda"]["train"][sample_id] = jsd_train.detach().cpu().numpy()
        jsd_d["noda"]["val"][sample_id] = jsd_val.detach().cpu().numpy()
        jsd_d["noda"]["test"][sample_id] = jsd_test.detach().cpu().numpy()


# %%
df_keys = [
    "Pseudospots (JS Divergence)",
    "RF50",
    "Real Spots (Mean AUC Ex1-10)",
]

if MILISI:
    df_keys.insert(2, "miLISI")


def gen_l_dfs(da):
    yield pd.DataFrame.from_dict(jsd_d[da], orient="columns")
    yield pd.DataFrame.from_dict(rf50_d[da], orient="columns")
    if MILISI:
        yield pd.DataFrame.from_dict(miLISI_d[da], orient="columns")
    yield pd.Series(realspots_d[da])
    return


results_df = pd.concat(
    [
        pd.concat(
            list(gen_l_dfs(da)),
            axis=1,
            keys=df_keys,
        )
        for da in ["noda", "da"]
    ],
    axis=0,
    keys=["Before DA", "After DA"],
)

results_df.to_csv(os.path.join(results_folder, "results.csv"))
results_df


# %%
