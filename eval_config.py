#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import os
import datetime
from collections import defaultdict
import warnings
import argparse
from joblib import parallel_backend, effective_n_jobs, Parallel, delayed


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import yaml

import scanpy as sc

from sklearn.metrics import RocCurveDisplay

from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import metrics

# import umap

from imblearn.ensemble import BalancedRandomForestClassifier


import torch
import harmonypy as hm

from src.da_models.adda import ADDAST
from src.da_models.dann import DANN

# from src.da_models.datasets import SpotDataset
from src.utils.evaluation import JSD

# from src.utils import data_loading
from src.utils.data_loading import (
    load_spatial,
    load_sc,
    get_selected_dir,
    get_model_rel_path,
)

script_start_time = datetime.datetime.now(datetime.timezone.utc)

SCALER_OPTS = ("minmax", "standard", "celldart")

parser = argparse.ArgumentParser(description="Evaluates.")
# parser.add_argument(
#     "--scaler",
#     "-s",
#     type=str,
#     default="celldart",
#     choices=SCALER_OPTS,
#     help="Scaler to use.",
# )
# parser.add_argument(
#     "-d",
#     type=str,
#     default="./data",
#     help="data directory",
# )
# parser.add_argument(
#     "--allgenes", "-a", action="store_true", help="Turn off marker selection."
# )
# parser.add_argument(
#     "--nmarkers",
#     type=int,
#     default=data_loading.DEFAULT_N_MARKERS,
#     help="Number of top markers in sc training data to used. Ignored if --allgenes flag is used.",
# )
# parser.add_argument(
#     "--nmix",
#     type=int,
#     default=data_loading.DEFAULT_N_MIX,
#     help="number of sc samples to use to generate pseudospots.",
# )
# parser.add_argument(
#     "--nspots",
#     type=int,
#     default=data_loading.DEFAULT_N_SPOTS,
#     help="Number of training pseudospots to use.",
# )
# parser.add_argument("--stsplit", action="store_true", help="Whether to split ST data.")
parser.add_argument("--pretraining", "-p", action="store_false", help="no pretraining")
parser.add_argument("--modelname", "-n", type=str, default="ADDA", help="model name")
parser.add_argument("--milisi", "-m", action="store_false", help="no milisi")
parser.add_argument(
    "--config_fname",
    "-f",
    type=str,
    help="Name of the config file to use",
)
# parser.add_argument("--modelversion", "-v", type=str, default="TESTING", help="model ver")
parser.add_argument(
    "--njobs",
    type=int,
    default=0,
    help="Number of jobs to use for parallel processing.",
)
parser.add_argument(
    "--cuda",
    "-c",
    default=None,
    help="gpu index to use",
)
# parser.add_argument("--seed", default="random", help="seed used for training")
args = parser.parse_args()


# %%
if args.cuda is not None:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    warnings.warn("Using CPU", stacklevel=2)


# TRAIN_USING_ALL_ST_SAMPLES = False

# SAMPLE_ID_N = "151673"

# BATCH_SIZE = 512
# NUM_WORKERS = 16


# PROCESSED_DATA_DIR = "data/preprocessed_markers_celldart"
# DATA_DIR = args.d

# ST_SPLIT = args.stsplit

MODEL_NAME = args.modelname
# MODEL_VERSION = args.modelversion
# PRETRAINING = True
PRETRAINING = args.pretraining

MILISI = True

print(f"Evaluating {MODEL_NAME} on {device} with {args.njobs} jobs")
print(f"Loading config {args.config_fname} ... ")

with open(os.path.join("configs", MODEL_NAME, args.config_fname), "r") as f:
    config = yaml.safe_load(f)

print(yaml.dump(config))

torch_params = config["torch_params"]
data_params = config["data_params"]
model_params = config["model_params"]
train_params = config["train_params"]

if "manual_seed" in torch_params:
    torch_seed = torch_params["manual_seed"]
    torch_seed_path = str(torch_params["manual_seed"])
else:
    torch_seed_path = "random"

# %%
# results_folder = os.path.join("results", MODEL_NAME, script_start_time)
# model_folder = os.path.join("model", MODEL_NAME, script_start_time)

model_rel_path = get_model_rel_path(
    MODEL_NAME,
    model_params["model_version"],
    scaler_name=data_params["scaler_name"],
    n_markers=data_params["n_markers"],
    all_genes=data_params["all_genes"],
    n_mix=data_params["n_mix"],
    n_spots=data_params["n_spots"],
    st_split=data_params["st_split"],
    torch_seed_path=torch_seed_path,
)

model_folder = os.path.join("model", model_rel_path)

# Check to make sure config file matches config file in model folder
with open(os.path.join(model_folder, "config.yml"), "r") as f:
    config_model_folder = yaml.safe_load(f)
if config_model_folder != config:
    raise ValueError("Config file does not match config file in model folder")

results_folder = os.path.join("results", model_rel_path)


if not os.path.isdir(results_folder):
    os.makedirs(results_folder)


# %%
# sc.logging.print_versions()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 3


# %% [markdown]
#   # Data load

selected_dir = get_selected_dir(
    data_params["data_dir"], data_params["n_markers"], data_params["all_genes"]
)

# %%
print("Loading Data")
# Load spatial data
mat_sp_d, mat_sp_train, st_sample_id_l = load_spatial(
    selected_dir,
    data_params["scaler_name"],
    train_using_all_st_samples=data_params["train_using_all_st_samples"],
    st_split=data_params["st_split"],
)

# Load sc data
sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2 = load_sc(
    selected_dir,
    data_params["scaler_name"],
    n_mix=data_params["n_mix"],
    n_spots=data_params["n_spots"],
)


# %% [markdown]
#   # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#   ## Prepare dataloaders

# %%
print("Setting up dataloaders:")
### source dataloaders
# source_train_set = SpotDataset(sc_mix_d["train"], lab_mix_d["train"])
# source_val_set = SpotDataset(sc_mix_d["val"], lab_mix_d["val"])
# source_test_set = SpotDataset(sc_mix_d["test"], lab_mix_d["test"])

# dataloader_source_train = torch.utils.data.DataLoader(
#     source_train_set,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     pin_memory=True,
# )
# dataloader_source_val = torch.utils.data.DataLoader(
#     source_val_set,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=True,
# )
# dataloader_source_test = torch.utils.data.DataLoader(
#     source_test_set,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=True,
# )

# ### target dataloaders
# target_test_set_d = {}
# for sample_id in st_sample_id_l:
#     target_test_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["test"])

# dataloader_target_test_d = {}
# for sample_id in st_sample_id_l:
#     dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
#         target_test_set_d[sample_id],
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#     )

# if TRAIN_USING_ALL_ST_SAMPLES:
#     target_train_set = SpotDataset(mat_sp_train)
#     dataloader_target_train = torch.utils.data.DataLoader(
#         target_train_set,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#     )
# else:
#     target_train_set_d = {}
#     dataloader_target_train_d = {}
#     for sample_id in st_sample_id_l:
#         target_train_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["train"])
#         dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
#             target_train_set_d[sample_id],
#             batch_size=BATCH_SIZE,
#             shuffle=True,
#             num_workers=NUM_WORKERS,
#             pin_memory=True,
#         )


# %% [markdown]
#   ## Define Model

# %%
pretrain_folder = os.path.join(model_folder, "pretrain")
advtrain_folder = os.path.join(model_folder, "advtrain")

pretrain_model_path = os.path.join(pretrain_folder, f"final_model.pth")


# %%
# st_sample_id_l = [SAMPLE_ID_N]


# %% [markdown]
#  ## Load Models

# %%
# checkpoints_da_d = {}
print("Getting predictions: ")
pred_sp_d = {}

if data_params["train_using_all_st_samples"]:
    check_point_da = torch.load(
        os.path.join(advtrain_folder, f"final_model.pth"), map_location=device
    )
    model = check_point_da["model"]
    model.to(device)

    model.eval()
    model.target_inference()
    with torch.no_grad():
        for sample_id in st_sample_id_l:
            out = model(torch.as_tensor(mat_sp_d[sample_id]["test"]).float().to(device))
            if isinstance(out, tuple):
                out = out[0]
            pred_sp_d[sample_id] = torch.exp(out).detach().cpu().numpy()

else:
    for sample_id in st_sample_id_l:
        check_point_da = torch.load(
            os.path.join(advtrain_folder, sample_id, f"final_model.pth"),
            map_location=device,
        )
        model = check_point_da["model"]
        model.to(device)

        model.eval()
        model.target_inference()

        with torch.no_grad():
            out = model(torch.as_tensor(mat_sp_d[sample_id]["test"]).float().to(device))
            if isinstance(out, tuple):
                out = out[0]
            pred_sp_d[sample_id] = torch.exp(out).detach().cpu().numpy()

if PRETRAINING:
    pred_sp_noda_d = {}
    checkpoint_noda = torch.load(pretrain_model_path, map_location=device)
    model_noda = checkpoint_noda["model"]
    model_noda.to(device)

    model_noda.eval()
    model_noda.set_encoder("source")

    with torch.no_grad():
        for sample_id in st_sample_id_l:
            out = model_noda(
                torch.as_tensor(mat_sp_d[sample_id]["test"]).float().to(device)
            )
            if isinstance(out, tuple):
                out = out[0]
            pred_sp_noda_d[sample_id] = torch.exp(out).detach().cpu().numpy()


# %% [markdown]
#  ## Evaluation of latent space

# %%
rf50_d = {"da": {}, "noda": {}}
splits = ["train", "val", "test"]
for split in splits:
    for k in rf50_d:
        rf50_d[k][split] = {}

if MILISI:
    miLISI_d = {"da": {}}
    if PRETRAINING:
        miLISI_d["noda"] = {}
    splits = ["train", "val", "test"]
    for split in splits:
        for k in miLISI_d:
            miLISI_d[k][split] = {}

Xs = [sc_mix_d["train"], sc_mix_d["val"], sc_mix_d["test"]]
random_states = np.asarray([225, 53, 92])

if PRETRAINING:
    checkpoint_noda = torch.load(pretrain_model_path, map_location=device)
    model_noda = checkpoint_noda["model"]
    model_noda.to(device)

    model_noda.eval()
    model_noda.set_encoder("source")

for sample_id in st_sample_id_l:
    print(f"Calculating domain shift for {sample_id}:", end=" ")
    random_states = random_states + 1

    check_point_da = torch.load(
        os.path.join(advtrain_folder, sample_id, f"final_model.pth"),
        map_location=device,
    )
    model = check_point_da["model"]
    model.to(device)

    model.eval()
    model.target_inference()

    for i, (split, X, rs) in enumerate(zip(splits, Xs, random_states)):
        print(split.upper(), end=" |")
        figs = []
        with torch.no_grad():
            X = torch.as_tensor(X).float().to(device)
            X_target = torch.as_tensor(mat_sp_d[sample_id]["test"]).float().to(device)

            y_dis = torch.cat(
                [
                    torch.zeros(X.shape[0], device=device, dtype=torch.long),
                    torch.ones(X_target.shape[0], device=device, dtype=torch.long),
                ]
            )

            try:
                source_emb = model.source_encoder(X)
            except AttributeError:
                source_emb = model.encoder(X)
            try:
                target_emb = model.target_encoder(X_target)
            except AttributeError:
                target_emb = model.encoder(X_target)

            emb = torch.cat([source_emb, target_emb])

            emb = emb.detach().cpu().numpy()
            y_dis = y_dis.detach().cpu().numpy()

            if PRETRAINING:
                try:
                    source_emb_noda = model_noda.source_encoder(X)
                    target_emb_noda = model_noda.source_encoder(X_target)
                except AttributeError:
                    source_emb_noda = model_noda.encoder(X)
                    target_emb_noda = model_noda.encoder(X_target)

                emb_noda = torch.cat([source_emb_noda, target_emb_noda])

                emb_noda = emb_noda.detach().cpu().numpy()

        n_cols = 2 if PRETRAINING else 1
        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 10, 10), squeeze=False)
        pca_da_df = pd.DataFrame(
            PCA(n_components=2).fit_transform(emb), columns=["PC1", "PC2"]
        )

        pca_da_df["domain"] = ["source" if x == 0 else "target" for x in y_dis]
        sns.scatterplot(
            data=pca_da_df, x="PC1", y="PC2", hue="domain", ax=axs[0][0], marker="."
        )

        if PRETRAINING:
            pca_noda_df = pd.DataFrame(
                PCA(n_components=2).fit_transform(emb_noda), columns=["PC1", "PC2"]
            )
            pca_noda_df["domain"] = pca_da_df["domain"]

            sns.scatterplot(
                data=pca_noda_df,
                x="PC1",
                y="PC2",
                hue="domain",
                ax=axs[0][1],
                marker=".",
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
        n_jobs = effective_n_jobs(args.njobs)
        with parallel_backend("loky", n_jobs=n_jobs):
            if MILISI:
                print(" milisi", end=" ")
                meta_df = pd.DataFrame(y_dis, columns=["Domain"])
                miLISI_d["da"][split][sample_id] = np.median(
                    hm.compute_lisi(emb, meta_df, ["Domain"])
                )

                if PRETRAINING:
                    miLISI_d["noda"][split][sample_id] = np.median(
                        hm.compute_lisi(emb_noda, meta_df, ["Domain"])
                    )
        print("rf50", end=" ")
        if PRETRAINING:
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
        else:
            (
                emb_train,
                emb_test,
                y_dis_train,
                y_dis_test,
            ) = model_selection.train_test_split(
                emb, y_dis, test_size=0.2, random_state=rs, stratify=y_dis
            )

        pca = PCA(n_components=min(50, emb_train.shape[1]))
        emb_train_50 = pca.fit_transform(emb_train)
        emb_test_50 = pca.transform(emb_test)

        clf = BalancedRandomForestClassifier(random_state=145, n_jobs=n_jobs)
        clf.fit(emb_train_50, y_dis_train)
        y_pred_test = clf.predict(emb_test_50)

        # bal_accu_train = metrics.balanced_accuracy_score(y_dis_train, y_pred_train)
        bal_accu_test = metrics.balanced_accuracy_score(y_dis_test, y_pred_test)

        rf50_d["da"][split][sample_id] = bal_accu_test

        if PRETRAINING:
            pca = PCA(n_components=50)
            emb_noda_train_50 = pca.fit_transform(emb_noda_train)
            emb_noda_test_50 = pca.transform(emb_noda_test)

            clf = BalancedRandomForestClassifier(random_state=145, n_jobs=n_jobs)
            clf.fit(emb_noda_train_50, y_dis_train)
            y_pred_noda_test = clf.predict(emb_noda_test_50)

            # bal_accu_train = metrics.balanced_accuracy_score(y_dis_train, y_pred_train)
            bal_accu_noda_test = metrics.balanced_accuracy_score(
                y_dis_test, y_pred_noda_test
            )

            rf50_d["noda"][split][sample_id] = bal_accu_noda_test
        print("|", end=" ")
    # newline at end of split
    print("")


# %% [markdown]
#   # 4. Predict cell fraction of spots and visualization

# %%
adata_spatialLIBD = sc.read_h5ad(os.path.join(selected_dir, "adata_spatialLIBD.h5ad"))

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
realspots_d = {"da": {}}
if PRETRAINING:
    realspots_d["noda"] = {}


def _plot_samples(sample_id):
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
    if PRETRAINING:
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
        if PRETRAINING:
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



    fig.suptitle(sample_id)
    fig.savefig(
        os.path.join(results_folder, f"{sample_id}_roc.png"),
        bbox_inches="tight",
        dpi=300,
    )
    # fig.show()
    plt.close()

    # realspots_d["da"][sample_id] = np.mean(da_aucs)
    # if PRETRAINING:
    #     realspots_d["noda"][sample_id] = np.mean(noda_aucs)
    return np.mean(da_aucs), np.mean(noda_aucs) if PRETRAINING else None
# for sample_id in st_sample_id_l:
#     plot_samples(sample_id)
n_jobs_samples = min(n_jobs, len(st_sample_id_l)) if n_jobs > 0 else n_jobs
aucs = Parallel(n_jobs=n_jobs_samples)(delayed(_plot_samples)(sid) for sid in st_sample_id_l)
for sample_id, auc in zip(st_sample_id_l, aucs):
    realspots_d["da"][sample_id] = auc[0]
    if PRETRAINING:
        realspots_d["noda"][sample_id] = auc[1]


# %%
metric_ctp = JSD()
# metric_ctp = lambda y_pred, y_true: torch.nn.KLDivLoss(reduction="batchmean")(y_pred.log(), y_true)


# %%
jsd_d = {"da": {}}
if PRETRAINING:
    jsd_d["noda"] = {}

for k in jsd_d:
    jsd_d[k] = {"train": {}, "val": {}, "test": {}}

if PRETRAINING:
    checkpoint_noda = torch.load(pretrain_model_path, map_location=device)
    model_noda = checkpoint_noda["model"]
    model_noda.to(device)

    model_noda.eval()
    model_noda.set_encoder("source")

for sample_id in st_sample_id_l:
    if data_params["train_using_all_st_samples"]:
        check_point_da = torch.load(
            os.path.join(advtrain_folder, f"final_model.pth"), map_location=device
        )

    else:
        check_point_da = torch.load(
            os.path.join(advtrain_folder, sample_id, f"final_model.pth"),
            map_location=device,
        )

    model = check_point_da["model"]
    model.to(device)

    model.eval()
    model.set_encoder("source")

    with torch.no_grad():
        pred_mix_train = model(torch.as_tensor(sc_mix_d["train"]).float().to(device))
        pred_mix_val = model(torch.as_tensor(sc_mix_d["val"]).float().to(device))
        pred_mix_test = model(torch.as_tensor(sc_mix_d["test"]).float().to(device))

        if isinstance(pred_mix_train, tuple):
            pred_mix_train = pred_mix_train[0]
            pred_mix_val = pred_mix_val[0]
            pred_mix_test = pred_mix_test[0]

        # target_names = [sc_sub_dict[i] for i in range(len(sc_sub_dict))]

        jsd_train = metric_ctp(
            torch.exp(pred_mix_train),
            torch.as_tensor(lab_mix_d["train"]).float().to(device),
        )
        jsd_val = metric_ctp(
            torch.exp(pred_mix_val),
            torch.as_tensor(lab_mix_d["val"]).float().to(device),
        )
        jsd_test = metric_ctp(
            torch.exp(pred_mix_test),
            torch.as_tensor(lab_mix_d["test"]).float().to(device),
        )

        jsd_d["da"]["train"][sample_id] = jsd_train.detach().cpu().numpy()
        jsd_d["da"]["val"][sample_id] = jsd_val.detach().cpu().numpy()
        jsd_d["da"]["test"][sample_id] = jsd_test.detach().cpu().numpy()

        if PRETRAINING:
            pred_mix_train = model_noda(
                torch.as_tensor(sc_mix_d["train"]).float().to(device)
            )
            pred_mix_val = model_noda(
                torch.as_tensor(sc_mix_d["val"]).float().to(device)
            )
            pred_mix_test = model_noda(
                torch.as_tensor(sc_mix_d["test"]).float().to(device)
            )

            if isinstance(pred_mix_train, tuple):
                pred_mix_train = pred_mix_train[0]
                pred_mix_val = pred_mix_val[0]
                pred_mix_test = pred_mix_test[0]

            # target_names = [sc_sub_dict[i] for i in range(len(sc_sub_dict))]

            jsd_train = metric_ctp(
                torch.exp(pred_mix_train),
                torch.as_tensor(lab_mix_d["train"]).float().to(device),
            )
            jsd_val = metric_ctp(
                torch.exp(pred_mix_val),
                torch.as_tensor(lab_mix_d["val"]).float().to(device),
            )
            jsd_test = metric_ctp(
                torch.exp(pred_mix_test),
                torch.as_tensor(lab_mix_d["test"]).float().to(device),
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
    """Generate a list of series for a given da"""
    df = pd.DataFrame.from_dict(jsd_d[da], orient="columns")
    df.columns.name = "SC Split"
    yield df
    df = pd.DataFrame.from_dict(rf50_d[da], orient="columns")
    df.columns.name = "SC Split"
    yield df
    if MILISI:
        df = pd.DataFrame.from_dict(miLISI_d[da], orient="columns")
        df.columns.name = "SC Split"
        yield df
    yield pd.Series(realspots_d[da])
    return


da_dict_keys = ["da"]
da_df_keys = ["After DA"]
if PRETRAINING:
    da_dict_keys.insert(0, "noda")
    da_df_keys.insert(0, "Before DA")

results_df = pd.concat(
    [
        pd.concat(
            list(gen_l_dfs(da)),
            axis=1,
            keys=df_keys,
        )
        for da in da_dict_keys
    ],
    axis=0,
    keys=da_df_keys,
)

results_df.to_csv(os.path.join(results_folder, "results.csv"))
print(results_df)

# %%
with open(os.path.join(results_folder, "config.yml"), "w") as f:
    yaml.dump(config, f)

print(
    "Script run time:", datetime.datetime.now(datetime.timezone.utc) - script_start_time
)
