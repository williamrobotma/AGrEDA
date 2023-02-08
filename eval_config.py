#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import os
import datetime
from collections import defaultdict
import logging
from multiprocessing import Pool, TimeoutError
import warnings
import argparse
from joblib import parallel_backend, effective_n_jobs, Parallel, delayed

parser = argparse.ArgumentParser(description="Evaluates.")
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
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = args.modelname
PRETRAINING = args.pretraining
MILISI = True

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

metric_ctp = JSD()

if args.cuda is not None:
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    warnings.warn("Using CPU", stacklevel=2)

# %%
def main():
    evaluator = Evaluator()
    evaluator.evaluate_embeddings()
    evaluator.eval_spots()
    evaluator.eval_sc(metric_ctp)

    evaluator.produce_results()


class Evaluator:
    def __init__(self):

        print(f"Evaluating {MODEL_NAME} on {device} with {args.njobs} jobs")
        print(f"Loading config {args.config_fname} ... ")

        with open(os.path.join("configs", MODEL_NAME, args.config_fname), "r") as f:
            config = yaml.safe_load(f)

        print(yaml.dump(config))

        self.torch_params = config["torch_params"]
        self.data_params = config["data_params"]
        self.model_params = config["model_params"]
        self.train_params = config["train_params"]

        if "manual_seed" in self.torch_params:
            torch_seed_path = str(self.torch_params["manual_seed"])
        else:
            torch_seed_path = "random"

        # %%
        model_rel_path = get_model_rel_path(
            MODEL_NAME,
            self.model_params["model_version"],
            scaler_name=self.data_params["scaler_name"],
            n_markers=self.data_params["n_markers"],
            all_genes=self.data_params["all_genes"],
            n_mix=self.data_params["n_mix"],
            n_spots=self.data_params["n_spots"],
            st_split=self.data_params["st_split"],
            torch_seed_path=torch_seed_path,
        )

        model_folder = os.path.join("model", model_rel_path)

        # Check to make sure config file matches config file in model folder
        with open(os.path.join(model_folder, "config.yml"), "r") as f:
            config_model_folder = yaml.safe_load(f)
        if config_model_folder != config:
            raise ValueError("Config file does not match config file in model folder")

        self.results_folder = os.path.join("results", model_rel_path)

        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        # %%
        # sc.logging.print_versions()
        sc.set_figure_params(facecolor="white", figsize=(8, 8))
        sc.settings.verbosity = 3

        # %% [markdown]
        #   # Data load

        self.selected_dir = get_selected_dir(
            self.data_params["data_dir"],
            self.data_params["n_markers"],
            self.data_params["all_genes"],
        )

        # %%
        print("Loading Data")
        # Load spatial data
        self.mat_sp_d, self.mat_sp_train, self.st_sample_id_l = load_spatial(
            self.selected_dir,
            self.data_params["scaler_name"],
            train_using_all_st_samples=self.data_params["train_using_all_st_samples"],
            st_split=self.data_params["st_split"],
        )

        # Load sc data
        self.sc_mix_d, self.lab_mix_d, self.sc_sub_dict, self.sc_sub_dict2 = load_sc(
            self.selected_dir,
            self.data_params["scaler_name"],
            n_mix=self.data_params["n_mix"],
            n_spots=self.data_params["n_spots"],
        )

        # %%
        pretrain_folder = os.path.join(model_folder, "pretrain")
        self.advtrain_folder = os.path.join(model_folder, "advtrain")
        self.pretrain_model_path = os.path.join(pretrain_folder, f"final_model.pth")

        # %%
        # st_sample_id_l = [SAMPLE_ID_N]

        # %% [markdown]
        #  ## Load Models
        
        self.splits = ("train", "val", "test")

    def gen_pca(self, sample_id, split, y_dis, emb, emb_noda=None):
        n_cols = 2 if emb_noda is not None else 1
        logger.debug("Generating PCA plots")
        fig, axs = plt.subplots(1, n_cols, figsize=(n_cols * 10, 10), squeeze=False)
        logger.debug("Calculating DA PCA")
        pca_da_df = pd.DataFrame(
            fit_pca(emb, n_components=2).transform(emb), columns=["PC1", "PC2"]
        )

        pca_da_df["domain"] = ["source" if x == 0 else "target" for x in y_dis]
        sns.scatterplot(
            data=pca_da_df, x="PC1", y="PC2", hue="domain", ax=axs[0][0], marker="."
        )

        if emb_noda is not None:
            logger.debug("Calculating no DA PCA")
            pca_noda_df = pd.DataFrame(
                fit_pca(emb_noda, n_components=2).transform(emb_noda),
                columns=["PC1", "PC2"],
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
            os.path.join(self.results_folder, f"PCA_{sample_id}_{split}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def rf50_score(self, emb_train, emb_test, y_dis_train, y_dis_test):
        logger.debug(f"emb_train dtype: {emb_train.dtype}")
        # pca = PCA(n_components=min(50, emb_train.shape[1]), svd_solver="full")
        logger.debug("fitting pca 50")
        pca = fit_pca(emb_train, n_components=min(50, emb_train.shape[1]))

        emb_train_50 = pca.transform(emb_train)
        # emb_train_50 = pca.fit_transform(emb_train)
        logger.debug("transforming pca 50 test")
        emb_test_50 = pca.transform(emb_test)

        logger.debug("initializie brfc")
        n_jobs = effective_n_jobs(args.njobs)
        clf = BalancedRandomForestClassifier(random_state=145, n_jobs=n_jobs)
        logger.debug("fit brfc")
        clf.fit(emb_train_50, y_dis_train)
        logger.debug("predict brfc")
        y_pred_test = clf.predict(emb_test_50)

        logger.debug("eval brfc")
        return metrics.balanced_accuracy_score(y_dis_test, y_pred_test)

    def evaluate_embeddings(self):
        random_states = np.asarray([225, 53, 92])

        self.rf50_d = {"da": {}, "noda": {}}

        for split in self.splits:
            for k in self.rf50_d:
                self.rf50_d[k][split] = {}

        if MILISI:
            self.miLISI_d = {"da": {}}
            if PRETRAINING:
                self.miLISI_d["noda"] = {}
            for split in self.splits:
                for k in self.miLISI_d:
                    self.miLISI_d[k][split] = {}

        if PRETRAINING:
            model_noda = get_model(self.pretrain_model_path)
        else:
            model_noda = None

        for sample_id in self.st_sample_id_l:
            print(f"Calculating domain shift for {sample_id}:", end=" ")
            random_states = random_states + 1

            self._eval_embeddings_1samp(sample_id, random_states, model_noda=model_noda)

    def _eval_embeddings_1samp(self, sample_id, random_states, model_noda=None):
        model = get_model(
            os.path.join(self.advtrain_folder, sample_id, f"final_model.pth")
        )
        n_jobs = effective_n_jobs(args.njobs)

        for split, rs in zip(self.splits, random_states):
            print(split.upper(), end=" |")
            Xs, Xt = (self.sc_mix_d[split], self.mat_sp_d[sample_id]["test"])
            logger.debug("Getting embeddings")
            source_emb = next(get_embeddings(model, Xs, source_encoder=True))
            target_emb = next(get_embeddings(model, Xt, source_encoder=False))
            emb = np.concatenate([source_emb, target_emb])
            if PRETRAINING:

                emb_gen = get_embeddings(model_noda, (Xs, Xt), source_encoder=True)
                source_emb_noda = next(emb_gen)
                target_emb_noda = next(emb_gen)

                emb_noda = np.concatenate([source_emb_noda, target_emb_noda])
            else:
                emb_noda = None

            y_dis = np.concatenate(
                [
                    np.zeros((source_emb.shape[0],), dtype=np.int_),
                    np.ones((target_emb.shape[0],), dtype=np.int_),
                ]
            )

            self.gen_pca(sample_id, split, y_dis, emb, emb_noda=emb_noda)

            if MILISI:
                logger.debug(
                    f"Using {n_jobs} jobs with parallel backend \"{'threading'}\""
                )
                with parallel_backend("threading", n_jobs=n_jobs):
                    print(" milisi", end=" ")
                    meta_df = pd.DataFrame(y_dis, columns=["Domain"])
                    self.miLISI_d["da"][split][sample_id] = np.median(
                        hm.compute_lisi(emb, meta_df, ["Domain"])
                    )
                    logger.debug(f"miLISI da: {self.miLISI_d['da'][split][sample_id]}")

                    if PRETRAINING:
                        self.miLISI_d["noda"][split][sample_id] = np.median(
                            hm.compute_lisi(emb_noda, meta_df, ["Domain"])
                        )
                        logger.debug(
                            f"miLISI noda: {self.miLISI_d['da'][split][sample_id]}"
                        )

            print("rf50", end=" ")
            if PRETRAINING:
                embs = (emb, emb_noda)
            else:
                embs = (emb,)
            split_data = model_selection.train_test_split(
                y_dis,
                *embs,
                test_size=0.2,
                random_state=rs,
                stratify=y_dis,
            )
            logger.debug(f"split shapes: {[x.shape for x in split_data]}")

            logger.debug("rf50 da")
            y_dis_train, y_dis_test = split_data[:2]
            emb_train, emb_test = split_data[2:4]
            self.rf50_d["da"][split][sample_id] = self.rf50_score(
                emb_train, emb_test, y_dis_train, y_dis_test
            )
            if PRETRAINING:
                logger.debug("rf50 noda")
                emb_noda_train, emb_noda_test = split_data[4:]
                self.rf50_d["noda"][split][sample_id] = self.rf50_score(
                    emb_noda_train, emb_noda_test, y_dis_train, y_dis_test
                )

            print("|", end=" ")
            # newline at end of split

        print("")

    def _plot_cellfraction(self, visnum, adata, pred_sp, ax=None):
        """Plot predicted cell fraction for a given visnum"""
        logging.debug(f"plotting cell fraction for {self.sc_sub_dict[visnum]}")
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
            title=f"{self.sc_sub_dict[visnum]}",
            spot_size=100,
            show=False,
            # vmin=vmin,
            # vmax=vmax,
            ax=ax,
        )

    def _plot_roc(self, visnum, adata, pred_sp, name, num_name_exN_l, numlist, ax=None):
        """Plot ROC for a given visnum"""
        logging.debug(f"plotting ROC for {self.sc_sub_dict[visnum]} and {name}")
        Ex_l = [t[2] for t in num_name_exN_l]
        num_to_ex_d = dict(zip(numlist, Ex_l))

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

    def _plot_layers(self, adata_spatialLIBD, adata_spatialLIBD_d):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(self.st_sample_id_l),
            figsize=(3 * len(self.st_sample_id_l), 3),
            squeeze=False,
            constrained_layout=True,
            dpi=50,
        )

        cmap = mpl.cm.get_cmap("Accent_r")

        color_range = list(
            np.linspace(
                0.125,
                1,
                len(adata_spatialLIBD.obs.spatialLIBD.cat.categories),
                endpoint=True,
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

        for i, sample_id in enumerate(self.st_sample_id_l):
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
        fig.savefig(
            os.path.join(self.results_folder, "layers.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def _plot_samples(
        self, sample_id, adata_spatialLIBD_d, pred_sp_d, pred_sp_noda_d=None
    ):
        logging.debug(f"Plotting {sample_id}")
        fig, ax = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, dpi=10)
        num_name_exN_l = []
        for k, v in self.sc_sub_dict.items():
            if "Ex" in v:
                num_name_exN_l.append((k, v, int(v.split("_")[1])))
        num_name_exN_l.sort(key=lambda a: a[2])
        num_name_exN_l

        # %%

        # %%
        numlist = [t[0] for t in num_name_exN_l]

        logging.debug(f"Plotting Cell Fractions")
        for i, num in enumerate(numlist):
            self._plot_cellfraction(
                num, adata_spatialLIBD_d[sample_id], pred_sp_d[sample_id], ax.flat[i]
            )
            ax.flat[i].axis("equal")
            ax.flat[i].set_xlabel("")
            ax.flat[i].set_ylabel("")
        fig.suptitle(sample_id)

        logging.debug(f"Saving Cell Fractions Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_cellfraction.png"),
            bbox_inches="tight",
            dpi=300,
        )
        # fig.show()
        plt.close()

        logging.debug(f"Plotting ROC")
        fig, ax = plt.subplots(
            2,
            5,
            figsize=(20, 8),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            dpi=10,
        )

        da_aucs = []
        if PRETRAINING:
            noda_aucs = []
        for i, num in enumerate(numlist):
            da_aucs.append(
                self._plot_roc(
                    num,
                    adata_spatialLIBD_d[sample_id],
                    pred_sp_d[sample_id],
                    MODEL_NAME,
                    num_name_exN_l,
                    numlist,
                    ax.flat[i],
                )
            )
            if PRETRAINING:
                noda_aucs.append(
                    self._plot_roc(
                        num,
                        adata_spatialLIBD_d[sample_id],
                        pred_sp_noda_d[sample_id],
                        f"{MODEL_NAME}_wo_da",
                        num_name_exN_l,
                        numlist,
                        ax.flat[i],
                    )
                )

            ax.flat[i].plot(
                [0, 1], [0, 1], transform=ax.flat[i].transAxes, ls="--", color="k"
            )
            ax.flat[i].set_aspect("equal")
            ax.flat[i].set_xlim([0, 1])
            ax.flat[i].set_ylim([0, 1])

            ax.flat[i].set_title(f"{self.sc_sub_dict[num]}")

            if i >= len(numlist) - 5:
                ax.flat[i].set_xlabel("FPR")
            else:
                ax.flat[i].set_xlabel("")
            if i % 5 == 0:
                ax.flat[i].set_ylabel("TPR")
            else:
                ax.flat[i].set_ylabel("")

        fig.suptitle(sample_id)
        logging.debug(f"Saving ROC Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_roc.png"),
            bbox_inches="tight",
            dpi=300,
        )
        # fig.show()
        plt.close()

        # realspots_d["da"][sample_id] = np.mean(da_aucs)
        # if PRETRAINING:
        #     realspots_d["noda"][sample_id] = np.mean(noda_aucs)
        return np.mean(da_aucs), np.mean(noda_aucs) if PRETRAINING else None

    # %%
    def eval_spots(self):
        print("Getting predictions: ")
        if self.data_params["train_using_all_st_samples"]:
            inputs = [self.mat_sp_d[sid]["test"] for sid in self.st_sample_id_l]
            path = os.path.join(self.advtrain_folder, f"final_model.pth")
            outputs = get_predictions(get_model(path), inputs)
            pred_sp_d = dict(zip(self.st_sample_id_l, outputs))

        else:
            pred_sp_d = {}
            for sample_id in self.st_sample_id_l:
                path = os.path.join(self.advtrain_folder, sample_id, f"final_model.pth")
                input = self.mat_sp_d[sample_id]["test"]
                pred_sp_d[sample_id] = next(get_predictions(get_model(path), input))

        if PRETRAINING:
            inputs = [self.mat_sp_d[sid]["test"] for sid in self.st_sample_id_l]
            outputs = get_predictions(
                get_model(self.pretrain_model_path), inputs, source_encoder=True
            )
            pred_sp_noda_d = dict(zip(self.st_sample_id_l, outputs))
        else:
            pred_sp_noda_d = None
        # %%
        adata_spatialLIBD = sc.read_h5ad(
            os.path.join(self.selected_dir, "adata_spatialLIBD.h5ad")
        )

        adata_spatialLIBD_d = {}
        print("Loading ST adata: ")
        for sample_id in self.st_sample_id_l:
            adata_spatialLIBD_d[sample_id] = adata_spatialLIBD[
                adata_spatialLIBD.obs.sample_id == sample_id
            ]
            adata_spatialLIBD_d[sample_id].obsm["spatial"] = (
                adata_spatialLIBD_d[sample_id].obs[["X", "Y"]].values
            )

        # %%

        # %%
        self._plot_layers(adata_spatialLIBD, adata_spatialLIBD_d)

        # %%
        realspots_d = {"da": {}}
        if PRETRAINING:
            realspots_d["noda"] = {}

        # for sample_id in st_sample_id_l:
        #     plot_samples(sample_id)
        print("Plotting Samples")
        n_jobs_samples = min(effective_n_jobs(args.njobs), len(self.st_sample_id_l))
        logging.debug(f"n_jobs_samples: {n_jobs_samples}")
        aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
            delayed(self._plot_samples)(
                sid, adata_spatialLIBD_d, pred_sp_d, pred_sp_noda_d
            )
            for sid in self.st_sample_id_l
        )
        for sample_id, auc in zip(self.st_sample_id_l, aucs):
            realspots_d["da"][sample_id] = auc[0]
            if PRETRAINING:
                realspots_d["noda"][sample_id] = auc[1]
        self.realspots_d = realspots_d

    def eval_sc(self, metric_ctp):
        self.jsd_d = {"da": {}}
        if PRETRAINING:
            self.jsd_d["noda"] = {}

        for k in self.jsd_d:
            self.jsd_d[k] = {split: {} for split in self.splits}

        if PRETRAINING:
            self._calc_jsd(
                metric_ctp,
                self.st_sample_id_l[0],
                model_path=self.pretrain_model_path,
                da="noda",
            )
            for split in self.splits:
                score = self.jsd_d["noda"][split][self.st_sample_id_l[0]]
                for sample_id in self.st_sample_id_l[1:]:
                    self.jsd_d["noda"][split][sample_id] = score

        for sample_id in self.st_sample_id_l:
            if self.data_params["train_using_all_st_samples"]:
                model_path = os.path.join(self.advtrain_folder, f"final_model.pth")
            else:
                model_path = os.path.join(
                    self.advtrain_folder, sample_id, f"final_model.pth"
                )

            self._calc_jsd(metric_ctp, sample_id, model_path=model_path, da="da")

    def _calc_jsd(self, metric_ctp, sample_id, model=None, model_path=None, da="da"):
        with torch.no_grad():
            inputs = (self.sc_mix_d[split] for split in self.splits)
            if model is None:
                model = get_model(model_path)
            pred_mix = get_predictions(model, inputs, source_encoder=True)
            for split, pred in zip(self.split, pred_mix):
                score = metric_ctp(
                    torch.exp(torch.as_tensor(pred).float()),
                    torch.as_tensor(self.lab_mix_d[split]).float(),
                )
                self.jsd_d[da][split][sample_id] = score.detach().cpu().numpy()

    def gen_l_dfs(self, da):
        """Generate a list of series for a given da"""
        df = pd.DataFrame.from_dict(self.jsd_d[da], orient="columns")
        df.columns.name = "SC Split"
        yield df
        df = pd.DataFrame.from_dict(self.rf50_d[da], orient="columns")
        df.columns.name = "SC Split"
        yield df
        if MILISI:
            df = pd.DataFrame.from_dict(self.miLISI_d[da], orient="columns")
            df.columns.name = "SC Split"
            yield df
        yield pd.Series(self.realspots_d[da])
        return

    def produce_results(self):
        df_keys = [
            "Pseudospots (JS Divergence)",
            "RF50",
            "Real Spots (Mean AUC Ex1-10)",
        ]

        if MILISI:
            df_keys.insert(2, "miLISI")

        da_dict_keys = ["da"]
        da_df_keys = ["After DA"]
        if PRETRAINING:
            da_dict_keys.insert(0, "noda")
            da_df_keys.insert(0, "Before DA")

        results_df = [
            pd.concat(list(self.gen_l_dfs(da)), axis=1, keys=df_keys)
            for da in da_dict_keys
        ]
        results_df = pd.concat(results_df, axis=0, keys=da_df_keys)

        results_df.to_csv(os.path.join(self.results_folder, "results.csv"))
        print(results_df)

        # %%
        with open(os.path.join(self.results_folder, "config.yml"), "w") as f:
            yaml.dump(self.config, f)


# %%
# checkpoints_da_d = {}
def get_model(model_path):
    check_point_da = torch.load(model_path, map_location=device)
    model = check_point_da["model"]
    model.to(device)
    model.eval()
    return model


def get_predictions(model, inputs, source_encoder=False):
    if source_encoder:
        model.set_encoder("source")
    else:
        model.target_inference()

    def out_func(x):
        out = model(torch.as_tensor(x).float().to(device))
        if isinstance(out, tuple):
            out = out[0]
        return torch.exp(out)

    try:
        inputs.shape
    except AttributeError:
        pass
    else:
        inputs = [inputs]

    with torch.no_grad():
        for input in inputs:
            yield out_func(input).detach().cpu().numpy()


def get_embeddings(model, inputs, source_encoder=False):
    if source_encoder:
        model.set_encoder("source")
    else:
        model.target_inference()

    try:
        encoder = model.encoder
    except AttributeError:
        if source_encoder:
            encoder = model.source_encoder
        else:
            encoder = model.target_encoder

    out_func = lambda x: encoder(torch.as_tensor(x).float().to(device))
    try:
        inputs.shape
    except AttributeError:
        inputs_iter = inputs
    else:
        logger.debug(f"Embeddings input is single array with shape {inputs.shape}")
        inputs_iter = [inputs]
    logger.debug(f"Embeddings input length: {len(inputs_iter)}")
    with torch.no_grad():
        for input in inputs_iter:
            yield out_func(input).detach().cpu().numpy()


# %% [markdown]
#  ## Evaluation of latent space
# %%
def _fit_pca(X, *args, **kwargs):
    return PCA(*args, **kwargs).fit(X)


def fit_pca(X, *args, **kwargs):

    # # try:
    # #     pca = PCA(*args, **kwargs)
    # #     pool = Pool(processes=1)
    # #     res = pool.apply_async(pca.fit, (X,))
    # #     pca = res.get(timeout=360)
    # # except TimeoutError:
    # #     logging.warning('PCA Timed out, retrying')
    # #     pool.terminate()
    # #     pca = fit_pca(X, *args, **kwargs)
    # # finally:
    # #     try:
    # #         pool.terminate()
    # #     except NameError:
    # #         pass
    # with Pool(processes=1) as pool:
    #     res = pool.apply_async(_fit_pca, (X, *args), kwargs)
    #     try:
    #         pca = res.get(timeout=360)
    #     except TimeoutError:
    #         logging.warning('PCA Timed out, retrying')
    #         pca = None
    # if pca is None:
    #     fit_pca(X, *args, **kwargs)

    # return pca
    return _fit_pca(X, *args, **kwargs)


# %%
if __name__ == "__main__":
    main()

print(
    "Script run time:", datetime.datetime.now(datetime.timezone.utc) - script_start_time
)
