#!/usr/bin/env python3
"""Runs evaluation on models."""
# %%
import argparse
import datetime
import logging
import math
import os
import pickle
import shutil
from collections import defaultdict

import harmonypy as hm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import yaml
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from sklearn import metrics, model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay

from src.da_models.utils import get_torch_device
from src.utils import data_loading

# from src.da_models.datasets import SpotDataset
from src.utils.evaluation import JSD
from src.utils.output_utils import TempFolderHolder

parser = argparse.ArgumentParser(description="Evaluates.")
parser.add_argument("--pretraining", "-p", action="store_true", help="force pretraining")
parser.add_argument("--modelname", "-n", type=str, default="ADDA", help="model name")
parser.add_argument("--milisi", "-m", action="store_false", help="no milisi")
parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
parser.add_argument(
    "--njobs", type=int, default=1, help="Number of jobs to use for parallel processing."
)
parser.add_argument("--cuda", "-c", default=None, help="GPU index to use")
parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary results directory")
parser.add_argument("--test", "-t", action="store_true", help="test mode")
args = parser.parse_args()


script_start_time = datetime.datetime.now(datetime.timezone.utc)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = args.modelname
MILISI = args.milisi

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

device = get_torch_device(args.cuda)


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
            self.config = yaml.safe_load(f)

        print(yaml.dump(self.config))

        self.torch_params = self.config["torch_params"]
        self.data_params = self.config["data_params"]
        self.model_params = self.config["model_params"]
        self.train_params = self.config["train_params"]

        torch_seed_path = str(self.torch_params.get("manual_seed", "random"))

        model_rel_path = data_loading.get_model_rel_path(
            MODEL_NAME,
            self.model_params["model_version"],
            torch_seed_path=torch_seed_path,
            **self.data_params,
        )
        model_folder = os.path.join("model", model_rel_path)

        if args.tmpdir:
            real_model_folder = model_folder
            model_folder = os.path.join(args.tmpdir, "model")

            shutil.copytree(real_model_folder, model_folder, dirs_exist_ok=True)

        # Check to make sure config file matches config file in model folder
        with open(os.path.join(model_folder, "config.yml"), "r") as f:
            config_model_folder = yaml.safe_load(f)
        if config_model_folder != self.config:
            raise ValueError("Config file does not match config file in model folder")

        self.pretraining = args.pretraining or self.train_params.get("pretraining", False)
        results_folder = os.path.join("results", model_rel_path)

        self.temp_folder_holder = TempFolderHolder()
        temp_results_folder = os.path.join(args.tmpdir, "results") if args.tmpdir else None
        self.results_folder = self.temp_folder_holder.set_output_folder(
            temp_results_folder, results_folder
        )

        sc.set_figure_params(facecolor="white", figsize=(8, 8))
        sc.settings.verbosity = 3

        self.selected_dir = data_loading.get_selected_dir(
            data_loading.get_dset_dir(
                self.data_params["data_dir"],
                dset=self.data_params.get("dset", "dlpfc"),
            ),
            **self.data_params,
        )

        print("Loading Data")
        # Load spatial data
        mat_sp_d, self.mat_sp_train, st_sample_id_l = data_loading.load_spatial(
            self.selected_dir,
            **self.data_params,
        )

        if args.test:
            self.splits = ("train", "val", "test")
        else:
            self.splits = ("train", "val")

        if self.data_params.get("samp_split", False):
            self.st_sample_id_d = {}
            for split in self.splits:
                self.st_sample_id_d[split] = [
                    sid for sid in st_sample_id_l if sid in mat_sp_d[split].keys()
                ]

            self.mat_sp_d = {}
            for split in self.splits:
                for sid in self.st_sample_id_d[split]:
                    self.mat_sp_d[sid] = mat_sp_d[split][sid]

        else:
            self.st_sample_id_d = {"": st_sample_id_l}
            self.mat_sp_d = {k: v["test"] for k, v in mat_sp_d.items()}

        # Load sc data
        (
            self.sc_mix_d,
            self.lab_mix_d,
            self.sc_sub_dict,
            self.sc_sub_dict2,
        ) = data_loading.load_sc(self.selected_dir, **self.data_params)

        pretrain_folder = os.path.join(model_folder, "pretrain")
        self.advtrain_folder = os.path.join(model_folder, "advtrain")
        self.pretrain_model_path = os.path.join(pretrain_folder, f"final_model.pth")

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
            data=pca_da_df,
            x="PC1",
            y="PC2",
            hue="domain",
            ax=axs[0][0],
            marker=".",
        )
        axs.flat[0].set_title("DA")

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
            axs.flat[1].set_title("No DA")

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
        logger.info("Running RF50")
        logger.debug(f"emb_train dtype: {emb_train.dtype}")
        logger.debug("fitting pca 50")
        pca = fit_pca(emb_train, n_components=min(50, emb_train.shape[1]))

        emb_train_50 = pca.transform(emb_train)
        logger.debug("transforming pca 50 test")
        emb_test_50 = pca.transform(emb_test)

        logger.debug("initialize brfc")
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
            if self.pretraining:
                self.miLISI_d["noda"] = {}
            for split in self.splits:
                for k in self.miLISI_d:
                    self.miLISI_d[k][split] = {}

        if self.pretraining:
            model_noda = get_model(self.pretrain_model_path)
        else:
            model_noda = None

        if "" in self.st_sample_id_d:
            splits = [""]
        else:
            splits = self.splits

        if self.data_params.get("samp_split", False):
            model = get_model(os.path.join(self.advtrain_folder, "samp_split/final_model.pth"))
        elif self.data_params.get("train_using_all_st_samples", False):
            model = get_model(os.path.join(self.advtrain_folder, "final_model.pth"))
        else:
            model = None

        for split in splits:
            for sample_id in self.st_sample_id_d[split]:
                print(f"Calculating domain shift for {sample_id}:", end=" ")
                random_states = random_states + 1

                self._eval_embeddings_1samp(
                    sample_id, random_states, model=model, model_noda=model_noda
                )

    def _eval_embeddings_1samp(self, sample_id, random_states, model=None, model_noda=None):
        if model is None:
            model = get_model(os.path.join(self.advtrain_folder, sample_id, f"final_model.pth"))

        n_jobs = effective_n_jobs(args.njobs)

        for split, rs in zip(self.splits, random_states):
            print(split.upper(), end=" |")
            Xs, Xt = (self.sc_mix_d[split], self.mat_sp_d[sample_id])
            logger.debug("Getting embeddings")
            source_emb = next(get_embeddings(model, Xs, source_encoder=True))
            target_emb = next(get_embeddings(model, Xt, source_encoder=False))
            emb = np.concatenate([source_emb, target_emb])
            if self.pretraining:

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
                self._run_milisi(sample_id, n_jobs, split, emb, emb_noda, y_dis)

            print("rf50", end=" ")
            if self.pretraining:
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
            if self.pretraining:
                logger.debug("rf50 noda")
                emb_noda_train, emb_noda_test = split_data[4:]
                self.rf50_d["noda"][split][sample_id] = self.rf50_score(
                    emb_noda_train, emb_noda_test, y_dis_train, y_dis_test
                )

            print("|", end=" ")
            # newline at end of split

        print("")

    def _run_milisi(self, sample_id, n_jobs, split, emb, emb_noda, y_dis):
        logger.debug(f"Using {n_jobs} jobs with parallel backend \"{'threading'}\"")

        all_embs = np.concatenate([emb, emb_noda], axis=1)
        all_embs_bal, y_dis_bal = RandomUnderSampler(random_state=int(sample_id)).fit_resample(
            all_embs, y_dis
        )
        emb_bal = all_embs_bal[:, : all_embs_bal.shape[1] // 2]
        emb_noda_bal = all_embs_bal[:, all_embs_bal.shape[1] // 2 :]

        print(" milisi", end=" ")
        meta_df = pd.DataFrame(y_dis_bal, columns=["Domain"])
        score = self._milisi_parallel(n_jobs, emb_bal, meta_df)

        self.miLISI_d["da"][split][sample_id] = np.median(score)
        logger.debug(f"miLISI da: {self.miLISI_d['da'][split][sample_id]}")

        if self.pretraining:
            score = self._milisi_parallel(n_jobs, emb_noda_bal, meta_df)
            self.miLISI_d["noda"][split][sample_id] = np.median(score)
            logger.debug(f"miLISI noda: {self.miLISI_d['da'][split][sample_id]}")

    def _milisi_parallel(self, n_jobs, emb, meta_df):
        if n_jobs > 1:
            with parallel_backend("threading", n_jobs=n_jobs):
                return hm.compute_lisi(emb, meta_df, ["Domain"])

        return hm.compute_lisi(emb, meta_df, ["Domain"])

    def _plot_cellfraction(self, visnum, adata, pred_sp, ax=None):
        """Plot predicted cell fraction for a given visnum"""
        try:
            cell_name = self.sc_sub_dict[visnum]
        except TypeError:
            cell_name = "Other"
        logging.debug(f"plotting cell fraction for {cell_name}")
        y_pred = pred_sp[:, visnum].squeeze()
        if y_pred.ndim > 1:
            y_pred = y_pred.sum(axis=1)
        adata.obs["Pred_label"] = y_pred

        sc.pl.spatial(
            adata,
            img_key="hires",
            color="Pred_label",
            palette="Set1",
            # size=1.5,
            legend_loc=None,
            title=cell_name,
            spot_size=1 if self.data_params.get("dset") == "pdac" else 150,
            show=False,
            ax=ax,
        )

    def _plot_roc(
        self,
        visnum,
        adata,
        pred_sp,
        name,
        num_name_exN_l,
        numlist,
        ax=None,
    ):
        """Plot ROC for a given visnum"""

        logging.debug(f"plotting ROC for {self.sc_sub_dict[visnum]} and {name}")
        Ex_l = [t[2] for t in num_name_exN_l]
        num_to_ex_d = dict(zip(numlist, Ex_l))

        def layer_to_layer_number(x):
            """Converts a string of layers to a list of layer numbers"""
            for char in x:
                if char.isdigit():
                    # if in (ordinal -> ex number -> layers)
                    if int(char) in Ex_to_L_d[num_to_ex_d[visnum]]:
                        return 1
            return 0

        y_pred = pred_sp[:, visnum]
        y_true = adata.obs["spatialLIBD"].map(layer_to_layer_number).fillna(0)
        # print(y_true)
        # print(y_true.isna().sum())
        RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)

        return metrics.roc_auc_score(y_true, y_pred)

    def _plot_roc_pdac(self, visnum, adata, pred_sp, name, sc_to_st_celltype, ax=None):
        """Plot ROC for a given visnum (PDAC)"""
        try:
            cell_name = self.sc_sub_dict[visnum]
        except TypeError:
            cell_name = "Other"
        logging.debug(f"plotting ROC for {cell_name} and {name}")

        def st_sc_bin(cell_subclass):
            return int(cell_subclass in sc_to_st_celltype.get(cell_name, set()))

        y_pred = pred_sp[:, visnum].squeeze()
        if y_pred.ndim > 1:
            y_pred = y_pred.sum(axis=1)
        y_true = adata.obs["cell_subclass"].map(st_sc_bin).fillna(0)

        if y_true.sum() > 0:
            RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_pred, name=name, ax=ax)
            return metrics.roc_auc_score(y_true, y_pred)

        return np.nan

    def _plot_layers(self, adata_st, adata_st_d):

        cmap = mpl.cm.get_cmap("Accent_r")

        color_range = list(
            np.linspace(
                0.125,
                1,
                len(adata_st.obs.spatialLIBD.cat.categories),
                endpoint=True,
            )
        )
        colors = [cmap(x) for x in color_range]

        color_dict = defaultdict(lambda: "lightgrey")
        for cat, color in zip(adata_st.obs.spatialLIBD.cat.categories, colors):
            color_dict[cat] = color

        color_dict["NA"] = "lightgrey"

        self._plot_spatial(adata_st_d, color_dict, color="spatialLIBD", fname="layers.png")

    def _plot_spatial(self, adata_st_d, color_dict, color="spatialLIBD", fname="layers.png"):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(self.mat_sp_d),
            figsize=(3 * len(self.mat_sp_d), 3),
            squeeze=False,
            constrained_layout=True,
            dpi=50,
        )
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
        fig.legend(
            bbox_to_anchor=(0, 0.5),
            handles=legend_elements,
            loc="center right",
        )

        if "" in self.st_sample_id_d:
            splits = [""]
        else:
            splits = self.splits
        for split in splits:
            for i, sample_id in enumerate(self.st_sample_id_d[split]):
                sc.pl.spatial(
                    adata_st_d[sample_id],
                    img_key=None,
                    color=color,
                    palette=color_dict,
                    size=1,
                    title=sample_id,
                    legend_loc=4,
                    na_color="lightgrey",
                    spot_size=1 if self.data_params.get("dset") == "pdac" else 100,
                    show=False,
                    ax=ax[0][i],
                )

            ax[0][i].axis("equal")
            ax[0][i].set_xlabel("")
            ax[0][i].set_ylabel("")

        fig.savefig(
            os.path.join(self.results_folder, fname),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    """
    def _plot_samples_pdac(self, sample_id, adata_st, pred_sp, pred_sp_noda=None):
        logging.debug(f"Plotting {sample_id}")

        if self.data_params.get("sc_id") == "CA001063":
            sc_to_st_celltype = {
                # Peng et al., 2019: Taken together, these results show that
                # type 2 ductal cells are the major source of malignant cells in
                # PDACs.
                "Ductal cell type 2": "Cancer region",
                "T cell": "Stroma",
                "Macrophage cell": "Stroma",
                "Fibroblast cell": "Cancer region",
                "B cell": "Stroma",
                "Ductal cell type 1": "Duct epithelium",
                "Endothelial cell": "Interstitium",
                "Stellate cell": "Stroma",
                "Acinar cell": "Pancreatic tissue",
                "Endocrine cell": "Pancreatic tissue",
            }
        else:
            sc_to_st_celltype = {
                # As expected, we found that all ductal subpopulations in PDAC-A
                # were enriched in the duct region of the tissue. In contrast,
                # only the hypoxic and terminal ductal cell populations were
                # significantly enriched in the cancer region
                "Ductal - MHC Class II": "Duct epithelium",
                "Ductal - CRISP3 high/centroacinar like": "Duct epithelium",
                "Ductal - terminal ductal like": "Cancer region",
                "Ductal - APOL1 high/hypoxic": "Cancer region",
                "Cancer clone": "Cancer region",
                "mDCs": "Stroma",
                "Macrophages": "Stroma",
                "T cells & NK cells": "Stroma",
                "Tuft cells": "Duct epithelium",
                "Monocytes": "Stroma",
                #'RBCs': 15,
                "Mast cells": "Stroma",
                "Acinar cells": "Pancreatic tissue",
                "Endocrine cells": "Pancreatic tissue",
                "pDCs": "Stroma",
                "Endothelial cells": "Interstitium",
            }

        celltypes = list(sc_to_st_celltype.keys()) + ["Other"]
        n_celltypes = len(celltypes)
        n_rows = int(math.ceil(n_celltypes / 5))

        numlist = [self.sc_sub_dict2.get(t) for t in celltypes[:-1]]
        numlist.append([v for k, v in self.sc_sub_dict2.items() if k not in celltypes[:-1]])
        # cluster_assignments = [
        #     "Cancer region",
        #     "Pancreatic tissue",
        #     "Interstitium",
        #     "Duct epithelium",
        #     "Stroma",
        # ]
        logging.debug(f"Plotting Cell Fractions")
        fig, ax = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows), constrained_layout=True, dpi=10)
        for i, num in enumerate(numlist):
            self._plot_cellfraction(num, adata_st, pred_sp, ax.flat[i])
            ax.flat[i].axis("equal")
            ax.flat[i].set_xlabel("")
            ax.flat[i].set_ylabel("")
        for i in range(n_celltypes, n_rows * 5):
            ax.flat[i].axis("off")
        fig.suptitle(sample_id)

        logging.debug(f"Saving Cell Fractions Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_cellfraction.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        logging.debug(f"Plotting ROC")

        st_to_sc_celltype = {}
        for k, v in sc_to_st_celltype.items():
            if v not in st_to_sc_celltype:
                st_to_sc_celltype[v] = set()
            st_to_sc_celltype[v].add(k)

        n_rows = int(math.ceil(len(sc_to_st_celltype) / 5))
        fig, ax = plt.subplots(
            n_rows,
            5,
            figsize=(20, 4 * n_rows),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            dpi=10,
        )

        da_aucs = []
        if self.pretraining:
            noda_aucs = []
        for i, num in enumerate(numlist[:-1]):
            da_aucs.append(
                self._plot_roc_pdac(
                    num,
                    adata_st,
                    pred_sp,
                    MODEL_NAME,
                    st_to_sc_celltype,
                    ax.flat[i],
                )
            )
            if self.pretraining:
                noda_aucs.append(
                    self._plot_roc_pdac(
                        num,
                        adata_st,
                        pred_sp_noda,
                        f"{MODEL_NAME}_wo_da",
                        st_to_sc_celltype,
                        ax.flat[i],
                    )
                )

            ax.flat[i].plot([0, 1], [0, 1], transform=ax.flat[i].transAxes, ls="--", color="k")
            ax.flat[i].set_aspect("equal")
            ax.flat[i].set_xlim([0, 1])
            ax.flat[i].set_ylim([0, 1])
            try:
                cell_name = self.sc_sub_dict[num]
            except TypeError:
                cell_name = "Other"
            ax.flat[i].set_title(cell_name)

            if i >= len(numlist) - 5:
                ax.flat[i].set_xlabel("FPR")
            else:
                ax.flat[i].set_xlabel("")
            if i % 5 == 0:
                ax.flat[i].set_ylabel("TPR")
            else:
                ax.flat[i].set_ylabel("")
        for i in range(len(numlist[:-1]), n_rows * 5):
            ax.flat[i].axis("off")
        fig.suptitle(sample_id)

        logging.debug(f"Saving ROC Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_roc.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        return np.nanmean(da_aucs), np.nanmean(noda_aucs) if self.pretraining else None
    """

    def _plot_samples_pdac(self, sample_id, adata_st, pred_sp, pred_sp_noda=None):
        logging.debug(f"Plotting {sample_id}")

        if self.data_params.get("sc_id") == "CA001063":
            sc_to_st_celltype = {
                # Peng et al., 2019: Taken together, these results show that
                # type 2 ductal cells are the major source of malignant cells in
                # PDACs.
                "Ductal cell type 2": {"Cancer region"},
                "T cell": {"Cancer region", "Stroma"},
                "Macrophage cell": {"Cancer region", "Stroma"},
                "Fibroblast cell": {"Cancer region", "Stroma"},
                "B cell": {"Cancer region", "Stroma"},
                "Ductal cell type 1": {"Duct epithelium"},
                "Endothelial cell": {"Interstitium"},
                "Stellate cell": {"Stroma", "Pancreatic tissue"},
                "Acinar cell": {"Pancreatic tissue"},
                "Endocrine cell": {"Pancreatic tissue"},
            }
        else:
            sc_to_st_celltype = {
                # As expected, we found that all ductal subpopulations in PDAC-A
                # were enriched in the duct region of the tissue. In contrast,
                # only the hypoxic and terminal ductal cell populations were
                # significantly enriched in the cancer region
                "Ductal - MHC Class II": {"Duct epithelium"},
                "Ductal - CRISP3 high/centroacinar like": {"Duct epithelium"},
                "Ductal - terminal ductal like": {"Duct epithelium", "Cancer region"},
                "Ductal - APOL1 high/hypoxic": {"Duct epithelium", "Cancer region"},
                "Cancer clone": {"Cancer region"},
                "mDCs": {"Cancer region", "Stroma"},
                "Macrophages": {"Cancer region", "Stroma"},
                "T cells & NK cells": {"Cancer region", "Stroma"},
                "Tuft cells": {"Pancreatic tissue"},
                "Monocytes": {"Cancer region", "Stroma"},
                # "RBCs": 15,
                "Mast cells": {"Cancer region", "Stroma"},
                "Acinar cells": {"Pancreatic tissue"},
                "Endocrine cells": {"Pancreatic tissue"},
                "pDCs": {"Cancer region", "Stroma"},
                "Endothelial cells": {"Interstitium"},
            }

        celltypes = list(sc_to_st_celltype.keys()) + ["Other"]
        n_celltypes = len(celltypes)
        n_rows = int(math.ceil(n_celltypes / 5))

        numlist = [self.sc_sub_dict2.get(t) for t in celltypes[:-1]]
        numlist.append([v for k, v in self.sc_sub_dict2.items() if k not in celltypes[:-1]])
        # cluster_assignments = [
        #     "Cancer region",
        #     "Pancreatic tissue",
        #     "Interstitium",
        #     "Duct epithelium",
        #     "Stroma",
        # ]
        logging.debug(f"Plotting Cell Fractions")
        fig, ax = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows), constrained_layout=True, dpi=10)
        for i, num in enumerate(numlist):
            self._plot_cellfraction(num, adata_st, pred_sp, ax.flat[i])
            ax.flat[i].axis("equal")
            ax.flat[i].set_xlabel("")
            ax.flat[i].set_ylabel("")
        for i in range(n_celltypes, n_rows * 5):
            ax.flat[i].axis("off")
        fig.suptitle(sample_id)

        logging.debug(f"Saving Cell Fractions Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_cellfraction.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        logging.debug(f"Plotting ROC")

        # st_to_sc_celltype = {}
        # for k, v in sc_to_st_celltype.items():
        #     if v not in st_to_sc_celltype:
        #         st_to_sc_celltype[v] = set()
        #     st_to_sc_celltype[v].add(k)

        n_rows = int(math.ceil(len(sc_to_st_celltype) / 5))
        fig, ax = plt.subplots(
            n_rows,
            5,
            figsize=(20, 4 * n_rows),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            dpi=10,
        )

        da_aucs = []
        if self.pretraining:
            noda_aucs = []
        for i, num in enumerate(numlist[:-1]):
            da_aucs.append(
                self._plot_roc_pdac(
                    num,
                    adata_st,
                    pred_sp,
                    MODEL_NAME,
                    sc_to_st_celltype,
                    ax.flat[i],
                )
            )
            if self.pretraining:
                noda_aucs.append(
                    self._plot_roc_pdac(
                        num,
                        adata_st,
                        pred_sp_noda,
                        f"{MODEL_NAME}_wo_da",
                        sc_to_st_celltype,
                        ax.flat[i],
                    )
                )

            ax.flat[i].plot([0, 1], [0, 1], transform=ax.flat[i].transAxes, ls="--", color="k")
            ax.flat[i].set_aspect("equal")
            ax.flat[i].set_xlim([0, 1])
            ax.flat[i].set_ylim([0, 1])
            try:
                cell_name = self.sc_sub_dict[num]
            except TypeError:
                cell_name = "Other"
            ax.flat[i].set_title(cell_name)

            if i >= len(numlist) - 5:
                ax.flat[i].set_xlabel("FPR")
            else:
                ax.flat[i].set_xlabel("")
            if i % 5 == 0:
                ax.flat[i].set_ylabel("TPR")
            else:
                ax.flat[i].set_ylabel("")
        for i in range(len(numlist[:-1]), n_rows * 5):
            ax.flat[i].axis("off")
        fig.suptitle(sample_id)

        logging.debug(f"Saving ROC Figure")
        fig.savefig(
            os.path.join(self.results_folder, f"{sample_id}_roc.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        return np.nanmean(da_aucs), np.nanmean(noda_aucs) if self.pretraining else None

    def _plot_samples(self, sample_id, adata_st_d, pred_sp_d, pred_sp_noda_d=None):
        logging.debug(f"Plotting {sample_id}")
        fig, ax = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, dpi=10)
        num_name_exN_l = []
        for k, v in self.sc_sub_dict.items():
            if "Ex" in v:
                # (clust_ordinal, clust_name, Ex_clust_num)
                num_name_exN_l.append((k, v, int(v.split("_")[1])))

        num_name_exN_l.sort(key=lambda a: a[2])

        numlist = [t[0] for t in num_name_exN_l]  # clust ordinals

        logging.debug(f"Plotting Cell Fractions")
        for i, num in enumerate(numlist):
            self._plot_cellfraction(num, adata_st_d[sample_id], pred_sp_d[sample_id], ax.flat[i])
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
        if self.pretraining:
            noda_aucs = []
        for i, num in enumerate(numlist):
            da_aucs.append(
                self._plot_roc(
                    num,
                    adata_st_d[sample_id],
                    pred_sp_d[sample_id],
                    MODEL_NAME,
                    num_name_exN_l,
                    numlist,
                    ax.flat[i],
                )
            )
            if self.pretraining:
                noda_aucs.append(
                    self._plot_roc(
                        num,
                        adata_st_d[sample_id],
                        pred_sp_noda_d[sample_id],
                        f"{MODEL_NAME}_wo_da",
                        num_name_exN_l,
                        numlist,
                        ax.flat[i],
                    )
                )

            ax.flat[i].plot(
                [0, 1],
                [0, 1],
                transform=ax.flat[i].transAxes,
                ls="--",
                color="k",
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
        plt.close()

        return np.nanmean(da_aucs), np.nanmean(noda_aucs) if self.pretraining else None

    # %%
    def eval_spots(self):
        print("Getting predictions: ")
        if "" in self.st_sample_id_d:
            splits = [""]
        else:
            splits = self.splits

        sids = [sid for split in splits for sid in self.st_sample_id_d[split]]

        if self.data_params.get("samp_split", False):
            path = os.path.join(self.advtrain_folder, "samp_split/final_model.pth")
        elif self.data_params["train_using_all_st_samples"]:
            path = os.path.join(self.advtrain_folder, "final_model.pth")
        else:
            path = None

        if path is not None:
            inputs = [self.mat_sp_d[sid] for sid in sids]
            outputs = get_predictions(get_model(path), inputs)
            pred_sp_d = dict(zip(sids, outputs))
        else:
            pred_sp_d = {}
            for sample_id in sids:
                path = os.path.join(self.advtrain_folder, sample_id, f"final_model.pth")
                input = self.mat_sp_d[sample_id]
                pred_sp_d[sample_id] = next(get_predictions(get_model(path), input))

        if self.pretraining:
            inputs = [self.mat_sp_d[sid] for sid in sids]
            outputs = get_predictions(
                get_model(self.pretrain_model_path), inputs, source_encoder=True
            )
            pred_sp_noda_d = dict(zip(sids, outputs))
        else:
            pred_sp_noda_d = None
        # %%
        adata_st = sc.read_h5ad(os.path.join(self.selected_dir, "st.h5ad"))

        adata_st_d = {}
        print("Loading ST adata: ")
        for sample_id in sids:
            adata_st_d[sample_id] = adata_st[adata_st.obs.sample_id == sample_id]
            adata_st_d[sample_id].obsm["spatial"] = adata_st_d[sample_id].obs[["X", "Y"]].values
        realspots_d = {"da": {}}
        if self.pretraining:
            realspots_d["noda"] = {}
        if self.data_params.get("dset") == "pdac":
            aucs = self.eval_pdac_spots(pred_sp_d, pred_sp_noda_d, adata_st_d)
        else:
            aucs = self.eval_dlpfc_spots(pred_sp_d, pred_sp_noda_d, adata_st, adata_st_d)

        for sample_id, auc in zip(sids, aucs):
            realspots_d["da"][sample_id] = auc[0]
            if self.pretraining:
                realspots_d["noda"][sample_id] = auc[1]
        self.realspots_d = realspots_d

    def eval_dlpfc_spots(self, pred_sp_d, pred_sp_noda_d, adata_st, adata_st_d):

        if "" in self.st_sample_id_d:
            splits = [""]
        else:
            splits = self.splits

        sids = [sid for split in splits for sid in self.st_sample_id_d[split]]

        self._plot_layers(adata_st, adata_st_d)

        print("Plotting Samples")
        n_jobs_samples = min(effective_n_jobs(args.njobs), len(sids))
        logging.debug(f"n_jobs_samples: {n_jobs_samples}")
        aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
            delayed(self._plot_samples)(sid, adata_st_d, pred_sp_d, pred_sp_noda_d) for sid in sids
        )
        return aucs

    def eval_pdac_spots(self, pred_sp_d, pred_sp_noda_d, adata_st_d):
        raw_pdac_dir = os.path.join(self.data_params["data_dir"], "pdac", "st_adata")
        ctr_fname = f"{self.data_params['st_id']}-cluster_to_rgb.pkl"
        with open(os.path.join(raw_pdac_dir, ctr_fname), "rb") as f:
            cluster_to_rgb = pickle.load(f)

        self._plot_spatial(
            adata_st_d, cluster_to_rgb, color="cell_subclass", fname="st_cell_types.png"
        )
        if "" in self.st_sample_id_d:
            splits = [""]
        else:
            splits = self.splits

        sids = [sid for split in splits for sid in self.st_sample_id_d[split]]

        print("Plotting Samples")
        n_jobs_samples = min(effective_n_jobs(args.njobs), len(sids))
        logging.debug(f"n_jobs_samples: {n_jobs_samples}")
        aucs = Parallel(n_jobs=n_jobs_samples, verbose=100)(
            delayed(self._plot_samples_pdac)(
                sid,
                adata_st_d[sid],
                pred_sp_d[sid],
                pred_sp_noda_d[sid] if pred_sp_noda_d else None,
            )
            for sid in sids
        )
        return aucs

    def eval_sc(self, metric_ctp):
        self.jsd_d = {"da": {}}
        if self.pretraining:
            self.jsd_d["noda"] = {}

        for k in self.jsd_d:
            self.jsd_d[k] = {split: {} for split in self.splits}

        if "" in self.st_sample_id_d:
            sids = self.st_sample_id_d[""]
        else:
            sids = [sid for split in self.splits for sid in self.st_sample_id_d[split]]

        if self.pretraining:
            model = get_model(self.pretrain_model_path)

            self._calc_jsd(
                metric_ctp,
                sids[0],
                model=model,
                da="noda",
            )
            for split in self.splits:
                score = self.jsd_d["noda"][split][sids[0]]
                for sample_id in sids[1:]:
                    self.jsd_d["noda"][split][sample_id] = score

        if self.data_params.get("samp_split", False):
            model = get_model(os.path.join(self.advtrain_folder, "samp_split/final_model.pth"))
        elif self.data_params["train_using_all_st_samples"]:
            model = get_model(os.path.join(self.advtrain_folder, f"final_model.pth"))
        else:
            model = None

        for sample_id in sids:
            if model is None:
                model = get_model(os.path.join(self.advtrain_folder, sample_id, "final_model.pth"))

            self._calc_jsd(metric_ctp, sample_id, model=model, da="da")

    def _calc_jsd(self, metric_ctp, sample_id, model=None, da="da"):
        with torch.no_grad():
            inputs = (self.sc_mix_d[split] for split in self.splits)
            pred_mix = get_predictions(model, inputs, source_encoder=True)
            for split, pred in zip(self.splits, pred_mix):
                score = metric_ctp(
                    torch.as_tensor(pred, dtype=torch.float32),
                    torch.as_tensor(self.lab_mix_d[split], dtype=torch.float32),
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
        if self.data_params.get("dset") == "pdac":
            real_spot_header = "Real Spots (Mean AUC celltype)"
        else:
            real_spot_header = "Real Spots (Mean AUC Ex1-10)"

        df_keys = [
            "Pseudospots (JS Divergence)",
            "RF50",
            real_spot_header,
        ]

        if MILISI:
            df_keys.insert(2, "miLISI")

        da_dict_keys = ["da"]
        da_df_keys = ["After DA"]
        if self.pretraining:
            da_dict_keys.insert(0, "noda")
            da_df_keys.insert(0, "Before DA")

        results_df = [
            pd.concat(list(self.gen_l_dfs(da)), axis=1, keys=df_keys) for da in da_dict_keys
        ]

        sid_to_split = {sid: split for split, sids_ in self.st_sample_id_d.items() for sid in sids_}
        for df in results_df:
            df.index = df.index.map(lambda x: (sid_to_split[x], x))
            df.index.set_names(["SC Split", "Sample ID"], inplace=True)

        results_df = pd.concat(results_df, axis=0, keys=da_df_keys)

        results_df.to_csv(os.path.join(self.results_folder, "results.csv"))
        print(results_df)

        with open(os.path.join(self.results_folder, "config.yml"), "w") as f:
            yaml.dump(self.config, f)

        self.temp_folder_holder.copy_out()


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
        out = model(torch.as_tensor(x, device=device, dtype=torch.float32))
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

    out_func = lambda x: encoder(torch.as_tensor(x, device=device, dtype=torch.float32))
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


def fit_pca(X, *args, **kwargs):
    return PCA(*args, **kwargs).fit(X)


if __name__ == "__main__":
    main()

print("Script run time:", datetime.datetime.now(datetime.timezone.utc) - script_start_time)
