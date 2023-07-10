#!/usr/bin/env python

# %%
import argparse
import glob
import math
import os

import pandas as pd
import yaml
from joblib import Parallel, delayed, effective_n_jobs

from src.da_utils import data_loading

# %%


def main(args):
    configs_dir = args.configs_dir
    model_name = args.modelname
    early_stopping = args.early_stopping
    n_jobs = args.njobs

    with open(os.path.join(configs_dir, model_name, "a_list.txt"), "r") as f:
        config_files = f.read().splitlines()

    batch_size = math.ceil(len(config_files) / effective_n_jobs(n_jobs))
    scores_epochs = Parallel(n_jobs=n_jobs, verbose=1, batch_size=batch_size)(
        delayed(get_score)(
            args.results_folder,
            configs_dir,
            model_name,
            early_stopping,
            config_file,
        )
        for config_file in config_files
    )

    df = pd.DataFrame.from_records(
        scores_epochs,
        columns=["score", "epoch"],
        index=config_files,
    )

    with open(os.path.join(configs_dir, model_name, config_files[0]), "r") as f:
        data_params = yaml.safe_load(f)["data_params"]

    if "spotless" in data_params.get("st_id", "spatialLIBD"):
        df = df.sort_values("score", ascending=True) # cos distance
    else:
        df = df.sort_values("score", ascending=False) # auc

    best_config = df.iloc[0].name

    print("Best config:", best_config)
    print("-" * 80)
    print("Top 5:")
    print(df.head(5))

    with open(os.path.join(configs_dir, model_name, best_config), "r") as f:
        config = yaml.safe_load(f)

    print("-" * 80)
    print(yaml.dump(config))

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        df.to_csv(os.path.join(args.output_dir, "final_scores.csv"))
        
        with open(os.path.join(args.output_dir, "best_config.yml"), "w") as f:
            yaml.safe_dump(config, f)


def get_score(rdir, configs_dir, model_name, early_stopping, config_file):
    with open(os.path.join(configs_dir, model_name, config_file), "r") as f:
        config = yaml.safe_load(f)

    lib_params = config["lib_params"]
    data_params = config["data_params"]
    model_params = config["model_params"]

    lib_seed_path = str(lib_params.get("manual_seed", "random"))
    model_rel_path = data_loading.get_model_rel_path(
        model_name,
        model_params["model_version"],
        lib_seed_path=lib_seed_path,
        **data_params,
    )
    results_folder = os.path.join(rdir, model_rel_path)

    if early_stopping:
        results_fname = glob.glob(
            os.path.join(results_folder, "results_checkpt-*.csv"),
        )
        if len(results_fname) != 1:
            raise OSError(
                f"{len(results_fname)} reverse_checkpt files found "
                f"in {results_folder}; expected 1"
            )
        results_fname = os.path.basename(results_fname[0])
    else:
        results_fname = "results.csv"

    results_df = pd.read_csv(
        os.path.join(results_folder, results_fname),
        header=[0, 1, 2],
        index_col=[0, 1, 2],
    )

    if early_stopping:
        after_da_idx = results_df.index.get_level_values(0).unique()
        after_da_idx = [idx for idx in after_da_idx if "After DA" in idx]
        if len(after_da_idx) != 1:
            raise ValueError(
                f"{len(after_da_idx)} unique 'After DA' epochs found "
                f"in {results_fname}; expected 1"
            )
        after_da_idx = after_da_idx[0]

        epoch = int(after_da_idx.lstrip("After DA (epoch ").rstrip(")"))
        epoch_fname = int(results_fname.rstrip(".csv").replace("results_checkpt-", ""))
        if epoch != epoch_fname:
            raise ValueError(f"Epoch mismatch: table: {epoch}, filename: {epoch_fname}")
    else:
        after_da_idx = "After DA (final model)"
        epoch = "final"

    if "spotless" in data_params.get("st_id", ""):
        score_col = "Real Spots (Cosine Distance)"
    elif "spatialLIBD" in data_params.get("st_id", ""):
        score_col = "Real Spots (Mean AUC Ex1-10)"
    else:
        score_col = "Real Spots (Mean AUC celltype)"

    score = results_df.loc[after_da_idx, score_col].to_numpy().mean()
    return score, epoch


if __name__ == "__main__":
    # %%
    parser = argparse.ArgumentParser(description="Chooses best result among generated configs.")
    parser.add_argument("--modelname", "-n", type=str, default="ADDA", help="model name")
    # parser.add_argument("--config_fname", "-f", type=str, help="Basic template config file to use")
    parser.add_argument("--configs_dir", "-cdir", type=str, default="configs", help="config dir")
    parser.add_argument(
        "--early_stopping",
        "-e",
        action="store_true",
        help="evaluate early stopping. Default: False",
    )
    parser.add_argument("--results_folder", type=str, default="results", help="results folder")
    parser.add_argument("--output_dir", default=None, help="Output dir")
    parser.add_argument(
        "--njobs", type=int, default=-1, help="Number of jobs to use for parallel processing."
    )

    args = parser.parse_args()

    main(args)
