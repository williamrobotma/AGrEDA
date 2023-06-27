#!/usr/bin/env python

# %%
import argparse
import glob
import os
import yaml

import pandas as pd

from src.da_utils import data_loading


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


# %%
# args = parser.parse_args([
#     "--modelname=CORAL",
#     "--configs_dir=configs/generated_spotless/",
# ])
args = parser.parse_args()

CONFIGS_DIR = args.configs_dir
MODEL_NAME = args.modelname
EARLY_STOPPING = args.early_stopping


# %%
with open(os.path.join(CONFIGS_DIR, MODEL_NAME, "a_list.txt"), "r") as f:
    config_files = f.read().splitlines()

scores_epochs = []
for config_file in config_files:
    with open(os.path.join(CONFIGS_DIR, MODEL_NAME, config_file), "r") as f:
        config = yaml.safe_load(f)
    lib_params = config["lib_params"]
    data_params = config["data_params"]
    model_params = config["model_params"]
    train_params = config["train_params"]

    lib_seed_path = str(lib_params.get("manual_seed", "random"))
    model_rel_path = data_loading.get_model_rel_path(
        MODEL_NAME,
        model_params["model_version"],
        lib_seed_path=lib_seed_path,
        **data_params,
    )
    results_folder = os.path.join("results", model_rel_path)

    if EARLY_STOPPING:
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
    if EARLY_STOPPING:
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

    score = results_df.loc[after_da_idx, "Real Spots (Cosine Distance)"].to_numpy().mean()

    scores_epochs.append((score, epoch))


# %%
df = pd.DataFrame.from_records(
    scores_epochs,
    columns=["score", "epoch"],
    index=config_files,
)
if "spotless" in data_params.get("st_id", "spatialLIBD"):
    best_config = df["score"].idxmin()  # cos distance
else:
    best_config = df["score"].idxmax()

print("Best config:", best_config)
print(df.loc[best_config])

with open(os.path.join(CONFIGS_DIR, MODEL_NAME, best_config), "r") as f:
    config = yaml.safe_load(f)

print(yaml.dump(config))
