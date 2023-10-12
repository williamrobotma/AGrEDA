#!/usr/bin/env python

# %%
import glob
import itertools
import os
from copy import deepcopy

import numpy as np
import yaml

# %%
# CONFIG_FNAME = "coral.yml"
MODEL_NAME = "CORAL"
CONFIG_DIR = "configs/generated_dlpfc"
CONFIG_FNAME_PREFIX = "gen_dlpfc_dlpfc"

# %%
config = {
    "lib_params": {},
    "data_params": {},
    "train_params": {},
    "model_params": {},
}

config["data_params"]["all_genes"] = False
config["data_params"]["data_dir"] = "data"
config["data_params"]["dset"] = "dlpfc"
config["data_params"]["n_spots"] = 100000
config["data_params"]["samp_split"] = True
config["data_params"]["sc_id"] = "GSE144136"
# config["data_params"]["scaler_name"] = "standard"
config["data_params"]["st_id"] = "spatialLIBD"
config["data_params"]["st_split"] = False

config["model_params"]["coral_kwargs"] = {}
config["model_params"]["coral_kwargs"]["batchnorm"] = True
config["model_params"]["coral_kwargs"]["batchnorm_after_act"] = True
config["model_params"]["coral_kwargs"]["predictor_hidden_layer_sizes"] = None
config["model_params"]["coral_kwargs"]["use_predictor"] = True
config["model_params"]["coral_kwargs"]["enc_out_act"] = True

config["train_params"]["epochs"] = 200
# config["train_params"]["reverse_val"] = False
config["train_params"]["reverse_val"] = True
config["train_params"]["val_samp"] = False
config["train_params"]["opt_kwargs"] = {}

if not os.path.exists(os.path.join(CONFIG_DIR, MODEL_NAME)):
    os.makedirs(os.path.join(CONFIG_DIR, MODEL_NAME))

with open(os.path.join(CONFIG_DIR, MODEL_NAME, f"{CONFIG_FNAME_PREFIX}.yml"), "w") as f:
    yaml.safe_dump(config, f)

# %%
## ADDA

# data_params
data_params_lists = dict(
    n_markers=[20, 40, 80],
    n_mix=[3, 5, 8, 10],
    scaler_name=["minmax", "standard"],
)

# model_params
model_params_lists = dict(
    dropout=[0.1, 0.2, 0.5],
    emb_dim=[32, 64],
    enc_hidden_layer_sizes=[
        (1024, 512),
        (512, 256),
        (256, 128),
        (512, 256, 128),
    ],
    hidden_act=["leakyrelu", "relu"],
)
# train_params
train_params_lists = {
    "batch_size": [256, 512, 1024],
    "lambda": [
        (0, 50),
        (0, 100),
        (0, 200),
        (50, 50),
        (100, 100),
        (200, 200),
    ],
}

opt_kwargs_lists = dict(
    lr=[1e-2, 1e-3, 1e-4],
    weight_decay=[0.1, 0.3, 0.5],
)

# %%
total_configs = 1
for value in data_params_lists.values():
    total_configs *= len(value)
for value in model_params_lists.values():
    total_configs *= len(value)
for value in train_params_lists.values():
    total_configs *= len(value)
for value in opt_kwargs_lists.values():
    total_configs *= len(value)
total_configs

# %%
config

# %%
rng = np.random.default_rng(5273)

yes_samples = set(rng.choice(total_configs, size=1000, replace=False))


data_params_l = []
for kv_tuples in itertools.product(
    *[[(k, v) for v in vlist] for k, vlist in data_params_lists.items()]
):
    data_params_l.append(dict(kv_tuples))

model_params_l = []
for kv_tuples in itertools.product(
    *[[(k, v) for v in vlist] for k, vlist in model_params_lists.items()]
):
    model_params_l.append(dict(kv_tuples))

train_params_l = []
for kv_tuples in itertools.product(
    *[[(k, v) for v in vlist] for k, vlist in train_params_lists.items()]
):
    train_params_l.append(dict(kv_tuples))

opt_kwargs_l = []
for kv_tuples in itertools.product(
    *[[(k, v) for v in vlist] for k, vlist in opt_kwargs_lists.items()]
):
    opt_kwargs_l.append(dict(kv_tuples))


for i, (data_params, model_params, train_params, opt_kwargs) in enumerate(
    itertools.product(data_params_l, model_params_l, train_params_l, opt_kwargs_l)
):
    if i not in yes_samples:
        continue

    new_config = deepcopy(config)

    new_config["data_params"].update(data_params)
    new_config["model_params"]["coral_kwargs"].update(model_params)
    new_config["train_params"].update(train_params)
    new_config["train_params"]["opt_kwargs"].update(opt_kwargs)

    new_config["lib_params"]["manual_seed"] = int(rng.integers(0, 2**32))

    version = f"{CONFIG_FNAME_PREFIX}-{i}"
    new_config["model_params"]["model_version"] = version

    with open(os.path.join(CONFIG_DIR, MODEL_NAME, f"{version}.yml"), "w") as f:
        yaml.safe_dump(new_config, f)

# %%
print(yaml.safe_dump(new_config))


# %%
lines = [
    os.path.basename(name)
    for name in sorted(
        glob.glob(os.path.join(CONFIG_DIR, MODEL_NAME, f"{CONFIG_FNAME_PREFIX}-*.yml"))
    )
]
with open(
    os.path.join(CONFIG_DIR, MODEL_NAME, "a_list.txt"),
    mode="wt",
    encoding="utf-8",
) as myfile:
    myfile.write("\n".join(lines))
    myfile.write("\n")

# %%
