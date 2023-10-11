#!/usr/bin/env python3

# %%
import glob
import itertools
import os
from copy import deepcopy

import numpy as np
import yaml

# %%
# CONFIG_FNAME = "dann.yml"
MODEL_NAME = "DANN"
CONFIG_DIR = "configs/generated_spotless"
CONFIG_FNAME_PREFIX = "gen_spotless10001000_oracle"

# %%
config = {
    "lib_params": {},
    "data_params": {},
    "train_params": {},
    "model_params": {},
}

config["data_params"]["all_genes"] = False
config["data_params"]["data_dir"] = "data"
config["data_params"]["dset"] = "mouse_cortex"
config["data_params"]["n_spots"] = 100000
config["data_params"]["samp_split"] = True
config["data_params"]["sc_id"] = "GSE115746"
# config["data_params"]["scaler_name"] = "standard"
config["data_params"]["st_id"] = "spotless_mouse_cortex"
config["data_params"]["st_split"] = False

config["model_params"]["dann_kwargs"] = {}
config["model_params"]["dann_kwargs"]["enc_out_act"] = True

config["train_params"]["adv_opt_kwargs"] = {}
config["train_params"]["adv_opt_kwargs"]["betas"] = (0.5, 0.999)
config["train_params"]["adv_opt_kwargs"]["eps"] = 1.0e-07

config["train_params"]["batch_size"] = 8
config["train_params"]["early_stop_crit"] = 100
config["train_params"]["epochs"] = 500
config["train_params"]["early_stop_crit_adv"] = 500
config["train_params"]["min_epochs"] = 40
config["train_params"]["min_epochs_adv"] = 100
config["train_params"]["pretraining"] = False
# config["train_params"]["reverse_val"] = False
config["train_params"]["reverse_val"] = True
config["train_params"]["val_samp"] = False


config["train_params"]["plateau_kwargs"] = {}
config["train_params"]["plateau_kwargs"]["factor"] = 0.5
config["train_params"]["plateau_kwargs"]["min_lr"] = 2.0e-06
config["train_params"]["plateau_kwargs"]["patience"] = 50
config["train_params"]["plateau_kwargs"]["verbose"] = True


if not os.path.exists(os.path.join(CONFIG_DIR, MODEL_NAME)):
    os.makedirs(os.path.join(CONFIG_DIR, MODEL_NAME))

with open(os.path.join(CONFIG_DIR, MODEL_NAME, f"{CONFIG_FNAME_PREFIX}.yml"), "w") as f:
    yaml.safe_dump(config, f)

# %%
## ADDA

# data_params
data_params_lists = dict(
    n_markers=[20, 40, 80],
    n_mix=[5, 8, 10, 15],
    scaler_name=["minmax", "standard"],
)

# model_params
model_params_lists = dict(
    alpha_=[1, 2, 3],
    dropout=[0.1, 0.2, 0.5],
    emb_dim=[32, 64],
    enc_hidden_layer_sizes=[
        (512,),
        (1024,),
        (512, 512),
        (512, 256),
        (256, 128),
    ],
    dis_dropout_factor=[0.5, 1],
)
# train_params
train_params_lists = {
    "lambda": [0.5, 1, 2],
    "two_step": [True, False],
    "source_first": [True, False],
}

adv_opt_kwargs_lists = dict(
    lr=[2e-3, 1e-3, 2e-4, 1e-4, 1e-5, 2e-5],
    weight_decay=[0.01, 0.1],
)

# %%
total_configs = 1
for value in data_params_lists.values():
    total_configs *= len(value)
for value in model_params_lists.values():
    total_configs *= len(value)
for value in train_params_lists.values():
    total_configs *= len(value)
for value in adv_opt_kwargs_lists.values():
    total_configs *= len(value)
total_configs

# %%
config

# %%
rng = np.random.default_rng(46758)

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

adv_opt_kwargs_l = []
for kv_tuples in itertools.product(
    *[[(k, v) for v in vlist] for k, vlist in adv_opt_kwargs_lists.items()]
):
    adv_opt_kwargs_l.append(dict(kv_tuples))


for i, (data_params, model_params, train_params, adv_opt_kwargs) in enumerate(
    itertools.product(data_params_l, model_params_l, train_params_l, adv_opt_kwargs_l)
):
    if i not in yes_samples:
        continue

    new_config = deepcopy(config)

    new_config["data_params"].update(data_params)
    new_config["model_params"]["dann_kwargs"].update(model_params)
    new_config["train_params"].update(train_params)
    new_config["train_params"]["adv_opt_kwargs"].update(adv_opt_kwargs)

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
