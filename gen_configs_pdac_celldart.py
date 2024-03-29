# %%
import glob
import itertools
import os
from copy import deepcopy

import numpy as np
import yaml

# %%
# CONFIG_FNAME = "bnfix_minmax_samp_split.yml"
MODEL_NAME = "CellDART"
CONFIG_DIR = "configs/generated_pdac"
CONFIG_FNAME_PREFIX = "gen_pdac"

# %%
config = {
    "lib_params": {},
    "data_params": {},
    "train_params": {},
    "model_params": {},
}

config["data_params"]["all_genes"] = False
config["data_params"]["data_dir"] = "data"
config["data_params"]["dset"] = "pdac"
config["data_params"]["n_spots"] = 100000
config["data_params"]["one_model"] = True
config["data_params"]["samp_split"] = False
config["data_params"]["sc_id"] = "CA001063"
# config["data_params"]["scaler_name"] = "standard"
config["data_params"]["st_id"] = "GSE111672"
config["data_params"]["st_split"] = False

config["model_params"]["celldart_kwargs"] = {}

config["train_params"]["initial_train_epochs"] = 10
config["train_params"]["n_iter"] = 15000
# config["train_params"]["reverse_val"] = False
config["train_params"]["reverse_val"] = True
config["data_params"]["val_samp"] = False
config["train_params"]["pretraining"] = True

if not os.path.exists(os.path.join(CONFIG_DIR, MODEL_NAME)):
    os.makedirs(os.path.join(CONFIG_DIR, MODEL_NAME))

with open(os.path.join(CONFIG_DIR, MODEL_NAME, f"{CONFIG_FNAME_PREFIX}.yml"), "w") as f:
    yaml.safe_dump(config, f)

# %%
## CellDART

# data_params
data_params_lists = dict(
    n_markers=[20, 40, 80],
    n_mix=[30, 50, 70],
    scaler_name=["minmax", "standard"],
)
# model_params
model_params_lists = dict(
    bn_momentum=[0.01, 0.1, 0.9, 0.99],
    emb_dim=[32, 64],
)


# train_params
train_params_lists = dict(
    alpha=[0.1, 0.6, 1.0, 2.0],
    alpha_lr=[1, 2, 5, 10],
    batch_size=[128, 256, 512],
    lr=[0.01, 0.001, 0.0001],
)


# %%
total_configs = 1
for value in data_params_lists.values():
    total_configs *= len(value)
for value in model_params_lists.values():
    total_configs *= len(value)
for value in train_params_lists.values():
    total_configs *= len(value)
total_configs

# %%
config

# %%
rng = np.random.default_rng(25780)

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

for i, (data_params, model_params, train_params) in enumerate(
    itertools.product(data_params_l, model_params_l, train_params_l)
):
    if i not in yes_samples:
        continue

    new_config = deepcopy(config)

    new_config["data_params"].update(data_params)
    new_config["model_params"]["celldart_kwargs"].update(model_params)
    new_config["train_params"].update(train_params)

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
