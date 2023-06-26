#!/usr/bin/env python3
"""something like CellDART but it actually follows Adda in PyTorch"""

# %% [markdown]
#  # ADDA for ST

# %% [markdown]
#  Creating something like CellDART but it actually follows Adda in PyTorch as a first step

# %%
import argparse
import datetime
import glob
import os
import shutil
import tarfile
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from tqdm.autonotebook import tqdm

from src.da_models.adda import ADDAST
from src.da_models.model_utils.datasets import SpotDataset
from src.da_models.model_utils.utils import (
    LibConfig,
    ModelWrapper,
    get_torch_device,
    initialize_weights,
    set_requires_grad,
)
from src.da_utils import data_loading, evaluation
from src.da_utils.output_utils import DupStdout, TempFolderHolder

script_start_time = datetime.datetime.now(datetime.timezone.utc)


# %%
parser = argparse.ArgumentParser(
    description="Creating something like CellDART but it actually follows Adda in PyTorch as a first step"
)
parser.add_argument("--config_fname", "-f", help="Name of the config file to use")
parser.add_argument("--configs_dir", "-cdir", type=str, default="configs", help="config dir")
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers to use for dataloaders."
)
parser.add_argument("--cuda", "-c", default=None, help="gpu index to use")
parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary model directory")
parser.add_argument("--log_fname", "-l", default=None, help="optional log file name")
parser.add_argument("--num_threads", "-t", default=None, help="number of threads to use")

# %%
args = parser.parse_args()
CONFIG_FNAME = args.config_fname
CONFIGS_DIR = args.configs_dir
CUDA_INDEX = args.cuda
NUM_WORKERS = args.num_workers
TMP_DIR = args.tmpdir
LOG_FNAME = args.log_fname
NUM_THREADS = int(args.num_threads) if args.num_threads else None
MIN_EVAL_BS = 512
# CONFIG_FNAME = "celldart1_bnfix.yml"
# NUM_WORKERS = 0
# CUDA_INDEX = None


# %%
# lib_params = {}

# lib_params["manual_seed"] = 72


# %%
# data_params = {}
# # Data path and parameters
# data_params["data_dir"] = "data"
# data_params["n_markers"] = 20
# data_params["all_genes"] = False

# # Pseudo-spot parameters
# data_params["n_spots"] = 20000
# data_params["n_mix"] = 8

# # ST spot parameters
# data_params["st_split"] = False
# data_params["sample_id_n"] = "151673"

# # Scaler parameter
# data_params["scaler_name"] = "celldart"


# %%
# model_params = {}

# # Model parameters
MODEL_NAME = "ADDA"
# model_params["model_version"] = "celldart_bnfix"

# model_params["adda_kwargs"] = {
#     "emb_dim": 64,
#     "bn_momentum": 0.01,
# }


# %%
# train_params = {}

# train_params["batch_size"] = 1024

# # Pretraining parameters
# # SAMPLE_ID_N = "151673"

# train_params["initial_train_epochs"] = 100

# train_params["early_stop_crit"] = 100
# train_params["min_epochs"] = 0.4 * train_params["initial_train_epochs"]

# # Adversarial training parameters
# train_params["epochs"] = 200
# train_params["early_stop_crit_adv"] = train_params["epochs"]
# train_params["min_epochs_adv"] =  0.4 * train_params["epochs"]


# train_params["enc_lr"] = 0.0002
# train_params["alpha"] = 2
# train_params["dis_loop_factor"] = 5
# train_params["adam_beta1"] = 0.5


# %%
# config = {
#     "lib_params": lib_params,
#     "data_params": data_params,
#     "model_params": model_params,
#     "train_params": train_params,
# }

# if not os.path.exists(os.path.join("configs", MODEL_NAME)):
#     os.makedirs(os.path.join("configs", MODEL_NAME))

# with open(os.path.join("configs", MODEL_NAME, CONFIG_FNAME), "w") as f:
#     yaml.safe_dump(config, f)

with open(os.path.join(CONFIGS_DIR, MODEL_NAME, CONFIG_FNAME), "r") as f:
    config = yaml.safe_load(f)

lib_params = config["lib_params"]
data_params = config["data_params"]
model_params = config["model_params"]
train_params = config["train_params"]

rewrite_config = False
if not "pretraining" in train_params:
    train_params["pretraining"] = True
    rewrite_config = True
if not "initial_train_lr" in train_params:
    train_params["initial_train_lr"] = 0.002
    rewrite_config = True

if rewrite_config:
    with open(os.path.join(CONFIGS_DIR, MODEL_NAME, CONFIG_FNAME), "w") as f:
        yaml.safe_dump(config, f)

print(yaml.safe_dump(config))


# %%
if NUM_THREADS:
    torch.set_num_threads(NUM_THREADS)

device = get_torch_device(CUDA_INDEX)
ModelWrapper.configurate(LibConfig(device, CUDA_INDEX))

# %%
torch_seed = lib_params.get("manual_seed", int(script_start_time.timestamp()))
lib_seed_path = str(torch_seed) if "manual_seed" in lib_params else "random"
torch.manual_seed(torch_seed)
np.random.seed(torch_seed)


# %%
model_folder = data_loading.get_model_rel_path(
    MODEL_NAME,
    model_params["model_version"],
    lib_seed_path=lib_seed_path,
    **data_params,
)
model_folder = os.path.join("model", model_folder)

temp_folder_holder = TempFolderHolder()
model_folder = temp_folder_holder.set_output_folder(TMP_DIR, model_folder)


# %% [markdown]
#  # Data load

# %%
selected_dir = data_loading.get_selected_dir(
    data_loading.get_dset_dir(
        data_params["data_dir"],
        dset=data_params.get("dset", "dlpfc"),
    ),
    **data_params,
)


# Load spatial data
mat_sp_d, mat_sp_meta_d, st_sample_id_l = data_loading.load_spatial(
    selected_dir,
    **data_params,
)

# Load sc data
sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2 = data_loading.load_sc(
    selected_dir,
    **data_params,
)


# %% [markdown]
#  # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#  ## Prepare dataloaders

# %%
### source dataloaders
source_dataloader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
alt_bs = max(train_params.get("batch_size", 0), MIN_EVAL_BS)

source_set_d = {}
dataloader_source_d = {}
for split in sc_mix_d:
    source_set_d[split] = SpotDataset(sc_mix_d[split], lab_mix_d[split])
    dataloader_source_d[split] = torch.utils.data.DataLoader(
        source_set_d[split],
        shuffle=(split == "train"),
        batch_size=train_params.get("batch_size", MIN_EVAL_BS) if "train" in split else alt_bs,
        **source_dataloader_kwargs,
    )
if train_params["reverse_val"]:
    dataloader_source_d["train_dis"] = torch.utils.data.DataLoader(
        SpotDataset(deepcopy(sc_mix_d["train"]), deepcopy(lab_mix_d["train"])),
        shuffle=True,
        batch_size=train_params.get("batch_size", MIN_EVAL_BS),
        **source_dataloader_kwargs,
    )

### target dataloaders
target_dataloader_kwargs = source_dataloader_kwargs

if data_params.get("samp_split", False) or data_params.get("one_model", False):
    target_d = {}
    if "train" in mat_sp_d:
        # keys of dict are splits
        for split in mat_sp_d:
            target_d[split] = np.concatenate(list(mat_sp_d[split].values()))
    else:
        # keys of subdicts are splits
        for split in next(iter(mat_sp_d.values())):
            target_d[split] = np.concatenate((v[split] for v in mat_sp_d.values()))
    target_d["train_dis"] = deepcopy(target_d["train"])

    dataloader_target_d = {}
    target_set_d = {}
    for split in target_d:
        target_set_d[split] = SpotDataset(target_d[split])
        dataloader_target_d[split] = torch.utils.data.DataLoader(
            target_set_d[split],
            shuffle=("train" in split),
            batch_size=train_params.get("batch_size", MIN_EVAL_BS) if "train" in split else alt_bs,
            **target_dataloader_kwargs,
        )
else:
    target_d = {}
    target_set_d = {}
    dataloader_target_d = {}

    for sample_id in st_sample_id_l:
        target_d[sample_id] = {}
        target_set_d[sample_id] = {}
        dataloader_target_d[sample_id] = {}
        for split, v in mat_sp_d[sample_id].items():
            if split == "train" and v is mat_sp_d[sample_id]["test"]:
                target_d[sample_id][split] = deepcopy(v)
            else:
                target_d[sample_id][split] = v

            target_set_d[sample_id][split] = SpotDataset(target_d[sample_id][split])

            dataloader_target_d[sample_id][split] = torch.utils.data.DataLoader(
                target_set_d[sample_id][split],
                shuffle=("train" in split),
                batch_size=train_params.get("batch_size", MIN_EVAL_BS)
                if "train" in split
                else alt_bs,
                **target_dataloader_kwargs,
            )
            if split == "train":
                target_d[sample_id]["train_dis"] = deepcopy(v)
                target_set_d[sample_id]["train_dis"] = SpotDataset(target_d[sample_id]["train_dis"])
                dataloader_target_d[sample_id]["train_dis"] = torch.utils.data.DataLoader(
                    target_set_d[sample_id]["train_dis"],
                    shuffle=True,
                    batch_size=train_params.get("batch_size", MIN_EVAL_BS),
                    **target_dataloader_kwargs,
                )


# %% [markdown]
#  ## Pretrain


# %%
criterion_clf = nn.KLDivLoss(reduction="batchmean")
to_inp_kwargs = dict(device=device, dtype=torch.float32, non_blocking=True)


# %%
def model_loss(x, y_true, model):
    x = x.to(**to_inp_kwargs)
    y_true = y_true.to(**to_inp_kwargs)

    y_pred = model(x)
    loss = criterion_clf(y_pred, y_true)

    return loss


def run_pretrain_epoch(
    model,
    dataloader,
    optimizer=None,
    scheduler=None,
    inner=None,
):
    loss_running = []
    mean_weights = []

    is_training = model.training and optimizer

    for _, batch in enumerate(dataloader):
        loss = model_loss(*batch, model)
        loss_running.append(loss.item())
        mean_weights.append(len(batch))  # we will weight average by batch size later

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        if inner:
            inner.update(1)
    return loss_running, mean_weights


def compute_acc(dataloader, model):
    model.eval()
    with torch.no_grad():
        loss_running, mean_weights = run_pretrain_epoch(model, dataloader)

    return np.average(loss_running, weights=mean_weights)


def pretrain(
    pretrain_folder,
    model,
    dataloader_source_train,
    dataloader_source_val=None,
):
    if dataloader_source_val is None:
        dataloader_source_val = dataloader_source_train

    pre_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["initial_train_lr"],
        betas=(0.9, 0.999),
        eps=1e-07,
    )

    pre_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        pre_optimizer,
        max_lr=train_params["initial_train_lr"],
        steps_per_epoch=len(dataloader_source_train),
        epochs=train_params["initial_train_epochs"],
    )

    model.pretraining()

    # Initialize lists to store loss and accuracy values
    loss_history = []
    loss_history_val = []

    loss_history_running = []

    # Early Stopping
    best_loss_val = np.inf
    early_stop_count = 0

    log_file_path = os.path.join(pretrain_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        # Train
        tqdm.write("Start pretrain...")
        outer = tqdm(total=train_params["initial_train_epochs"], desc="Epochs")
        inner = tqdm(total=len(dataloader_source_train), desc=f"Batch")

        tqdm.write(" Epoch | Train Loss | Val Loss   ")
        tqdm.write("---------------------------------")
        checkpoint = {
            "epoch": -1,
            "model": model,
            "optimizer": pre_optimizer,
            "scheduler": pre_scheduler,
            # 'scaler': scaler
        }
        for epoch in range(train_params["initial_train_epochs"]):
            inner.refresh()  # force print final state
            inner.reset()  # reuse bar

            checkpoint["epoch"] = epoch

            # Train mode
            model.train()

            loss_running, mean_weights = run_pretrain_epoch(
                model,
                dataloader_source_train,
                optimizer=pre_optimizer,
                scheduler=pre_scheduler,
                inner=inner,
            )

            loss_history.append(np.average(loss_running, weights=mean_weights))
            loss_history_running.append(loss_running)

            # Evaluate mode
            model.eval()
            with torch.no_grad():
                curr_loss_val = compute_acc(dataloader_source_val, model)
                loss_history_val.append(curr_loss_val)

            # Print the results
            outer.update(1)
            out_string = f" {epoch:5d} | {loss_history[-1]:<10.8f} | {curr_loss_val:<10.8f} "
            # Save the best weights
            if curr_loss_val < best_loss_val:
                best_loss_val = curr_loss_val
                torch.save(
                    checkpoint,
                    os.path.join(pretrain_folder, f"best_model.pth"),
                )
                early_stop_count = 0

                out_string += "<-- new best val loss"

            tqdm.write(out_string)

            early_stop_count += 1

        inner.close()
        outer.close()
        # Save final model
        best_checkpoint = torch.load(os.path.join(pretrain_folder, f"best_model.pth"))
        torch.save(best_checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))

        return best_checkpoint["model"]


# %% [markdown]
#  ## Define Model

# %%
pretrain_folder = os.path.join(model_folder, "pretrain")

if not os.path.isdir(pretrain_folder):
    os.makedirs(pretrain_folder)

# %%
model = ADDAST(
    inp_dim=sc_mix_d["train"].shape[1],
    ncls_source=lab_mix_d["train"].shape[1],
    is_adda=True,
    **model_params["adda_kwargs"],
)
model.apply(initialize_weights)
model.to(device)

pretrain(
    pretrain_folder,
    model,
    dataloader_source_d["train"],
    dataloader_source_d["val"],
)


# %% [markdown]
#  ## Adversarial Adaptation


# %%
criterion_dis = nn.BCEWithLogitsLoss()


# %%
def discrim_loss_accu(x, domain, model):
    x = x.to(**to_inp_kwargs)

    if domain == "source":
        y_dis = torch.zeros(x.shape[0], device=device, dtype=x.dtype).view(-1, 1)
        emb = model.source_encoder(x)  # .view(x.shape[0], -1)
    elif domain == "target":
        y_dis = torch.ones(x.shape[0], device=device, dtype=x.dtype).view(-1, 1)
        emb = model.target_encoder(x)  # .view(x.shape[0], -1)
    else:
        raise (
            ValueError,
            f"invalid domain {domain} given, must be 'source' or 'target'",
        )

    y_pred = model.dis(emb)

    loss = criterion_dis(y_pred, y_dis)
    accu = (
        (torch.round(torch.sigmoid(y_pred)).to(torch.long) == y_dis).to(torch.float32).mean().cpu()
    )

    return loss, accu


def compute_acc_dis(dataloader_source, dataloader_target, model):
    results_history = {
        "dis": {
            "source": {},
            "target": {},
        }
    }

    model.eval()
    model.dis.eval()
    model.target_encoder.eval()
    model.source_encoder.eval()
    with torch.no_grad():
        loss_running, accu_running, mean_weights = run_adv_epoch_dis(
            model, dataloader_source, "source"
        )
        results_history["dis"]["source"]["loss"] = np.average(loss_running, weights=mean_weights)
        results_history["dis"]["source"]["accu"] = np.average(accu_running, weights=mean_weights)

        loss_running, accu_running, mean_weights = run_adv_epoch_dis(
            model, dataloader_target, "target"
        )
        results_history["dis"]["target"]["loss"] = np.average(
            loss_running,
            weights=mean_weights,
        )
        results_history["dis"]["target"]["accu"] = np.average(
            accu_running,
            weights=mean_weights,
        )

        return results_history


def run_adv_epoch_dis(model, dataloader, domain):
    loss_running = []
    accu_running = []
    mean_weights = []
    for _, (X, _) in enumerate(dataloader):
        loss, accu = discrim_loss_accu(X, domain, model)
        loss_running.append(loss.item())
        accu_running.append(accu)
        mean_weights.append(len(X))
    return loss_running, accu_running, mean_weights


def encoder_loss(x_target, model):
    x_target = x_target.to(device)

    # flip label
    y_dis = torch.zeros(x_target.shape[0], device=device, dtype=x_target.dtype).view(-1, 1)

    emb_target = model.target_encoder(x_target)  # .view(x_target.shape[0], -1)
    y_pred = model.dis(emb_target)
    loss = criterion_dis(y_pred, y_dis)
    accu = (
        (torch.round(torch.sigmoid(y_pred)).to(torch.long) == y_dis).to(torch.float32).mean().cpu()
    )

    return loss, accu


# %%
def train_adversarial_iters(
    model,
    save_folder,
    dataloader_source_train,
    dataloader_source_val,
    dataloader_target_train,
    dataloader_target_train_dis,
    dataloader_target_val=None,
    checkpoints=False,
    fname_prefix="",
    epoch_override=None,
):
    if dataloader_target_val is None:
        dataloader_target_val = dataloader_target_train

    model.to(device)
    model.advtraining()

    target_optimizer = torch.optim.Adam(
        model.target_encoder.parameters(),
        lr=train_params["enc_lr"],
        betas=(train_params["adam_beta1"], 0.999),
        eps=1e-07,
    )
    dis_optimizer = torch.optim.Adam(
        model.dis.parameters(),
        lr=train_params["alpha"] * train_params["enc_lr"],
        betas=(train_params["adam_beta1"], 0.999),
        eps=1e-07,
    )

    dataloader_lengths = [
        # len(dataloader_source_train),
        len(dataloader_target_train),
        len(dataloader_target_train_dis) * train_params["dis_loop_factor"],
    ]
    max_len_dataloader = np.amax(dataloader_lengths)

    # dis_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     dis_optimizer, max_lr=0.0005, steps_per_epoch=iters, epochs=EPOCHS
    # )
    # target_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     target_optimizer, max_lr=0.0005, steps_per_epoch=iters, epochs=EPOCHS
    # )

    results_template = {
        "dis": {
            "source": {"loss": [], "accu": [], "weights": []},
            "target": {"loss": [], "accu": [], "weights": []},
        },
        "gen": {
            "target": {"loss": [], "accu": [], "weights": []},
        },
    }
    results_history = deepcopy(results_template)
    results_history_val = deepcopy(results_template)
    results_history_running = deepcopy(results_template)

    epochs = epoch_override if epoch_override is not None else train_params["epochs"]
    # Early Stopping
    early_stop_count = 0
    log_file_path = os.path.join(save_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        # Train
        tqdm.write("Start adversarial training...")
        outer = tqdm(total=epochs, desc="Epochs")
        inner = tqdm(total=max_len_dataloader, desc=f"Batch")

        tqdm.write(" Epoch ||| Generator       ||| Discriminator ")
        tqdm.write(
            "       ||| Train           ||| Train                             || Validation    "
        )
        tqdm.write(
            "       ||| Loss   | Accu   ||| Loss            | Accu            || Loss            | Accu  "
        )
        tqdm.write(
            "       ||| Target - Target ||| Source - Target | Source - Target || Source - Target | Source - Target "
        )
        tqdm.write(
            "------------------------------------------------------------------------------------------------------"
        )
        checkpoint = {
            "epoch": -1,
            "model": model,
            "dis_optimizer": dis_optimizer,
            "target_optimizer": target_optimizer,
            # "dis_scheduler": dis_scheduler,
            # "target_scheduler": target_scheduler,
        }
        if checkpoints:
            torch.save(
                {"model": model.state_dict(), "epoch": -1},
                os.path.join(save_folder, f"{fname_prefix}checkpt--1.pth"),
            )
        for epoch in range(epochs):
            inner.refresh()  # force print final state
            inner.reset()  # reuse bar

            checkpoint["epoch"] = epoch

            # Train mode
            model.train()
            model.target_encoder.train()
            model.source_encoder.eval()
            model.dis.train()

            results_running = deepcopy(results_template)

            s_train_iter = iter(dataloader_source_train)
            t_train_iter = iter(dataloader_target_train)
            t_train_dis_iter = iter(dataloader_target_train_dis)
            for i in range(max_len_dataloader):
                try:
                    x_source, _ = next(s_train_iter)
                except StopIteration:
                    s_train_iter = iter(dataloader_source_train)
                    x_source, _ = next(s_train_iter)
                try:
                    x_target, _ = next(t_train_iter)
                except StopIteration:
                    t_train_iter = iter(dataloader_target_train)
                    x_target, _ = next(t_train_iter)

                inner_loop_idx = i % train_params["dis_loop_factor"]
                train_encoder_step = inner_loop_idx == train_params["dis_loop_factor"] - 1

                model.train_discriminator()

                set_requires_grad(model.target_encoder, False)
                set_requires_grad(model.source_encoder, False)
                set_requires_grad(model.dis, True)

                # lr_history_running.append(scheduler.get_last_lr())
                dis_optimizer.zero_grad()

                loss, accu = discrim_loss_accu(x_source, "source", model)
                results_running["dis"]["source"]["loss"].append(loss.item())
                results_running["dis"]["source"]["accu"].append(accu)
                results_running["dis"]["source"]["weights"].append(len(x_source))

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                loss.backward()
                dis_optimizer.step()

                dis_optimizer.zero_grad()

                loss, accu = discrim_loss_accu(x_target, "target", model)
                results_running["dis"]["target"]["loss"].append(loss.item())
                results_running["dis"]["target"]["accu"].append(accu)
                results_running["dis"]["target"]["weights"].append(len(x_target))

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                loss.backward()
                dis_optimizer.step()
                # dis_scheduler.step()

                if train_encoder_step:
                    try:
                        x_target_enc, _ = next(t_train_dis_iter)
                    except StopIteration:
                        t_train_dis_iter = iter(dataloader_target_train_dis)
                        x_target_enc, _ = next(t_train_dis_iter)
                    model.train_target_encoder()

                    set_requires_grad(model.target_encoder, True)
                    set_requires_grad(model.source_encoder, False)
                    set_requires_grad(model.dis, False)

                    target_optimizer.zero_grad()

                    loss, accu = encoder_loss(x_target_enc, model)

                    results_running["gen"]["target"]["loss"].append(loss.item())
                    results_running["gen"]["target"]["accu"].append(accu)
                    results_running["gen"]["target"]["weights"].append(len(x_target_enc))

                    loss.backward()
                    target_optimizer.step()
                # target_scheduler.step()

                inner.update(1)

            evaluation.recurse_avg_dict(results_running, results_history)
            evaluation.recurse_running_dict(results_running, results_history_running)

            model.eval()
            model.dis.eval()
            model.target_encoder.eval()
            model.source_encoder.eval()

            set_requires_grad(model, True)
            set_requires_grad(model.target_encoder, True)
            set_requires_grad(model.source_encoder, True)
            set_requires_grad(model.dis, True)

            with torch.no_grad():
                results_val = compute_acc_dis(dataloader_source_val, dataloader_target_val, model)
            evaluation.recurse_running_dict(results_val, results_history_val)

            if (epoch % 10 == 9 or epoch >= epochs - 1 or epoch < 20) and checkpoints:
                torch.save(
                    {"model": model.state_dict()},
                    os.path.join(save_folder, f"{fname_prefix}checkpt-{epoch}.pth"),
                )

            # Print the results
            outer.update(1)
            out_string = (
                f" {epoch:5d} "
                f"||| {results_history['gen']['target']['loss'][-1]:6.4f} "
                f"- {results_history['gen']['target']['accu'][-1]:6.4f} "
                f"||| {results_history['dis']['source']['loss'][-1]:6.4f} "
                f"- {results_history['dis']['target']['loss'][-1]:6.4f} "
                f"| {results_history['dis']['source']['accu'][-1]:6.4f} "
                f"- {results_history['dis']['target']['accu'][-1]:6.4f} "
                f"|| {results_history_val['dis']['source']['loss'][-1]:6.4f} "
                f"- {results_history_val['dis']['target']['loss'][-1]:6.4f} "
                f"| {results_history_val['dis']['source']['accu'][-1]:6.4f} "
                f"- {results_history_val['dis']['target']['accu'][-1]:6.4f} "
            )

            tqdm.write(out_string)

            early_stop_count += 1

    outer.close()
    inner.close()

    # Save final model
    torch.save(checkpoint, os.path.join(save_folder, f"{fname_prefix}final_model.pth"))

    return (results_history, results_history_running, results_history_val), checkpoint


# %%
def plot_results(results_history, results_history_running, results_history_val, save_folder):
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(9, 12), layout="constrained")

    # loss
    axs[0].plot(
        *evaluation.format_iters(results_history_running["dis"]["source"]["loss"]),
        label="d-source",
        linewidth=0.5,
    )
    axs[0].plot(
        *evaluation.format_iters(results_history_running["dis"]["target"]["loss"]),
        label="d-target",
        linewidth=0.5,
    )
    axs[0].plot(
        *evaluation.format_iters(results_history_running["gen"]["target"]["loss"]),
        label="g-target",
        linewidth=0.5,
    )

    axs[0].set_ylim(bottom=0, top=2)
    axs[0].grid(which="major")
    axs[0].minorticks_on()
    axs[0].grid(which="minor", alpha=0.2)

    axs[0].set_title("Training BCE Loss")
    axs[0].legend()

    # accuracy
    axs[1].plot(
        *evaluation.format_iters(results_history_running["dis"]["source"]["accu"]),
        label="d-source",
        linewidth=0.5,
    )
    axs[1].plot(
        *evaluation.format_iters(results_history_running["dis"]["target"]["accu"]),
        label="d-target",
        linewidth=0.5,
    )
    axs[1].plot(
        *evaluation.format_iters(results_history_running["gen"]["target"]["accu"]),
        label="g-target",
        linewidth=0.5,
    )

    axs[1].set_ylim(bottom=0, top=1)
    axs[1].grid(which="major")
    axs[1].minorticks_on()
    axs[1].grid(which="minor", alpha=0.2)

    axs[1].set_title("Training Accuracy")
    axs[1].legend()

    # val loss
    axs[2].plot(results_history["dis"]["source"]["loss"], label="d-source")
    axs[2].plot(results_history["dis"]["target"]["loss"], label="d-target")

    axs[2].set_ylim(bottom=0, top=2)
    axs[2].grid(which="major")
    axs[2].minorticks_on()
    axs[2].grid(which="minor", alpha=0.2)

    axs[2].set_title("Validation BCE Loss")
    axs[2].legend()

    # val accuracy
    axs[3].plot(results_history["dis"]["source"]["accu"], label="d-source")
    axs[3].plot(results_history["dis"]["target"]["accu"], label="d-target")

    axs[3].set_ylim(bottom=0, top=1)
    axs[3].grid(which="major")
    axs[3].minorticks_on()
    axs[3].grid(which="minor", alpha=0.2)

    axs[3].set_title("Validation Accuracy")
    axs[3].legend()

    plt.savefig(os.path.join(save_folder, "adv_train.png"))

    plt.show(block=False)


# %%
# st_sample_id_l = [SAMPLE_ID_N]


# %%
def load_train_plot(
    pretrain_folder_or_model,
    save_folder,
    dataloader_source_train,
    dataloader_source_val,
    dataloader_target_train,
    dataloader_target_train_dis,
    dataloader_target_val,
    **adv_train_kwargs,
):
    if isinstance(pretrain_folder_or_model, str):
        fname = f"{adv_train_kwargs.get('fname_prefix', '')}final_model.pth"
        best_checkpoint = torch.load(os.path.join(pretrain_folder, fname))

        model = ADDAST(
            sc_mix_d["train"].shape[1],
            ncls_source=lab_mix_d["train"].shape[1],
            is_adda=True,
            **model_params["adda_kwargs"],
        )

        model.apply(initialize_weights)

        # load state dicts
        # this makes it easier, if, say, the discriminator changes
        model.source_encoder.load_state_dict(best_checkpoint["model"].source_encoder.state_dict())

        model.clf.load_state_dict(best_checkpoint["model"].clf.state_dict())
    else:
        model = pretrain_folder_or_model

    model.init_adv()
    model.dis.apply(initialize_weights)

    model.to(device)

    model.advtraining()

    results, checkpoint = train_adversarial_iters(
        model,
        save_folder,
        dataloader_source_train,
        dataloader_source_val,
        dataloader_target_train,
        dataloader_target_train_dis,
        dataloader_target_val,
        **adv_train_kwargs,
    )
    plot_results(*results, save_folder)

    return checkpoint


# %%
advtrain_folder = os.path.join(model_folder, "advtrain")

if not os.path.isdir(advtrain_folder):
    os.makedirs(advtrain_folder)


def reverse_val(
    target_d,
    save_folder,
):
    rv_scores_d = OrderedDict()
    for name in sorted(glob.glob(os.path.join(save_folder, f"checkpt-*.pth"))):
        model_name = os.path.basename(name).rstrip(".pth")
        epoch = int(model_name[len("checkpt-") :])
        tqdm.write(f"  {model_name}")

        model = ModelWrapper(save_folder, model_name)

        dataloader_target_now_source_d = {}
        pred_target_d = OrderedDict()
        for split in target_d:
            pred_target_d[split] = model.get_predictions(target_d[split])

            dataloader_target_now_source_d[split] = torch.utils.data.DataLoader(
                SpotDataset(target_d[split], pred_target_d[split]),
                shuffle=("train" in split),
                batch_size=train_params.get("batch_size", MIN_EVAL_BS)
                if "train" in split
                else alt_bs,
                **source_dataloader_kwargs,
            )

        model = ADDAST(
            inp_dim=sc_mix_d["train"].shape[1],
            ncls_source=lab_mix_d["train"].shape[1],
            is_adda=True,
            **model_params["adda_kwargs"],
        )
        model.apply(initialize_weights)
        model.to(device)

        rv_pretrain_folder = os.path.join(save_folder, f"reverse_val-{model_name}-pretrain")
        if not os.path.isdir(rv_pretrain_folder):
            os.makedirs(rv_pretrain_folder)

        model = pretrain(
            rv_pretrain_folder,
            model,
            dataloader_target_now_source_d["train"],
            dataloader_target_now_source_d.get("val", None),
        )

        rv_save_folder = os.path.join(save_folder, f"reverse_val-{model_name}")
        if not os.path.isdir(rv_save_folder):
            os.makedirs(rv_save_folder)

        checkpoint = load_train_plot(
            model,
            rv_save_folder,
            dataloader_target_now_source_d["train"],
            dataloader_target_now_source_d.get("val", dataloader_target_now_source_d["train"]),
            dataloader_source_d["train"],
            dataloader_source_d["train_dis"],
            dataloader_source_d["val"],
            checkpoints=False,
            epoch_override=epoch + 1,
        )

        # model = ModelWrapper(model)
        rv_scores_d[epoch] = OrderedDict()
        for split in ("train", "val"):
            rv_scores_d[epoch][split] = compute_acc(
                dataloader_source_d[split],
                checkpoint["model"],
            )

    rv_scores_df = pd.DataFrame.from_dict(rv_scores_d, orient="index").sort_index()

    tqdm.write("Reverse validation scores: ")
    tqdm.write(repr(rv_scores_df))
    best_epoch = rv_scores_df["val"].idxmin()
    tqdm.write(f"Best epoch: {best_epoch} ({rv_scores_df['val'].min():.4f})")

    rv_scores_df.to_csv(os.path.join(save_folder, f"reverse_val_scores.csv"))

    best_epoch_df = rv_scores_df.loc[[best_epoch]]
    best_epoch_df.index = pd.Index([model_params["model_version"]])
    best_epoch_df["best_epoch"] = best_epoch
    best_epoch_df["best_epoch_rv"] = best_epoch
    best_epoch_df["config_fname"] = CONFIG_FNAME

    tqdm.write("Best Epoch: ")
    tqdm.write(repr(best_epoch_df))

    best_epoch_df.to_csv(os.path.join(save_folder, f"reverse_val_best_epoch.csv"))

    best_checkpoint = torch.load(os.path.join(save_folder, f"checkpt-{best_epoch}.pth"))
    final_checkpoint = torch.load(os.path.join(save_folder, f"final_model.pth"))

    model = final_checkpoint["model"]
    model.load_state_dict(best_checkpoint["model"])

    torch.save(
        {"model": model, "epoch": best_epoch},
        os.path.join(save_folder, f"best_model.pth"),
    )
    shutil.copyfile(
        os.path.join(save_folder, f"best_model.pth"),
        os.path.join(save_folder, f"final_model.pth"),
    )
    with tarfile.open(os.path.join(save_folder, f"reverse_val.tar.gz"), "w:gz") as tar:
        for name in glob.glob(os.path.join(save_folder, f"reverse_val*")):
            if os.path.isdir(name):
                tar.add(name, arcname=os.path.basename(name))
                shutil.rmtree(name)


if data_params.get("samp_split", False) or data_params.get("one_model", False):
    if "train" in mat_sp_d:
        tqdm.write(f"Adversarial training for slides {mat_sp_d['train'].keys()}: ")
        save_folder = os.path.join(advtrain_folder, "samp_split")
    else:
        tqdm.write(f"Adversarial training for slides {next(iter(mat_sp_d.values()))}: ")
        save_folder = os.path.join(advtrain_folder, "one_model")

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    load_train_plot(
        pretrain_folder,
        save_folder,
        dataloader_source_d["train"],
        dataloader_source_d["val"],
        dataloader_target_d["train"],
        dataloader_target_d["train_dis"],
        dataloader_target_d.get("val", None),
        checkpoints=True,
    )

    if train_params["reverse_val"]:
        tqdm.write(f"Reverse validating ...")

        reverse_val(
            target_d,
            save_folder,
        )

    with tarfile.open(os.path.join(save_folder, f"checkpts.tar.gz"), "w:gz") as tar:
        for name in glob.glob(os.path.join(save_folder, f"checkpt*.pth")):
            tar.add(name, arcname=os.path.basename(name))
            os.unlink(name)

else:
    for sample_id in st_sample_id_l:
        tqdm.write(f"Adversarial training for ST slide {sample_id}: ")

        save_folder = os.path.join(advtrain_folder, sample_id)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        load_train_plot(
            pretrain_folder,
            save_folder,
            dataloader_source_d["train"],
            dataloader_source_d["val"],
            dataloader_target_d[sample_id]["train"],
            dataloader_target_d[sample_id]["train_dis"],
            dataloader_target_d[sample_id].get("val", None),
            checkpoints=True,
        )

        if train_params["reverse_val"]:
            tqdm.write(f"Reverse validating ...")

            reverse_val(
                target_d[sample_id],
                save_folder,
            )

        with tarfile.open(os.path.join(save_folder, f"checkpts.tar.gz"), "w:gz") as tar:
            for name in glob.glob(os.path.join(save_folder, f"checkpt*.pth")):
                tar.add(name, arcname=os.path.basename(name))
                os.unlink(name)


# %%
with open(os.path.join(model_folder, "config.yml"), "w") as f:
    yaml.safe_dump(config, f)

temp_folder_holder.copy_out()
