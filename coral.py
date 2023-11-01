#!/usr/bin/env python3
"""Creating something like CellDART but just using coral loss"""

# %% [markdown]
#   # CORAL for ST

# %% [markdown]
#   Creating something like CellDART but just using coral loss

# %%
import argparse
import datetime
import glob
import os
import pickle

import tarfile
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from tqdm.autonotebook import tqdm

from src.da_models.coral import CORAL
from src.da_models.model_utils.datasets import SpotDataset
from src.da_models.model_utils.losses import coral_loss
from src.da_models.model_utils.utils import (
    LibConfig,
    ModelWrapper,
    get_torch_device,
    initialize_weights,
)
from src.da_utils import data_loading, evaluation
from src.da_utils.output_utils import DupStdout, TempFolderHolder

# datetime object containing current date and time
script_start_time = datetime.datetime.now(datetime.timezone.utc)


# %%
parser = argparse.ArgumentParser(
    description="Creating something like CellDART but just using coral loss"
)
parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
parser.add_argument("--configs_dir", "-cdir", type=str, default="configs", help="config dir")
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers to use for dataloaders."
)
parser.add_argument("--cuda", "-c", default=None, help="gpu index to use")
parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary model directory")
parser.add_argument("--log_fname", "-l", default=None, help="optional log file name")
parser.add_argument("--num_threads", "-t", default=None, help="number of threads to use")
parser.add_argument("--model_dir", default="model", help="model directory")
parser.add_argument(
    "--seed_override",
    default=None,
    help="seed to use for torch and numpy; overrides that in config file",
)
parser.add_argument(
    "--ps_seed",
    default=-1,
    help="specific pseudospot seed to use; default of -1 corresponds to 623",
)

# %%
args = parser.parse_args()
PS_SEED = int(args.ps_seed)
MODEL_DIR = args.model_dir
SEED_OVERRIDE = args.seed_override
CONFIG_FNAME = args.config_fname
CONFIGS_DIR = args.configs_dir
CUDA_INDEX = args.cuda
NUM_WORKERS = args.num_workers
TMP_DIR = args.tmpdir
LOG_FNAME = args.log_fname
NUM_THREADS = int(args.num_threads) if args.num_threads else None
MIN_EVAL_BS = 512

# %%
# CONFIG_FNAME = "coral.yml"
# CUDA_INDEX = None
# NUM_WORKERS = 0
# TMP_DIR = None
# LOG_FNAME = "log.txt"
# %%
# lib_params = {}

# lib_params["manual_seed"] = 3583


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
# data_params["scaler_name"] = "standard"


# %%
MODEL_NAME = "CORAL"

# model_params = {}

# # Model parameters

# model_params["model_version"] = "v1"

# model_params["coral_kwargs"] = {
#     "enc_hidden_layer_sizes": (1024, 512, 64),
#     "hidden_act": "leakyrelu",
#     "dropout": 0.5,
#     "batchnorm": True,
#     "batchnorm_after_act": True,
# }


# %%
# train_params = {}

# train_params["batch_size"] = 512

# # Pretraining parameters
# # SAMPLE_ID_N = "151673"

# # train_params["initial_train_epochs"] = 100

# # train_params["early_stop_crit"] = 100
# # train_params["min_epochs"] = 0.4 * train_params["initial_train_epochs"]

# # Adversarial training parameters
# train_params["epochs"] = 100
# train_params["early_stop_crit_adv"] = train_params["epochs"]
# train_params["min_epochs_adv"] = 0.4 * train_params["epochs"]


# # train_params["enc_lr"] = 0.0002
# # train_params["alpha"] = 2
# # train_params["dis_loop_factor"] = 5
# # train_params["adam_beta1"] = 0.5
# train_params["lambda"] = 100
# train_params["opt_kwargs"] = {"lr": 0.001, "weight_decay": 0.3}
# train_params["two_step"] = True


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

tqdm.write(yaml.safe_dump(config))

lib_params = config["lib_params"]
data_params = config["data_params"]
model_params = config["model_params"]
train_params = config["train_params"]


# %%
if NUM_THREADS:
    torch.set_num_threads(NUM_THREADS)

device = get_torch_device(CUDA_INDEX)
ModelWrapper.configurate(LibConfig(device, CUDA_INDEX))

# %%
if SEED_OVERRIDE is None:
    torch_seed = lib_params.get("manual_seed", int(script_start_time.timestamp()))
    lib_seed_path = str(torch_seed) if "manual_seed" in lib_params else "random"
else:
    torch_seed = int(SEED_OVERRIDE)
    lib_seed_path = str(torch_seed)

torch.manual_seed(torch_seed)
np.random.seed(torch_seed)


# %%
model_folder = data_loading.get_model_rel_path(
    MODEL_NAME,
    model_params["model_version"],
    lib_seed_path=lib_seed_path,
    **data_params,
)
model_folder = os.path.join(MODEL_DIR, model_folder)
temp_folder_holder = TempFolderHolder()
model_folder = temp_folder_holder.set_output_folder(TMP_DIR, model_folder)


# %% [markdown]
#   # Data load

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
    seed_int=PS_SEED,
    **data_params,
)


# %% [markdown]
#   # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#   ## Prepare dataloaders

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


# %% [markdown]
#   ## Pretrain

# %%
criterion_clf = nn.KLDivLoss(reduction="batchmean")

to_inp_kwargs = dict(device=device, dtype=torch.float32, non_blocking=True)


# %%
def model_loss(x, y_true, model):
    x = x.to(**to_inp_kwargs)
    y_true = y_true.to(**to_inp_kwargs)

    y_pred, _ = model(x)
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
    lr_running = []
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
                lr_running.append(scheduler.get_last_lr()[-1])
                scheduler.step()
        if inner:
            inner.update(1)
    return loss_running, mean_weights, lr_running


def compute_acc(dataloader, model):
    model.eval()
    with torch.no_grad():
        loss_running, mean_weights, _ = run_pretrain_epoch(model, dataloader)

    return np.average(loss_running, weights=mean_weights)


# %%
def pretrain(
    pretrain_folder,
    model,
    dataloader_source_train,
    dataloader_source_val,
):
    pre_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params["initial_train_lr"],
        betas=(0.9, 0.999),
        eps=1e-07,
    )

    pre_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        pre_optimizer,
        max_lr=train_params["initial_train_lr"],
        steps_per_epoch=len(dataloader_source_d["train"]),
        epochs=train_params["initial_train_epochs"],
    )

    model.pretraining()

    if not os.path.isdir(pretrain_folder):
        os.makedirs(pretrain_folder)
    # Initialize lists to store loss and accuracy values
    loss_history = []
    loss_history_val = []

    loss_history_running = []

    lr_history_running = []

    # Early Stopping
    best_loss_val = np.inf
    early_stop_count = 0

    # Train
    log_file_path = os.path.join(pretrain_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        tqdm.write("Start pretrain...")
        outer = tqdm(total=train_params["initial_train_epochs"], desc="Epochs", position=0)
        inner = tqdm(total=len(dataloader_source_d["train"]), desc=f"Batch", position=1)

        tqdm.write(" Epoch | Train Loss | Val Loss   | Next LR    ")
        tqdm.write("----------------------------------------------")
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

            loss_running, mean_weights, lr_running = run_pretrain_epoch(
                model,
                dataloader_source_d["train"],
                optimizer=pre_optimizer,
                scheduler=pre_scheduler,
                inner=inner,
            )

            loss_history.append(np.average(loss_running, weights=mean_weights))
            loss_history_running.append(loss_running)
            lr_history_running.append(lr_running)

            # Evaluate mode
            model.eval()
            with torch.no_grad():
                curr_loss_val = compute_acc(dataloader_source_d["val"], model)
                loss_history_val.append(curr_loss_val)

            # Print the results
            outer.update(1)

            out_string = (
                f" {epoch:5d} "
                f"| {loss_history[-1]:<10.8f} "
                f"| {curr_loss_val:<10.8f} "
                f"| {pre_scheduler.get_last_lr()[-1]:<10.5} "
            )

            # Save the best weights
            if curr_loss_val < best_loss_val:
                best_loss_val = curr_loss_val
                torch.save(checkpoint, os.path.join(pretrain_folder, f"best_model.pth"))
                early_stop_count = 0

                out_string += "<-- new best val loss"

            tqdm.write(out_string)

            # Save checkpoint every 10
            # if epoch % 10 == 0 or epoch >= train_params["initial_train_epochs"] - 1:
            #     torch.save(checkpoint, os.path.join(pretrain_folder, f"checkpt{epoch}.pth"))

            # check to see if validation loss has plateau'd
            if (
                early_stop_count >= train_params["early_stop_crit"]
                and epoch >= train_params["min_epochs"] - 1
            ):
                tqdm.write(f"Validation loss plateaued after {early_stop_count} at epoch {epoch}")
                torch.save(checkpoint, os.path.join(pretrain_folder, f"earlystop{epoch}.pth"))
                break

            early_stop_count += 1

    lr_history_running[-1].append(pre_scheduler.get_last_lr()[-1])

    # Save final model
    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"best_model.pth"))
    torch.save(best_checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))
    return loss_history, loss_history_val, loss_history_running, lr_history_running


pretrain_folder = os.path.join(model_folder, "pretrain")
if train_params.get("pretraining", False):
    if not os.path.isdir(pretrain_folder):
        os.makedirs(pretrain_folder)

    model = CORAL(
        inp_dim=sc_mix_d["train"].shape[1],
        ncls_source=lab_mix_d["train"].shape[1],
        **model_params["coral_kwargs"],
    )
    model.apply(initialize_weights)
    model.to(device)

    loss_history, loss_history_val, loss_history_running, lr_history_running = pretrain(
        pretrain_folder,
        model,
        dataloader_source_d["train"],
        dataloader_source_d["val"],
    )


# %%
if train_params.get("pretraining", False):
    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))

    best_epoch = best_checkpoint["epoch"]
    best_loss_val = loss_history_val[best_epoch]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 4), layout="constrained")

    axs[0].plot(
        *evaluation.format_iters(loss_history_running),
        label="Training",
        linewidth=0.5,
    )
    axs[0].plot(loss_history_val, label="Validation")
    axs[0].axvline(best_epoch, color="tab:green")

    axs[0].set_ylim(bottom=0)
    axs[0].grid(which="major")
    axs[0].minorticks_on()
    axs[0].grid(which="minor", alpha=0.2)

    axs[0].text(
        x=best_epoch + (2 if best_epoch < len(loss_history) * 0.75 else -2),
        y=max(loss_history + loss_history_val) * 0.5,
        s=f"Best val. loss:\n{best_loss_val:.4f} at epoch {best_epoch}",
        ha="left" if best_epoch < len(loss_history) * 0.75 else "right",
        size="x-small",
    )

    # axs[0].set_xlabel("Epoch")
    axs[0].set_title("Cross-Entropy Loss")
    axs[0].legend()

    # lr history
    iters_by_epoch, lr_history_running_flat = evaluation.format_iters(
        lr_history_running, startpoint=True
    )
    axs[1].plot(iters_by_epoch, lr_history_running_flat)
    axs[1].axvline(best_checkpoint["epoch"], ymax=2, clip_on=False, color="tab:green")

    axs[1].set_ylim(bottom=0)
    axs[1].grid(which="major")
    axs[1].minorticks_on()
    axs[1].grid(which="minor", alpha=0.2)

    best_epoch_idx = np.where(iters_by_epoch == best_epoch)[0][0]
    axs[1].text(
        x=best_epoch + (2 if best_epoch < len(loss_history) * 0.75 else -2),
        y=np.median(lr_history_running_flat),
        s=f"LR:\n{lr_history_running_flat[best_epoch_idx]:.4} at epoch {best_epoch}",
        ha="left" if best_epoch < len(loss_history) * 0.75 else "right",
        size="x-small",
    )

    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Learning Rate")

    plt.savefig(os.path.join(pretrain_folder, "train_plots.png"), bbox_inches="tight")

    plt.show(block=False)


# %% [markdown]
#   ## Adversarial Adaptation

# %%
advtrain_folder = os.path.join(model_folder, "advtrain")

if not os.path.isdir(advtrain_folder):
    os.makedirs(advtrain_folder)


# %%
criterion_dis = coral_loss  # lambda s, t: coral_loss(torch.exp(s), torch.exp(t))

loss_lambdas = train_params["lambda"]
try:
    iter(loss_lambdas)
except TypeError:
    loss_lambdas = [0, loss_lambdas]


# %%
def model_adv_loss(
    x_source,
    x_target,
    y_source,
    model,
    two_step=False,
    source_first=True,
    optimizer=None,
):
    if two_step:
        if source_first:
            y_pred_source, logits_source = model(x_source)
            _, logits_target = model(x_target)
        else:
            _, logits_target = model(x_target)
            y_pred_source, logits_source = model(x_source)
        logits = zip(logits_source, logits_target)
    else:
        y_pred, logits = model(torch.cat([x_source, x_target], dim=0))
        y_pred_source, _ = torch.split(y_pred, [len(x_source), len(x_target)])
        logits = [torch.split(logit, [len(x_source), len(x_target)]) for logit in logits]

    loss_clf = criterion_clf(y_pred_source, y_source)
    loss_dis = []
    for logit_source, logit_target in logits:
        loss_dis.append(criterion_dis(logit_source, logit_target))

    loss_dis = loss_dis[0] * loss_lambdas[0] + loss_dis[1] * loss_lambdas[1]
    loss = loss_clf + loss_dis
    update_weights(optimizer, loss)

    return loss, loss_dis, loss_clf


def update_weights(optimizer, loss):
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run_epoch(
    dataloader_source,
    dataloader_target,
    model,
    tqdm_bar=None,
    scheduler=None,
    iter_override=None,
    **kwargs,
):
    results_running = {
        "clf": {"loss": [], "weights": []},
        "dis": {"loss": [], "weights": []},
        "ovr": {"loss": [], "weights": []},
    }

    if scheduler is not None:
        results_running["ovr"]["lr"] = []

    n_iters = len(dataloader_target)
    if iter_override is not None:
        n_iters = min(n_iters, iter_override)

    s_iter = iter(dataloader_source)
    t_iter = iter(dataloader_target)
    for i in range(n_iters):
        try:
            x_source, y_source = next(s_iter)
        except StopIteration:
            s_iter = iter(dataloader_source)
            x_source, y_source = next(s_iter)
        try:
            x_target, _ = next(t_iter)
        except StopIteration:
            t_iter = iter(dataloader_target)
            x_target, _ = next(t_iter)

        x_source = x_source.to(**to_inp_kwargs)
        x_target = x_target.to(**to_inp_kwargs)
        y_source = y_source.to(**to_inp_kwargs)

        loss, loss_dis, loss_clf = model_adv_loss(x_source, x_target, y_source, model, **kwargs)

        results_running["dis"]["loss"].append(loss_dis.item())
        results_running["clf"]["loss"].append(loss_clf.item())
        results_running["ovr"]["loss"].append(loss.item())

        results_running["dis"]["weights"].append((len(x_source) + len(x_target)) / 2)
        results_running["clf"]["weights"].append(len(x_source))
        results_running["ovr"]["weights"].append(
            (len(x_source) + len(x_target)) / 2 + len(x_source)
        )
        if scheduler is not None:
            results_running["ovr"]["lr"].append(scheduler.get_last_lr()[-1])
            scheduler.step()

        if tqdm_bar is not None:
            tqdm_bar.update(1)

    return results_running


# %%
def train_adversarial_iters(
    model,
    save_folder,
    dataloader_source_train,
    dataloader_source_val,
    dataloader_target_train,
    dataloader_target_val=None,
    checkpoints=False,
    iter_override=None,
):
    if dataloader_target_val is None:
        dataloader_target_val = dataloader_target_train

    model.to(device)
    model.advtraining()

    max_len_dataloader = len(dataloader_target_train)
    if iter_override is not None:
        max_len_dataloader = min(max_len_dataloader, iter_override)

    optimizer = torch.optim.AdamW(model.parameters(), **train_params["opt_kwargs"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_params["opt_kwargs"]["lr"],
        steps_per_epoch=max_len_dataloader,
        epochs=train_params["epochs"],
    )

    results_template = {
        "clf": {"loss": [], "weights": []},
        "dis": {"loss": [], "weights": []},
        "ovr": {"loss": [], "lr": [], "weights": []},
    }
    results_history = deepcopy(results_template)
    results_history_val = deepcopy(results_template)
    results_history_running = deepcopy(results_template)

    # Early Stopping
    best_loss_val = np.inf
    early_stop_count = 0
    log_file_path = os.path.join(save_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        # Train
        tqdm.write("Start adversarial training...")
        outer = tqdm(total=train_params["epochs"], desc="Epochs", position=0)
        inner = tqdm(total=max_len_dataloader, desc=f"Batch", position=1)

        tqdm.write(" Epoch || KLDiv.          || Coral           || Overall         || Next LR    ")
        tqdm.write("       || Train  | Val.   || Train  | Val.   || Train  | Val.   ||            ")
        tqdm.write("------------------------------------------------------------------------------")
        checkpoint = {
            "epoch": -1,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        for epoch in range(train_params["epochs"]):
            inner.refresh()  # force print final state
            inner.reset()  # reuse bar

            checkpoint["epoch"] = epoch

            results_running = run_epoch(
                dataloader_source_train,
                dataloader_target_train,
                model,
                tqdm_bar=inner,
                optimizer=optimizer,
                scheduler=scheduler,
                iter_override=iter_override,
                two_step=train_params.get("two_step", False),
                source_first=train_params.get("source_first", True),
            )

            evaluation.recurse_avg_dict(results_running, results_history)
            evaluation.recurse_running_dict(results_running, results_history_running)

            model.eval()
            with torch.no_grad():
                results_val = run_epoch(dataloader_source_val, dataloader_target_val, model)

            evaluation.recurse_avg_dict(results_val, results_history_val)

            # Print the results
            outer.update(1)

            out_string = (
                f" {epoch:5d} "
                f"|| {results_history['clf']['loss'][-1]:6.4f} "
                f"| {results_history_val['clf']['loss'][-1]:6.4f} "
                f"|| {results_history['dis']['loss'][-1]:6.4f} "
                f"| {results_history_val['dis']['loss'][-1]:6.4f} "
                f"|| {results_history['ovr']['loss'][-1]:6.4f} "
                f"| {results_history_val['ovr']['loss'][-1]:6.4f} "
                f"|| {scheduler.get_last_lr()[-1]:<10.5} "
            )

            better_val_loss = results_history_val["ovr"]["loss"][-1] < best_loss_val
            if better_val_loss:
                best_loss_val = results_history_val["ovr"]["loss"][-1]
                torch.save(checkpoint, os.path.join(save_folder, f"best_model.pth"))
                early_stop_count = 0
                out_string += f"<-- new best val loss"

            tqdm.write(out_string)

            early_stop_count += 1

    inner.close()
    outer.close()

    results_history_running["ovr"]["lr"][-1].append(scheduler.get_last_lr()[-1])
    # Save final model
    best_checkpoint = torch.load(os.path.join(save_folder, f"best_model.pth"))
    torch.save(best_checkpoint, os.path.join(save_folder, f"final_model.pth"))

    results_history_out = {
        "train": {"avg": results_history, "running": results_history_running},
        "val": {"avg": results_history_val},
    }
    # Save results
    with open(os.path.join(save_folder, f"results_history_out.pkl"), "wb") as f:
        pickle.dump(results_history_out, f)

    return results_history_out, best_checkpoint


# %%
def plot_results(save_folder, results_history_out=None):
    if results_history_out is None:
        with open(os.path.join(save_folder, f"results_history_out.pkl"), "rb") as f:
            results_history_out = pickle.load(results_history_out, f)
    results_history = results_history_out["train"]["avg"]
    results_history_running = results_history_out["train"]["running"]
    results_history_val = results_history_out["val"]["avg"]

    n_epochs = len(results_history["ovr"]["loss"])
    best_checkpoint = torch.load(os.path.join(save_folder, f"final_model.pth"))
    best_epoch = best_checkpoint["epoch"]

    if best_epoch < 0:
        return  # no training happened

    best_coral_loss = results_history_val["dis"]["loss"][best_epoch]
    best_kld_loss = results_history_val["clf"]["loss"][best_epoch]
    best_overall_loss = results_history_val["ovr"]["loss"][best_epoch]

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(9, 12), layout="constrained")

    x_text = best_epoch + (2 if best_epoch < n_epochs * 0.75 else -2)
    ha_text = "left" if best_epoch < n_epochs * 0.75 else "right"

    # Coral
    axs[0].plot(
        *evaluation.format_iters(results_history_running["dis"]["loss"]),
        label="training",
        linewidth=0.5,
    )
    axs[0].plot(results_history_val["dis"]["loss"], label="validation")
    axs[0].axvline(best_epoch, color="tab:green")

    axs[0].set_ylim(bottom=0, top=max(results_history_val["dis"]["loss"]))
    axs[0].grid(which="major")
    axs[0].minorticks_on()
    axs[0].grid(which="minor", alpha=0.2)
    axs[0].text(
        x=x_text,
        y=max(results_history_val["dis"]["loss"]) * 0.5,
        s=f"CORAL val. loss:\n{best_coral_loss:.4f} at epoch {best_epoch}",
        ha=ha_text,
        size="x-small",
    )
    axs[0].set_title("CORAL Loss")
    axs[0].legend()

    # KLDiv
    axs[1].plot(
        *evaluation.format_iters(results_history_running["clf"]["loss"]),
        label="training",
        linewidth=0.5,
    )
    axs[1].plot(results_history_val["clf"]["loss"], label="validation")
    axs[1].axvline(best_epoch, ymax=2, clip_on=False, color="tab:green")

    axs[1].set_ylim(bottom=0, top=max(results_history_val["clf"]["loss"]))
    axs[1].grid(which="major")
    axs[1].minorticks_on()
    axs[1].grid(which="minor", alpha=0.2)
    axs[1].text(
        x=x_text,
        y=max(results_history_val["clf"]["loss"]) * 0.5,
        s=f"KLD val. loss:\n{best_kld_loss:.4f} at epoch {best_epoch}",
        ha=ha_text,
        size="x-small",
    )
    axs[1].set_title("KL-Divergence Loss")
    axs[1].legend()

    # Overall
    axs[2].plot(
        *evaluation.format_iters(results_history_running["ovr"]["loss"]),
        label="training",
        linewidth=0.5,
    )
    axs[2].plot(results_history_val["ovr"]["loss"], label="validation")
    axs[2].axvline(best_epoch, ymax=2, clip_on=False, color="tab:green")

    axs[2].set_ylim(bottom=0, top=max(results_history_val["ovr"]["loss"]))
    axs[2].grid(which="major")
    axs[2].minorticks_on()
    axs[2].grid(which="minor", alpha=0.2)
    axs[2].text(
        x=x_text,
        y=max(results_history_val["ovr"]["loss"]) * 0.5,
        s=f"Best overall val. loss:\n{best_overall_loss:.4f} at epoch {best_epoch}",
        ha=ha_text,
        size="x-small",
    )
    axs[2].set_title("Overall Loss")
    axs[2].legend()

    # lr history
    iters_by_epoch, lr_history_running_flat = evaluation.format_iters(
        results_history_running["ovr"]["lr"], startpoint=True
    )
    axs[3].plot(iters_by_epoch, lr_history_running_flat)
    axs[3].axvline(best_epoch, ymax=2, clip_on=False, color="tab:green")

    axs[3].set_ylim(bottom=0)
    axs[3].grid(which="major")
    axs[3].minorticks_on()
    axs[3].grid(which="minor", alpha=0.2)

    best_epoch_idx = np.where(iters_by_epoch == best_epoch)[0][0]
    axs[3].text(
        x=x_text,
        y=np.median(lr_history_running_flat),
        s=f"LR:\n{lr_history_running_flat[best_epoch_idx]:.4} at epoch {best_epoch}",
        ha=ha_text,
        size="x-small",
    )

    axs[3].set_xlabel("Epoch")
    axs[3].set_title("Learning Rate")

    if save_folder is not None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, "adv_train.png"))

    plt.show(block=False)


# %%
def load_train_plot(
    pretrain_folder,
    save_folder,
    dataloader_source_train,
    dataloader_source_val,
    dataloader_target_train,
    dataloader_target_val,
    **adv_train_kwargs,
):
    model = CORAL(
        sc_mix_d["train"].shape[1],
        ncls_source=lab_mix_d["train"].shape[1],
        **model_params["coral_kwargs"],
    )
    model.apply(initialize_weights)

    if train_params.get("pretraining", False):
        best_pre_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
        model.load_state_dict(best_pre_checkpoint["model"].state_dict())
    model.to(device)

    model.advtraining()

    if adv_train_kwargs.get("checkpoints", False):
        tqdm.write(repr(model))

    results, checkpoint = train_adversarial_iters(
        model,
        save_folder,
        dataloader_source_train,
        dataloader_source_val,
        dataloader_target_train,
        dataloader_target_val,
        **adv_train_kwargs,
    )
    plot_results(save_folder, results)

    return checkpoint


def reverse_val(
    target_d,
    save_folder,
):
    model = ModelWrapper(save_folder)
    best_epoch = model.epoch
    dataloader_target_now_source_d = {}
    pred_target_d = OrderedDict()
    for split in target_d:
        pred_target_d[split] = model.get_predictions(target_d[split])

        dataloader_target_now_source_d[split] = torch.utils.data.DataLoader(
            SpotDataset(target_d[split], pred_target_d[split]),
            shuffle=("train" in split),
            batch_size=train_params.get("batch_size", MIN_EVAL_BS) if "train" in split else alt_bs,
            **source_dataloader_kwargs,
        )

    if train_params.get("pretraining", False):
        rv_pretrain_folder = os.path.join(save_folder, f"reverse_val-pretrain")
        if not os.path.isdir(rv_pretrain_folder):
            os.makedirs(rv_pretrain_folder)
        model = CORAL(
            inp_dim=sc_mix_d["train"].shape[1],
            ncls_source=lab_mix_d["train"].shape[1],
            **model_params["coral_kwargs"],
        )
        model.apply(initialize_weights)
        model.to(device)

        pretrain(
            rv_pretrain_folder,
            model,
            dataloader_target_now_source_d["train"],
            dataloader_target_now_source_d.get("val", None),
        )
    else:
        rv_pretrain_folder = pretrain_folder

    rv_save_folder = os.path.join(save_folder, f"reverse_val")
    if not os.path.isdir(rv_save_folder):
        os.makedirs(rv_save_folder)

    checkpoint = load_train_plot(
        model,
        rv_save_folder,
        dataloader_target_now_source_d["train"],
        dataloader_target_now_source_d.get("val", dataloader_target_now_source_d["train"]),
        dataloader_source_d["train"],
        dataloader_source_d["val"],
        iter_override=len(dataloader_target_now_source_d["train"]),
    )

    # model = ModelWrapper(model)
    rv_scores_d = OrderedDict()
    for split in ("train", "val"):
        rv_scores_d[split] = compute_acc(
            dataloader_source_d[split],
            checkpoint["model"],
        )

    rv_scores_df = pd.DataFrame(rv_scores_d, index=[model_params["model_version"]])

    # rv_scores_df.index = pd.Index([model_params["model_version"]])
    rv_scores_df["best_epoch"] = best_epoch
    rv_scores_df["best_epoch_rv"] = checkpoint["epoch"]
    rv_scores_df["config_fname"] = CONFIG_FNAME

    tqdm.write("Best Epoch: ")
    tqdm.write(repr(rv_scores_df))

    rv_scores_df.to_csv(os.path.join(save_folder, f"reverse_val_best_epoch.csv"))


if data_params.get("samp_split") or data_params.get("one_model"):
    if data_params.get("samp_split"):
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
        tqdm.write(f"Adversarial training for ST slide {sample_id}:")

        save_folder = os.path.join(advtrain_folder, sample_id)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        load_train_plot(
            pretrain_folder,
            save_folder,
            dataloader_source_d["train"],
            dataloader_source_d["val"],
            dataloader_target_d[sample_id]["train"],
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

# %%
tqdm.write(f"Script run time: {datetime.datetime.now(datetime.timezone.utc) - script_start_time}")
