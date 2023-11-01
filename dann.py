#!/usr/bin/env python3
"""Creating something like CellDART but it actually follows DANN in PyTorch"""

# %% [markdown]
#  # DANN for ST

# %% [markdown]
#  Creating something like CellDART but it actually follows DANN in PyTorch

import argparse
import datetime
import glob
import os
import pickle
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
from tqdm.autonotebook import tqdm

from src.da_models.dann import DANN
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

# datetime object containing current date and time
script_start_time = datetime.datetime.now(datetime.timezone.utc)


# %%
parser = argparse.ArgumentParser(
    description=("Creating something like CellDART but it actually follows DANN in PyTorch")
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

# CUDA_INDEX = 2
# NUM_WORKERS = 4
# CONFIG_FNAME = "dann.yml"

# %%
# lib_params = {}
# lib_params["manual_seed"] = 25098

# CUDA_INDEX = 2
# NUM_WORKERS = 4
# CONFIG_FNAME = "dann.yml"

MODEL_NAME = "DANN"

with open(os.path.join(CONFIGS_DIR, MODEL_NAME, CONFIG_FNAME), "r") as f:
    config = yaml.safe_load(f)


print(yaml.safe_dump(config))

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
    seed_int=PS_SEED,
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
#  ## Pretrain

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


def compute_acc(dataloader, model):
    loss_running = []
    mean_weights = []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            loss = model_loss(*batch, model)

            loss_running.append(loss.item())

            # we will weight average by batch size later
            mean_weights.append(len(batch))

    return np.average(loss_running, weights=mean_weights)


# %%
def pretrain(
    pretrain_folder,
    model,
    dataloader_source_train,
    dataloader_source_val=None,
):
    if dataloader_source_val is None:
        dataloader_source_val = dataloader_source_train

    pre_optimizer = torch.optim.AdamW(
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

    lr_history_running = []

    # Train
    log_file_path = os.path.join(pretrain_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        tqdm.write("Start pretrain...")
        outer = tqdm(total=train_params["initial_train_epochs"], desc="Epochs")
        inner = tqdm(total=len(dataloader_source_train), desc=f"Batch")

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
            checkpoint["epoch"] = epoch

            # Train mode
            model.train()
            loss_running = []
            mean_weights = []
            lr_running = []

            inner.refresh()  # force print final state
            inner.reset()  # reuse bar
            for _, batch in enumerate(dataloader_source_train):
                lr_running.append(pre_scheduler.get_last_lr()[-1])

                pre_optimizer.zero_grad()
                loss = model_loss(*batch, model)
                loss_running.append(loss.item())
                mean_weights.append(len(batch))  # we will weight average by batch size later

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                loss.backward()
                pre_optimizer.step()
                pre_scheduler.step()

                inner.update(1)

            loss_history.append(np.average(loss_running, weights=mean_weights))
            loss_history_running.append(loss_running)
            lr_history_running.append(lr_running)

            # Evaluate mode
            model.eval()
            with torch.no_grad():
                curr_loss_val = compute_acc(dataloader_source_val, model)
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

                out_string += "<-- new best val loss"

            tqdm.write(out_string)

    inner.close()
    outer.close()
    lr_history_running[-1].append(pre_scheduler.get_last_lr()[-1])

    # Save final model
    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"best_model.pth"))
    torch.save(best_checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))
    return (
        loss_history,
        loss_history_val,
        loss_history_running,
        lr_history_running,
    ), best_checkpoint


pretrain_folder = os.path.join(model_folder, "pretrain")

if train_params["pretraining"]:
    if not os.path.isdir(pretrain_folder):
        os.makedirs(pretrain_folder)

    model = DANN(
        inp_emb=sc_mix_d["train"].shape[1],
        ncls_source=lab_mix_d["train"].shape[1],
        **model_params["dann_kwargs"],
    )
    model.apply(initialize_weights)
    model.to(device)

    (
        loss_history,
        loss_history_val,
        loss_history_running,
        lr_history_running,
    ), _ = pretrain(
        pretrain_folder,
        model,
        dataloader_source_d["train"],
        dataloader_source_d["val"],
    )


# %%
if train_params["pretraining"]:
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
#  ## Adversarial Adaptation

# %%
advtrain_folder = os.path.join(model_folder, "advtrain")

if not os.path.isdir(advtrain_folder):
    os.makedirs(advtrain_folder)


# %%
criterion_dis = nn.BCEWithLogitsLoss()


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
            y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source = source_step(
                x_source, y_source, model, optimizer
            )
            y_dis_target, y_dis_target_pred, loss_dis_target = target_step(
                x_target, model, optimizer
            )
        else:
            y_dis_target, y_dis_target_pred, loss_dis_target = target_step(
                x_target, model, optimizer
            )
            y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source = source_step(
                x_source, y_source, model, optimizer
            )
    else:
        y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source = source_pass(
            x_source, y_source, model
        )
        y_dis_target, y_dis_target_pred, loss_dis_target = target_pass(x_target, model)
        loss = loss_clf + (loss_dis_source + loss_dis_target) * train_params["lambda"]
        update_weights(optimizer, loss)

    accu_source = logits_to_accu(y_dis_source_pred, y_dis_source)
    accu_target = logits_to_accu(y_dis_target_pred, y_dis_target)

    return (loss_dis_source, loss_dis_target, loss_clf), (accu_source, accu_target)


def logits_to_accu(y_pred, y_true):
    accu = (
        (torch.round(torch.sigmoid(y_pred.detach())).to(torch.long) == y_true.detach())
        .to(torch.float32)
        .mean()
        .cpu()
    )

    return accu


def target_step(x_target, model, optimizer):
    if optimizer is not None:
        clf_rq_bak = dict(
            ((name, param.requires_grad) for name, param in model.clf.named_parameters())
        )
        set_requires_grad(model.clf, False)

    y_dis_target, y_dis_target_pred, loss_dis_target = target_pass(x_target, model)
    loss = loss_dis_target * train_params["lambda"]

    if optimizer is not None:
        update_weights(optimizer, loss)
        for name, param in model.clf.named_parameters():
            param.requires_grad = clf_rq_bak[name]

    return y_dis_target, y_dis_target_pred, loss_dis_target


def source_step(x_source, y_source, model, optimizer):
    y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source = source_pass(
        x_source, y_source, model
    )
    loss = loss_clf + loss_dis_source * train_params["lambda"]
    update_weights(optimizer, loss)
    return y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source


def update_weights(optimizer, loss):
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def target_pass(x_target, model):
    y_dis_target = torch.ones(x_target.shape[0], device=device, dtype=x_target.dtype).view(-1, 1)
    _, y_dis_target_pred = model(x_target, clf=False)
    loss_dis_target = criterion_dis(y_dis_target_pred, y_dis_target)
    return y_dis_target, y_dis_target_pred, loss_dis_target


def source_pass(x_source, y_source, model):
    y_dis_source = torch.zeros(x_source.shape[0], device=device, dtype=x_source.dtype).view(-1, 1)
    y_clf, y_dis_source_pred = model(x_source, clf=True)
    loss_clf = criterion_clf(y_clf, y_source)
    loss_dis_source = criterion_dis(y_dis_source_pred, y_dis_source)
    return y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source


def run_epoch(
    dataloader_source,
    dataloader_target,
    model,
    tqdm_bar=None,
    iter_override=None,
    **kwargs,
):
    source_results = {}
    target_results = {}

    source_results["clf_loss"] = []
    source_results["dis_loss"] = []
    source_results["dis_accu"] = []
    source_results["weights"] = []

    target_results["dis_loss"] = []
    target_results["dis_accu"] = []
    target_results["weights"] = []

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

        (
            (loss_dis_source, loss_dis_target, loss_clf),
            (accu_source, accu_target),
        ) = model_adv_loss(x_source, x_target, y_source, model, **kwargs)

        source_results["dis_loss"].append(loss_dis_source.item())
        target_results["dis_loss"].append(loss_dis_target.item())
        source_results["clf_loss"].append(loss_clf.item())

        source_results["dis_accu"].append(accu_source)
        target_results["dis_accu"].append(accu_target)

        source_results["weights"].append(len(x_source))
        target_results["weights"].append(len(x_target))

        if tqdm_bar is not None:
            tqdm_bar.update(1)

    return source_results, target_results


# %%
def train_adversarial_iters(
    model,
    save_folder,
    dataloader_source_train,
    dataloader_source_val,
    dataloader_target_train,
    dataloader_target_val=None,
    checkpoints=False,
    epoch_override=None,
    iter_override=None,
):
    if dataloader_target_val is None:
        dataloader_target_val = dataloader_target_train
    model.to(device)
    model.advtraining()

    optimizer = torch.optim.AdamW(model.parameters(), **train_params["adv_opt_kwargs"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **train_params["plateau_kwargs"]
    )

    max_len_dataloader = len(dataloader_target_train)

    iters_val = max(len(dataloader_source_val), len(dataloader_target_train))

    # Initialize lists to store loss and accuracy values
    results_running_history_source = {}
    results_running_history_source["clf_loss"] = []
    results_running_history_source["dis_loss"] = []
    results_running_history_source["dis_accu"] = []

    results_running_history_target = {}
    results_running_history_target["dis_loss"] = []
    results_running_history_target["dis_accu"] = []

    results_history_source = {}
    results_history_source["clf_loss"] = []
    results_history_source["dis_loss"] = []
    results_history_source["dis_accu"] = []

    results_history_target = {}
    results_history_target["dis_loss"] = []
    results_history_target["dis_accu"] = []

    results_history_source_val = {}
    results_history_source_val["clf_loss"] = []
    results_history_source_val["dis_loss"] = []
    results_history_source_val["dis_accu"] = []

    results_history_target_val = {}
    results_history_target_val["dis_loss"] = []
    results_history_target_val["dis_accu"] = []

    best_loss_val = np.inf
    best_loss_val_unstable = np.inf
    dis_stable_found = False
    early_stop_count = 0

    epochs = epoch_override if epoch_override is not None else train_params["epochs"]
    log_file_path = os.path.join(save_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        # Train
        tqdm.write("Start adversarial training...")
        outer = tqdm(total=epochs, desc="Epochs", position=0)
        inner = tqdm(total=max_len_dataloader, desc=f"Batch", position=1)
        tqdm.write(" Epoch ||| Predictor       ||| Discriminator ")
        tqdm.write(
            "       ||| Loss            ||| Loss                              || Accuracy      "
        )
        tqdm.write(
            "       ||| Source          ||| Source          | Target          || Source          | Target          "
        )
        tqdm.write(
            "       ||| Train  | Val.   ||| Train  | Val.   | Train  | Val.   || Train  | Val.   | Train  | Val.   "
        )
        tqdm.write(
            "------------------------------------------------------------------------------------------------------"
        )
        checkpoint = {
            "epoch": -1,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        if checkpoints:
            torch.save(
                {"model": model.state_dict(), "epoch": -1},
                os.path.join(save_folder, f"checkpt--1.pth"),
            )
        for epoch in range(epochs):
            checkpoint["epoch"] = epoch

            # Train mode
            model.train()
            inner.refresh()  # force print final state
            inner.reset()  # reuse bar

            source_results, target_results = run_epoch(
                dataloader_source_train,
                dataloader_target_train,
                model,
                tqdm_bar=inner,
                iter_override=iter_override,
                optimizer=optimizer,
                two_step=train_params["two_step"],
                source_first=train_params.get("source_first", True),
            )

            for k in results_running_history_source:
                results_running_history_source[k].append(source_results[k])
            for k in results_running_history_target:
                results_running_history_target[k].append(target_results[k])

            for k in results_history_source:
                results_history_source[k].append(
                    np.average(source_results[k], weights=source_results["weights"])
                )
            for k in results_history_target:
                results_history_target[k].append(
                    np.average(target_results[k], weights=target_results["weights"])
                )

            model.eval()
            with torch.no_grad():
                source_results_val, target_results_val = run_epoch(
                    dataloader_source_val, dataloader_target_val, model
                )

            for k in results_history_source_val:
                results_history_source_val[k].append(
                    np.average(source_results_val[k], weights=source_results_val["weights"])
                )
            for k in results_history_target_val:
                results_history_target_val[k].append(
                    np.average(target_results_val[k], weights=target_results_val["weights"])
                )

            # Print the results
            outer.update(1)

            out_string = (
                f" {epoch:5d} "
                f"||| {results_history_source['clf_loss'][-1]:6.4f} "
                f"| {results_history_source_val['clf_loss'][-1]:6.4f} "
                f"||| {results_history_source['dis_loss'][-1]:6.4f} "
                f"| {results_history_source_val['dis_loss'][-1]:6.4f} "
                f"| {results_history_target['dis_loss'][-1]:6.4f} "
                f"| {results_history_target_val['dis_loss'][-1]:6.4f} "
                f"|| {results_history_source['dis_accu'][-1]:6.4f} "
                f"| {results_history_source_val['dis_accu'][-1]:6.4f} "
                f"| {results_history_target['dis_accu'][-1]:6.4f} "
                f"| {results_history_target_val['dis_accu'][-1]:6.4f} "
            )
            # Save the best weights

            dis_train_accu_stable = (
                results_history_source["dis_accu"][-1] > 0.5
                and results_history_target["dis_accu"][-1] > 0.5
                and results_history_source["dis_accu"][-1] < 0.6
                and results_history_target["dis_accu"][-1] < 0.6
            )
            dis_val_accu_stable = (
                results_history_source_val["dis_accu"][-1] > 0.3
                and results_history_target_val["dis_accu"][-1] > 0.3
                and results_history_source_val["dis_accu"][-1] < 0.8
                and results_history_target_val["dis_accu"][-1] < 0.8
            )
            dis_stable = dis_train_accu_stable and dis_val_accu_stable

            better_val_loss = results_history_source_val["clf_loss"][-1] < best_loss_val
            if (dis_stable and not dis_stable_found) or (
                better_val_loss and (dis_stable or not dis_stable_found)
            ):
                best_loss_val = results_history_source_val["clf_loss"][-1]
                if epoch_override is None:
                    torch.save(checkpoint, os.path.join(save_folder, f"best_model.pth"))
                early_stop_count = 0
                out_string += f"<-- new best {'stable' if dis_stable else 'unstable'} val clf loss"

            tqdm.write(out_string)

            # Save checkpoint every 10
            if (epoch % 10 == 9 or epoch >= train_params["epochs"] - 1) and checkpoints:
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch},
                    os.path.join(save_folder, f"checkpt-{epoch}.pth"),
                )

            early_stop_count += 1

            # Found first stable epoch
            if dis_stable and not dis_stable_found:
                dis_stable_found = True
                # reset scheduler
                scheduler.best = scheduler.mode_worse

            # Only update if dis is stable
            # or if haven't found a stable epoch yet and haven't reached min epochs
            if dis_stable or (epoch < train_params["min_epochs_adv"] and not dis_stable_found):
                scheduler.step(results_history_source_val["clf_loss"][-1])
                # tqdm.write(scheduler.best)
            else:
                scheduler.step(scheduler.mode_worse)

    inner.close()
    outer.close()
    # Save final model
    if epoch_override is None:
        checkpoint = torch.load(os.path.join(save_folder, f"best_model.pth"))
    torch.save(checkpoint, os.path.join(save_folder, f"final_model.pth"))

    return (
        results_running_history_source,
        results_running_history_target,
        results_history_source,
        results_history_target,
        results_history_source_val,
        results_history_target_val,
    ), checkpoint


# %%
def plot_results(
    results_running_history_source,
    results_running_history_target,
    results_history_source,
    results_history_target,
    results_history_source_val,
    results_history_target_val,
    save_folder,
):
    n_epochs = len(results_history_source_val["clf_loss"])
    best_checkpoint = torch.load(os.path.join(save_folder, f"final_model.pth"))
    best_epoch = best_checkpoint["epoch"]
    if best_epoch < 0:
        return  # no training happened
    # best_acc_val = accu_history_val[best_epoch]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 9), layout="constrained")

    # prediction loss
    axs[0].plot(
        *evaluation.format_iters(results_running_history_source["clf_loss"]),
        label="Training",
        ls="-",
        c="tab:blue",
        linewidth=0.5,
        alpha=0.5,
    )
    axs[0].plot(
        results_history_source_val["clf_loss"],
        label="Validation",
        ls="--",
        c="tab:blue",
    )
    axs[0].axvline(best_epoch, color="tab:green")

    axs[0].set_ylim(bottom=0, top=2)
    axs[0].grid(which="major")
    axs[0].minorticks_on()
    axs[0].grid(which="minor", alpha=0.2)

    best_clf_loss_val = results_history_source_val["clf_loss"][best_epoch]
    axs[0].text(
        x=best_epoch + (2 if best_epoch < n_epochs * 0.75 else -2),
        y=1.1,
        s=f"Best clf val. loss:\n{best_clf_loss_val:.4f} at epoch {best_epoch}",
        ha="left" if best_epoch < n_epochs * 0.75 else "right",
        size="x-small",
    )

    axs[0].set_title("Source KL-Divergence Prediction Loss")
    axs[0].legend()

    # discriminator loss
    axs[1].plot(
        *evaluation.format_iters(results_running_history_source["dis_loss"]),
        label="Source train",
        ls="-",
        c="tab:blue",
        linewidth=0.5,
        alpha=0.5,
    )
    axs[1].plot(
        *evaluation.format_iters(results_running_history_target["dis_loss"]),
        label="Target train",
        ls="-",
        c="tab:orange",
        linewidth=0.5,
        alpha=0.5,
    )
    axs[1].plot(
        results_history_source_val["dis_loss"],
        label="Source val",
        ls="--",
        c="tab:blue",
    )
    axs[1].plot(
        results_history_target_val["dis_loss"],
        label="Target train eval",
        ls="--",
        c="tab:orange",
    )

    axs[1].set_ylim(bottom=0, top=2)
    axs[1].grid(which="major")
    axs[1].minorticks_on()
    axs[1].grid(which="minor", alpha=0.2)

    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Discriminator BCE Loss")
    axs[1].legend()

    # discriminator accuracy
    axs[2].plot(
        *evaluation.format_iters(results_running_history_source["dis_accu"]),
        label="Source train",
        ls="-",
        c="tab:blue",
        linewidth=0.5,
        alpha=0.5,
    )
    axs[2].plot(
        *evaluation.format_iters(results_running_history_target["dis_accu"]),
        label="Target train",
        ls="-",
        c="tab:orange",
        linewidth=0.5,
        alpha=0.5,
    )
    axs[2].plot(
        results_history_source_val["dis_accu"],
        label="Source val",
        ls="--",
        c="tab:blue",
    )
    axs[2].plot(
        results_history_target_val["dis_accu"],
        label="Target train eval",
        ls="--",
        c="tab:orange",
    )

    axs[2].set_ylim([0, 1])
    axs[2].grid(which="major")
    axs[2].minorticks_on()
    axs[2].grid(which="minor", alpha=0.2)

    axs[2].set_xlabel("Epoch")
    axs[2].set_title("Discriminator Accuracy")
    axs[2].legend()

    plt.savefig(os.path.join(save_folder, "adv_train.png"))

    plt.show(block=False)


def load_train_plot(
    pretrain_folder,
    save_folder,
    dataloader_source_train,
    dataloader_source_val,
    dataloader_target_train,
    dataloader_target_val,
    **adv_train_kwargs,
):
    model = DANN(
        sc_mix_d["train"].shape[1],
        ncls_source=lab_mix_d["train"].shape[1],
        **model_params["dann_kwargs"],
    )
    model.apply(initialize_weights)

    if train_params["pretraining"]:
        best_pre_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
        model.load_state_dict(best_pre_checkpoint["model"].state_dict())
    model.to(device)

    model.advtraining()

    if adv_train_kwargs.get("checkpoints", False):
        tqdm.write(repr(model))

    training_history, checkpoint = train_adversarial_iters(
        model,
        save_folder,
        dataloader_source_train,
        dataloader_source_val,
        dataloader_target_train,
        dataloader_target_val,
        **adv_train_kwargs,
    )

    with open(os.path.join(save_folder, "training_history.pkl"), "wb") as f:
        pickle.dump(training_history, f)

    plot_results(*training_history, save_folder)

    return checkpoint


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

        rv_pretrain_folder = os.path.join(save_folder, f"reverse_val-{model_name}-pretrain")
        if not os.path.isdir(rv_pretrain_folder):
            os.makedirs(rv_pretrain_folder)

        if train_params.get("pretraining", False):
            rv_pretrain_folder = os.path.join(save_folder, f"reverse_val-{model_name}-pretrain")
            if not os.path.isdir(rv_pretrain_folder):
                os.makedirs(rv_pretrain_folder)
            model = DANN(
                inp_emb=sc_mix_d["train"].shape[1],
                ncls_source=lab_mix_d["train"].shape[1],
                **model_params["dann_kwargs"],
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

        rv_save_folder = os.path.join(save_folder, f"reverse_val-{model_name}")
        if not os.path.isdir(rv_save_folder):
            os.makedirs(rv_save_folder)

        checkpoint = load_train_plot(
            model,
            rv_save_folder,
            dataloader_target_now_source_d["train"],
            dataloader_target_now_source_d.get("val", dataloader_target_now_source_d["train"]),
            dataloader_source_d["train"],
            dataloader_source_d["val"],
            checkpoints=False,
            epoch_override=epoch + 1,
            iter_override=len(dataloader_target_now_source_d["train"]),
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
        print(f"Adversarial training for ST slide {sample_id}: ")

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

tqdm.write(f"Script run time: {datetime.datetime.now(datetime.timezone.utc) - script_start_time}")
