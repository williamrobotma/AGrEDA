#!/usr/bin/env python3
"""Reproduce CellDART for ST"""

# %% [markdown]
#  # Reproduce CellDART

# %%
import argparse
import datetime
import glob
import os
from copy import deepcopy
from itertools import chain
import tarfile
from collections import OrderedDict
import shutil

import pandas as pd
import numpy as np
import torch
import yaml
from torch import nn
from tqdm.autonotebook import tqdm

from src.da_models.adda import ADDAST
from src.da_models.model_utils.datasets import SpotDataset
from src.da_models.model_utils.utils import (
    get_torch_device,
    set_requires_grad,
    LibConfig,
    ModelWrapper,
    initialize_weights,
)
from src.da_utils import data_loading
from src.da_utils.output_utils import DupStdout, TempFolderHolder

script_start_time = datetime.datetime.now(datetime.timezone.utc)

parser = argparse.ArgumentParser(description="Reproduce CellDART for ST")
parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
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
NUM_THREADS = args.num_threads

# CONFIG_FNAME = "celldart1_bnfix.yml"
# CUDA_INDEX = None
# NUM_WORKERS = 0
# %%
# lib_params = {}

# lib_params["manual_seed"] = 1205
# lib_params["cuda_i"] = 0


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
MODEL_NAME = "CellDART"
# model_params["model_version"] = "celldart1_bnfix"

# model_params["celldart_kwargs"] = {
#     "emb_dim": 64,
#     "bn_momentum": 0.01,
# }


# %%
# train_params = {}

# train_params["batch_size"] = 512
# train_params["num_workers"] = 16
# # Pretraining parameters
# # SAMPLE_ID_N = "151673"

# train_params["initial_train_epochs"] = 10

# train_params["early_stop_crit"] = 100
# train_params["min_epochs"] = train_params["initial_train_epochs"]

# # Adversarial training parameters
# train_params["early_stop_crit_adv"] = 10
# train_params["min_epochs_adv"] = 10

# train_params["n_iter"] = 3000
# train_params["alpha_lr"] = 5
# train_params["alpha"] = 0.6


# %%
# config = {
#     "lib_params": lib_params,
#     "data_params": data_params,
#     "model_params": model_params,
#     "train_params": train_params,
# }

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
if not "lr" in train_params:
    train_params["lr"] = 0.001
    rewrite_config = True

if rewrite_config:
    with open(os.path.join(CONFIGS_DIR, MODEL_NAME, CONFIG_FNAME), "w") as f:
        yaml.safe_dump(config, f)

tqdm.write(yaml.safe_dump(config))


# %%
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
#

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
    num_workers=NUM_WORKERS, pin_memory=True, batch_size=train_params["batch_size"]
)
source_set_d = {}
dataloader_source_d = {}
for split in sc_mix_d:
    source_set_d[split] = SpotDataset(sc_mix_d[split], lab_mix_d[split])
    dataloader_source_d[split] = torch.utils.data.DataLoader(
        source_set_d[split],
        shuffle=(split == "train"),
        **source_dataloader_kwargs,
    )

target_dataloader_kwargs = dict(
    num_workers=NUM_WORKERS, pin_memory=False, batch_size=train_params["batch_size"]
)

### target dataloaders
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
        target_set_d[split] = SpotDataset(deepcopy(target_d[split]))
        dataloader_target_d[split] = torch.utils.data.DataLoader(
            target_set_d[split],
            shuffle=("train" in split),
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
            target_d[sample_id][split] = v
            target_set_d[sample_id][split] = SpotDataset(deepcopy(target_d[sample_id][split]))

            dataloader_target_d[sample_id][split] = torch.utils.data.DataLoader(
                target_set_d[sample_id][split],
                shuffle=("train" in split),
                **target_dataloader_kwargs,
            )


# %% [markdown]
#  ## Define Model


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


def run_pretrain_epoch(model, dataloader, optimizer=None, inner=None):
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
        lr=train_params["lr"],
        betas=(0.9, 0.999),
        eps=1e-07,
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
        tqdm.write("Start pretrain...", file=dup_stdout)
        outer = tqdm(total=train_params["initial_train_epochs"], desc="Epochs")
        inner = tqdm(total=len(dataloader_source_d["train"]), desc=f"Batch")

        tqdm.write(" Epoch | Train Loss | Val Loss   ", file=dup_stdout)
        tqdm.write("---------------------------------", file=dup_stdout)
        checkpoint = {
            "epoch": -1,
            "model": model,
            "optimizer": pre_optimizer,
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

            tqdm.write(out_string, file=dup_stdout)

            early_stop_count += 1

        model.eval()
        with torch.no_grad():
            curr_loss_train = compute_acc(dataloader_source_train, model)

        tqdm.write(f"Final train loss: {curr_loss_train}", file=dup_stdout)
        outer.close()
        inner.close()
        # Save final model
        torch.save(checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))

        return checkpoint["model"]


pretrain_folder = os.path.join(model_folder, "pretrain")

if not os.path.isdir(pretrain_folder):
    os.makedirs(pretrain_folder)

# %%
model = ADDAST(
    inp_dim=sc_mix_d["train"].shape[1],
    ncls_source=lab_mix_d["train"].shape[1],
    **model_params["celldart_kwargs"],
)


## CellDART uses just one encoder!
model.target_encoder = model.source_encoder

model.apply(initialize_weights)
tqdm.write(repr(model.to(device)))

pretrain(
    pretrain_folder,
    model,
    dataloader_source_d["train"],
    dataloader_source_d["val"],
)


# %% [markdown]
#  ## Adversarial Adaptation

# %%
advtrain_folder = os.path.join(model_folder, "advtrain")

if not os.path.isdir(advtrain_folder):
    os.makedirs(advtrain_folder)


# %%
def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of
    the same size.
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(
            all_examples_indices,
            size=batch_size,
            replace=False,
        )
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


# %%
criterion_dis = nn.CrossEntropyLoss()
criterion_clf_nored = nn.KLDivLoss(reduction="none")


def discrim_loss_accu(x, y_dis, model):
    x = x.to(**to_inp_kwargs)

    emb = model.source_encoder(x)
    y_pred = model.dis(emb)

    loss = criterion_dis(y_pred, y_dis)

    accu = torch.mean((torch.flatten(torch.argmax(y_pred, dim=1)) == y_dis).to(torch.float32)).cpu()

    return loss, accu


def compute_acc_dis(dataloader_source, dataloader_target, model):
    len_target = len(dataloader_target)
    len_source = len(dataloader_source)

    loss_running = []
    accu_running = []
    mean_weights = []
    model.eval()
    model.source_encoder.eval()
    model.dis.eval()
    with torch.no_grad():
        for y_val, dl in zip([1, 0], [dataloader_target, dataloader_source]):
            for _, (x, _) in enumerate(dl):
                y_dis = torch.full((x.shape[0],), y_val, device=device, dtype=torch.long)

                loss, accu = discrim_loss_accu(x, y_dis, model)

                accu_running.append(accu)
                loss_running.append(loss.item())

                # we will weight average by batch size later
                mean_weights.append(len(x))

    return (
        np.average(loss_running, weights=mean_weights),
        np.average(accu_running, weights=mean_weights),
    )


# %%
def train_adversarial(
    model,
    save_folder,
    sc_mix_train_s,
    lab_mix_train_s,
    mat_sp_train_s,
    dataloader_source_train_eval,
    dataloader_target_train_eval,
    dataloader_source_val=None,
    dataloader_target_val=None,
    checkpoints=False,
    n_iter_override=None,
):
    if dataloader_source_val is None:
        dataloader_source_val = dataloader_source_train_eval
    if dataloader_target_val is None:
        dataloader_target_val = dataloader_target_train_eval

    model.to(device)
    model.advtraining()
    model.set_encoder("source")

    S_batches = batch_generator(
        [sc_mix_train_s.copy(), lab_mix_train_s.copy()],
        train_params["batch_size"],
    )
    T_batches = batch_generator(
        [mat_sp_train_s.copy(), np.ones(shape=(len(mat_sp_train_s), 2))],
        train_params["batch_size"],
    )

    enc_optimizer = torch.optim.Adam(
        chain(
            model.source_encoder.parameters(),
            model.dis.parameters(),
            model.clf.parameters(),
        ),
        lr=train_params["lr"],
        betas=(0.9, 0.999),
        eps=1e-07,
    )

    dis_optimizer = torch.optim.Adam(
        chain(model.source_encoder.parameters(), model.dis.parameters()),
        lr=train_params["alpha_lr"] * train_params["lr"],
        betas=(0.9, 0.999),
        eps=1e-07,
    )

    # Initialize lists to store loss and accuracy values
    loss_history_running = []
    log_file_path = os.path.join(save_folder, LOG_FNAME) if LOG_FNAME else None

    n_iter = n_iter_override if n_iter_override is not None else train_params["n_iter"]
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        # Train
        tqdm.write("Start adversarial training...", file=dup_stdout)
        outer = tqdm(total=n_iter, desc="Iterations")

        checkpoint = {
            "epoch": -1,
            "model": model,
            "dis_optimizer": dis_optimizer,
            "enc_optimizer": enc_optimizer,
        }
        if checkpoints:
            torch.save(
                {"model": model.state_dict(), "epoch": -1},
                os.path.join(save_folder, f"checkpt--1.pth"),
            )
        for iters in range(n_iter):
            checkpoint["epoch"] = iters

            model.train()
            model.dis.train()
            model.clf.train()
            model.source_encoder.train()

            x_source, y_true = next(S_batches)
            x_target, _ = next(T_batches)

            ## Train encoder
            set_requires_grad(model.source_encoder, True)
            set_requires_grad(model.clf, True)
            set_requires_grad(model.dis, True)

            # save discriminator weights
            dis_weights = deepcopy(model.dis.state_dict())
            new_dis_weights = {}
            for k in dis_weights:
                # if "num_batches_tracked" not in k:
                new_dis_weights[k] = dis_weights[k]

            dis_weights = new_dis_weights

            (
                x_source,
                x_target,
                y_true,
            ) = (
                torch.as_tensor(x_source, dtype=torch.float32, device=device),
                torch.as_tensor(x_target, dtype=torch.float32, device=device),
                torch.as_tensor(y_true, dtype=torch.float32, device=device),
            )

            x = torch.cat((x_source, x_target))

            # save for discriminator later
            x_d = x.detach()

            # y_dis is the REAL one
            y_dis = torch.cat(
                [
                    torch.zeros(x_source.shape[0], device=device, dtype=torch.long),
                    torch.ones(x_target.shape[0], device=device, dtype=torch.long),
                ]
            )
            y_dis_flipped = 1 - y_dis.detach()

            emb = model.source_encoder(x).view(x.shape[0], -1)

            y_dis_pred = model.dis(emb)
            y_clf_pred = model.clf(emb)

            # we use flipped because we want to confuse discriminator
            loss_dis = criterion_dis(y_dis_pred, y_dis_flipped)

            # Set true = predicted for target samples since we don't know what it is
            y_clf_true = torch.cat((y_true, torch.zeros_like(y_clf_pred[y_true.shape[0] :])))

            # Loss fn does mean over all samples including target
            loss_clf = criterion_clf_nored(y_clf_pred, y_clf_true)
            clf_sample_weights = torch.cat(
                (
                    torch.ones((x_source.shape[0],), device=device),
                    torch.zeros((x_target.shape[0],), device=device),
                )
            )
            # batchmean with sample weights
            loss_clf = (loss_clf.sum(dim=1) * clf_sample_weights).mean()

            # Scale back up loss so mean doesn't include target
            # loss = (x.shape[0] / x_source.shape[0]) * loss_clf + ALPHA * loss_dis
            loss = loss_clf + train_params["alpha"] * loss_dis

            # loss = loss_clf + ALPHA * loss_dis

            enc_optimizer.zero_grad()
            loss.backward()
            enc_optimizer.step()

            model.dis.load_state_dict(dis_weights, strict=False)

            ## Train discriminator
            set_requires_grad(model.source_encoder, True)
            set_requires_grad(model.clf, True)
            set_requires_grad(model.dis, True)

            # save encoder and clf weights
            source_encoder_weights = deepcopy(model.source_encoder.state_dict())
            clf_weights = deepcopy(model.clf.state_dict())

            new_clf_weights = {}
            for k in clf_weights:
                # if "num_batches_tracked" not in k:
                new_clf_weights[k] = clf_weights[k]

            clf_weights = new_clf_weights

            new_source_encoder_weights = {}
            for k in source_encoder_weights:
                # if "num_batches_tracked" not in k:
                new_source_encoder_weights[k] = source_encoder_weights[k]

            source_encoder_weights = new_source_encoder_weights

            emb = model.source_encoder(x_d).view(x_d.shape[0], -1)
            y_pred = model.dis(emb)

            # we use the real domain labels to train discriminator
            loss = criterion_dis(y_pred, y_dis)

            dis_optimizer.zero_grad()
            loss.backward()
            dis_optimizer.step()

            model.clf.load_state_dict(clf_weights, strict=False)
            model.source_encoder.load_state_dict(source_encoder_weights, strict=False)

            # Save checkpoint every 100
            if (iters % 1000 == 999 or iters >= n_iter - 1) and checkpoints:
                torch.save(
                    {"model": model.state_dict()},
                    os.path.join(save_folder, f"checkpt-{iters}.pth"),
                )

                model.eval()
                source_loss = compute_acc(dataloader_source_train_eval, model)
                _, dis_accu = compute_acc_dis(
                    dataloader_source_train_eval,
                    dataloader_target_train_eval,
                    model,
                )

                source_loss_val = compute_acc(dataloader_source_val, model)
                _, dis_accu_val = compute_acc_dis(
                    dataloader_source_val,
                    dataloader_target_val,
                    model,
                )

                # Print the results

                out_string = (
                    f"iters: {iters} "
                    f"source loss: {source_loss:.4f} dis accu: {dis_accu:.4f} -- "
                    f"val source loss: {source_loss_val:.4f} val dis accu: {dis_accu_val:.4f}"
                )
                tqdm.write(
                    out_string,
                    file=dup_stdout,
                )

            outer.update(1)

        outer.close()

    torch.save(checkpoint, os.path.join(save_folder, f"final_model.pth"))

    return checkpoint


# %%
# st_sample_id_l = [SAMPLE_ID_N]


def reverse_val(
    target_d,
    save_folder,
):
    rv_scores_d = OrderedDict()
    for name in sorted(glob.glob(os.path.join(save_folder, f"checkpt-*.pth"))):
        model_name = os.path.basename(name).rstrip(".pth")
        iters = int(model_name[len("checkpt-") :])
        tqdm.write(f"  {model_name}")

        model = ModelWrapper(save_folder, model_name)

        dataloader_target_now_source_d = {}
        pred_target_d = OrderedDict()
        for split in target_d:
            pred_target_d[split] = model.get_predictions(target_d[split])

            dataloader_target_now_source_d[split] = torch.utils.data.DataLoader(
                SpotDataset(target_d[split], pred_target_d[split]),
                shuffle=("train" in split),
                **source_dataloader_kwargs,
            )

        model = ADDAST(
            inp_dim=sc_mix_d["train"].shape[1],
            ncls_source=lab_mix_d["train"].shape[1],
            **model_params["celldart_kwargs"],
        )

        model.to(device)
        ## CellDART uses just one encoder!
        model.target_encoder = model.source_encoder

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

        checkpoint = train_adversarial(
            model,
            rv_save_folder,
            target_d["train"],
            pred_target_d["train"],
            sc_mix_d["train"],
            dataloader_target_now_source_d["train"],
            dataloader_source_d["train"],
            dataloader_target_now_source_d.get("val", dataloader_target_now_source_d["train"]),
            dataloader_source_d.get("val", None),
            checkpoints=False,
            n_iter_override=iters + 1,
        )

        # model = ModelWrapper(model)
        rv_scores_d[iters] = OrderedDict()
        for split in ("train", "val"):
            rv_scores_d[iters][split] = compute_acc(
                dataloader_source_d[split],
                checkpoint["model"],
            )

    rv_scores_df = pd.DataFrame.from_dict(rv_scores_d, orient="index").sort_index()

    tqdm.write("Reverse validation scores: ")
    tqdm.write(repr(rv_scores_df))
    best_iter = rv_scores_df["val"].idxmin()
    tqdm.write(f"Best iter: {best_iter} ({rv_scores_df['val'].min():.4f})")

    rv_scores_df.to_csv(os.path.join(save_folder, f"reverse_val_scores.csv"))

    best_epoch_df = rv_scores_df.loc[[best_iter]]
    best_epoch_df.index = pd.Index([model_params["model_version"]])
    best_epoch_df["best_epoch"] = best_iter
    best_epoch_df["best_epoch_rv"] = best_iter
    best_epoch_df["config_fname"] = CONFIG_FNAME

    tqdm.write("Best Epoch: ")
    tqdm.write(repr(best_epoch_df))

    best_epoch_df.to_csv(os.path.join(save_folder, f"reverse_val_best_epoch.csv"))

    best_checkpoint = torch.load(os.path.join(save_folder, f"checkpt-{best_iter}.pth"))
    final_checkpoint = torch.load(os.path.join(save_folder, f"final_model.pth"))

    model = final_checkpoint["model"]
    model.load_state_dict(best_checkpoint["model"])

    torch.save(
        {"model": model, "epoch": best_iter},
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


# %%
if data_params.get("samp_split", False) or data_params.get("one_model", False):
    if "train" in mat_sp_d:
        tqdm.write(f"Adversarial training for slides {mat_sp_d['train'].keys()}: ")
        save_folder = os.path.join(advtrain_folder, "samp_split")
    else:
        tqdm.write(f"Adversarial training for slides {next(iter(mat_sp_d.values()))}: ")
        save_folder = os.path.join(advtrain_folder, "one_model")

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
    model = best_checkpoint["model"]
    model.to(device)
    model.advtraining()

    train_adversarial(
        model,
        save_folder,
        sc_mix_d["train"],
        lab_mix_d["train"],
        target_d["train"],
        dataloader_source_d["train"],
        dataloader_target_d["train"],
        dataloader_source_d["val"],
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
        for name in glob.glob(os.path.join(save_folder, "checkpt*.pth")):
            tar.add(name, arcname=os.path.basename(name))
            os.unlink(name)
else:
    for sample_id in st_sample_id_l:
        tqdm.write(f"Adversarial training for ST slide {sample_id}: ")

        save_folder = os.path.join(advtrain_folder, sample_id)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        best_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
        model = best_checkpoint["model"]
        model.to(device)
        model.advtraining()

        train_adversarial(
            model,
            save_folder,
            sc_mix_d["train"],
            lab_mix_d["train"],
            target_d["train"],
            dataloader_source_d["train"],
            dataloader_target_d[sample_id]["train"],
            dataloader_source_d["val"],
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
            for name in glob.glob(os.path.join(save_folder, "checkpt*.pth")):
                tar.add(name, arcname=os.path.basename(name))
                os.unlink(name)

# %%
with open(os.path.join(model_folder, "config.yml"), "w") as f:
    yaml.safe_dump(config, f)

temp_folder_holder.copy_out()

tqdm.write(f"Script run time: {datetime.datetime.now(datetime.timezone.utc) - script_start_time}")
