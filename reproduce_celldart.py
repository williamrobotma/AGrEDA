#!/usr/bin/env python3

# %% [markdown]
#  # Reproduce CellDART

# %%
import argparse
import os
import datetime
from itertools import chain
from copy import deepcopy
import warnings

from tqdm.autonotebook import tqdm

import yaml
import numpy as np

import torch
from torch import nn

from src.da_models.adda import ADDAST
from src.da_models.datasets import SpotDataset
from src.da_models.utils import set_requires_grad
from src.utils.dupstdout import DupStdout

# from src.utils.data_loading import (
#     load_spatial,
#     load_sc,
#     get_selected_dir,
#     get_model_rel_path,
# )
from src.utils import data_loading

script_start_time = datetime.datetime.now(datetime.timezone.utc)

parser = argparse.ArgumentParser(description="Reproduce CellDART for ST")
parser.add_argument(
    "--config_fname",
    "-f",
    type=str,
    help="Name of the config file to use",
)
parser.add_argument(
    "--njobs",
    type=int,
    default=0,
    help="Number of jobs to use for parallel processing.",
)
parser.add_argument(
    "--cuda",
    "-c",
    default=None,
    help="gpu index to use",
)

# %%
args = parser.parse_args()
CONFIG_FNAME = args.config_fname
CUDA_INDEX = args.cuda
NUM_WORKERS = args.njobs

# CONFIG_FNAME = "celldart1_bnfix.yml"
# CUDA_INDEX = None
# NUM_WORKERS = 0
# %%
# torch_params = {}

# torch_params["manual_seed"] = 1205
# torch_params["cuda_i"] = 0


# %%
# data_params = {}
# # Data path and parameters
# data_params["data_dir"] = "data"
# data_params["train_using_all_st_samples"] = False
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
#     "torch_params": torch_params,
#     "data_params": data_params,
#     "model_params": model_params,
#     "train_params": train_params,
# }

with open(os.path.join("configs", MODEL_NAME, CONFIG_FNAME), "r") as f:
    config = yaml.safe_load(f)

torch_params = config["torch_params"]
data_params = config["data_params"]
model_params = config["model_params"]
train_params = config["train_params"]

if not "pretraining" in train_params:
    train_params["pretraining"] = True
    with open(os.path.join("configs", MODEL_NAME, CONFIG_FNAME), "w") as f:
        yaml.safe_dump(config, f)

print(yaml.safe_dump(config))


# %%
if CUDA_INDEX is not None:
    device = torch.device(f"cuda:{CUDA_INDEX}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    warnings.warn("Using CPU", stacklevel=2)

# %%
if "manual_seed" in torch_params:
    torch_seed = torch_params["manual_seed"]
    torch_seed_path = str(torch_params["manual_seed"])
else:
    torch_seed = int(script_start_time.timestamp())
    # torch_seed_path = script_start_time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
    torch_seed_path = "random"

torch.manual_seed(torch_seed)
np.random.seed(torch_seed)


# %%
model_folder = data_loading.get_model_rel_path(
    MODEL_NAME,
    model_params["model_version"],
    dset=data_params.get("dset", "dlpfc"),
    sc_id=data_params.get("sc_id", data_loading.DEF_SC_ID),
    st_id=data_params.get("st_id", data_loading.DEF_ST_ID),
    n_markers=data_params["n_markers"],
    all_genes=data_params["all_genes"],
    n_mix=data_params["n_mix"],
    n_spots=data_params["n_spots"],
    st_split=data_params["st_split"],
    scaler_name=data_params["scaler_name"],
    torch_seed_path=torch_seed_path,
)
model_folder = os.path.join("model", model_folder)

if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
    print(model_folder)


# %% [markdown]
#  # Data load
#

# %%
selected_dir = data_loading.get_selected_dir(
    data_loading.get_dset_dir(
        data_params["data_dir"], dset=data_params.get("dset", "dlpfc")
    ),
    sc_id=data_params.get("sc_id", data_loading.DEF_SC_ID),
    st_id=data_params.get("st_id", data_loading.DEF_ST_ID),
    n_markers=data_params["n_markers"],
    all_genes=data_params["all_genes"],
)

# Load spatial data
mat_sp_d, mat_sp_train, st_sample_id_l = data_loading.load_spatial(
    selected_dir,
    data_params["scaler_name"],
    train_using_all_st_samples=data_params["train_using_all_st_samples"],
    st_split=data_params["st_split"],
)

# Load sc data
sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2 = data_loading.load_sc(
    selected_dir,
    data_params["scaler_name"],
    n_mix=data_params["n_mix"],
    n_spots=data_params["n_spots"],
)


# %% [markdown]
#  # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#  ## Prepare dataloaders

# %%
### source dataloaders
source_train_set = SpotDataset(
    deepcopy(sc_mix_d["train"]), deepcopy(lab_mix_d["train"])
)
source_val_set = SpotDataset(deepcopy(sc_mix_d["val"]), deepcopy(lab_mix_d["val"]))
source_test_set = SpotDataset(deepcopy(sc_mix_d["test"]), deepcopy(lab_mix_d["test"]))

dataloader_source_train = torch.utils.data.DataLoader(
    source_train_set,
    batch_size=train_params["batch_size"],
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
dataloader_source_val = torch.utils.data.DataLoader(
    source_val_set,
    batch_size=train_params["batch_size"],
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)
dataloader_source_test = torch.utils.data.DataLoader(
    source_test_set,
    batch_size=train_params["batch_size"],
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

# ### target dataloaders
# target_test_set_d = {}
# for sample_id in st_sample_id_l:
#     target_test_set_d[sample_id] = SpotDataset(deepcopy(mat_sp_d["test"][sample_id]))

# dataloader_target_test_d = {}
# for sample_id in st_sample_id_l:
#     dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
#         target_test_set_d[sample_id],
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=False,
#     )

# if TRAIN_USING_ALL_ST_SAMPLES:
#     target_train_set = SpotDataset(deepcopy(mat_sp_train_s))
#     dataloader_target_train = torch.utils.data.DataLoader(
#         target_train_set,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#     )
# else:
#     target_train_set_d = {}
#     dataloader_target_train_d = {}
#     for sample_id in st_sample_id_l:
#         target_train_set_d[sample_id] = SpotDataset(
#             deepcopy(mat_sp_d["train"][sample_id])
#         )
#         dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
#             target_train_set_d[sample_id],
#             batch_size=BATCH_SIZE,
#             shuffle=True,
#             num_workers=NUM_WORKERS,
#             pin_memory=True,
#         )

# if TRAIN_USING_ALL_ST_SAMPLES:
#     target_train_set = SpotDataset(deepcopy(mat_sp_train))
#     dataloader_target_train = torch.utils.data.DataLoader(
#         target_train_set,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#     )

### target dataloaders
target_train_set_d = {}
dataloader_target_train_d = {}
if data_params["st_split"]:
    target_val_set_d = {}
    target_test_set_d = {}

    dataloader_target_val_d = {}
    dataloader_target_test_d = {}
    for sample_id in st_sample_id_l:
        target_train_set_d[sample_id] = SpotDataset(
            deepcopy(mat_sp_d[sample_id]["train"])
        )
        target_val_set_d[sample_id] = SpotDataset(deepcopy(mat_sp_d[sample_id]["val"]))
        target_test_set_d[sample_id] = SpotDataset(
            deepcopy(mat_sp_d[sample_id]["test"])
        )

        dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
            target_train_set_d[sample_id],
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
        dataloader_target_val_d[sample_id] = torch.utils.data.DataLoader(
            target_val_set_d[sample_id],
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )
        dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
            target_test_set_d[sample_id],
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

else:
    target_test_set_d = {}
    dataloader_target_test_d = {}
    for sample_id in st_sample_id_l:
        target_train_set_d[sample_id] = SpotDataset(
            deepcopy(mat_sp_d[sample_id]["train"])
        )
        dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
            target_train_set_d[sample_id],
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        target_test_set_d[sample_id] = SpotDataset(
            deepcopy(mat_sp_d[sample_id]["test"])
        )
        dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
            target_test_set_d[sample_id],
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )


if data_params["train_using_all_st_samples"]:
    target_train_set = SpotDataset(mat_sp_train)
    dataloader_target_train = torch.utils.data.DataLoader(
        target_train_set,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# %% [markdown]
#  ## Define Model

# %%
model = ADDAST(
    inp_dim=sc_mix_d["train"].shape[1],
    ncls_source=lab_mix_d["train"].shape[1],
    **model_params["celldart_kwargs"],
)

## CellDART uses just one encoder!
model.target_encoder = model.source_encoder
print(model.to(device))


# %% [markdown]
#  ## Pretrain

# %%
pretrain_folder = os.path.join(model_folder, "pretrain")

if not os.path.isdir(pretrain_folder):
    os.makedirs(pretrain_folder)


# %%
pre_optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07
)

criterion_clf = nn.KLDivLoss(reduction="batchmean")


# %%
def model_loss(x, y_true, model):
    x = x.to(torch.float32).to(device)
    y_true = y_true.to(torch.float32).to(device)

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


# %%
model.pretraining()


# %%
# Initialize lists to store loss and accuracy values
loss_history = []
loss_history_val = []

loss_history_running = []

# Early Stopping
best_loss_val = np.inf
early_stop_count = 0


with DupStdout().dup_to_file(os.path.join(pretrain_folder, "log.txt"), "w") as f_log:
    # Train
    print("Start pretrain...")
    outer = tqdm(total=train_params["initial_train_epochs"], desc="Epochs", position=0)
    inner = tqdm(total=len(dataloader_source_train), desc=f"Batch", position=1)

    print(" Epoch | Train Loss | Val Loss   ")
    print("---------------------------------")
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
            model, dataloader_source_train, optimizer=pre_optimizer, inner=inner
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
        out_string = (
            f" {epoch:5d} " f"| {loss_history[-1]:<10.8f} " f"| {curr_loss_val:<10.8f} "
        )

        # Save the best weights
        if curr_loss_val < best_loss_val:
            best_loss_val = curr_loss_val
            torch.save(checkpoint, os.path.join(pretrain_folder, f"best_model.pth"))
            early_stop_count = 0

            out_string += "<-- new best val loss"

        tqdm.write(out_string)

        # # Save checkpoint every 10
        # if epoch % 10 == 0 or epoch >= train_params["initial_train_epochs"] - 1:
        #     torch.save(checkpoint, os.path.join(pretrain_folder, f"checkpt{epoch}.pth"))

        # # check to see if validation loss has plateau'd
        # if (
        #     early_stop_count >= train_params["early_stop_crit"]
        #     and epoch >= train_params["min_epochs"] - 1
        # ):
        #     print(
        #         f"Validation loss plateaued after {early_stop_count} at epoch {epoch}"
        #     )
        #     torch.save(
        #         checkpoint, os.path.join(pretrain_folder, f"earlystop{epoch}.pth")
        #     )
        #     break

        early_stop_count += 1

    model.eval()
    with torch.no_grad():
        curr_loss_train = compute_acc(dataloader_source_train, model)

    print(f"Final train loss: {curr_loss_train}")
    # Save final model
    torch.save(checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))


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
            all_examples_indices, size=batch_size, replace=False
        )
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


# %%
criterion_dis = nn.CrossEntropyLoss()
criterion_clf_nored = nn.KLDivLoss(reduction="none")


def discrim_loss_accu(x, y_dis, model):
    x = x.to(torch.float32).to(device)

    emb = model.source_encoder(x)
    y_pred = model.dis(emb)

    loss = criterion_dis(y_pred, y_dis)

    accu = torch.mean(
        (torch.flatten(torch.argmax(y_pred, dim=1)) == y_dis).to(torch.float32)
    ).cpu()

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

                y_dis = torch.full(
                    (x.shape[0],), y_val, device=device, dtype=torch.long
                )

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
):

    model.to(device)
    model.advtraining()
    model.set_encoder("source")

    S_batches = batch_generator(
        [sc_mix_train_s.copy(), lab_mix_train_s.copy()], train_params["batch_size"]
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
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-07,
    )

    dis_optimizer = torch.optim.Adam(
        chain(model.source_encoder.parameters(), model.dis.parameters()),
        lr=train_params["alpha_lr"] * 0.001,
        betas=(0.9, 0.999),
        eps=1e-07,
    )

    # Initialize lists to store loss and accuracy values
    loss_history_running = []
    with DupStdout().dup_to_file(os.path.join(save_folder, "log.txt"), "w") as f_log:
        # Train
        print("Start adversarial training...")
        outer = tqdm(total=train_params["n_iter"], desc="Iterations", position=0)

        checkpoint = {
            "epoch": -1,
            "model": model,
            "dis_optimizer": dis_optimizer,
            "enc_optimizer": enc_optimizer,
        }
        for iters in range(train_params["n_iter"]):
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

            x_source, x_target, y_true, = (
                torch.Tensor(x_source),
                torch.Tensor(x_target),
                torch.Tensor(y_true),
            )
            x_source, x_target, y_true, = (
                x_source.to(device),
                x_target.to(device),
                y_true.to(device),
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
            y_clf_true = torch.cat(
                (y_true, torch.zeros_like(y_clf_pred[y_true.shape[0] :]))
            )

            # Loss fn does mean over all samples including target
            loss_clf = criterion_clf_nored(y_clf_pred, y_clf_true)
            clf_sample_weights = torch.cat(
                (torch.ones((x_source.shape[0],)), torch.zeros((x_target.shape[0],)))
            ).to(device)
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
            if iters % 100 == 99 or iters >= train_params["n_iter"] - 1:
                # torch.save(checkpoint, os.path.join(save_folder, f"checkpt{iters}.pth"))

                model.eval()
                source_loss = compute_acc(dataloader_source_train_eval, model)
                _, dis_accu = compute_acc_dis(
                    dataloader_source_train_eval, dataloader_target_train_eval, model
                )

                # Print the results
                tqdm.write(
                    f"iter: {iters} source loss: {source_loss} dis accu: {dis_accu}"
                )

            outer.update(1)

    torch.save(checkpoint, os.path.join(save_folder, f"final_model.pth"))


# %%
# st_sample_id_l = [SAMPLE_ID_N]


# %%
if data_params["train_using_all_st_samples"]:
    print(f"Adversarial training for all ST slides")
    save_folder = advtrain_folder

    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))
    model = best_checkpoint["model"]
    model.to(device)
    model.advtraining()

    train_adversarial(
        model,
        save_folder,
        sc_mix_d["train"],
        lab_mix_d["train"],
        mat_sp_train,
        dataloader_source_train,
        dataloader_target_train,
    )

else:
    for sample_id in st_sample_id_l:
        print(f"Adversarial training for ST slide {sample_id}: ")

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
            mat_sp_d[sample_id]["train"],
            dataloader_source_train,
            dataloader_target_train_d[sample_id],
        )

# %%
with open(os.path.join(model_folder, "config.yml"), "w") as f:
    yaml.safe_dump(config, f)
