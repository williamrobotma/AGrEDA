#!/usr/bin/env python3
"""Creating something like CellDART but it actually follows DANN in PyTorch"""

# %% [markdown]
#  # DANN for ST

# %% [markdown]
#  Creating something like CellDART but it actually follows DANN in PyTorch

import argparse

# %%
import datetime
import os
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from tqdm.autonotebook import tqdm

from src.da_models.dann import DANN
from src.da_models.datasets import SpotDataset
from src.da_models.utils import get_torch_device, initialize_weights, set_requires_grad
from src.utils import data_loading, evaluation
from src.utils.output_utils import DupStdout, TempFolderHolder

# datetime object containing current date and time
script_start_time = datetime.datetime.now(datetime.timezone.utc)


# %%
parser = argparse.ArgumentParser(
    description=("Creating something like CellDART but it actually follows DANN in PyTorch")
)
parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers to use for dataloaders."
)
parser.add_argument("--cuda", "-c", default=None, help="gpu index to use")
parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary model directory")
parser.add_argument("--log_fname", "-l", default=None, help="optional log file name")

# %%
# CUDA_INDEX = 2
# NUM_WORKERS = 4
# CONFIG_FNAME = "dann.yml"

args = parser.parse_args()
CONFIG_FNAME = args.config_fname
CUDA_INDEX = args.cuda
NUM_WORKERS = args.num_workers
TMP_DIR = args.tmpdir
LOG_FNAME = args.log_fname

# %%
# torch_params = {}
# torch_params["manual_seed"] = 25098

# CUDA_INDEX = 2
# NUM_WORKERS = 4
# CONFIG_FNAME = "dann.yml"


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
# data_params["scaler_name"] = "standard"

MODEL_NAME = "DANN"


# %%
# model_params = {}

# # Model parameters
# model_params["model_version"] = "Standard1"
# model_params["dann_kwargs"] = {
#     "emb_dim": 64,
#     "alpha_": 1,
# }

# train_params = {}

# train_params["batch_size"] = 512

# # Pretraining parameters
# # SAMPLE_ID_N = "151673"
# train_params["initial_train_lr"] = 0.001
# train_params["initial_train_epochs"] = 100

# train_params["early_stop_crit"] = 100
# train_params["min_epochs"] = 0.4 * train_params["initial_train_epochs"]

# # Adversarial training parameters
# train_params["epochs"] = 500
# train_params["early_stop_crit_adv"] = 500
# train_params["min_epochs_adv"] =  0.4 * 500


# train_params["adv_lr"] = 2e-4
# train_params["lambda"] = 1
# train_params["pretraining"] = False

# train_params["adv_opt_kwargs"] = {"lr": train_params["adv_lr"], "betas": (0.5, 0.999), "eps": 1e-07}

# train_params["plateau_kwargs"] = {
#     "patience": 50,
#     "factor": 0.5,
#     "min_lr": train_params["adv_lr"] / 10,
#     "verbose": True,
# }

# train_params["two_step"] = False


# %%
# config = {
#     "torch_params": torch_params,
#     "data_params": data_params,
#     "model_params": model_params,
#     "train_params": train_params,
# }

# if not os.path.exists(os.path.join("configs", MODEL_NAME)):
#     os.makedirs(os.path.join("configs", MODEL_NAME))

# with open(os.path.join("configs", MODEL_NAME, CONFIG_FNAME), "w") as f:
#     yaml.safe_dump(config, f)

with open(os.path.join("configs", MODEL_NAME, CONFIG_FNAME), "r") as f:
    config = yaml.safe_load(f)


print(yaml.safe_dump(config))

torch_params = config["torch_params"]
data_params = config["data_params"]
model_params = config["model_params"]
train_params = config["train_params"]


# %%
device = get_torch_device(CUDA_INDEX)


# %%
torch_seed = torch_params.get("manual_seed", int(script_start_time.timestamp()))
torch_seed_path = str(torch_seed) if "manual_seed" in torch_params else "random"

torch.manual_seed(torch_seed)
np.random.seed(torch_seed)


# %%
model_folder = data_loading.get_model_rel_path(
    MODEL_NAME,
    model_params["model_version"],
    torch_seed_path=torch_seed_path,
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
mat_sp_d, mat_sp_train, st_sample_id_l = data_loading.load_spatial(selected_dir, **data_params)

# Load sc data
sc_mix_d, lab_mix_d, sc_sub_dict, sc_sub_dict2 = data_loading.load_sc(selected_dir, **data_params)


# %% [markdown]
#  # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#  ## Prepare dataloaders

# %%
### source dataloaders
source_train_set = SpotDataset(sc_mix_d["train"], lab_mix_d["train"])
source_val_set = SpotDataset(sc_mix_d["val"], lab_mix_d["val"])
source_test_set = SpotDataset(sc_mix_d["test"], lab_mix_d["test"])

source_dataloader_kwargs = dict(
    num_workers=NUM_WORKERS, pin_memory=True, batch_size=train_params["batch_size"]
)

dataloader_source_train = torch.utils.data.DataLoader(
    source_train_set, shuffle=True, **source_dataloader_kwargs
)
dataloader_source_val = torch.utils.data.DataLoader(
    source_val_set, shuffle=False, **source_dataloader_kwargs
)
dataloader_source_test = torch.utils.data.DataLoader(
    source_test_set, shuffle=False, **source_dataloader_kwargs
)

### target dataloaders
target_dataloader_kwargs = source_dataloader_kwargs

target_train_set_d = {}
dataloader_target_train_d = {}
if data_params["st_split"]:
    target_val_set_d = {}
    target_test_set_d = {}

    dataloader_target_val_d = {}
    dataloader_target_test_d = {}
    for sample_id in st_sample_id_l:
        target_train_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["train"])
        target_val_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["val"])
        target_test_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["test"])

        dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
            target_train_set_d[sample_id],
            shuffle=True,
            **target_dataloader_kwargs,
        )
        dataloader_target_val_d[sample_id] = torch.utils.data.DataLoader(
            target_val_set_d[sample_id],
            shuffle=False,
            **target_dataloader_kwargs,
        )
        dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
            target_test_set_d[sample_id],
            shuffle=False,
            **target_dataloader_kwargs,
        )

else:
    target_test_set_d = {}
    dataloader_target_test_d = {}
    for sample_id in st_sample_id_l:
        target_train_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["train"])
        dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
            target_train_set_d[sample_id],
            shuffle=True,
            **target_dataloader_kwargs,
        )

        target_test_set_d[sample_id] = SpotDataset(deepcopy(mat_sp_d[sample_id]["test"]))
        dataloader_target_test_d[sample_id] = torch.utils.data.DataLoader(
            target_test_set_d[sample_id],
            shuffle=False,
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
if train_params["pretraining"]:

    pretrain_folder = os.path.join(model_folder, "pretrain")

    model = DANN(
        inp_emb=sc_mix_d["train"].shape[1],
        ncls_source=lab_mix_d["train"].shape[1],
        **model_params["dann_kwargs"],
    )
    model.apply(initialize_weights)
    model.to(device)

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
                early_stop_count = 0

                out_string += "<-- new best val loss"

            tqdm.write(out_string)

            # Save checkpoint every 10
            if epoch % 10 == 0 or epoch >= train_params["initial_train_epochs"] - 1:
                torch.save(checkpoint, os.path.join(pretrain_folder, f"checkpt{epoch}.pth"))

            # check to see if validation loss has plateau'd
            if (
                early_stop_count >= train_params["early_stop_crit"]
                and epoch >= train_params["min_epochs"] - 1
            ):
                tqdm.write(f"Validation loss plateaued after {early_stop_count} at epoch {epoch}")
                torch.save(checkpoint, os.path.join(pretrain_folder, f"earlystop{epoch}.pth"))
                break

            early_stop_count += 1
    inner.close()
    outer.close()
    lr_history_running[-1].append(pre_scheduler.get_last_lr()[-1])

    # Save final model
    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"best_model.pth"))
    torch.save(best_checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))


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

    n_iters = max(len(dataloader_source), len(dataloader_target))

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
):
    model.to(device)
    model.advtraining()

    optimizer = torch.optim.AdamW(model.parameters(), **train_params["adv_opt_kwargs"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **train_params["plateau_kwargs"]
    )

    max_len_dataloader = max(len(dataloader_source_train), len(dataloader_target_train))

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
    log_file_path = os.path.join(save_folder, LOG_FNAME) if LOG_FNAME else None
    with DupStdout().dup_to_file(log_file_path, "w") as dup_stdout:
        # Train
        tqdm.write("Start adversarial training...")
        outer = tqdm(total=train_params["epochs"], desc="Epochs", position=0)
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
        for epoch in range(train_params["epochs"]):
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
                    dataloader_source_val, dataloader_target_train, model
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
                torch.save(checkpoint, os.path.join(save_folder, f"best_model.pth"))
                early_stop_count = 0
                out_string += f"<-- new best {'stable' if dis_stable else 'unstable'} val clf loss"

            tqdm.write(out_string)

            # Save checkpoint every 10
            if epoch % 10 == 0 or epoch >= train_params["epochs"] - 1:
                torch.save(checkpoint, os.path.join(save_folder, f"checkpt{epoch}.pth"))

            # check to see if validation loss has plateau'd
            # if early_stop_count >= EARLY_STOP_CRIT_ADV and epoch > MIN_EPOCHS_ADV - 1:
            #     tqdm.write(
            #         f"Loss plateaued after {early_stop_count} at epoch {epoch}"
            #     )
            #     torch.save(checkpoint, os.path.join(save_folder, f"earlystop_{epoch}.pth"))
            #     break

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
    best_checkpoint = torch.load(os.path.join(save_folder, f"best_model.pth"))
    torch.save(best_checkpoint, os.path.join(save_folder, f"final_model.pth"))

    return (
        results_running_history_source,
        results_running_history_target,
        results_history_source,
        results_history_target,
        results_history_source_val,
        results_history_target_val,
    )


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
        y=max(results_history_source_val["clf_loss"]) * 0.5,
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


# %%
# st_sample_id_l = [SAMPLE_ID_N]


# %%
if data_params["train_using_all_st_samples"]:
    print(f"Adversarial training for all ST slides")
    save_folder = advtrain_folder

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

    tqdm.write(repr(model))

    train_adversarial_iters(
        model,
        save_folder,
        dataloader_source_train,
        dataloader_source_val,
        dataloader_target_train,
    )

else:
    for sample_id in st_sample_id_l:
        print(f"Adversarial training for ST slide {sample_id}: ")

        save_folder = os.path.join(advtrain_folder, sample_id)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

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
        tqdm.write(repr(model))

        training_history = train_adversarial_iters(
            model,
            save_folder,
            dataloader_source_train,
            dataloader_source_val,
            dataloader_target_train_d[sample_id],
        )

        with open(os.path.join(save_folder, "training_history.pkl"), "wb") as f:
            pickle.dump(training_history, f)

        plot_results(*training_history, save_folder)


# %%
with open(os.path.join(model_folder, "config.yml"), "w") as f:
    yaml.safe_dump(config, f)

temp_folder_holder.copy_out()

tqdm.write(f"Script run time: {datetime.datetime.now(datetime.timezone.utc) - script_start_time}")
