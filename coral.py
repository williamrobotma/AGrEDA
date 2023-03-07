#!/usr/bin/env python3
"""Creating something like CellDART but just using coral loss"""

# %% [markdown]
#   # CORAL for ST

# %% [markdown]
#   Creating something like CellDART but just using coral loss

# %%
import argparse
import datetime
import os
import pickle
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from tqdm.autonotebook import tqdm

from src.da_models.coral import CORAL
from src.da_models.datasets import SpotDataset
from src.da_models.utils import initialize_weights
from src.utils import data_loading
from src.utils.evaluation import coral_loss, format_iters
from src.utils.output_utils import DupStdout, TempFolderHolder

# datetime object containing current date and time
script_start_time = datetime.datetime.now(datetime.timezone.utc)


# %%
parser = argparse.ArgumentParser(
    description="Creating something like CellDART but just using coral loss"
)
parser.add_argument("--config_fname", "-f", type=str, help="Name of the config file to use")
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers to use for dataloaders."
)
parser.add_argument("--cuda", "-c", default=None, help="gpu index to use")
parser.add_argument("--tmpdir", "-d", default=None, help="optional temporary model directory")


# %%
# CONFIG_FNAME = "coral.yml"
# NUM_WORKERS = 16
# CUDA_INDEX = None

args = parser.parse_args()
CONFIG_FNAME = args.config_fname
CUDA_INDEX = args.cuda
NUM_WORKERS = args.num_workers
TMP_DIR = args.tmpdir


# %%
# torch_params = {}

# torch_params["manual_seed"] = 3583


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
    scaler_name=data_params["scaler_name"],
    n_markers=data_params["n_markers"],
    all_genes=data_params["all_genes"],
    n_mix=data_params["n_mix"],
    n_spots=data_params["n_spots"],
    st_split=data_params["st_split"],
    torch_seed_path=torch_seed_path,
)
model_folder = os.path.join("model", model_folder)


temp_folder_holder = TempFolderHolder()
model_folder = temp_folder_holder.set_output_folder(TMP_DIR, model_folder)


# %% [markdown]
#   # Data load

# %%
selected_dir = data_loading.get_selected_dir(
    data_params["data_dir"], data_params["n_markers"], data_params["all_genes"]
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
#   # Training: Adversarial domain adaptation for cell fraction estimation

# %% [markdown]
#   ## Prepare dataloaders

# %%
### source dataloaders
source_train_set = SpotDataset(sc_mix_d["train"], lab_mix_d["train"])
source_val_set = SpotDataset(sc_mix_d["val"], lab_mix_d["val"])
source_test_set = SpotDataset(sc_mix_d["test"], lab_mix_d["test"])

dataloader_source_train = torch.utils.data.DataLoader(
    source_train_set,
    batch_size=train_params["batch_size"],
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
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

### target dataloaders
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
        target_train_set_d[sample_id] = SpotDataset(mat_sp_d[sample_id]["train"])
        dataloader_target_train_d[sample_id] = torch.utils.data.DataLoader(
            target_train_set_d[sample_id],
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False,
        )

        target_test_set_d[sample_id] = SpotDataset(deepcopy(mat_sp_d[sample_id]["test"]))
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
#   ## Define Model

# %%


# %% [markdown]
#   ## Pretrain

# %%
criterion_clf = nn.KLDivLoss(reduction="batchmean")


# %%
def model_loss(x, y_true, model):
    x = x.to(torch.float32).to(device)
    y_true = y_true.to(torch.float32).to(device)

    y_pred, _ = model(x)

    loss = criterion_clf(y_pred, y_true)

    return loss


def run_pretrain_epoch(model, dataloader, optimizer=None, scheduler=None, inner=None):
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
if train_params.get("pretraining", False):

    pretrain_folder = os.path.join(model_folder, "pretrain")

    model = CORAL(
        inp_dim=sc_mix_d["train"].shape[1],
        ncls_source=lab_mix_d["train"].shape[1],
        **model_params["coral_kwargs"],
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
    with DupStdout().dup_to_file(os.path.join(pretrain_folder, "log.txt"), "w") as f_log:
        print("Start pretrain...")
        outer = tqdm(total=train_params["initial_train_epochs"], desc="Epochs", position=0)
        inner = tqdm(total=len(dataloader_source_train), desc=f"Batch", position=1)

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
                dataloader_source_train,
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

    lr_history_running[-1].append(pre_scheduler.get_last_lr()[-1])

    # Save final model
    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"best_model.pth"))
    torch.save(best_checkpoint, os.path.join(pretrain_folder, f"final_model.pth"))


# %%
if train_params.get("pretraining", False):

    best_checkpoint = torch.load(os.path.join(pretrain_folder, f"final_model.pth"))

    best_epoch = best_checkpoint["epoch"]
    best_loss_val = loss_history_val[best_epoch]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 4), layout="constrained")

    axs[0].plot(*format_iters(loss_history_running), label="Training", linewidth=0.5)
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
    iters_by_epoch, lr_history_running_flat = format_iters(lr_history_running, startpoint=True)
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
            y_pred_target, logits_target = model(x_target)
        else:
            y_pred_target, logits_source = model(x_target)
            y_pred_source, logits_target = model(x_source)
    else:
        y_pred, logits = model(torch.cat([x_source, x_target], dim=0))
        y_pred_source, y_pred_target = torch.split(y_pred, [len(x_source), len(x_target)])
        logits_source, logits_target = torch.split(logits, [len(x_source), len(x_target)])

    loss_clf = criterion_clf(y_pred_source, y_source)
    loss_dis = criterion_dis(logits_source, logits_target)
    loss = loss_clf + loss_dis * train_params["lambda"]
    update_weights(optimizer, loss)

    return loss, loss_dis, loss_clf


# def target_step(x_target, model, optimizer):
#     if optimizer is not None:
#         clf_rq_bak = dict(
#             (
#                 (name, param.requires_grad)
#                 for name, param in model.clf.named_parameters()
#             )
#         )

#     y_dis_target, y_dis_target_pred, loss_dis_target = target_pass(x_target, model)
#     loss = loss_dis_target * train_params["lambda"]

#     if optimizer is not None:
#         update_weights(optimizer, loss)
#         for name, param in model.clf.named_parameters():
#             param.requires_grad = clf_rq_bak[name]

#     return y_dis_target, y_dis_target_pred, loss_dis_target


# def source_step(x_source, y_source, model, optimizer):
#     y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source = source_pass(
#         x_source, y_source, model
#     )
#     loss = loss_clf + loss_dis_source * train_params["lambda"]
#     update_weights(optimizer, loss)
#     return y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source


def update_weights(optimizer, loss):
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# def target_pass(x_target, model):
#     y_dis_target = torch.ones(
#         x_target.shape[0], device=device, dtype=x_target.dtype
#     ).view(-1, 1)
#     _, y_dis_target_pred = model(x_target, clf=False)
#     loss_dis_target = criterion_dis(y_dis_target_pred, y_dis_target)
#     return y_dis_target, y_dis_target_pred, loss_dis_target


# def source_pass(x_source, y_source, model):
#     y_dis_source = torch.zeros(
#         x_source.shape[0], device=device, dtype=x_source.dtype
#     ).view(-1, 1)
#     y_clf, y_dis_source_pred = model(x_source, clf=True)
#     loss_clf = criterion_clf(y_clf, y_source)
#     loss_dis_source = criterion_dis(y_dis_source_pred, y_dis_source)
#     return y_dis_source, y_dis_source_pred, loss_clf, loss_dis_source


def run_epoch(
    dataloader_source,
    dataloader_target,
    model,
    tqdm_bar=None,
    scheduler=None,
    **kwargs,
):
    results_running = {
        "clf": {"loss": [], "weights": []},
        "dis": {"loss": [], "weights": []},
        "ovr": {"loss": [], "weights": []},
    }

    if scheduler is not None:
        results_running["ovr"]["lr"] = []
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

        x_source = x_source.to(torch.float32).to(device)
        x_target = x_target.to(torch.float32).to(device)
        y_source = y_source.to(torch.float32).to(device)

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
):
    model.to(device)
    model.advtraining()

    max_len_dataloader = max(len(dataloader_source_train), len(dataloader_target_train))

    optimizer = torch.optim.AdamW(model.parameters(), **train_params["opt_kwargs"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, **train_params["plateau_kwargs"]
    # )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_params["opt_kwargs"]["lr"],
        steps_per_epoch=max_len_dataloader,
        epochs=train_params["epochs"],
    )

    # iters = -(max_len_dataloader // -(1 + DIS_LOOP_FACTOR))  # ceiling divide

    iters_val = max(len(dataloader_source_val), len(dataloader_target_train))

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
    with DupStdout().dup_to_file(os.path.join(save_folder, "log.txt"), "w") as f_log:
        # Train
        print("Start adversarial training...")
        outer = tqdm(total=train_params["epochs"], desc="Epochs", position=0)
        inner1 = tqdm(total=max_len_dataloader, desc=f"Batch", position=1)

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
            inner1.refresh()  # force print final state
            inner1.reset()  # reuse bar

            checkpoint["epoch"] = epoch

            results_running = run_epoch(
                dataloader_source_train,
                dataloader_target_train,
                model,
                tqdm_bar=inner1,
                optimizer=optimizer,
                scheduler=scheduler,
                two_step=train_params.get("two_step", False),
                source_first=train_params.get("source_first", True),
            )

            for goal_k in results_running:
                for metric_k in results_running[goal_k]:
                    results_history[goal_k][metric_k].append(
                        np.average(
                            results_running[goal_k][metric_k],
                            weights=results_running[goal_k]["weights"],
                        )
                    )
            for goal_k in results_running:
                for metric_k in results_running[goal_k]:
                    results_history_running[goal_k][metric_k].append(
                        results_running[goal_k][metric_k]
                    )

            model.eval()
            with torch.no_grad():
                results_val = run_epoch(dataloader_source_val, dataloader_target_train, model)
            for goal_k in results_val:
                for metric_k in results_val[goal_k]:
                    results_history_val[goal_k][metric_k].append(
                        np.average(
                            results_val[goal_k][metric_k],
                            weights=results_val[goal_k]["weights"],
                        )
                    )
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

    return results_history_out


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

    best_coral_loss = results_history_val["dis"]["loss"][best_epoch]
    best_kld_loss = results_history_val["clf"]["loss"][best_epoch]
    best_overall_loss = results_history_val["ovr"]["loss"][best_epoch]

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(9, 12), layout="constrained")

    x_text = best_epoch + (2 if best_epoch < n_epochs * 0.75 else -2)
    ha_text = "left" if best_epoch < n_epochs * 0.75 else "right"

    # Coral
    axs[0].plot(
        *(x_y_coral_iters := format_iters(results_history_running["dis"]["loss"])),
        label="training",
        linewidth=0.5,
    )
    axs[0].plot(results_history_val["dis"]["loss"], label="validation")
    axs[1].axvline(best_epoch, color="tab:green")

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
        *(x_y_kld_iters := format_iters(results_history_running["clf"]["loss"])),
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
        *(x_y_ovr_iters := format_iters(results_history_running["ovr"]["loss"])),
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
    iters_by_epoch, lr_history_running_flat = format_iters(
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
# st_sample_id_l = [SAMPLE_ID_N]


# %%
if data_params["train_using_all_st_samples"]:
    print(f"Adversarial training for all ST slides")
    save_folder = advtrain_folder

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

        print(model)
        results = train_adversarial_iters(
            model,
            save_folder,
            dataloader_source_train,
            dataloader_source_val,
            dataloader_target_train_d[sample_id],
        )
        plot_results(save_folder, results)


# %%
with open(os.path.join(model_folder, "config.yml"), "w") as f:
    yaml.safe_dump(config, f)

temp_folder_holder.copy_out()

# %%
print("Script run time:", datetime.datetime.now(datetime.timezone.utc) - script_start_time)
