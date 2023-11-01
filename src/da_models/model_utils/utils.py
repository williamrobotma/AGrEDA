"""Model utility functions."""
import glob
import os
import warnings
from dataclasses import dataclass
from typing import Union

import pandas as pd
import torch
import yaml
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import nn

from src.da_models.model_utils.losses import JSD
from src.da_utils import data_loading


def set_requires_grad(model, requires_grad=True):
    """Sets requires_grad for all parameters in model."""
    for param in model.parameters():
        param.requires_grad = requires_grad


def initialize_weights(m):
    """Initialize weights.

    Args:
        m (:obj:, nn.Module): Module to initialize weights for.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def _untorch_metric(metric_):
    def metric(y_pred, y_true):
        return (
            metric_(
                torch.as_tensor(y_pred, dtype=torch.float32),
                torch.as_tensor(y_true, dtype=torch.float32),
            )
            .detach()
            .cpu()
            .numpy()
        )

    return metric


def get_metric_ctp(metric_name):
    """Get metric class.

    Args:
        metric_name (str): Metric name.

    Returns:
        Callable Metric.

    """
    if metric_name == "jsd":
        return _untorch_metric(JSD())
    elif metric_name == "cos":

        def avg_cosine_distance(x, y):
            return paired_cosine_distances(x, y).mean()

        return avg_cosine_distance
    raise ValueError(f"Unknown metric: {metric_name}")


@dataclass
class LibConfig:
    device: torch.device
    cuda_index: Union[int, None] = None


def dict_to_lib_config(args_dict):
    return LibConfig(
        device=get_torch_device(cuda_index=args_dict["cuda"]),
        cuda_index=args_dict["cuda"],
    )


def get_torch_device(cuda_index=None):
    if not torch.cuda.is_available():
        warnings.warn("Using CPU", category=UserWarning, stacklevel=2)
        return torch.device("cpu")
    if cuda_index is not None and 0 <= int(cuda_index) < torch.cuda.device_count():
        warnings.warn(
            f"Invalid torch.device ordinal {cuda_index}", category=UserWarning, stacklevel=2
        )
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cuda")


def get_model(model_dir, name, lib_config):
    model_path = os.path.join(model_dir, f"{name}.pth")
    check_point_da = torch.load(model_path, map_location=lib_config.device)
    model = check_point_da["model"]
    model.to(lib_config.device)
    model.eval()
    return model


def _out_func(x, model, device=None):
    out = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    if isinstance(out, tuple):
        out = out[0]
    return torch.exp(out)


class ModelWrapper:
    @classmethod
    def configurate(cls, lib_config):
        cls.lib_config = lib_config

        if cls.lib_config.cuda_index is not None and f"index={lib_config.cuda_index}" not in repr(
            cls.lib_config.device
        ):
            try:
                actual_index = cls.lib_config.device.idx
            except AttributeError:
                actual_index = None

            warnings.warn(f"Device index mismatch: {actual_index} != {lib_config.cuda_index}")
            lib_config.cuda_index = actual_index

    @classmethod
    def _get_lib_config_device(cls):
        try:
            return cls.lib_config.device
        except AttributeError:
            cls.lib_config = LibConfig(
                device=get_torch_device(cuda_index=None),
                cuda_index=None,
            )

            warnings.warn(
                "PyTorch device not yet configured, "
                f"initializing using {repr(cls.lib_config.device)}",
                UserWarning,
            )

            return cls.lib_config.device

    def __init__(self, model_or_model_dir, name="final_model"):
        if isinstance(model_or_model_dir, str):
            model_path = os.path.join(model_or_model_dir, f"{name}.pth")

            check_point_da = torch.load(model_path, map_location=self._get_lib_config_device())
            try:
                check_point_da["model"].to(self.lib_config.device)
            except AttributeError:
                final_model_path = os.path.join(model_or_model_dir, "final_model.pth")
                final_model_chkpt = torch.load(
                    final_model_path, map_location=self.lib_config.device
                )
                self.model = final_model_chkpt["model"]
                self.model.load_state_dict(check_point_da["model"])
                self.model.to(self.lib_config.device)
            else:
                self.model = check_point_da["model"]
            try:
                self.epoch = check_point_da["epoch"]
            except KeyError:
                self.epoch = check_point_da.get("iters")
        else:
            self.model = model_or_model_dir
            self.model.to(self.lib_config.device)

        self.model.eval()

    def get_predictions(self, input, source_encoder=False):
        if source_encoder:
            self.model.set_encoder("source")
        else:
            self.model.target_inference()

        with torch.no_grad():
            return (
                _out_func(input, self.model, device=self.lib_config.device).detach().cpu().numpy()
            )

    def get_embeddings(self, input, source_encoder=False):
        if source_encoder:
            self.model.set_encoder("source")
        else:
            self.model.target_inference()

        try:
            encoder = self.model.encoder
        except AttributeError:
            if source_encoder:
                encoder = self.model.source_encoder
            else:
                encoder = self.model.target_encoder

        with torch.no_grad():
            return (
                encoder(torch.as_tensor(input, dtype=torch.float32, device=self.lib_config.device))
                .detach()
                .cpu()
                .numpy()
            )


def get_best_params_file(model_name, dset, sc_id, st_id, configs_dir="configs"):
    pattern = os.path.join(
        "model",
        model_name,
        dset,
        f"{sc_id}_{st_id}",
        "**",
        "reverse_val_best_epoch.csv",
    )

    results = []
    for rv_result_path in glob.glob(pattern, recursive=True):
        results.append(pd.read_csv(rv_result_path, index_col=0))

    results_df = pd.concat(results, axis=0)
    best_hp = results_df["val"].idxmin()
    config_fname = results_df.loc[best_hp, "config_fname"]
    with open(os.path.join(configs_dir, model_name, config_fname), "r") as f:
        config = yaml.safe_load(f)

    lib_params = config["lib_params"]
    data_params = config["data_params"]
    model_params = config["model_params"]

    torch_seed = lib_params.get("manual_seed")
    lib_seed_path = str(torch_seed) if "manual_seed" in lib_params else "random"

    model_folder = data_loading.get_model_rel_path(
        model_name,
        model_params["model_version"],
        lib_seed_path=lib_seed_path,
        **data_params,
    )

    model_folder

    model_path = os.path.join(
        "model",
        model_folder,
        "advtrain",
        "samp_split" if data_params.get("samp_split") else "",
        "final_model.pth",
    )
    checkpoint = torch.load(model_path, map_location=ModelWrapper._get_lib_config_device())

    try:
        epoch = checkpoint["epoch"]
    except KeyError:
        epoch = checkpoint.get("iters")

    if int(epoch) != int(results_df.loc[best_hp, "best_epoch"]):
        raise ValueError("Epoch mismatch")

    return config_fname, results_df, best_hp
