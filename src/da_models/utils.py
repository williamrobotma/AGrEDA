"""Model utility functions."""
import warnings

import torch
from torch import nn


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


def get_torch_device(cuda_index=None):
    if not torch.cuda.is_available():
        warnings.warn("Using CPU", category=UserWarning, stacklevel=2)
        return torch.device("cpu")
    if cuda_index is not None:
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cuda")
